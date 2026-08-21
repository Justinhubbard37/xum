import { describe, expect, spyOn, test } from "bun:test";
import type { BlobRef } from "@/common/types/durableEvent";
import { REFINEMENT_INVERSE_BLOB_QUOTA_BYTES } from "@/common/types/refinement";
import { DisposableTempDir } from "@/node/services/tempDir";
import {
  DurableEventJournal,
  sharedDurableEventJournal,
} from "@/node/utils/journal/durableEventJournal";
import { appendRefinementEvent, reclaimExcessRefinementInverseBlobs } from "./refinementJournal";

/** Append one blob-backed restore-files refinement row (put+append locked). */
async function publishInverseRow(
  journal: DurableEventJournal,
  content: string
): Promise<{ ref: BlobRef; size: number }> {
  return await journal.withBlobLock(async () => {
    const { ref, size } = await journal.blobs.put(content);
    await journal.append({
      workspaceId: "ws-refine",
      kind: "refinement",
      data: {
        kind: "memory",
        action: { op: "str_replace", path: "/memories/global/notes.md" },
        inverse: { op: "restore-files", files: [{ path: "/m/notes.md", blobRef: ref }] },
        evidence: { workspaceId: "ws-refine", toolName: "test" },
      },
    });
    return { ref, size };
  });
}

describe("reclaimExcessRefinementInverseBlobs", () => {
  test("quota eviction keeps newest inverse payloads and never re-attempts old deletions", async () => {
    using tmp = new DisposableTempDir("refinement-journal-test");
    const journal = new DurableEventJournal(tmp.path);
    const deleteSpy = spyOn(journal.blobs, "delete");
    // Initialize the per-journal state (recovery sweep over an empty journal)
    // so the fabricated over-quota sizes below drive the incremental path
    // deterministically (payload bytes are tiny; the sweep would stat them).
    await reclaimExcessRefinementInverseBlobs(journal, []);

    const fakeSize = Math.ceil(REFINEMENT_INVERSE_BLOB_QUOTA_BYTES * 0.4);
    const refs: BlobRef[] = [];
    for (let i = 1; i <= 4; i++) {
      const { ref } = await publishInverseRow(journal, `inverse-payload-${i}`);
      refs.push(ref);
      await reclaimExcessRefinementInverseBlobs(journal, [{ ref, size: fakeSize }]);
    }

    // 0.4x quota each: the third publish evicts the first, the fourth evicts
    // the second — and refs already deleted are never re-attempted.
    expect(deleteSpy.mock.calls.map((call) => call[0])).toEqual([refs[0], refs[1]]);
    expect(await journal.blobs.has(refs[0])).toBe(false);
    expect(await journal.blobs.has(refs[1])).toBe(false);
    expect(await journal.blobs.has(refs[2])).toBe(true);
    expect(await journal.blobs.has(refs[3])).toBe(true);
    deleteSpy.mockRestore();
  });

  test("a payload hash shared with another event kind survives eviction", async () => {
    using tmp = new DisposableTempDir("refinement-journal-test");
    const journal = new DurableEventJournal(tmp.path);
    await reclaimExcessRefinementInverseBlobs(journal, []);

    // Identical content stored by a result-handle event: content addressing
    // shares one blob across kinds, so refinement eviction must skip it.
    const { ref: sharedRef } = await journal.publishWithBlob("shared-bytes", (blobHash, size) => ({
      workspaceId: "ws-refine",
      kind: "result-handle",
      data: { handle: "vars.__h1", preview: "p", blobHash, size },
    }));
    const { ref } = await publishInverseRow(journal, "shared-bytes");
    expect(ref).toBe(sharedRef);
    // An over-quota fabricated size makes the shared payload evictable by the
    // quota walk; only reference safety keeps it alive.
    await reclaimExcessRefinementInverseBlobs(journal, [
      { ref, size: REFINEMENT_INVERSE_BLOB_QUOTA_BYTES + 1 },
    ]);
    expect(await journal.blobs.has(sharedRef)).toBe(true);
  });

  test("appendRefinementEvent bounds aggregate inverse bytes (unique large versions loop)", async () => {
    using tmp = new DisposableTempDir("refinement-journal-test");
    // The reported attack: a loop mutating a large file with a changing
    // suffix journals each unique prior version as a blob. Three unique
    // versions of ~0.4x quota cross it on the third edit.
    const versionBytes = Math.ceil(REFINEMENT_INVERSE_BLOB_QUOTA_BYTES * 0.4);
    const journal = sharedDurableEventJournal(tmp.path);
    const priorVersion = (i: number) => `${"v".repeat(versionBytes)}-${i}`;
    for (let i = 1; i <= 3; i++) {
      await appendRefinementEvent({
        sessionDir: tmp.path,
        workspaceId: "ws-refine",
        kind: "memory",
        action: { op: "str_replace", path: "/memories/global/big.md" },
        inverse: {
          op: "restore-files",
          files: [{ path: "/m/big.md", content: priorVersion(i) }],
        },
        evidence: { toolName: "test" },
      });
    }
    const rows = (await journal.read()).filter((e) => e.kind === "refinement");
    expect(rows).toHaveLength(3);
    const refOf = (row: (typeof rows)[number]) =>
      (row.data.inverse as { files: Array<{ blobRef: BlobRef }> }).files[0].blobRef;
    // Rows all survive as audit records; only the oldest payload is evicted.
    expect(await journal.blobs.has(refOf(rows[0]))).toBe(false);
    expect(await journal.blobs.has(refOf(rows[1]))).toBe(true);
    expect(await journal.blobs.has(refOf(rows[2]))).toBe(true);
  });

  test("recovery sweep after a restart evicts over-quota payloads by real blob size", async () => {
    using tmp = new DisposableTempDir("refinement-journal-test");
    // "Process 1" journals two large inverse payloads and crashes before any
    // reclamation (real bytes: two fit only 1x under the quota together).
    const bigBytes = Math.ceil((REFINEMENT_INVERSE_BLOB_QUOTA_BYTES * 2) / 3);
    const journal1 = new DurableEventJournal(tmp.path);
    const older = await publishInverseRow(journal1, "a".repeat(bigBytes));
    const newer = await publishInverseRow(journal1, "b".repeat(bigBytes));

    // "Process 2" (fresh instance = fresh state): the first pass sweeps the
    // journal, stats the blobs (rows record no sizes), and evicts oldest-first.
    const journal2 = new DurableEventJournal(tmp.path);
    await reclaimExcessRefinementInverseBlobs(journal2, []);
    expect(await journal2.blobs.has(older.ref)).toBe(false);
    expect(await journal2.blobs.has(newer.ref)).toBe(true);
  });
});
