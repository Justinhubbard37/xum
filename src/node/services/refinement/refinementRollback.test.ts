import { describe, expect, it } from "bun:test";

import { spawnSync } from "node:child_process";
import * as fsPromises from "node:fs/promises";
import * as path from "node:path";
import { REFINEMENT_INLINE_MAX_CHARS } from "@/common/types/refinement";
import { Config } from "@/node/config";
import { LocalRuntime } from "@/node/runtime/LocalRuntime";
import { MemoryMetaService } from "@/node/services/memoryMeta";
import { MemoryService, type MemoryScopeContext } from "@/node/services/memoryService";
import { TestTempDir } from "@/node/services/tools/testHelpers";
import { sharedDurableEventJournal } from "@/node/utils/journal/durableEventJournal";
import { appendRefinementEvent } from "./refinementJournal";
import {
  acquireRollbackFileLock,
  listRefinements,
  reclaimStaleRollbackLock,
  rollbackRefinement,
  type RefinementEvent,
} from "./refinementRollback";

function pathExists(target: string): Promise<boolean> {
  return fsPromises.access(target).then(
    () => true,
    () => false
  );
}

interface RollbackFixture extends Disposable {
  muxHome: string;
  checkout: string;
  sessionDir: string;
  service: MemoryService;
  ctx: MemoryScopeContext;
}

const WORKSPACE_ID = "ws-rollback";
const EVIDENCE = { toolName: "test" };

/** Real MemoryService against a temp mux home: rollbacks consume real r2 rows. */
async function createFixture(): Promise<RollbackFixture> {
  const tempDir = new TestTempDir("test-refinement-rollback");
  const muxHome = path.join(tempDir.path, "mux-home");
  const checkout = path.join(tempDir.path, "checkout");
  await fsPromises.mkdir(muxHome, { recursive: true });
  await fsPromises.mkdir(checkout, { recursive: true });
  const config = new Config(muxHome);
  const service = new MemoryService(config, new MemoryMetaService(muxHome));
  return {
    muxHome,
    checkout,
    sessionDir: config.getSessionDir(WORKSPACE_ID),
    service,
    ctx: {
      runtime: new LocalRuntime(checkout),
      checkoutCwd: checkout,
      workspaceId: WORKSPACE_ID,
      projectPath: "/stable/project-id",
    },
    [Symbol.dispose]() {
      tempDir[Symbol.dispose]();
    },
  };
}

async function lastRow(sessionDir: string): Promise<RefinementEvent> {
  const rows = await listRefinements(sessionDir);
  expect(rows.length).toBeGreaterThan(0);
  return rows[rows.length - 1];
}

describe("refinementRollback", () => {
  it("create → edit → rollback restores byte-identical prior content (inline)", async () => {
    using fixture = await createFixture();
    const prior = "# Notes\n\noriginal content with unicode: ünïcödé ✓\n";
    await fixture.service.create(fixture.ctx, "/memories/global/notes.md", prior, "agent");
    await fixture.service.strReplace(
      fixture.ctx,
      "/memories/global/notes.md",
      "original content",
      "edited content",
      "agent"
    );
    const editRow = await lastRow(fixture.sessionDir);
    const physicalPath = path.join(fixture.muxHome, "memory", "global", "notes.md");
    expect(await fsPromises.readFile(physicalPath, "utf-8")).toContain("edited content");

    const result = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: editRow.id,
      evidence: EVIDENCE,
    });
    expect(result.success).toBe(true);
    expect(await fsPromises.readFile(physicalPath, "utf-8")).toBe(prior);
  });

  it("restores blob-backed prior content byte-identically and journals rollbackOf", async () => {
    using fixture = await createFixture();
    // Above the inline cap → the r2 inverse offloads prior content to a blob.
    const prior = `start\n${"x".repeat(REFINEMENT_INLINE_MAX_CHARS + 100)}\nend\n`;
    await fixture.service.create(fixture.ctx, "/memories/global/big.md", prior, "agent");
    await fixture.service.strReplace(fixture.ctx, "/memories/global/big.md", "start", "s", "agent");
    const editRow = await lastRow(fixture.sessionDir);
    const inverse = editRow.data.inverse as { op: string; files: Array<{ blobRef?: string }> };
    expect(inverse.files[0].blobRef).toBeDefined();

    const result = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: editRow.id,
      evidence: EVIDENCE,
    });
    expect(result.success).toBe(true);
    const physicalPath = path.join(fixture.muxHome, "memory", "global", "big.md");
    expect(await fsPromises.readFile(physicalPath, "utf-8")).toBe(prior);

    const rollbackRow = await lastRow(fixture.sessionDir);
    expect(rollbackRow.data.rollbackOf).toBe(editRow.id);
    expect(rollbackRow.data.kind).toBe("memory");
    expect(rollbackRow.data.action).toMatchObject({ op: "rollback", of: editRow.id });
    if (!result.success) throw new Error("unreachable");
    expect(result.data.rollbackRowId).toBe(rollbackRow.id);
  });

  it("aborts a multi-file restore before any write when a blob is missing", async () => {
    using fixture = await createFixture();
    // Two files under one memory dir; the big one's captured content is
    // blob-backed in the delete row's inverse. Sorted capture order puts
    // a-small.md first, so a sequential apply would restore it before the
    // blob failure.
    await fixture.service.create(fixture.ctx, "/memories/global/notes/a-small.md", "sm\n", "agent");
    const big = "x".repeat(REFINEMENT_INLINE_MAX_CHARS + 100);
    await fixture.service.create(fixture.ctx, "/memories/global/notes/z-big.md", big, "agent");
    await fixture.service.deletePath(fixture.ctx, "/memories/global/notes", "agent");
    const deleteRow = await lastRow(fixture.sessionDir);
    const inverse = deleteRow.data.inverse as {
      op: string;
      files: Array<{ path: string; blobRef?: string }>;
    };
    const blobbed = inverse.files.find((file) => file.blobRef !== undefined);
    expect(blobbed?.blobRef).toBeDefined();
    // Corrupt the journal: drop the blob payload backing z-big.md
    // (blobs live at blobs/<hash[0:2]>/<hash> with the ref's sha256: prefix stripped).
    const hash = blobbed!.blobRef!.slice("sha256:".length);
    await fsPromises.rm(path.join(fixture.sessionDir, "blobs", hash.slice(0, 2), hash));

    const result = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: deleteRow.id,
      evidence: EVIDENCE,
    });
    expect(result.success).toBe(false);
    if (result.success) throw new Error("unreachable");
    expect(result.error).toContain("Blob");
    // Phase 1 failed before any write: the small file must NOT be restored...
    const smallPath = path.join(fixture.muxHome, "memory", "global", "notes", "a-small.md");
    expect(await pathExists(smallPath)).toBe(false);
    // ...and no rollback row was appended.
    const rows = await listRefinements(fixture.sessionDir);
    expect(rows.some((row) => row.data.rollbackOf === deleteRow.id)).toBe(false);
  });

  it("compensates already-written files when a multi-file restore fails midway", async () => {
    using fixture = await createFixture();
    await fixture.service.create(fixture.ctx, "/memories/global/notes/a/first.md", "1\n", "agent");
    await fixture.service.create(fixture.ctx, "/memories/global/notes/z/second.md", "2\n", "agent");
    await fixture.service.deletePath(fixture.ctx, "/memories/global/notes", "agent");
    const deleteRow = await lastRow(fixture.sessionDir);

    // Sabotage the SECOND destination: a regular file where its parent dir
    // must be created makes phase 2 fail after the first file was written.
    const notesDir = path.join(fixture.muxHome, "memory", "global", "notes");
    await fsPromises.mkdir(notesDir, { recursive: true });
    await fsPromises.writeFile(path.join(notesDir, "z"), "not a dir\n", "utf-8");

    const result = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: deleteRow.id,
      evidence: EVIDENCE,
    });
    expect(result.success).toBe(false);
    // Compensation removed the already-restored first file (it was absent
    // pre-rollback), so a later retry sees no divergence from this failure.
    expect(await pathExists(path.join(notesDir, "a", "first.md"))).toBe(false);
    const rows = await listRefinements(fixture.sessionDir);
    expect(rows.some((row) => row.data.rollbackOf === deleteRow.id)).toBe(false);
  });

  it("refuses a double rollback of the same id, but allows rolling back the rollback", async () => {
    using fixture = await createFixture();
    await fixture.service.create(fixture.ctx, "/memories/global/a.md", "v1\n", "agent");
    await fixture.service.strReplace(fixture.ctx, "/memories/global/a.md", "v1", "v2", "agent");
    const editRow = await lastRow(fixture.sessionDir);
    const first = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: editRow.id,
      evidence: EVIDENCE,
    });
    expect(first.success).toBe(true);

    const second = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: editRow.id,
      evidence: EVIDENCE,
    });
    expect(second.success).toBe(false);
    if (second.success) throw new Error("unreachable");
    expect(second.error).toContain("already rolled back");

    // Rolling back the rollback re-applies the edit (double inversion).
    const rollbackRow = await lastRow(fixture.sessionDir);
    const undo = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: rollbackRow.id,
      evidence: EVIDENCE,
    });
    expect(undo.success).toBe(true);
    const physicalPath = path.join(fixture.muxHome, "memory", "global", "a.md");
    expect(await fsPromises.readFile(physicalPath, "utf-8")).toBe("v2\n");
  });

  it("refuses on divergence (file deleted since the edit) and applies with force", async () => {
    using fixture = await createFixture();
    await fixture.service.create(fixture.ctx, "/memories/global/gone.md", "v1\n", "agent");
    await fixture.service.strReplace(fixture.ctx, "/memories/global/gone.md", "v1", "v2", "agent");
    const editRow = await lastRow(fixture.sessionDir);
    const physicalPath = path.join(fixture.muxHome, "memory", "global", "gone.md");
    // Out-of-band deletion: the inverse expects the edited file to exist.
    await fsPromises.rm(physicalPath);

    const refused = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: editRow.id,
      evidence: EVIDENCE,
    });
    expect(refused.success).toBe(false);
    if (refused.success) throw new Error("unreachable");
    expect(refused.error).toContain("diverges");
    expect(refused.error).toContain(physicalPath);
    expect(await pathExists(physicalPath)).toBe(false);

    const forced = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: editRow.id,
      force: true,
      evidence: EVIDENCE,
    });
    expect(forced.success).toBe(true);
    expect(await fsPromises.readFile(physicalPath, "utf-8")).toBe("v1\n");
  });

  it("refuses when the file was manually edited after the refinement, applies with force", async () => {
    using fixture = await createFixture();
    await fixture.service.create(fixture.ctx, "/memories/global/hand.md", "v1\n", "agent");
    await fixture.service.strReplace(fixture.ctx, "/memories/global/hand.md", "v1", "v2", "agent");
    const editRow = await lastRow(fixture.sessionDir);
    const physicalPath = path.join(fixture.muxHome, "memory", "global", "hand.md");
    // Out-of-band edit (user editor, other workspace): the file still exists,
    // so presence checks pass — only the recorded postState hash detects it.
    await fsPromises.writeFile(physicalPath, "manually edited\n", "utf-8");

    const refused = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: editRow.id,
      evidence: EVIDENCE,
    });
    expect(refused.success).toBe(false);
    if (refused.success) throw new Error("unreachable");
    expect(refused.error).toContain("modified after the target refinement");
    // The manual edit is untouched by a refused rollback.
    expect(await fsPromises.readFile(physicalPath, "utf-8")).toBe("manually edited\n");

    const forced = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: editRow.id,
      force: true,
      evidence: EVIDENCE,
    });
    expect(forced.success).toBe(true);
    expect(await fsPromises.readFile(physicalPath, "utf-8")).toBe("v1\n");
  });

  it("fails while the cross-process lockfile is held by a live process", async () => {
    using fixture = await createFixture();
    await fixture.service.create(fixture.ctx, "/memories/global/lock.md", "v1\n", "agent");
    await fixture.service.strReplace(fixture.ctx, "/memories/global/lock.md", "v1", "v2", "agent");
    const editRow = await lastRow(fixture.sessionDir);

    // Simulate another process's in-flight rollback: our own PID is live, so
    // the lock must never be broken and the call must fail with a clear error.
    const lockPath = path.join(fixture.sessionDir, "refinement-rollback.lock");
    await fsPromises.writeFile(lockPath, String(process.pid), { encoding: "utf-8", flag: "wx" });

    const refused = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: editRow.id,
      evidence: EVIDENCE,
    });
    expect(refused.success).toBe(false);
    if (refused.success) throw new Error("unreachable");
    expect(refused.error).toContain("Another rollback is in progress");
    // The live owner's lockfile survives the refusal.
    expect(await fsPromises.readFile(lockPath, "utf-8")).toBe(String(process.pid));

    await fsPromises.unlink(lockPath);
    const retried = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: editRow.id,
      evidence: EVIDENCE,
    });
    expect(retried.success).toBe(true);
  });

  it("reclaims a stale lockfile whose owner is provably dead", async () => {
    using fixture = await createFixture();
    await fixture.service.create(fixture.ctx, "/memories/global/stale.md", "v1\n", "agent");
    await fixture.service.strReplace(fixture.ctx, "/memories/global/stale.md", "v1", "v2", "agent");
    const editRow = await lastRow(fixture.sessionDir);

    // A short-lived child that has already exited gives a provably dead PID
    // (ESRCH from kill(pid, 0)); crash remnants must not block rollbacks.
    const child = spawnSync(process.execPath, ["--version"]);
    expect(child.pid).toBeGreaterThan(0);
    const lockPath = path.join(fixture.sessionDir, "refinement-rollback.lock");
    await fsPromises.writeFile(lockPath, String(child.pid), { encoding: "utf-8", flag: "wx" });

    const result = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: editRow.id,
      evidence: EVIDENCE,
    });
    expect(result.success).toBe(true);
    // The reclaimed lock was released after the rollback.
    expect(await pathExists(lockPath)).toBe(false);
  });

  it("reclaim cannot unlink a live lock created after the stale read (double-reclaim race)", async () => {
    using fixture = await createFixture();
    await fixture.service.create(fixture.ctx, "/memories/global/race2.md", "v1\n", "agent");
    await fixture.service.strReplace(fixture.ctx, "/memories/global/race2.md", "v1", "v2", "agent");
    const editRow = await lastRow(fixture.sessionDir);
    const lockPath = path.join(fixture.sessionDir, "refinement-rollback.lock");

    // Reclaimer A observed a stale (dead-owner) lock...
    const child = spawnSync(process.execPath, ["--version"]);
    const deadToken = `${child.pid}:dead-owner-uuid`;
    // ...but before A's reclaim executes, a competitor finished its own
    // reclaim and acquired a fresh LIVE lock at the same pathname (the
    // interleaving that made unconditional unlink destroy the live lock).
    const liveToken = `${process.pid}:live-owner-uuid`;
    await fsPromises.writeFile(lockPath, liveToken, { encoding: "utf-8", flag: "wx" });

    // A's reclaim re-reads under the guard, detects the token mismatch, and
    // must abort without ever touching the canonical path (the rename-aside
    // design displaced B's live lock here, letting a third wx-create enter
    // the critical section alongside B).
    let threw: unknown = null;
    try {
      await reclaimStaleRollbackLock(lockPath, deadToken, `${process.pid}:reclaimer-a-uuid`);
    } catch (error) {
      threw = error;
    }
    expect(String(threw)).toContain("changed owners mid-reclaim");
    // B's live lock is untouched at the canonical pathname, byte-identical...
    expect(await fsPromises.readFile(lockPath, "utf-8")).toBe(liveToken);
    // ...with no guard or renamed-aside residue left behind.
    const residue = (await fsPromises.readdir(fixture.sessionDir)).filter((name) =>
      name.includes(".reclaim")
    );
    expect(residue).toEqual([]);

    // The restored live lock (live PID) still refuses a full rollback.
    const refused = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: editRow.id,
      evidence: EVIDENCE,
    });
    expect(refused.success).toBe(false);
    if (refused.success) throw new Error("unreachable");
    expect(refused.error).toContain("Another rollback is in progress");
  });

  it("release leaves the lockfile alone when its token no longer matches", async () => {
    using fixture = await createFixture();
    // Materialize the session dir (acquire creates it, but be explicit).
    await fsPromises.mkdir(fixture.sessionDir, { recursive: true });
    const lockPath = path.join(fixture.sessionDir, "refinement-rollback.lock");

    const lock = await acquireRollbackFileLock(fixture.sessionDir);
    // Simulate a wrongful reclaim while we hold the lock: the pathname now
    // carries another acquisition's token.
    const foreignToken = `${process.pid}:foreign-uuid`;
    await fsPromises.writeFile(lockPath, foreignToken, "utf-8");

    await lock[Symbol.asyncDispose]();
    // Ownership-verified release must not unlink the new owner's lock.
    expect(await fsPromises.readFile(lockPath, "utf-8")).toBe(foreignToken);

    // Sanity: a matching token still releases (same acquire/dispose path).
    await fsPromises.unlink(lockPath);
    const lock2 = await acquireRollbackFileLock(fixture.sessionDir);
    await lock2[Symbol.asyncDispose]();
    expect(await pathExists(lockPath)).toBe(false);
  });

  it("reclaims a plain stale lock under the guard and claims it atomically", async () => {
    using fixture = await createFixture();
    await fsPromises.mkdir(fixture.sessionDir, { recursive: true });
    const lockPath = path.join(fixture.sessionDir, "refinement-rollback.lock");

    const child = spawnSync(process.execPath, ["--version"]);
    const deadToken = `${child.pid}:dead-owner-uuid`;
    await fsPromises.writeFile(lockPath, deadToken, { encoding: "utf-8", flag: "wx" });

    const myToken = `${process.pid}:reclaimer-uuid`;
    expect(await reclaimStaleRollbackLock(lockPath, deadToken, myToken)).toBe(true);
    // The canonical lock now carries the reclaimer's token; the guard is gone.
    expect(await fsPromises.readFile(lockPath, "utf-8")).toBe(myToken);
    expect(await pathExists(`${lockPath}.reclaim-guard`)).toBe(false);
  });

  it("a crash-remnant reclaim guard (dead PID) does not deadlock reclamation", async () => {
    using fixture = await createFixture();
    await fixture.service.create(fixture.ctx, "/memories/global/guard.md", "v1\n", "agent");
    await fixture.service.strReplace(fixture.ctx, "/memories/global/guard.md", "v1", "v2", "agent");
    const editRow = await lastRow(fixture.sessionDir);
    const lockPath = path.join(fixture.sessionDir, "refinement-rollback.lock");

    // A crashed reclaimer left BOTH files behind: a stale canonical lock and
    // a stale guard. The guard must be reclaimed one level deep by the same
    // dead-PID rule instead of wedging every future rollback.
    const child = spawnSync(process.execPath, ["--version"]);
    await fsPromises.writeFile(lockPath, `${child.pid}:dead-lock-uuid`, {
      encoding: "utf-8",
      flag: "wx",
    });
    await fsPromises.writeFile(`${lockPath}.reclaim-guard`, `${child.pid}:dead-guard-uuid`, {
      encoding: "utf-8",
      flag: "wx",
    });

    const result = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: editRow.id,
      evidence: EVIDENCE,
    });
    expect(result.success).toBe(true);
    // Both remnants were cleaned up by the successful acquisition + release.
    expect(await pathExists(lockPath)).toBe(false);
    expect(await pathExists(`${lockPath}.reclaim-guard`)).toBe(false);
  });

  it("a live reclaim guard fails reclamation conservatively", async () => {
    using fixture = await createFixture();
    await fsPromises.mkdir(fixture.sessionDir, { recursive: true });
    const lockPath = path.join(fixture.sessionDir, "refinement-rollback.lock");

    const child = spawnSync(process.execPath, ["--version"]);
    const deadToken = `${child.pid}:dead-owner-uuid`;
    await fsPromises.writeFile(lockPath, deadToken, { encoding: "utf-8", flag: "wx" });
    // Another process's reclamation is in flight (live guard owner).
    const liveGuardToken = `${process.pid}:live-guard-uuid`;
    await fsPromises.writeFile(`${lockPath}.reclaim-guard`, liveGuardToken, {
      encoding: "utf-8",
      flag: "wx",
    });

    let threw: unknown = null;
    try {
      await reclaimStaleRollbackLock(lockPath, deadToken, `${process.pid}:reclaimer-uuid`);
    } catch (error) {
      threw = error;
    }
    expect(String(threw)).toContain("reclamation is in progress");
    // Neither the canonical lock nor the live guard was touched.
    expect(await fsPromises.readFile(lockPath, "utf-8")).toBe(deadToken);
    expect(await fsPromises.readFile(`${lockPath}.reclaim-guard`, "utf-8")).toBe(liveGuardToken);
  });

  it("commit-point ownership loss aborts, compensates mutations, and appends no row", async () => {
    using fixture = await createFixture();
    await fixture.service.create(fixture.ctx, "/memories/global/entry.md", "v1\n", "agent");
    await fixture.service.strReplace(fixture.ctx, "/memories/global/entry.md", "v1", "v2", "agent");
    const editRow = await lastRow(fixture.sessionDir);
    const physicalPath = path.join(fixture.muxHome, "memory", "global", "entry.md");
    const lockPath = path.join(fixture.sessionDir, "refinement-rollback.lock");

    // Simulate the theoretical double-entry: another process wrongly judged
    // us dead and reclaimed the canonical lock AFTER our mutation but before
    // our journal append. The commit-point re-check must catch it.
    const result = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: editRow.id,
      evidence: EVIDENCE,
      testOnlyBeforeCommit: async () => {
        // Mutation already applied at this point (v1 back on disk).
        expect(await fsPromises.readFile(physicalPath, "utf-8")).toBe("v1\n");
        await fsPromises.writeFile(lockPath, `${process.pid}:foreign-uuid`, "utf-8");
      },
    });
    expect(result.success).toBe(false);
    if (result.success) throw new Error("unreachable");
    expect(result.error).toContain("lost ownership");
    // The losing entrant compensated: the file is back to its post-edit state...
    expect(await fsPromises.readFile(physicalPath, "utf-8")).toBe("v2\n");
    // ...and no rollbackOf row was committed.
    const rows = await listRefinements(fixture.sessionDir);
    expect(rows.some((row) => row.data.rollbackOf === editRow.id)).toBe(false);

    // With the foreign lock removed, a clean retry sees no divergence.
    await fsPromises.unlink(lockPath);
    const retry = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: editRow.id,
      evidence: EVIDENCE,
    });
    expect(retry.success).toBe(true);
    expect(await fsPromises.readFile(physicalPath, "utf-8")).toBe("v1\n");
  });

  it("serializes concurrent rollbacks of the same row: one succeeds, one rollbackOf row", async () => {
    using fixture = await createFixture();
    await fixture.service.create(fixture.ctx, "/memories/global/race.md", "v1\n", "agent");
    await fixture.service.strReplace(fixture.ctx, "/memories/global/race.md", "v1", "v2", "agent");
    const editRow = await lastRow(fixture.sessionDir);

    // Model tool + debug CLI (or two tool invocations) racing on the same row:
    // without the per-session lock both pass the already-rolled-back check.
    const opts = { sessionDir: fixture.sessionDir, id: editRow.id, evidence: EVIDENCE };
    const results = await Promise.all([rollbackRefinement(opts), rollbackRefinement(opts)]);

    const successes = results.filter((result) => result.success);
    const failures = results.filter((result) => !result.success);
    expect(successes).toHaveLength(1);
    expect(failures).toHaveLength(1);
    if (failures[0].success) throw new Error("unreachable");
    expect(failures[0].error).toContain("already rolled back");

    const rollbackRows = (await listRefinements(fixture.sessionDir)).filter(
      (row) => row.data.rollbackOf === editRow.id
    );
    expect(rollbackRows).toHaveLength(1);
    const physicalPath = path.join(fixture.muxHome, "memory", "global", "race.md");
    expect(await fsPromises.readFile(physicalPath, "utf-8")).toBe("v1\n");
  });

  it("refuses when a later refinement row touched the same path (roll back newest first)", async () => {
    using fixture = await createFixture();
    await fixture.service.create(fixture.ctx, "/memories/global/stack.md", "v1\n", "agent");
    const createRow = await lastRow(fixture.sessionDir);
    await fixture.service.strReplace(fixture.ctx, "/memories/global/stack.md", "v1", "v2", "agent");

    const refused = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: createRow.id,
      evidence: EVIDENCE,
    });
    expect(refused.success).toBe(false);
    if (refused.success) throw new Error("unreachable");
    expect(refused.error).toContain("later refinement row");
  });

  it("unrolls multiple edits LIFO without force once later rows are rolled back", async () => {
    using fixture = await createFixture();
    await fixture.service.create(fixture.ctx, "/memories/global/lifo.md", "v1\n", "agent");
    await fixture.service.strReplace(fixture.ctx, "/memories/global/lifo.md", "v1", "v2", "agent");
    const edit1 = await lastRow(fixture.sessionDir);
    await fixture.service.strReplace(fixture.ctx, "/memories/global/lifo.md", "v2", "v3", "agent");
    const edit2 = await lastRow(fixture.sessionDir);

    const newest = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: edit2.id,
      evidence: EVIDENCE,
    });
    expect(newest.success).toBe(true);
    // edit2 is rolled back (and its rollback row rewound past nothing older),
    // so unrolling edit1 next must not flag divergence or require force.
    const older = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: edit1.id,
      evidence: EVIDENCE,
    });
    expect(older.success).toBe(true);
    const physicalPath = path.join(fixture.muxHome, "memory", "global", "lifo.md");
    expect(await fsPromises.readFile(physicalPath, "utf-8")).toBe("v1\n");
  });

  it("still refuses when a rolled-back rollback re-applied a later edit", async () => {
    using fixture = await createFixture();
    await fixture.service.create(fixture.ctx, "/memories/global/reapply.md", "v1\n", "agent");
    await fixture.service.strReplace(
      fixture.ctx,
      "/memories/global/reapply.md",
      "v1",
      "v2",
      "agent"
    );
    const edit1 = await lastRow(fixture.sessionDir);
    await fixture.service.strReplace(
      fixture.ctx,
      "/memories/global/reapply.md",
      "v2",
      "v3",
      "agent"
    );
    const edit2 = await lastRow(fixture.sessionDir);

    const undo = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: edit2.id,
      evidence: EVIDENCE,
    });
    expect(undo.success).toBe(true);
    const undoRow = await lastRow(fixture.sessionDir);
    // Roll back the rollback: edit2's content ("v3") is live on disk again.
    const redo = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: undoRow.id,
      evidence: EVIDENCE,
    });
    expect(redo.success).toBe(true);

    const refused = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: edit1.id,
      evidence: EVIDENCE,
    });
    expect(refused.success).toBe(false);
    if (refused.success) throw new Error("unreachable");
    expect(refused.error).toContain("later refinement row");
  });

  it("rolls back a skill write via the delete-files inverse and back again", async () => {
    using fixture = await createFixture();
    const skillFile = path.join(fixture.checkout, ".mux", "skills", "my-skill", "SKILL.md");
    const content = "---\nname: my-skill\n---\n\nbody\n";
    await fsPromises.mkdir(path.dirname(skillFile), { recursive: true });
    await fsPromises.writeFile(skillFile, content, "utf-8");
    // Same emitter the skill tools use: a write that created the file journals
    // a delete-files inverse.
    await appendRefinementEvent({
      sessionDir: fixture.sessionDir,
      workspaceId: WORKSPACE_ID,
      kind: "skill",
      action: { op: "write", skillName: "my-skill", filePath: "SKILL.md" },
      inverse: { op: "delete-files", paths: [skillFile] },
      evidence: { toolName: "agent_skill_write" },
    });
    const writeRow = await lastRow(fixture.sessionDir);

    const result = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: writeRow.id,
      evidence: EVIDENCE,
    });
    expect(result.success).toBe(true);
    expect(await pathExists(skillFile)).toBe(false);

    // The rollback row restores the deleted file byte-identically.
    const rollbackRow = await lastRow(fixture.sessionDir);
    expect(rollbackRow.data.rollbackOf).toBe(writeRow.id);
    const undo = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: rollbackRow.id,
      evidence: EVIDENCE,
    });
    expect(undo.success).toBe(true);
    expect(await fsPromises.readFile(skillFile, "utf-8")).toBe(content);
  });

  it("rolls back a GLOBAL skill write (path under <muxHome>/skills)", async () => {
    using fixture = await createFixture();
    // Global-scope skills live at <muxHome>/skills (agent_skill_write/delete
    // resolve path.join(muxScope.muxHome, "skills")), NOT under a .mux/skills
    // segment — confinement must accept this root.
    const skillFile = path.join(fixture.muxHome, "skills", "my-skill", "SKILL.md");
    const content = "---\nname: my-skill\n---\n\nbody\n";
    await fsPromises.mkdir(path.dirname(skillFile), { recursive: true });
    await fsPromises.writeFile(skillFile, content, "utf-8");
    await appendRefinementEvent({
      sessionDir: fixture.sessionDir,
      workspaceId: WORKSPACE_ID,
      kind: "skill",
      action: { op: "write", skillName: "my-skill", filePath: "SKILL.md" },
      inverse: { op: "delete-files", paths: [skillFile] },
      evidence: { toolName: "agent_skill_write" },
    });
    const writeRow = await lastRow(fixture.sessionDir);

    const result = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: writeRow.id,
      evidence: EVIDENCE,
    });
    expect(result.success).toBe(true);
    expect(await pathExists(skillFile)).toBe(false);

    // The global skills ROOT itself is still not a legal target.
    await appendRefinementEvent({
      sessionDir: fixture.sessionDir,
      workspaceId: WORKSPACE_ID,
      kind: "skill",
      action: { op: "write", skillName: "x", filePath: "SKILL.md" },
      inverse: { op: "delete-files", paths: [path.join(fixture.muxHome, "skills", "loose-file")] },
      evidence: { toolName: "agent_skill_write" },
    });
    const rootRow = await lastRow(fixture.sessionDir);
    const refused = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: rootRow.id,
      force: true,
      evidence: EVIDENCE,
    });
    expect(refused.success).toBe(false);
    if (refused.success) throw new Error("unreachable");
    expect(refused.error).toContain("global skills root");
  });

  it("undoes a memory rename via the mirrored rename inverse", async () => {
    using fixture = await createFixture();
    await fixture.service.create(fixture.ctx, "/memories/global/old.md", "v1\n", "agent");
    await fixture.service.rename(
      fixture.ctx,
      "/memories/global/old.md",
      "/memories/global/new.md",
      "agent"
    );
    const renameRow = await lastRow(fixture.sessionDir);
    const result = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: renameRow.id,
      evidence: EVIDENCE,
    });
    expect(result.success).toBe(true);
    const oldPath = path.join(fixture.muxHome, "memory", "global", "old.md");
    expect(await fsPromises.readFile(oldPath, "utf-8")).toBe("v1\n");
    expect(await pathExists(path.join(fixture.muxHome, "memory", "global", "new.md"))).toBe(false);
  });

  describe("confinement guard rails", () => {
    it("refuses inverse paths outside every legal root, even with force", async () => {
      using fixture = await createFixture();
      // Corrupted row: a memory-kind inverse pointing at a repo AGENTS.md.
      const evilPath = path.join(fixture.checkout, "AGENTS.md");
      await appendRefinementEvent({
        sessionDir: fixture.sessionDir,
        workspaceId: WORKSPACE_ID,
        kind: "memory",
        action: { op: "str_replace", path: "/memories/global/x.md" },
        inverse: { op: "restore-files", files: [{ path: evilPath, content: "pwned" }] },
        evidence: { toolName: "memory" },
      });
      const row = await lastRow(fixture.sessionDir);
      const result = await rollbackRefinement({
        sessionDir: fixture.sessionDir,
        id: row.id,
        force: true,
        evidence: EVIDENCE,
      });
      expect(result.success).toBe(false);
      if (result.success) throw new Error("unreachable");
      expect(result.error).toContain("outside every memory scope root");
      expect(await pathExists(evilPath)).toBe(false);
    });

    it("refuses workspace memory paths that target another session's memory", async () => {
      using fixture = await createFixture();
      // Corrupted row: a memory-kind inverse pointing into a DIFFERENT
      // workspace's memory dir under the same sessions root.
      const foreign = path.join(fixture.muxHome, "sessions", "other-ws", "memory", "notes.md");
      await appendRefinementEvent({
        sessionDir: fixture.sessionDir,
        workspaceId: WORKSPACE_ID,
        kind: "memory",
        action: { op: "str_replace", path: "/memories/workspace/notes.md" },
        inverse: { op: "restore-files", files: [{ path: foreign, content: "pwned" }] },
        evidence: { toolName: "memory" },
      });
      const row = await lastRow(fixture.sessionDir);
      const result = await rollbackRefinement({
        sessionDir: fixture.sessionDir,
        id: row.id,
        force: true,
        evidence: EVIDENCE,
      });
      expect(result.success).toBe(false);
      if (result.success) throw new Error("unreachable");
      expect(result.error).toContain("outside every memory scope root");
      expect(await pathExists(foreign)).toBe(false);
    });

    it("rolls back workspace-scope memory inside the current session", async () => {
      using fixture = await createFixture();
      await fixture.service.create(fixture.ctx, "/memories/workspace/w.md", "v1\n", "agent");
      await fixture.service.strReplace(
        fixture.ctx,
        "/memories/workspace/w.md",
        "v1",
        "v2",
        "agent"
      );
      const editRow = await lastRow(fixture.sessionDir);
      const result = await rollbackRefinement({
        sessionDir: fixture.sessionDir,
        id: editRow.id,
        evidence: EVIDENCE,
      });
      expect(result.success).toBe(true);
      const physicalPath = path.join(fixture.sessionDir, "memory", "w.md");
      expect(await fsPromises.readFile(physicalPath, "utf-8")).toBe("v1\n");
    });

    it("refuses traversal that escapes the memory root lexically", async () => {
      using fixture = await createFixture();
      // Literal traversal in the stored path (path.join would pre-collapse it).
      const escapePath = `${fixture.muxHome}/memory/global/../../config.json`;
      await appendRefinementEvent({
        sessionDir: fixture.sessionDir,
        workspaceId: WORKSPACE_ID,
        kind: "memory",
        action: { op: "delete", path: "/memories/global/x.md" },
        inverse: { op: "restore-files", files: [{ path: escapePath, content: "pwned" }] },
        evidence: { toolName: "memory" },
      });
      const row = await lastRow(fixture.sessionDir);
      const result = await rollbackRefinement({
        sessionDir: fixture.sessionDir,
        id: row.id,
        force: true,
        evidence: EVIDENCE,
      });
      expect(result.success).toBe(false);
      if (result.success) throw new Error("unreachable");
      expect(result.error).toContain("Refusing rollback");
      expect(await pathExists(path.join(fixture.muxHome, "config.json"))).toBe(false);
    });

    it("refuses skill paths without a .mux/skills or .agents/skills root", async () => {
      using fixture = await createFixture();
      const evilPath = path.join(fixture.checkout, "src", "main.ts");
      await appendRefinementEvent({
        sessionDir: fixture.sessionDir,
        workspaceId: WORKSPACE_ID,
        kind: "skill",
        action: { op: "write", skillName: "x", filePath: "SKILL.md" },
        inverse: { op: "delete-files", paths: [evilPath] },
        evidence: { toolName: "agent_skill_write" },
      });
      await fsPromises.mkdir(path.dirname(evilPath), { recursive: true });
      await fsPromises.writeFile(evilPath, "code", "utf-8");
      const row = await lastRow(fixture.sessionDir);
      const result = await rollbackRefinement({
        sessionDir: fixture.sessionDir,
        id: row.id,
        force: true,
        evidence: EVIDENCE,
      });
      expect(result.success).toBe(false);
      if (result.success) throw new Error("unreachable");
      expect(result.error).toContain("outside every skills root");
      expect(await fsPromises.readFile(evilPath, "utf-8")).toBe("code");
    });

    it("refuses symlink escapes out of the skills root", async () => {
      using fixture = await createFixture();
      const skillsRoot = path.join(fixture.checkout, ".mux", "skills");
      const outside = path.join(fixture.checkout, "outside");
      await fsPromises.mkdir(outside, { recursive: true });
      await fsPromises.mkdir(skillsRoot, { recursive: true });
      // <skillsRoot>/evil → symlink to a directory outside the root.
      await fsPromises.symlink(outside, path.join(skillsRoot, "evil"));
      const target = path.join(skillsRoot, "evil", "SKILL.md");
      await appendRefinementEvent({
        sessionDir: fixture.sessionDir,
        workspaceId: WORKSPACE_ID,
        kind: "skill",
        action: { op: "write", skillName: "evil", filePath: "SKILL.md" },
        inverse: { op: "restore-files", files: [{ path: target, content: "pwned" }] },
        evidence: { toolName: "agent_skill_write" },
      });
      const row = await lastRow(fixture.sessionDir);
      const result = await rollbackRefinement({
        sessionDir: fixture.sessionDir,
        id: row.id,
        force: true,
        evidence: EVIDENCE,
      });
      expect(result.success).toBe(false);
      if (result.success) throw new Error("unreachable");
      expect(result.error).toContain("symlink");
      expect(await pathExists(path.join(outside, "SKILL.md"))).toBe(false);
    });

    it("refuses non-rollbackable refinement kinds", async () => {
      using fixture = await createFixture();
      await sharedDurableEventJournal(fixture.sessionDir).append({
        workspaceId: WORKSPACE_ID,
        kind: "refinement",
        data: { kind: "other", action: {}, inverse: { op: "delete-files", paths: ["/x"] } },
      });
      const row = await lastRow(fixture.sessionDir);
      const result = await rollbackRefinement({
        sessionDir: fixture.sessionDir,
        id: row.id,
        evidence: EVIDENCE,
      });
      expect(result.success).toBe(false);
      if (result.success) throw new Error("unreachable");
      expect(result.error).toContain("not rollbackable");
    });
  });

  it("refuses unknown ids", async () => {
    using fixture = await createFixture();
    const result = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: "does-not-exist",
      evidence: EVIDENCE,
    });
    expect(result.success).toBe(false);
    if (result.success) throw new Error("unreachable");
    expect(result.error).toContain("No refinement row");
  });
});
