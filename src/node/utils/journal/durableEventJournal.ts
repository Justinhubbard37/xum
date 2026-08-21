/**
 * DurableEventJournal: the concrete per-session journal for the shared
 * durable-event schema (src/common/types/durableEvent.ts), pairing the generic
 * Journal with a content-addressed BlobStore.
 *
 * Layout inside a session dir:
 * - `durable-events.jsonl` — one DurableEvent per line
 * - `blobs/<hash[0:2]>/<hash>` — content-addressed payloads
 *
 * This is the durable leg of the event spine's three-way split. New consumers
 * (turn envelopes, refinement journal, result handles) append here; the
 * HistoryService/chat.jsonl family intentionally stays as-is.
 */

import assert from "node:assert";
import crypto from "node:crypto";
import * as fs from "fs/promises";
import * as path from "path";
import {
  DurableEventSchema,
  DURABLE_EVENT_VERSION,
  type BlobRef,
  type DurableEvent,
  type DurableEventDraft,
} from "@/common/types/durableEvent";
import { AsyncMutex } from "@/node/utils/concurrency/asyncMutex";
import { acquireProcessFileLock } from "@/node/utils/concurrency/fileLock";
import { Journal } from "./journal";
import { BlobStore } from "./blobStore";

export const DURABLE_EVENTS_FILE_NAME = "durable-events.jsonl";
export const BLOBS_DIR_NAME = "blobs";
export const BLOB_LOCK_FILE_NAME = "blobs.lock";

/**
 * Bound on waiting for the cross-process blob lock. Holders include
 * once-per-process recovery sweeps (journal read + per-blob stats), so this
 * is more generous than the append lock's 5s; hitting it means another
 * process is wedged, and failing the operation (all blob-lock consumers are
 * best-effort or self-healing) beats deciding reclamation from an
 * unserialized view.
 */
const BLOB_LOCK_TIMEOUT_MS = 10_000;

/**
 * Process-wide journal registry keyed by resolved session dir. Multiple
 * producers (turn envelopes, hook context, sandbox vars snapshots) append to
 * the same durable-events.jsonl; independent instances would each cache their
 * own next sequence number and could reuse or regress `seq`, corrupting the
 * journal's global event ordering. All live writers must obtain their journal
 * here — blob reclamation additionally relies on it (the blob lock and the
 * blob-mention index are per-instance). Entries are tiny and live for the
 * process.
 */
const sharedJournals = new Map<string, DurableEventJournal>();

export function sharedDurableEventJournal(sessionDir: string): DurableEventJournal {
  const key = path.resolve(sessionDir);
  let journal = sharedJournals.get(key);
  if (!journal) {
    journal = new DurableEventJournal(sessionDir);
    sharedJournals.set(key, journal);
  }
  return journal;
}

/**
 * Which events mention a blob ref, summarized for reclamation decisions.
 * `kinds` answers "which event kinds reference this blob"; snapshot
 * reclamation is additionally per-scope, so sandbox-vars-snapshot mentions
 * also record their scopeKey. Journal rows are never removed, so mentions
 * only accumulate and plain sets (not counts) suffice.
 */
export interface BlobMentions {
  kinds: Set<DurableEvent["kind"]>;
  /** scopeKeys of sandbox-vars-snapshot rows mentioning the ref. */
  snapshotScopes: Set<string>;
}

/** Matches BlobRefSchema refs anywhere inside a serialized row. */
const BLOB_REF_MENTION_PATTERN = /sha256:[0-9a-f]{64}/g;

/**
 * Record every blob ref mentioned by `event`. Serialized containment (rather
 * than a per-kind field list) so every current and future event kind that
 * embeds a blob hash is honored; 64-hex-char refs make false positives a
 * non-concern (a false positive merely retains a blob).
 */
function indexBlobMentions(index: Map<BlobRef, BlobMentions>, event: DurableEvent): void {
  const serialized = JSON.stringify(event);
  for (const match of serialized.matchAll(BLOB_REF_MENTION_PATTERN)) {
    const ref = match[0];
    let mentions = index.get(ref);
    if (!mentions) {
      mentions = { kinds: new Set(), snapshotScopes: new Set() };
      index.set(ref, mentions);
    }
    mentions.kinds.add(event.kind);
    if (event.kind === "sandbox-vars-snapshot") {
      mentions.snapshotScopes.add(event.data.scopeKey);
    }
  }
}

export class DurableEventJournal {
  private readonly journal: Journal<DurableEvent>;
  /** Blob store for content-addressed payloads referenced from rows. */
  public readonly blobs: BlobStore;
  private readonly journalFilePath: string;
  private readonly blobLockPath: string;
  /** In-process leg of the blob lock (fairness + reentrancy assertions);
   * the cross-process leg is the blobs.lock file (see withBlobLock). */
  private readonly blobLock = new AsyncMutex();
  /**
   * Lazily built blob-mention index (see indexBlobMentions), maintained
   * incrementally on append so reclamation passes do O(1) reference lookups
   * instead of re-reading the journal on every persist. Entries are tiny and
   * bounded by journal size; rows are never removed, so it only grows.
   */
  private blobMentions: Map<BlobRef, BlobMentions> | null = null;
  /**
   * Journal file size up to which blobMentions is verifiably complete; null
   * while no index exists. Our own appends advance it contiguously (see the
   * onAppended hook); a stat mismatch at blobMentionIndex() means a FOREIGN
   * instance/process appended rows we never indexed, forcing a rebuild —
   * without this, a reclamation pass could delete a blob whose referencing
   * row was appended by the debug CLI after our index was built.
   */
  private mentionSyncSize: number | null = null;

  constructor(sessionDir: string) {
    this.journalFilePath = path.join(sessionDir, DURABLE_EVENTS_FILE_NAME);
    this.blobLockPath = path.join(sessionDir, BLOB_LOCK_FILE_NAME);
    this.journal = new Journal<DurableEvent>({
      filePath: this.journalFilePath,
      schema: DurableEventSchema,
      getSeq: (row) => row.seq,
      getId: (row) => row.id,
      onAppended: (row, sizes) => {
        // Keep the lazily-built blob-mention index current (see
        // blobMentionIndex). Runs synchronously inside the append's exclusive
        // section so the index can never expose an appended-but-unindexed row.
        if (this.blobMentions === null) return;
        indexBlobMentions(this.blobMentions, row);
        // Advance the freshness watermark only when this append extended the
        // exact file state we had indexed; any gap (foreign bytes) leaves the
        // watermark behind so the next blobMentionIndex() stat forces a
        // rebuild. A mid-rebuild append leaves mentionSyncSize null and is
        // covered by the rebuild's own read + this idempotent indexing.
        if (this.mentionSyncSize !== null && this.mentionSyncSize === sizes.preAppendFileSize) {
          this.mentionSyncSize = sizes.postAppendFileSize;
        }
      },
    });
    this.blobs = new BlobStore(path.join(sessionDir, BLOBS_DIR_NAME));
  }

  /** Append a draft; the journal assigns v/seq/ts (and id unless provided). */
  async append(draft: DurableEventDraft): Promise<DurableEvent> {
    return await this.journal.append((seq) => {
      const built = {
        ...draft,
        v: DURABLE_EVENT_VERSION,
        seq,
        id: draft.id ?? crypto.randomUUID(),
        ts: Date.now(),
      };
      // The spread of a distributive draft union does not re-narrow to the
      // discriminated union; the journal schema-validates the row on append.
      return built as DurableEvent;
    });
  }

  /** Read all events (self-healed: malformed/duplicate rows dropped, seq order). */
  async read(): Promise<DurableEvent[]> {
    return this.journal.read();
  }

  /**
   * Run `fn` while holding this journal's blob lock. Producers pairing
   * `blobs.put()` with a later `append()` MUST do both inside one locked
   * section: content addressing means a concurrent reclamation pass could
   * otherwise observe the blob during the put→append window, find no event
   * referencing its hash, and delete it — permanently breaking the event
   * about to be appended. Reclamation passes hold the same lock across their
   * whole decide→delete window. Non-reentrant: do not nest (including
   * publishWithBlob, which takes the lock itself).
   *
   * Two-level like the journal's append serialization: the in-process mutex
   * orders callers on this instance cheaply, and a cross-process lockfile
   * (blobs.lock, same protocol as the append lock) excludes OTHER journal
   * instances — the debug rollback CLI publishes inverse blobs from its own
   * process, and without the file lock the live app's reclamation could
   * observe (and delete inside) that publisher's put→append window.
   * Lock order is blob → append (fn's appends take the append lock);
   * nothing acquires them in the opposite order, so no deadlock.
   */
  async withBlobLock<T>(fn: () => Promise<T>): Promise<T> {
    await using _mutex = await this.blobLock.acquire();
    await using _fileLock = await acquireProcessFileLock({
      lockPath: this.blobLockPath,
      timeoutMs: BLOB_LOCK_TIMEOUT_MS,
      label: "blob lock",
    });
    return await fn();
  }

  /**
   * Store a blob and append the event referencing it as one atomic unit with
   * respect to blob reclamation (see withBlobLock).
   */
  async publishWithBlob(
    content: string | Uint8Array,
    buildDraft: (ref: BlobRef, size: number) => DurableEventDraft
  ): Promise<{ event: DurableEvent; ref: BlobRef; size: number }> {
    return await this.withBlobLock(async () => {
      const { ref, size } = await this.blobs.put(content);
      const event = await this.append(buildDraft(ref, size));
      return { event, ref, size };
    });
  }

  /**
   * Blob-mention index for reclamation decisions. Callers MUST hold the blob
   * lock: decisions on the index are only race-free while publishers are
   * excluded. Freshness is verified against the journal file size
   * (mentionSyncSize): our own appends advance the watermark incrementally,
   * while foreign appends (a second in-process instance, or the debug CLI in
   * another process) leave a size gap that forces a rebuild here — foreign
   * publishers hold the cross-process blob lock, so their rows are fully
   * appended (and thus visible to the rebuild's read) before we run.
   */
  async blobMentionIndex(): Promise<ReadonlyMap<BlobRef, BlobMentions>> {
    assert(this.blobLock.isLocked, "blobMentionIndex requires holding withBlobLock");
    const fileSize = await this.journalFileSize();
    if (this.blobMentions !== null && this.mentionSyncSize === fileSize) {
      return this.blobMentions;
    }
    // Install the map BEFORE the read: own appends that interleave with the
    // read index themselves into it (see onAppended), and set semantics make
    // the potential double-indexing of one row idempotent. The watermark is
    // set only AFTER the read completes so an interleaved own append (whose
    // watermark advance sees null and skips) triggers at most a harmless
    // extra rebuild, never a stale-marked-fresh index.
    const index = new Map<BlobRef, BlobMentions>();
    this.blobMentions = index;
    this.mentionSyncSize = null;
    for (const event of await this.read()) {
      indexBlobMentions(index, event);
    }
    this.mentionSyncSize = await this.journalFileSize();
    return index;
  }

  /** Journal file size in bytes; 0 when the file does not exist yet. */
  private async journalFileSize(): Promise<number> {
    try {
      return (await fs.stat(this.journalFilePath)).size;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") {
        return 0;
      }
      throw error;
    }
  }
}
