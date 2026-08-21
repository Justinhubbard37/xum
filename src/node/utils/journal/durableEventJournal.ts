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
import * as path from "path";
import {
  DurableEventSchema,
  DURABLE_EVENT_VERSION,
  type BlobRef,
  type DurableEvent,
  type DurableEventDraft,
} from "@/common/types/durableEvent";
import { AsyncMutex } from "@/node/utils/concurrency/asyncMutex";
import { Journal } from "./journal";
import { BlobStore } from "./blobStore";

export const DURABLE_EVENTS_FILE_NAME = "durable-events.jsonl";
export const BLOBS_DIR_NAME = "blobs";

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
  /** Serializes blob publication (put+append) against blob reclamation. */
  private readonly blobLock = new AsyncMutex();
  /**
   * Lazily built blob-mention index (see indexBlobMentions), maintained
   * incrementally on append so reclamation passes do O(1) reference lookups
   * instead of re-reading the journal on every persist. Entries are tiny and
   * bounded by journal size; rows are never removed, so it only grows.
   */
  private blobMentions: Map<BlobRef, BlobMentions> | null = null;

  constructor(sessionDir: string) {
    this.journal = new Journal<DurableEvent>({
      filePath: path.join(sessionDir, DURABLE_EVENTS_FILE_NAME),
      schema: DurableEventSchema,
      getSeq: (row) => row.seq,
      getId: (row) => row.id,
    });
    this.blobs = new BlobStore(path.join(sessionDir, BLOBS_DIR_NAME));
  }

  /** Append a draft; the journal assigns v/seq/ts (and id unless provided). */
  async append(draft: DurableEventDraft): Promise<DurableEvent> {
    const row = await this.journal.append((seq) => {
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
    // Keep the lazily-built blob-mention index current (see blobMentionIndex).
    if (this.blobMentions !== null) {
      indexBlobMentions(this.blobMentions, row);
    }
    return row;
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
   */
  async withBlobLock<T>(fn: () => Promise<T>): Promise<T> {
    await using _lock = await this.blobLock.acquire();
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
   * lock: the first call builds the index from a full read, and decisions on
   * it are only race-free while publishers are excluded. Correctness also
   * relies on all live writers sharing this instance (see sharedJournals):
   * appends through a second instance would bypass the incremental index
   * maintenance in append().
   */
  async blobMentionIndex(): Promise<ReadonlyMap<BlobRef, BlobMentions>> {
    assert(this.blobLock.isLocked, "blobMentionIndex requires holding withBlobLock");
    if (this.blobMentions === null) {
      // Install the map BEFORE the read: appends that interleave with the
      // read index themselves into it, and set semantics make the potential
      // double-indexing of one row idempotent.
      const index = new Map<BlobRef, BlobMentions>();
      this.blobMentions = index;
      for (const event of await this.read()) {
        indexBlobMentions(index, event);
      }
    }
    return this.blobMentions;
  }
}
