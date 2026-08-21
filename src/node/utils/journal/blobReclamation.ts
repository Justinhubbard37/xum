/**
 * Shared blob-reclamation helpers for durable-event journals (journal kit
 * companion). Consumers (sandbox vars snapshots, result handles, refinement
 * inverses) each keep their own per-journal incremental state; these helpers
 * hold the two rules every reclamation pass must share:
 * - reference safety across event kinds (content addressing can share one
 *   payload between kinds — see blobOnlyMentionedBy), and
 * - newest-first byte-quota retention (see walkBlobQuota).
 * Every decide→delete window must run under the journal's blob lock
 * (DurableEventJournal.withBlobLock) so publishers' put→append windows can
 * never be observed.
 */

import type { BlobRef, DurableEvent } from "@/common/types/durableEvent";
import type { BlobMentions } from "./durableEventJournal";

/** One reclaimable blob payload as quota accounting sees it. */
export interface BlobQuotaEntry {
  ref: BlobRef;
  /** Payload size in bytes (recorded on the event or measured at publish). */
  size: number;
}

/**
 * Reference safety: a blob may be deleted only when every event mentioning
 * its hash belongs to the reclaiming pass's own kind (and, for snapshots, its
 * own scope) — content addressing means identical content shares one blob,
 * and deleting a payload referenced by any other event would corrupt that
 * event. Backed by the journal's blob-mention index (O(1) per candidate)
 * instead of a per-persist journal scan.
 */
export function blobOnlyMentionedBy(
  mentions: BlobMentions | undefined,
  kind: DurableEvent["kind"],
  snapshotScope?: string
): boolean {
  // Candidates come from journal events, so an unindexed ref means the index
  // and the journal disagree — retain, never guess.
  if (mentions === undefined) return false;
  for (const mentionKind of mentions.kinds) {
    if (mentionKind !== kind) return false;
  }
  if (snapshotScope !== undefined) {
    for (const scope of mentions.snapshotScopes) {
      if (scope !== snapshotScope) return false;
    }
  }
  return true;
}

/**
 * The newest-first quota walk shared by recovery sweeps (all journal rows)
 * and incremental passes (previous retained list + newly published entries).
 * Content addressing can repeat a ref; its NEWEST occurrence decides
 * retention (duplicates are one blob, counted once). Note the walk keeps
 * accumulating after an entry fails to fit, so an older-but-smaller payload
 * can stay retained past a newer oversized one — retention is per-entry
 * "fits the remaining quota", not a suffix cut.
 */
export function walkBlobQuota(
  entries: BlobQuotaEntry[],
  quotaBytes: number
): { retained: BlobQuotaEntry[]; evictable: Set<BlobRef> } {
  const seen = new Set<BlobRef>();
  const retained: BlobQuotaEntry[] = [];
  const evictable = new Set<BlobRef>();
  let retainedBytes = 0;
  for (const entry of entries) {
    if (seen.has(entry.ref)) continue;
    seen.add(entry.ref);
    if (retainedBytes + entry.size <= quotaBytes) {
      retainedBytes += entry.size;
      retained.push(entry);
    } else {
      evictable.add(entry.ref);
    }
  }
  return { retained, evictable };
}
