/**
 * RLM keep-recent compaction floor (rlm-mode experiment).
 *
 * When RLM mode is on, compaction preserves a recent tail of messages
 * verbatim instead of summarizing the whole epoch: the tail is excluded from
 * the summarization request and re-appended (as sanitized copies) after the
 * durable boundary. Everything here is a pure function over durable history
 * rows so live request assembly, compaction completion, and replay derive the
 * exact same tail — no request-time injection of live state.
 */

import type { MuxMessage, MuxMessageMetadata } from "@/common/types/message";
import assert from "@/common/utils/assert";
import { isNonNegativeInteger } from "@/common/utils/numbers";
import { safeStringifyForCounting } from "@/common/utils/tokens/safeStringifyForCounting";
import { hasProviderEligibleMessages } from "@/common/utils/messages/compactionBoundary";
import { RLM_COMPACTION_CHARS_PER_TOKEN } from "@/constants/rlmCompaction";

/**
 * Provider-agnostic token estimate for one history row (chars / 4 heuristic).
 * Used only for the keep-recent floor cut, never for provider payloads.
 */
export function estimateMuxMessageTokens(message: MuxMessage): number {
  assert(message != null, "estimateMuxMessageTokens requires a message");
  return Math.ceil(safeStringifyForCounting(message.parts).length / RLM_COMPACTION_CHARS_PER_TOKEN);
}

/**
 * Select the start index of the keep-recent tail: the oldest suffix of
 * `messages` whose estimated token size fits under `floorTokens`.
 *
 * Safe boundaries: a tail may only start on a non-synthetic user row with a
 * valid historySequence. Assistant rows embed their tool call/result pairs as
 * parts of a single row, so any row boundary is pairing-safe at the provider
 * level; starting on a real user turn additionally keeps a turn's assistant
 * steps and synthetic continuations attached to the prompt that produced them.
 *
 * Clamp-down: when even the newest safe suffix exceeds the floor (or no safe
 * boundary exists), returns -1 — the tail is dropped entirely rather than
 * shrunk below a turn boundary. Forced compaction must always be able to make
 * progress: preserving the floor is best-effort, and an over-floor tail would
 * defeat the point of compacting near the context limit.
 *
 * The head (rows before the returned index) must contain at least one
 * provider-eligible message so the summarization request has something to
 * summarize; candidates that would leave an empty head are skipped.
 */
export function selectKeepRecentTailStartIndex(
  // Mutable array type (repo convention for message helpers): Array.isArray on a
  // readonly array parameter would narrow it to any[] and poison type safety.
  messages: MuxMessage[],
  floorTokens: number
): number {
  assert(Array.isArray(messages), "selectKeepRecentTailStartIndex requires a message array");
  assert(
    Number.isFinite(floorTokens) && floorTokens > 0,
    "selectKeepRecentTailStartIndex requires a positive floor"
  );

  let suffixTokens = 0;
  let bestStartIndex = -1;

  for (let i = messages.length - 1; i >= 1; i--) {
    const message = messages[i];
    suffixTokens += estimateMuxMessageTokens(message);
    if (suffixTokens > floorTokens) {
      break;
    }

    const isSafeBoundary =
      message.role === "user" &&
      message.metadata?.synthetic !== true &&
      isNonNegativeInteger(message.metadata?.historySequence);
    if (!isSafeBoundary) {
      continue;
    }

    if (!hasProviderEligibleMessages(messages.slice(0, i))) {
      // An empty head would leave the summarizer with nothing to summarize.
      break;
    }

    bestStartIndex = i;
  }

  return bestStartIndex;
}

/**
 * Validated accessor for the durable keep-recent stamp on a compaction-request
 * row. Self-healing read path: malformed persisted stamps degrade to
 * "no tail" instead of crashing request assembly.
 */
export function getKeepRecentTailStartHistorySequence(
  muxMetadata: MuxMessageMetadata | undefined
): number | undefined {
  if (muxMetadata?.type !== "compaction-request") {
    return undefined;
  }
  const start = muxMetadata.keepRecentTail?.startHistorySequence;
  return isNonNegativeInteger(start) ? start : undefined;
}

/**
 * Exclude the keep-recent tail from a compaction summarization request.
 *
 * When the last user row is a compaction-request stamped with a keep-recent
 * start sequence, rows before the request whose historySequence is at or after
 * the stamp are dropped so the model summarizes only the older head. Rows at
 * or after the request row (e.g. a partial continuation) always survive, as do
 * rows without a valid historySequence (conservative self-healing).
 *
 * Returns the input array unchanged (same reference) when no stamp applies —
 * with RLM off no row ever carries a stamp, so this is byte-identical to
 * today's behavior for both live requests and replay.
 */
export function excludeKeepRecentTailForCompactionRequest(messages: MuxMessage[]): MuxMessage[] {
  assert(Array.isArray(messages), "excludeKeepRecentTailForCompactionRequest requires an array");

  let requestIndex = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "user") {
      requestIndex = i;
      break;
    }
  }
  if (requestIndex === -1) {
    return messages;
  }

  const startHistorySequence = getKeepRecentTailStartHistorySequence(
    messages[requestIndex].metadata?.muxMetadata
  );
  if (startHistorySequence === undefined) {
    return messages;
  }

  const filtered = messages.filter((message, index) => {
    if (index >= requestIndex) {
      return true;
    }
    const sequence = message.metadata?.historySequence;
    if (!isNonNegativeInteger(sequence)) {
      return true;
    }
    return sequence < startHistorySequence;
  });

  return filtered.length === messages.length ? messages : filtered;
}
