/**
 * Branch summarization on fork/truncate (rlm-mode experiment, nested under
 * Programmatic Tool Calling). When RLM mode is on and history branches (fork
 * from an earlier message or edit-resend truncation), the abandoned tail is
 * summarized via a cheap side-channel model call and appended to the new
 * branch as a durable labeled row. With RLM off these constants are unused
 * and forks/truncations behave exactly as before.
 */

/**
 * Minimum estimated token size (chars/4 heuristic over serialized parts) of
 * the abandoned segment before a summary is worth a model call. Tiny tails
 * (a quick retry of the last message, a one-line answer) carry no context
 * worth preserving.
 */
export const BRANCH_SUMMARY_MIN_SEGMENT_TOKENS = 1_000;

/** Output budget for the summary call; also drives the prompt's word target. */
export const BRANCH_SUMMARY_MAX_OUTPUT_TOKENS = 1_024;

/**
 * Hard wall-clock bound for the whole summary generation (all candidate
 * models share one deadline). Generation is synchronous inside fork/edit —
 * see maybeAppendAbandonedBranchSummary for why — so this caps how long the
 * user-facing operation can be delayed.
 */
export const BRANCH_SUMMARY_TIMEOUT_MS = 10_000;

/**
 * Input cap for the thinking-stripped transcript fed to the summarizer.
 * Oldest messages are dropped first: the newest abandoned work carries the
 * most context worth preserving. ~40k tokens at the chars/4 heuristic keeps
 * the side-channel call cheap even for a large abandoned tail.
 */
export const BRANCH_SUMMARY_MAX_TRANSCRIPT_CHARS = 160_000;
