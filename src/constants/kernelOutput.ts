/**
 * RLM kernel-mode model-visible output bounds (Track 2 context isolation).
 *
 * In kernel mode (persistent mount) the model's only data channels out of a
 * code_execution call are its return value (r4 handle offload applies),
 * console output, and compact per-call summaries. Console output is the
 * model's deliberate debug/print channel, so it stays visible — but it must
 * be bounded so a stray `console.log(bigValue)` cannot reopen the context
 * leak that record suppression closed.
 */

/** Cap on total model-visible console bytes per execution (kernel mode only). */
export const KERNEL_CONSOLE_CAP_BYTES = 16 * 1024;

/** Bounded head shown for a mux.load ingestion ({key, bytes, lines, preview}). */
export const KERNEL_LOAD_PREVIEW_CHARS = 512;
