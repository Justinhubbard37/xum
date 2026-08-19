/**
 * RLM result-handle offloading limits (Track 2 context offloading).
 *
 * Under an RLM persistent kernel mount, tool results and code_execution
 * return values whose JSON serialization exceeds the threshold stop entering
 * the model context: the model-visible record is replaced by
 * { handle, preview, size } while the full value stays in the guest `vars`
 * namespace (vars.__hN), the content-addressed blob store, and one
 * `result-handle` durable event.
 */

/** Serialized-size threshold above which a value is offloaded to a handle. */
export const RESULT_HANDLE_OFFLOAD_THRESHOLD_BYTES = 16 * 1024;

/** Head/tail excerpt lengths for the bounded model-visible preview. */
export const RESULT_HANDLE_PREVIEW_HEAD_CHARS = 1024;
export const RESULT_HANDLE_PREVIEW_TAIL_CHARS = 256;

/**
 * Cap on the TOTAL bytes retained by handle vars in one scope. Handles live
 * in `vars`, which is snapshotted after every call — without a cap the
 * snapshot (and guest memory) would grow unboundedly. Oldest handles are
 * evicted first; the blob store keeps the durable copy of every offloaded
 * value, so eviction only trades guest-local convenience for bounded state.
 */
export const RESULT_HANDLE_VARS_CAP_BYTES = 4 * 1024 * 1024;
