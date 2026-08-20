/**
 * Host-side bulk file ingestion for the RLM kernel (mux.load, r12).
 *
 * mux.file_read caps at ~16KB/1000 lines per call, so bulk reads paginate
 * into N model-visible records — exactly the context leak RLM exists to
 * close. mux.load reads the WHOLE file host-side and hands the content
 * straight to the guest `vars` namespace; the guest return value and the
 * model-visible record only ever carry {key, bytes, lines, preview}.
 */

import type { Runtime } from "@/node/runtime/Runtime";
import { readFileString } from "@/node/utils/runtime/helpers";
import { resolvePathWithinCwd, validateFileSize } from "./fileCommon";
import { KERNEL_LOAD_PREVIEW_CHARS } from "@/constants/kernelOutput";

/** Full content + bounded model-visible summary of one loaded file. */
export interface KernelLoadedFile {
  /** Full file content — guest-only (destined for vars[key]); never model-visible. */
  content: string;
  bytes: number;
  lines: number;
  /** Bounded head of the content. */
  preview: string;
}

/** Host closure resolving + reading a file with the workspace's cwd/runtime. */
export type KernelFileLoader = (args: { path: string }) => Promise<KernelLoadedFile>;

/**
 * Build the loader from the same cwd/runtime pair the file tools use, so
 * absolute/relative path resolution is consistent with mux.file_read.
 * Errors are thrown (not returned) so the tool bridge surfaces them as
 * catchable guest errors recorded by the compact call record.
 */
export function createKernelFileLoader(config: {
  cwd: string;
  runtime: Runtime;
}): KernelFileLoader {
  return async ({ path }) => {
    const { resolvedPath } = resolvePathWithinCwd(path, config.cwd, config.runtime);
    // stat throws a RuntimeError with a clear message for missing paths.
    const stat = await config.runtime.stat(resolvedPath);
    if (stat.isDirectory) {
      throw new Error(`Path is a directory, not a file: ${resolvedPath}`);
    }
    // Keep file_read's file-size ceiling (per-operation sanity bound). The
    // 16KB/1000-line PAGINATION caps do not apply — that is the point of
    // load — but loads land in `vars`, which is snapshotted after every call
    // and subject to the 4MB retention policy, so a single load must stay
    // well under that budget.
    const sizeValidation = validateFileSize(stat);
    if (sizeValidation) {
      throw new Error(sizeValidation.error);
    }
    const content = await readFileString(config.runtime, resolvedPath);
    const bytes = Buffer.byteLength(content, "utf8");
    const lines = content === "" ? 0 : content.split("\n").length;
    const preview = content.slice(0, KERNEL_LOAD_PREVIEW_CHARS);
    return { content, bytes, lines, preview };
  };
}
