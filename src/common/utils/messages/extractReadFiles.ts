import type { MuxMessage } from "@/common/types/message";
import { FILE_READ_TOOL_NAMES } from "@/common/types/tools";
import { MAX_POST_COMPACTION_READ_FILES } from "@/constants/rlmCompaction";
import { extractToolFilePath } from "@/common/utils/tools/toolInputFilePath";

/**
 * Structural view of one nested tool-call record inside a code_execution
 * output (PTCToolCallRecord). Declared here because src/common must not
 * import node-side PTC types; only the fields this extractor reads.
 */
interface NestedToolCallRecord {
  toolName?: unknown;
  args?: unknown;
  error?: unknown;
  ok?: unknown;
}

/**
 * Nested read-flavored calls inside a code_execution part (RLM/PTC): in the
 * exclusive posture file access happens as nested xum.file_read / xum.load
 * calls, so the outer part is named "code_execution" and the reads live in
 * its output's toolCalls records. Success = no error, and for kernel compact
 * records ok !== false (supplement-mode records carry no ok field).
 */
function collectNestedReadPaths(output: unknown): string[] {
  if (typeof output !== "object" || output === null) return [];
  const toolCalls = (output as { toolCalls?: unknown }).toolCalls;
  if (!Array.isArray(toolCalls)) return [];

  const paths: string[] = [];
  for (const record of toolCalls as NestedToolCallRecord[]) {
    if (typeof record !== "object" || record === null) continue;
    const isRead =
      FILE_READ_TOOL_NAMES.includes(record.toolName as (typeof FILE_READ_TOOL_NAMES)[number]) ||
      record.toolName === "load";
    if (!isRead) continue;
    if (record.error !== undefined || record.ok === false) continue;
    const filePath = extractToolFilePath(record.args);
    if (filePath) paths.push(filePath);
  }
  return paths;
}

/**
 * Extract unique file paths successfully READ during the given messages
 * (RLM post-compaction read tracking). Mirrors extractEditedFilePaths but for
 * read-flavored tools: paths only, never contents.
 *
 * Returns most recently read paths first, capped at
 * MAX_POST_COMPACTION_READ_FILES.
 */
export function extractReadFilePaths(messages: readonly MuxMessage[]): string[] {
  const readFiles: string[] = [];
  const seen = new Set<string>();

  const add = (filePath: string): boolean => {
    const trimmed = filePath.trim();
    if (trimmed.length === 0 || seen.has(trimmed)) return false;
    seen.add(trimmed);
    readFiles.push(trimmed);
    return readFiles.length >= MAX_POST_COMPACTION_READ_FILES;
  };

  // Iterate in reverse to get most recent reads first.
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    if (message.role !== "assistant") continue;

    for (const part of message.parts) {
      if (part.type !== "dynamic-tool") continue;
      if (part.state !== "output-available") continue;

      if (part.toolName === "code_execution") {
        // The execution's overall success is irrelevant: nested reads that
        // completed before a later failure still loaded those files.
        for (const nested of collectNestedReadPaths(part.output)) {
          if (add(nested)) return readFiles;
        }
        continue;
      }

      if (!FILE_READ_TOOL_NAMES.includes(part.toolName as (typeof FILE_READ_TOOL_NAMES)[number])) {
        continue;
      }

      // Only count completed reads that actually returned content.
      const output = part.output as { success?: boolean } | undefined;
      if (output?.success !== true) continue;

      const filePath = extractToolFilePath(part.input);
      if (!filePath) continue;
      if (add(filePath)) return readFiles;
    }
  }

  return readFiles;
}

/**
 * Merge read-file paths cumulatively across compactions: incoming (newer)
 * paths first, then previously tracked paths, deduped and capped. Mirrors
 * mergeFileEditDiffs so successive compactions keep older reads until the cap
 * evicts them newest-first.
 */
export function mergeReadFilePaths(
  existing: readonly string[],
  incoming: readonly string[]
): string[] {
  const merged: string[] = [];
  const seen = new Set<string>();

  for (const path of [...incoming, ...existing]) {
    if (typeof path !== "string") continue;
    const trimmed = path.trim();
    if (trimmed.length === 0 || seen.has(trimmed)) continue;
    seen.add(trimmed);
    merged.push(trimmed);
    if (merged.length >= MAX_POST_COMPACTION_READ_FILES) {
      break;
    }
  }

  return merged;
}
