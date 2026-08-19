import type { MuxMessage } from "@/common/types/message";
import { FILE_READ_TOOL_NAMES } from "@/common/types/tools";
import { MAX_POST_COMPACTION_READ_FILES } from "@/constants/rlmCompaction";
import { extractToolFilePath } from "@/common/utils/tools/toolInputFilePath";

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

  // Iterate in reverse to get most recent reads first.
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    if (message.role !== "assistant") continue;

    for (const part of message.parts) {
      if (part.type !== "dynamic-tool") continue;
      if (!FILE_READ_TOOL_NAMES.includes(part.toolName as (typeof FILE_READ_TOOL_NAMES)[number])) {
        continue;
      }

      // Only count completed reads that actually returned content.
      if (part.state !== "output-available") continue;
      const output = part.output as { success?: boolean } | undefined;
      if (output?.success !== true) continue;

      const filePath = extractToolFilePath(part.input);
      if (!filePath) continue;
      const trimmed = filePath.trim();
      if (trimmed.length === 0 || seen.has(trimmed)) continue;

      seen.add(trimmed);
      readFiles.push(trimmed);
      if (readFiles.length >= MAX_POST_COMPACTION_READ_FILES) {
        return readFiles;
      }
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
