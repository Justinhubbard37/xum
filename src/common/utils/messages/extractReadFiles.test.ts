import { describe, expect, it } from "bun:test";

import type { MuxMessage } from "@/common/types/message";
import { MAX_POST_COMPACTION_READ_FILES } from "@/constants/rlmCompaction";

import { extractReadFilePaths, mergeReadFilePaths } from "./extractReadFiles";

function createAssistantMessage(
  toolCalls: Array<{
    toolName: string;
    filePath?: string;
    success?: boolean;
    state?: "output-available" | "input-available";
  }>
): MuxMessage {
  return {
    id: `msg-${Math.random().toString(36).slice(2)}`,
    role: "assistant",
    parts: toolCalls.map((tc) =>
      tc.state === "input-available"
        ? {
            type: "dynamic-tool" as const,
            toolCallId: `tc-${Math.random().toString(36).slice(2)}`,
            toolName: tc.toolName,
            state: "input-available" as const,
            input: { path: tc.filePath },
          }
        : {
            type: "dynamic-tool" as const,
            toolCallId: `tc-${Math.random().toString(36).slice(2)}`,
            toolName: tc.toolName,
            state: "output-available" as const,
            input: { path: tc.filePath },
            output: { success: tc.success ?? true },
          }
    ),
  };
}

describe("extractReadFilePaths", () => {
  it("extracts successful file_read paths newest-first, deduped", () => {
    const messages: MuxMessage[] = [
      createAssistantMessage([
        { toolName: "file_read", filePath: "/a.ts" },
        { toolName: "file_read", filePath: "/b.ts" },
      ]),
      createAssistantMessage([{ toolName: "file_read", filePath: "/a.ts" }]),
      createAssistantMessage([{ toolName: "file_read", filePath: "/c.ts" }]),
    ];

    expect(extractReadFilePaths(messages)).toEqual(["/c.ts", "/a.ts", "/b.ts"]);
  });

  it("ignores failed reads, interrupted calls, and non-read tools", () => {
    const messages: MuxMessage[] = [
      createAssistantMessage([
        { toolName: "file_read", filePath: "/failed.ts", success: false },
        { toolName: "file_read", filePath: "/interrupted.ts", state: "input-available" },
        { toolName: "file_edit_insert", filePath: "/edited.ts" },
        { toolName: "file_read", filePath: "/ok.ts" },
      ]),
    ];

    expect(extractReadFilePaths(messages)).toEqual(["/ok.ts"]);
  });

  it("caps the extracted list", () => {
    const messages = [
      createAssistantMessage(
        Array.from({ length: MAX_POST_COMPACTION_READ_FILES + 20 }, (_, i) => ({
          toolName: "file_read",
          filePath: `/file-${i}.ts`,
        }))
      ),
    ];

    expect(extractReadFilePaths(messages)).toHaveLength(MAX_POST_COMPACTION_READ_FILES);
  });
});

describe("mergeReadFilePaths", () => {
  it("puts incoming (newer) paths first and dedupes against existing", () => {
    expect(mergeReadFilePaths(["/old.ts", "/both.ts"], ["/new.ts", "/both.ts"])).toEqual([
      "/new.ts",
      "/both.ts",
      "/old.ts",
    ]);
  });

  it("caps the merged list, evicting the oldest entries", () => {
    const existing = Array.from({ length: MAX_POST_COMPACTION_READ_FILES }, (_, i) => `/old-${i}`);
    const incoming = ["/new-1", "/new-2"];

    const merged = mergeReadFilePaths(existing, incoming);
    expect(merged).toHaveLength(MAX_POST_COMPACTION_READ_FILES);
    expect(merged.slice(0, 2)).toEqual(incoming);
    expect(merged).not.toContain(`/old-${MAX_POST_COMPACTION_READ_FILES - 1}`);
    expect(merged).not.toContain(`/old-${MAX_POST_COMPACTION_READ_FILES - 2}`);
  });
});
