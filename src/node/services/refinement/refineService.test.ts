import { describe, expect, it } from "bun:test";

import * as fsPromises from "node:fs/promises";
import * as path from "node:path";
import { MockLanguageModelV3, simulateReadableStream } from "ai/test";
import type { LanguageModelV3CallOptions, LanguageModelV3StreamPart } from "@ai-sdk/provider";

import { EXPERIMENT_IDS, type ExperimentId } from "@/common/constants/experiments";
import { createMuxMessage, type MuxMessage } from "@/common/types/message";
import type { WorkspaceMetadata } from "@/common/types/workspace";
import { Err, Ok } from "@/common/types/result";
import { REFINE_SUMMARY_LABEL } from "@/constants/refine";
import { Config } from "@/node/config";
import { HistoryService } from "@/node/services/historyService";
import { MemoryMetaService } from "@/node/services/memoryMeta";
import { MemoryService } from "@/node/services/memoryService";
import { attachLanguageModelCleanup } from "@/node/services/languageModelCleanup";
import { listRefinements, rollbackRefinement } from "./refinementRollback";
import { RefineService } from "./refineService";
import { TestTempDir } from "../tools/testHelpers";

/**
 * Behavior under test: the /refine orchestration rails — RLM gating (backend
 * refusal), one-run-at-a-time rejection, journal-row correlation with r2
 * inverses, r6 rollback of a refine edit, the labeled summary row, and the
 * first-class no-op. The model is a scripted mock.
 */

// fsPromises.access rejects with a plain value in bun's typings, tripping
// @typescript-eslint/await-thenable on `expect(...).rejects`; assert existence
// via a boolean instead (same pattern as refinementRollback.test.ts).
function pathExists(target: string): Promise<boolean> {
  return fsPromises.access(target).then(
    () => true,
    () => false
  );
}

const WORKSPACE_ID = "ws-refine";
const LESSON_PATH = "/memories/workspace/refine-lessons.md";

function finishChunk(reason: "stop" | "tool-calls"): LanguageModelV3StreamPart {
  return {
    type: "finish",
    finishReason: { unified: reason, raw: reason },
    usage: {
      inputTokens: { total: 10, noCache: 10, cacheRead: 0, cacheWrite: 0 },
      outputTokens: { total: 5, text: 5, reasoning: 0 },
    },
  };
}

function textChunks(text: string): LanguageModelV3StreamPart[] {
  return [
    { type: "text-start", id: "t1" },
    { type: "text-delta", id: "t1", delta: text },
    { type: "text-end", id: "t1" },
    finishChunk("stop"),
  ];
}

function userPromptText(options: LanguageModelV3CallOptions): string {
  const parts: string[] = [];
  for (const message of options.prompt) {
    if (message.role !== "user") continue;
    for (const part of message.content) {
      if (part.type === "text") parts.push(part.text);
    }
  }
  return parts.join("\n");
}

/** Model that makes no edits ("nothing worth distilling"). */
function noOpModel(capturePrompt?: (prompt: string) => void): MockLanguageModelV3 {
  return new MockLanguageModelV3({
    doStream: (options) => {
      capturePrompt?.(userPromptText(options));
      return Promise.resolve({
        stream: simulateReadableStream({ chunks: textChunks("Nothing worth distilling.") }),
      });
    },
  });
}

/** Model that scripts the given tool calls on step 1, then closes with text. */
function toolCallModel(
  calls: Array<{ toolCallId: string; toolName: string; input: Record<string, unknown> }>,
  closingText: string
): MockLanguageModelV3 {
  let streamCount = 0;
  return new MockLanguageModelV3({
    doStream: () => {
      streamCount++;
      const chunks: LanguageModelV3StreamPart[] =
        streamCount === 1
          ? [
              ...calls.map(
                (call): LanguageModelV3StreamPart => ({
                  type: "tool-call",
                  toolCallId: call.toolCallId,
                  toolName: call.toolName,
                  input: JSON.stringify(call.input),
                })
              ),
              finishChunk("tool-calls"),
            ]
          : textChunks(closingText);
      return Promise.resolve({ stream: simulateReadableStream({ chunks }) });
    },
  });
}

interface Fixture extends Disposable {
  muxHome: string;
  workspacePath: string;
  sessionDir: string;
  config: Config;
  service: RefineService;
  historyService: HistoryService;
  memoryService: MemoryService;
  modelCalls: string[];
  emittedMessages: MuxMessage[];
  seedTrajectory: (lines?: string[]) => Promise<void>;
  readChat: () => Promise<MuxMessage[]>;
}

async function createFixture(options?: {
  modelFactory?: () => MockLanguageModelV3;
  /** Holds every model creation open until resolved (in-flight race tests). */
  modelGate?: Promise<void>;
  enabledExperiments?: ExperimentId[];
  /** Provide workspace metadata so the skill-write tool is available. */
  withSkillTool?: boolean;
  timelineEvents?: Array<{ kind: string; description: string }>;
  /** Shortens the pass deadline (wedged-provider tests). */
  timeoutMs?: number;
}): Promise<Fixture> {
  const tempDir = new TestTempDir("test-refine-service");
  const muxHome = path.join(tempDir.path, "mux-home");
  const workspacePath = path.join(tempDir.path, "checkout");
  await fsPromises.mkdir(path.join(muxHome, "memory"), { recursive: true });
  await fsPromises.mkdir(workspacePath, { recursive: true });

  const config = new Config(muxHome);
  await config.editConfig((cfg) => {
    cfg.projects.set("/projects/demo", {
      workspaces: [{ id: WORKSPACE_ID, name: WORKSPACE_ID, path: workspacePath }],
    });
    return cfg;
  });

  const historyService = new HistoryService(config);
  const metaService = new MemoryMetaService(muxHome);
  const memoryService = new MemoryService(config, metaService);

  const enabled = new Set<ExperimentId>(
    options?.enabledExperiments ?? [EXPERIMENT_IDS.RLM, EXPERIMENT_IDS.PROGRAMMATIC_TOOL_CALLING]
  );
  const modelCalls: string[] = [];
  const emittedMessages: MuxMessage[] = [];
  const metadata: WorkspaceMetadata = {
    id: WORKSPACE_ID,
    name: WORKSPACE_ID,
    projectName: "demo",
    projectPath: "/projects/demo",
    runtimeConfig: { type: "local" },
  };

  const service = new RefineService(
    config,
    memoryService,
    metaService,
    historyService,
    {
      createModelWithPinnedMetadata: async (modelString: string) => {
        modelCalls.push(modelString);
        if (options?.modelGate) await options.modelGate;
        return Ok({
          model: options?.modelFactory?.() ?? noOpModel(),
          metadataModel: modelString,
        });
      },
      getWorkspaceMetadata: () =>
        Promise.resolve(
          options?.withSkillTool === true ? Ok(metadata) : Err("no metadata in this fixture")
        ),
    },
    { isExperimentEnabled: (id) => enabled.has(id) },
    {
      emitChatMessage: (_workspaceId, message) => {
        emittedMessages.push(message);
      },
      ...(options?.timeoutMs !== undefined ? { timeoutMs: options.timeoutMs } : {}),
      timelineService:
        options?.timelineEvents !== undefined
          ? {
              list: () =>
                Promise.resolve({
                  events: options.timelineEvents!.map((event, index) => ({
                    v: 1 as const,
                    seq: index + 1,
                    id: `tl-${index}`,
                    ts: 1_700_000_000_000 + index,
                    kind: event.kind,
                    source: { system: "test" },
                    data: { description: event.description },
                  })),
                  nextCursor: null,
                  hasOlder: false,
                }),
            }
          : undefined,
    }
  );

  return {
    muxHome,
    workspacePath,
    sessionDir: config.getSessionDir(WORKSPACE_ID),
    config,
    service,
    historyService,
    memoryService,
    modelCalls,
    emittedMessages,
    seedTrajectory: async (lines) => {
      const texts = lines ?? [
        "Please run the tests for this repo.",
        "Lesson learned: in this repo you must run 'bun install' before 'make test' or module resolution fails.",
      ];
      for (const [index, text] of texts.entries()) {
        await historyService.appendToHistory(
          WORKSPACE_ID,
          createMuxMessage(`user-${index}`, "user", text, { timestamp: Date.now() })
        );
      }
    },
    readChat: async () => {
      const result = await historyService.getHistoryFromLatestBoundary(WORKSPACE_ID);
      if (!result.success) throw new Error(result.error);
      return result.data;
    },
    [Symbol.dispose]() {
      tempDir[Symbol.dispose]();
    },
  };
}

describe("RefineService", () => {
  it("refuses when the rlm-mode experiment is off (and never calls the model)", async () => {
    using fixture = await createFixture({ enabledExperiments: [] });
    await fixture.seedTrajectory();

    const result = await fixture.service.run(WORKSPACE_ID);
    expect(result.success).toBe(false);
    if (!result.success) expect(result.error).toContain("rlm-mode experiment is disabled");
    expect(fixture.modelCalls).toHaveLength(0);
  });

  it("refuses when RLM is on but no PTC parent flag is (sub-experiment gating)", async () => {
    using fixture = await createFixture({ enabledExperiments: [EXPERIMENT_IDS.RLM] });
    await fixture.seedTrajectory();

    const result = await fixture.service.run(WORKSPACE_ID);
    expect(result.success).toBe(false);
    expect(fixture.modelCalls).toHaveLength(0);
  });

  it("rejects a concurrent invocation while a pass is in flight", async () => {
    let releaseGate: () => void = () => undefined;
    const gate = new Promise<void>((resolve) => {
      releaseGate = resolve;
    });
    using fixture = await createFixture({ modelGate: gate });
    await fixture.seedTrajectory();

    const first = fixture.service.run(WORKSPACE_ID);
    const second = await fixture.service.run(WORKSPACE_ID);
    expect(second.success).toBe(false);
    if (!second.success) expect(second.error).toContain("already running");

    releaseGate();
    const firstResult = await first;
    expect(firstResult.success).toBe(true);
    // After the first run settles, the lock is released.
    const third = await fixture.service.run(WORKSPACE_ID);
    expect(third.success).toBe(true);
    expect(fixture.modelCalls).toHaveLength(2);
  });

  it("releases the run lock at the deadline even when the provider ignores abort", async () => {
    // A wedged stream: never yields, never closes, ignores the abort signal
    // entirely. The pass must still settle at the deadline and release the
    // per-workspace lock; previously the consumer stayed pinned in read()
    // forever and every later /refine was rejected as already running.
    const wedgedModel = () =>
      new MockLanguageModelV3({
        doStream: () =>
          Promise.resolve({
            stream: new ReadableStream<LanguageModelV3StreamPart>({
              pull: () => new Promise<never>(() => undefined),
            }),
          }),
      });
    using fixture = await createFixture({ modelFactory: wedgedModel, timeoutMs: 150 });
    await fixture.seedTrajectory();

    const result = await fixture.service.run(WORKSPACE_ID);
    expect(result.success).toBe(false);
    if (!result.success) expect(result.error).toContain("refine stream failed");

    // The lock was released: a second invocation starts a fresh pass instead
    // of being rejected as already running.
    const second = await fixture.service.run(WORKSPACE_ID);
    expect(second.success).toBe(false);
    if (!second.success) expect(second.error).not.toContain("already running");
    expect(fixture.modelCalls).toHaveLength(2);
  });

  it("releases model resources after successful and failed passes", async () => {
    // Providers attach cleanup hooks (e.g. WebSocket transports) via
    // attachLanguageModelCleanup; every pass must release its model or
    // repeated /refine runs accumulate live transports.
    let cleanups = 0;
    const withCleanup = (model: MockLanguageModelV3): MockLanguageModelV3 => {
      attachLanguageModelCleanup(model, () => {
        cleanups += 1;
      });
      return model;
    };

    {
      using fixture = await createFixture({ modelFactory: () => withCleanup(noOpModel()) });
      await fixture.seedTrajectory();
      expect((await fixture.service.run(WORKSPACE_ID)).success).toBe(true);
      expect(cleanups).toBe(1);
    }

    {
      // Failure path: the stream errors immediately, and the finally must
      // still release the model.
      const failingModel = () =>
        withCleanup(
          new MockLanguageModelV3({ doStream: () => Promise.reject(new Error("provider boom")) })
        );
      using fixture = await createFixture({ modelFactory: failingModel });
      await fixture.seedTrajectory();
      const result = await fixture.service.run(WORKSPACE_ID);
      expect(result.success).toBe(false);
      expect(cleanups).toBe(2);
    }
  });

  it("returns a no-op without a model call for an empty trajectory", async () => {
    using fixture = await createFixture();

    const result = await fixture.service.run(WORKSPACE_ID);
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.noOp).toBe(true);
      expect(result.data.applied).toHaveLength(0);
    }
    expect(fixture.modelCalls).toHaveLength(0);
  });

  it("treats a lesson-free trajectory as a clean no-op: no rows, no chat summary", async () => {
    using fixture = await createFixture({ modelFactory: () => noOpModel() });
    await fixture.seedTrajectory(["Just chatting, nothing durable here."]);
    const chatBefore = await fixture.readChat();

    const result = await fixture.service.run(WORKSPACE_ID);
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.noOp).toBe(true);
      expect(result.data.applied).toHaveLength(0);
      expect(result.data.summary).toBe("Nothing worth distilling.");
    }
    expect(await listRefinements(fixture.sessionDir)).toHaveLength(0);
    expect(await fixture.readChat()).toHaveLength(chatBefore.length);
    expect(fixture.emittedMessages).toHaveLength(0);
  });

  it("applies a memory edit with a journaled inverse, posts the summary row, and rolls back via r6", async () => {
    using fixture = await createFixture({
      modelFactory: () =>
        toolCallModel(
          [
            {
              toolCallId: "refine-edit-1",
              toolName: "memory",
              input: {
                command: "create",
                path: LESSON_PATH,
                file_text: "Run 'bun install' before 'make test' in this repo.\n",
              },
            },
          ],
          `${LESSON_PATH}: repo tests need bun install first.`
        ),
    });
    await fixture.seedTrajectory();

    const result = await fixture.service.run(WORKSPACE_ID);
    expect(result.success).toBe(true);
    if (!result.success) return;
    expect(result.data.noOp).toBe(false);
    expect(result.data.applied).toHaveLength(1);
    expect(result.data.applied[0].description).toBe(`memory create ${LESSON_PATH}`);

    // The edit landed on disk.
    const lessonFile = path.join(
      fixture.muxHome,
      "sessions",
      WORKSPACE_ID,
      "memory",
      "refine-lessons.md"
    );
    expect(await fsPromises.readFile(lessonFile, "utf-8")).toContain("bun install");

    // r2: exactly one journaled refinement row with an invertible payload,
    // attributed to this pass's tool call.
    const rows = await listRefinements(fixture.sessionDir);
    expect(rows).toHaveLength(1);
    expect(rows[0].id).toBe(result.data.applied[0].refinementId);
    expect(rows[0].data.inverse).toEqual({ op: "delete-files", paths: [lessonFile] });

    // Completion UX: durable, labeled summary row listing the refinement id
    // and the rollback hint; also emitted to the live session.
    const chat = await fixture.readChat();
    const summaryRow = chat[chat.length - 1];
    expect(summaryRow.metadata?.muxMetadata?.type).toBe("refine-summary");
    const summaryText = summaryRow.parts
      .map((part) => (part.type === "text" ? part.text : ""))
      .join("");
    expect(summaryText).toContain(REFINE_SUMMARY_LABEL);
    expect(summaryText).toContain(result.data.applied[0].refinementId);
    expect(summaryText).toContain("refinement_rollback");
    expect(fixture.emittedMessages).toHaveLength(1);
    expect(fixture.emittedMessages[0].id).toBe(summaryRow.id);

    // r6: rolling the refine edit back restores the pre-edit state.
    const rollback = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: result.data.applied[0].refinementId,
      evidence: { toolName: "test" },
    });
    expect(rollback.success).toBe(true);
    expect(await pathExists(lessonFile)).toBe(false);
  });

  it("rejects guard-rail escapes: invalid memory paths apply nothing and journal nothing", async () => {
    using fixture = await createFixture({
      modelFactory: () =>
        toolCallModel(
          [
            {
              toolCallId: "refine-escape-1",
              toolName: "memory",
              input: {
                command: "create",
                path: "/memories/../AGENTS.md",
                file_text: "must never land\n",
              },
            },
          ],
          "attempted escape"
        ),
    });
    await fixture.seedTrajectory();

    const result = await fixture.service.run(WORKSPACE_ID);
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.noOp).toBe(true);
      expect(result.data.applied).toHaveLength(0);
    }
    expect(await listRefinements(fixture.sessionDir)).toHaveLength(0);
    expect(await pathExists(path.join(fixture.muxHome, "AGENTS.md"))).toBe(false);
  });

  it("writes project skills through the standard tool (journaled) but refuses path escapes", async () => {
    const skillMarkdown = [
      "---",
      "name: distilled-lesson",
      "description: Run bun install before make test in this repo.",
      "---",
      "",
      "Run `bun install` before `make test`.",
      "",
    ].join("\n");
    using fixture = await createFixture({
      withSkillTool: true,
      modelFactory: () =>
        toolCallModel(
          [
            {
              toolCallId: "refine-skill-1",
              toolName: "agent_skill_write",
              input: { name: "distilled-lesson", content: skillMarkdown },
            },
            {
              toolCallId: "refine-skill-escape",
              toolName: "agent_skill_write",
              input: {
                name: "distilled-lesson",
                filePath: "../../AGENTS.md",
                content: "must never land\n",
              },
            },
          ],
          "distilled-lesson: repo test setup procedure."
        ),
    });
    await fixture.seedTrajectory();

    const result = await fixture.service.run(WORKSPACE_ID);
    expect(result.success).toBe(true);
    if (!result.success) return;
    expect(result.data.applied).toHaveLength(1);
    expect(result.data.applied[0].description).toBe("skill write distilled-lesson/SKILL.md");

    const skillFile = path.join(
      fixture.workspacePath,
      ".mux",
      "skills",
      "distilled-lesson",
      "SKILL.md"
    );
    expect(await fsPromises.readFile(skillFile, "utf-8")).toContain("bun install");
    // The escape attempt landed nowhere (workspace AGENTS.md untouched).
    expect(await pathExists(path.join(fixture.workspacePath, "AGENTS.md"))).toBe(false);

    // Journal row carries the delete inverse; rollback removes the skill file.
    const rows = await listRefinements(fixture.sessionDir);
    expect(rows).toHaveLength(1);
    const rollback = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: rows[0].id,
      evidence: { toolName: "test" },
    });
    expect(rollback.success).toBe(true);
    expect(await pathExists(skillFile)).toBe(false);
  });

  it("includes timeline events in the prompt only when the Timeline experiment is on", async () => {
    const prompts: string[] = [];
    const timelineEvents = [{ kind: "milestone", description: "shipped the fix" }];

    {
      using fixture = await createFixture({
        modelFactory: () => noOpModel((prompt) => prompts.push(prompt)),
        timelineEvents,
        enabledExperiments: [
          EXPERIMENT_IDS.RLM,
          EXPERIMENT_IDS.PROGRAMMATIC_TOOL_CALLING,
          EXPERIMENT_IDS.TIMELINE,
        ],
      });
      await fixture.seedTrajectory();
      expect((await fixture.service.run(WORKSPACE_ID)).success).toBe(true);
      expect(prompts[0]).toContain("shipped the fix");
    }

    {
      using fixture = await createFixture({
        modelFactory: () => noOpModel((prompt) => prompts.push(prompt)),
        timelineEvents,
      });
      await fixture.seedTrajectory();
      expect((await fixture.service.run(WORKSPACE_ID)).success).toBe(true);
      expect(prompts[1]).not.toContain("shipped the fix");
    }
  });
});
