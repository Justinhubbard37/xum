import { describe, expect, it, spyOn } from "bun:test";

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
import { sharedDurableEventJournal } from "@/node/utils/journal/durableEventJournal";
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
  /** Captures recordHeadlessUsage calls (usage accounting tests). */
  onHeadlessUsage?: (usage: { inputTokens?: number; outputTokens?: number }) => void;
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
      ...(options?.onHeadlessUsage !== undefined
        ? {
            sessionUsageService: {
              recordHeadlessUsage: (
                _workspaceId: string,
                _modelString: string,
                usage: { inputTokens?: number; outputTokens?: number } | undefined
              ) => {
                if (usage) options.onHeadlessUsage!(usage);
                return Promise.resolve(undefined);
              },
            },
          }
        : {}),
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

  it("reports applied-but-unjournaled edits instead of classifying them as a no-op", async () => {
    // At APPLY time the memory write succeeds but its r2 journal append fails
    // (swallowed by design so user writes stay self-healing). The file
    // changed with no rollback id: the apply must say so — not report a
    // no-op while leaving a silent, untracked edit behind.
    using fixture = await createFixture({
      modelFactory: () =>
        toolCallModel(
          [
            {
              toolCallId: "refine-unjournaled-1",
              toolName: "memory",
              input: {
                command: "create",
                path: LESSON_PATH,
                file_text: "An edit whose journal row never lands.\n",
              },
            },
          ],
          `${LESSON_PATH}: applied without a journal row.`
        ),
    });
    await fixture.seedTrajectory();
    const stagedResult = await fixture.service.run(WORKSPACE_ID);
    expect(stagedResult.success).toBe(true);

    // Same process-wide journal instance the service and MemoryService use.
    const journal = sharedDurableEventJournal(fixture.sessionDir);
    // Lazy rejection (not mockRejectedValue): bun creates that rejected
    // promise eagerly, which trips unhandled-rejection detection before any
    // caller can catch it.
    const appendSpy = spyOn(journal, "append").mockImplementation(() =>
      Promise.reject(new Error("journal unavailable"))
    );
    try {
      const result = await fixture.service.apply(WORKSPACE_ID);
      expect(result.success).toBe(true);
      if (!result.success) return;
      // No journal row landed...
      expect(await listRefinements(fixture.sessionDir)).toHaveLength(0);
      expect(result.data.applied).toHaveLength(0);
      // ...but the edit is real, so the apply is NOT a no-op and the
      // untracked count is surfaced.
      expect(result.data.noOp).toBe(false);
      expect(result.data.untrackedApplied).toBe(1);
      // The chat summary warns that rollback is unavailable for these edits
      // (the staged proposal row from the run is emittedMessages[0]).
      expect(fixture.emittedMessages).toHaveLength(2);
      const text = fixture.emittedMessages[1].parts.find((part) => part.type === "text");
      expect(text?.type === "text" && text.text).toContain("could not be journaled");
      expect(text?.type === "text" && text.text).not.toContain("Rollback with:");
    } finally {
      appendSpy.mockRestore();
    }
  });

  it("records completed-step usage when a later step errors", async () => {
    // Step 1 completes (tool call + finish with real usage); step 2 errors.
    // The completed step billed real tokens — the error must not make that
    // spend vanish from accounting.
    let streamCount = 0;
    const errorOnStepTwoModel = () =>
      new MockLanguageModelV3({
        doStream: () => {
          streamCount++;
          if (streamCount === 1) {
            return Promise.resolve({
              stream: simulateReadableStream({
                chunks: [
                  {
                    type: "tool-call",
                    toolCallId: "usage-step-1",
                    toolName: "memory",
                    input: JSON.stringify({
                      command: "create",
                      path: LESSON_PATH,
                      file_text: "Lesson recorded before the provider failure.\n",
                    }),
                  },
                  finishChunk("tool-calls"),
                ] satisfies LanguageModelV3StreamPart[],
              }),
            });
          }
          return Promise.reject(new Error("provider exploded on step 2"));
        },
      });
    const usages: Array<{ inputTokens?: number; outputTokens?: number }> = [];
    using fixture = await createFixture({
      modelFactory: errorOnStepTwoModel,
      onHeadlessUsage: (usage) => usages.push(usage),
    });
    await fixture.seedTrajectory();

    const result = await fixture.service.run(WORKSPACE_ID);
    // The pass still fails (edits stay journaled + rollbackable)...
    expect(result.success).toBe(false);
    if (!result.success) expect(result.error).toContain("refine stream failed");
    // ...but the completed step's tokens were recorded (finishChunk reports
    // 10 in / 5 out per step).
    expect(usages).toHaveLength(1);
    expect(usages[0].inputTokens).toBeGreaterThan(0);
    expect(usages[0].outputTokens).toBeGreaterThan(0);
  });

  it("does not resolve a cancelled pass while a tool execution is still settling", async () => {
    // The deadline fires while a staging tool execution is mid-flight (the
    // memory tool's pin guard awaits metaService.getEntries for deletes).
    // The pass must not settle (releasing the run lock and letting removal
    // delete the session directory) until that execution has fully settled;
    // a detached late execution could otherwise write session state after
    // removal.
    let releaseGuard: () => void = () => undefined;
    const guardGate = new Promise<void>((resolve) => {
      releaseGuard = resolve;
    });
    using fixture = await createFixture({
      timeoutMs: 150,
      modelFactory: () =>
        toolCallModel(
          [
            {
              toolCallId: "refine-slow-guard-1",
              toolName: "memory",
              input: { command: "delete", path: LESSON_PATH },
            },
          ],
          `${LESSON_PATH}: deletion proposed slowly.`
        ),
    });
    await fixture.seedTrajectory();
    const metaService = (
      fixture.service as unknown as {
        metaService: { getEntries: () => Promise<Map<string, never>> };
      }
    ).metaService;
    const entriesSpy = spyOn(metaService, "getEntries").mockImplementation(async () => {
      await guardGate;
      return new Map<string, never>();
    });
    try {
      let settled = false;
      const runPromise = fixture.service.run(WORKSPACE_ID).then((result) => {
        settled = true;
        return result;
      });
      // Wait for the guard to start, then let the 150ms deadline pass well by.
      const spinDeadline = Date.now() + 5_000;
      while (entriesSpy.mock.calls.length === 0 && Date.now() < spinDeadline) {
        await new Promise((resolve) => setTimeout(resolve, 5));
      }
      expect(entriesSpy.mock.calls.length).toBe(1);
      await new Promise((resolve) => setTimeout(resolve, 400));
      // The pass is deadline-cancelled but the tool execution has not
      // settled: the run must still be pending.
      expect(settled).toBe(false);

      releaseGuard();
      const result = await runPromise;
      expect(result.success).toBe(false);
    } finally {
      entriesSpy.mockRestore();
    }
  });

  it("cancelInFlightRefinePass aborts a running pass so no writes or summary land", async () => {
    // Removal races a pass that WOULD apply a memory edit and post a summary
    // row. Gate model creation to hold the race window open deterministically;
    // cancellation must then stop the pass before any write.
    let releaseGate: () => void = () => undefined;
    const gate = new Promise<void>((resolve) => {
      releaseGate = resolve;
    });
    using fixture = await createFixture({
      modelGate: gate,
      modelFactory: () =>
        toolCallModel(
          [
            {
              toolCallId: "refine-cancelled-1",
              toolName: "memory",
              input: {
                command: "create",
                path: LESSON_PATH,
                file_text: "A lesson that must never land after removal.\n",
              },
            },
          ],
          `${LESSON_PATH}: must never be written.`
        ),
    });
    await fixture.seedTrajectory();
    const chatBefore = await fixture.readChat();

    const runPromise = fixture.service.run(WORKSPACE_ID);
    // Removal races in while the pass is gated; both waiters must settle once
    // the gate opens.
    const cancelPromise = fixture.service.cancelInFlightRefinePass(WORKSPACE_ID);
    releaseGate();
    await cancelPromise;

    const result = await runPromise;
    expect(result.success).toBe(false);

    // No tool-driven writes, no journal rows, no summary row, no emission —
    // and nothing staged: a later apply must find nothing to execute.
    expect(await listRefinements(fixture.sessionDir)).toHaveLength(0);
    expect(await fixture.readChat()).toHaveLength(chatBefore.length);
    expect(fixture.emittedMessages).toHaveLength(0);
    const applyAfterCancel = await fixture.service.apply(WORKSPACE_ID);
    expect(applyAfterCancel.success).toBe(false);
    if (!applyAfterCancel.success) {
      expect(applyAfterCancel.error).toContain("no staged refine edits");
    }

    // The lock is cleared: a later invocation is not rejected as running.
    const second = await fixture.service.run(WORKSPACE_ID);
    if (!second.success) expect(second.error).not.toContain("already running");
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

  it("stages a memory edit, applies it only on approval, and rolls back via r6", async () => {
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

    // SECURITY contract: the run only STAGES the model-proposed edit.
    const staged = await fixture.service.run(WORKSPACE_ID);
    expect(staged.success).toBe(true);
    if (!staged.success) return;
    expect(staged.data.noOp).toBe(false);
    expect(staged.data.applied).toHaveLength(0);
    expect(staged.data.staged).toEqual([{ description: `memory create ${LESSON_PATH}` }]);

    const lessonFile = path.join(
      fixture.muxHome,
      "sessions",
      WORKSPACE_ID,
      "memory",
      "refine-lessons.md"
    );
    // NOTHING landed yet: no file, no journal row. The staged summary row
    // tells the user how to approve.
    expect(await pathExists(lessonFile)).toBe(false);
    expect(await listRefinements(fixture.sessionDir)).toHaveLength(0);
    expect(fixture.emittedMessages).toHaveLength(1);
    const stagedText = fixture.emittedMessages[0].parts
      .map((part) => (part.type === "text" ? part.text : ""))
      .join("");
    expect(stagedText).toContain(REFINE_SUMMARY_LABEL);
    expect(stagedText).toContain("/refine apply");

    // Explicit approval applies through the journaled tool path.
    const result = await fixture.service.apply(WORKSPACE_ID);
    expect(result.success).toBe(true);
    if (!result.success) return;
    expect(result.data.noOp).toBe(false);
    expect(result.data.applied).toHaveLength(1);
    expect(result.data.applied[0].description).toBe(`memory create ${LESSON_PATH}`);
    expect(await fsPromises.readFile(lessonFile, "utf-8")).toContain("bun install");

    // r2: exactly one journaled refinement row with an invertible payload,
    // attributed to the staged tool call.
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
    expect(fixture.emittedMessages).toHaveLength(2);

    // The staged set is consumed: a second apply has nothing to do.
    const reapply = await fixture.service.apply(WORKSPACE_ID);
    expect(reapply.success).toBe(false);
    if (!reapply.success) expect(reapply.error).toContain("no staged refine edits");

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

    // Both writes (including the escape attempt) are STAGED — the standard
    // tool's containment runs at apply time and refuses the escape there.
    const stagedResult = await fixture.service.run(WORKSPACE_ID);
    expect(stagedResult.success).toBe(true);
    if (!stagedResult.success) return;
    expect(stagedResult.data.staged).toHaveLength(2);
    expect(await listRefinements(fixture.sessionDir)).toHaveLength(0);

    const result = await fixture.service.apply(WORKSPACE_ID);
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
