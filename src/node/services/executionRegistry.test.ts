import assert from "node:assert/strict";
import { afterEach, beforeEach, describe, expect, spyOn, test } from "bun:test";
import * as fsPromises from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";

import { DEFAULT_RUNTIME_CONFIG } from "@/common/constants/workspace";
import type { ExecutionHandle } from "@/common/types/execution";
import type { Workspace } from "@/common/types/project";
import { Config } from "@/node/config";
import { ExecutionRegistry } from "@/node/services/executionRegistry";
import { ExecutionStore } from "@/node/services/executionStore";
import { TaskHandleStore } from "@/node/services/taskHandleStore";
import { upsertSubagentFailureArtifact } from "@/node/services/subagentFailureArtifacts";
import { upsertSubagentGitPatchArtifact } from "@/node/services/subagentGitPatchArtifacts";
import { upsertSubagentReportArtifact } from "@/node/services/subagentReportArtifacts";

const OWNER = "owner";
const CREATED_AT = "2026-08-06T00:00:00.000Z";

function canonicalHandle(overrides: Partial<ExecutionHandle> = {}): ExecutionHandle {
  return {
    version: 1,
    executionId: "exe_canonical",
    aliases: ["canonical-workspace"],
    ownerSessionId: OWNER,
    requesterWorkspaceId: OWNER,
    target: { kind: "workspace", workspaceId: "canonical-workspace", origin: "created" },
    launchPolicy: { kind: "agent_task", agentId: "exec", prompt: "Implement" },
    completionPolicy: { kind: "final_assistant_message" },
    retentionPolicy: { kind: "delete_workspace_on_completion" },
    attentionPolicy: "blocking_until_terminal",
    status: "starting",
    createdAt: CREATED_AT,
    updatedAt: CREATED_AT,
    ...overrides,
  };
}

async function addAgentTask(
  config: Config,
  taskId: string,
  taskStatus: Workspace["taskStatus"],
  overrides: Partial<Workspace> = {}
): Promise<void> {
  await config.addWorkspace("/repo", {
    id: taskId,
    name: taskId,
    title: `${taskId} title`,
    projectName: "repo",
    projectPath: "/repo",
    createdAt: CREATED_AT,
    runtimeConfig: DEFAULT_RUNTIME_CONFIG,
    parentWorkspaceId: OWNER,
    agentId: "exec",
    taskStatus,
    taskPrompt: `${taskId} prompt`,
    ...(taskStatus === "reported" ? { reportedAt: "2026-08-06T00:00:05.000Z" } : {}),
    ...overrides,
  });
  if (overrides.executionId != null) {
    await config.editConfig((projectsConfig) => {
      for (const project of projectsConfig.projects.values()) {
        const workspace = project.workspaces.find((candidate) => candidate.id === taskId);
        if (workspace != null) workspace.executionId = overrides.executionId;
      }
      return projectsConfig;
    });
  }
}

describe("ExecutionRegistry canonical lifecycle", () => {
  let rootDir: string;
  let config: Config;
  let store: ExecutionStore;
  let registry: ExecutionRegistry;

  beforeEach(async () => {
    rootDir = await fsPromises.mkdtemp(path.join(os.tmpdir(), "mux-execution-registry-"));
    config = new Config(rootDir);
    store = new ExecutionStore(config);
    registry = new ExecutionRegistry(config, { executionStore: store });
  });

  afterEach(async () => {
    await fsPromises.rm(rootDir, { recursive: true, force: true });
  });

  test("snapshots active updates by alias and supports timeout and abort", async () => {
    await registry.upsert(canonicalHandle());
    const running = canonicalHandle({
      status: "running",
      phase: "awaiting_report",
      startedAt: "2026-08-06T00:00:01.000Z",
      updatedAt: "2026-08-06T00:00:01.000Z",
    });
    await registry.upsert(running);

    expect(await registry.snapshot(OWNER, "canonical-workspace")).toEqual(running);
    expect(await registry.waitForTerminal(OWNER, "canonical-workspace", { timeoutMs: 0 })).toEqual({
      kind: "timeout",
      snapshot: running,
    });

    const abortController = new AbortController();
    abortController.abort();
    expect(
      await registry.waitForTerminal(OWNER, "exe_canonical", {
        abortSignal: abortController.signal,
      })
    ).toEqual({ kind: "aborted", snapshot: running });
    expect(await registry.waitForTerminal(OWNER, "exe_missing", { timeoutMs: 0 })).toEqual({
      kind: "not_found",
    });
  });

  test("persists terminal results before resolving all canonical waiters", async () => {
    await registry.upsert(canonicalHandle({ status: "running", startedAt: CREATED_AT }));

    let releaseWrite: (() => void) | undefined;
    const writeGate = new Promise<void>((resolve) => {
      releaseWrite = resolve;
    });
    let terminalWriteStarted: (() => void) | undefined;
    const terminalWriteStart = new Promise<void>((resolve) => {
      terminalWriteStarted = resolve;
    });
    const originalUpsert = store.upsert.bind(store);
    spyOn(store, "upsert").mockImplementation(async (handle) => {
      if (handle.status === "completed") {
        terminalWriteStarted?.();
        await writeGate;
      }
      await originalUpsert(handle);
    });

    const waiterById = registry.waitForTerminal(OWNER, "exe_canonical");
    const waiterByAlias = registry.waitForTerminal(OWNER, "canonical-workspace");
    const settling = registry.settle(
      OWNER,
      "canonical-workspace",
      { kind: "completed", reportMarkdown: "Done" },
      { terminalAt: "2026-08-06T00:00:02.000Z" }
    );
    await terminalWriteStart;

    let waiterResolved = false;
    void waiterById.then(() => {
      waiterResolved = true;
    });
    await Promise.resolve();
    expect(waiterResolved).toBe(false);
    expect(await new ExecutionStore(config).get(OWNER, "exe_canonical")).toMatchObject({
      status: "running",
    });

    releaseWrite?.();
    const [settled, byId, byAlias] = await Promise.all([settling, waiterById, waiterByAlias]);
    if (settled == null) throw new Error("Expected canonical execution to settle");
    expect(settled).toMatchObject({
      status: "completed",
      result: { kind: "completed", reportMarkdown: "Done" },
      terminalAt: "2026-08-06T00:00:02.000Z",
    });
    expect(byId).toEqual({ kind: "terminal", handle: settled });
    expect(byAlias).toEqual({ kind: "terminal", handle: settled });
    expect(await new ExecutionStore(config).get(OWNER, "exe_canonical")).toEqual(settled);
  });

  test("keeps terminal settlement immutable and returns it after restart", async () => {
    await registry.upsert(canonicalHandle({ status: "running", startedAt: CREATED_AT }));
    const completed = await registry.settle(
      OWNER,
      "exe_canonical",
      { kind: "completed", reportMarkdown: "First" },
      { terminalAt: "2026-08-06T00:00:02.000Z" }
    );
    if (completed == null) throw new Error("Expected canonical execution to settle");

    expect(
      await registry.settle(
        OWNER,
        "canonical-workspace",
        { kind: "error", error: "Late failure" },
        { terminalAt: "2026-08-06T00:00:03.000Z" }
      )
    ).toEqual(completed);
    expect(
      await registry.upsert(
        canonicalHandle({
          status: "running",
          updatedAt: "2026-08-06T00:00:04.000Z",
          startedAt: CREATED_AT,
        })
      )
    ).toEqual(completed);

    const restarted = new ExecutionRegistry(config);
    expect(await restarted.waitForTerminal(OWNER, "canonical-workspace", { timeoutMs: 0 })).toEqual(
      {
        kind: "terminal",
        handle: completed,
      }
    );
  });

  test("reconciliation can replace a stale terminal projection without weakening normal settlement", async () => {
    await registry.upsert(canonicalHandle({ status: "running", startedAt: CREATED_AT }));
    const staleTerminal = await registry.settle(
      OWNER,
      "exe_canonical",
      { kind: "interrupted", message: "Restart assumed the turn was stale" },
      { terminalAt: "2026-08-06T00:00:02.000Z" }
    );
    assert(staleTerminal != null);

    const revived = canonicalHandle({
      status: "running",
      startedAt: CREATED_AT,
      updatedAt: "2026-08-06T00:00:03.000Z",
    });
    expect(await registry.overwriteForReconciliation(revived)).toEqual(revived);
    expect(await registry.get(OWNER, "exe_canonical")).toEqual(revived);

    const repaired = await registry.settle(
      OWNER,
      "exe_canonical",
      { kind: "completed", reportMarkdown: "Recovered completion" },
      { terminalAt: "2026-08-06T00:00:04.000Z" }
    );
    expect(repaired).toMatchObject({
      status: "completed",
      result: { kind: "completed", reportMarkdown: "Recovered completion" },
    });
    expect(
      await registry.settle(
        OWNER,
        "exe_canonical",
        { kind: "error", error: "Late failure" },
        { terminalAt: "2026-08-06T00:00:05.000Z" }
      )
    ).toEqual(repaired);
  });

  test("queries canonical agent execution depth, descendants, owner root, and active statuses", async () => {
    const root = canonicalHandle({
      executionId: "exe_root",
      aliases: ["root-workspace"],
      target: { kind: "workspace", workspaceId: "root-workspace", origin: "created" },
      status: "running",
      startedAt: CREATED_AT,
    });
    const child = canonicalHandle({
      executionId: "exe_child",
      aliases: ["child-workspace"],
      parentExecutionId: root.executionId,
      requesterWorkspaceId: "root-workspace",
      target: { kind: "workspace", workspaceId: "child-workspace", origin: "created" },
      status: "starting",
      createdAt: "2026-08-06T00:00:01.000Z",
      updatedAt: "2026-08-06T00:00:01.000Z",
    });
    const grandchild = canonicalHandle({
      executionId: "exe_grandchild",
      aliases: ["grandchild-workspace"],
      parentExecutionId: child.executionId,
      requesterWorkspaceId: "child-workspace",
      target: { kind: "workspace", workspaceId: "grandchild-workspace", origin: "created" },
      status: "completed",
      result: { kind: "completed", reportMarkdown: "Done" },
      createdAt: "2026-08-06T00:00:02.000Z",
      updatedAt: "2026-08-06T00:00:03.000Z",
      terminalAt: "2026-08-06T00:00:03.000Z",
    });
    await Promise.all([registry.upsert(root), registry.upsert(child), registry.upsert(grandchild)]);

    expect((await registry.listAgentExecutions(OWNER)).map((handle) => handle.executionId)).toEqual(
      [root.executionId, child.executionId, grandchild.executionId]
    );
    expect(
      (await registry.listAgentExecutions(OWNER, { statuses: ["completed"] })).map(
        (handle) => handle.executionId
      )
    ).toEqual([grandchild.executionId]);
    expect(
      (await registry.listActiveAgentExecutions(OWNER)).map((handle) => handle.executionId)
    ).toEqual([root.executionId, child.executionId]);
    expect(await registry.getAgentExecutionDepth(OWNER, "root-workspace")).toBe(1);
    expect(await registry.getAgentExecutionDepth(OWNER, child.executionId)).toBe(2);
    expect(await registry.getAgentExecutionDepth(OWNER, "grandchild-workspace")).toBe(3);
    expect(await registry.getAgentExecutionDepth(OWNER, "missing")).toBeNull();
    expect(
      (await registry.listDescendantAgentExecutions(OWNER, "root-workspace")).map(
        (handle) => handle.executionId
      )
    ).toEqual([child.executionId, grandchild.executionId]);
    expect(
      (await registry.listDescendantAgentExecutions(OWNER, child.executionId)).map(
        (handle) => handle.executionId
      )
    ).toEqual([grandchild.executionId]);
    expect(
      (await registry.listDescendantAgentExecutions(OWNER)).map((handle) => handle.executionId)
    ).toEqual([root.executionId, child.executionId, grandchild.executionId]);
  });
});

describe("ExecutionRegistry legacy adapters", () => {
  let rootDir: string;
  let config: Config;
  let registry: ExecutionRegistry;

  beforeEach(async () => {
    rootDir = await fsPromises.mkdtemp(path.join(os.tmpdir(), "mux-execution-registry-"));
    config = new Config(rootDir);
    registry = new ExecutionRegistry(config);
  });

  afterEach(async () => {
    await fsPromises.rm(rootDir, { recursive: true, force: true });
  });

  test("golden maps legacy agent lifecycle states and preserves workspace ID aliases", async () => {
    const fixtures = [
      ["queued-task", "queued"],
      ["running-task", "running"],
      ["awaiting-task", "awaiting_report"],
      ["reported-task", "reported"],
      ["interrupted-task", "interrupted"],
    ] as const;
    for (const [taskId, status] of fixtures) {
      await addAgentTask(config, taskId, status);
    }

    const sessionDir = config.getSessionDir(OWNER);
    await upsertSubagentReportArtifact({
      workspaceId: OWNER,
      workspaceSessionDir: sessionDir,
      childTaskId: "reported-task",
      parentWorkspaceId: OWNER,
      ancestorWorkspaceIds: [OWNER],
      reportMarkdown: "Completed report",
      structuredOutput: { ok: true },
      nowMs: Date.parse("2026-08-06T00:00:05.000Z"),
    });
    await upsertSubagentGitPatchArtifact({
      workspaceId: OWNER,
      workspaceSessionDir: sessionDir,
      childTaskId: "reported-task",
      updater: () => ({
        childTaskId: "reported-task",
        parentWorkspaceId: OWNER,
        createdAtMs: Date.parse(CREATED_AT),
        updatedAtMs: Date.parse("2026-08-06T00:00:05.000Z"),
        status: "ready",
        projectArtifacts: [
          {
            projectPath: "/repo",
            projectName: "repo",
            storageKey: "repo",
            status: "ready",
            baseCommitSha: "base",
            headCommitSha: "head",
            commitCount: 1,
            mboxPath: "/tmp/report.mbox",
          },
        ],
        readyProjectCount: 1,
        failedProjectCount: 0,
        skippedProjectCount: 0,
        totalCommitCount: 1,
      }),
    });

    const byAlias = new Map(
      await Promise.all(
        fixtures.map(async ([taskId]) => [taskId, await registry.get(OWNER, taskId)] as const)
      )
    );

    expect(byAlias.get("queued-task")).toMatchObject({
      aliases: ["queued-task"],
      status: "queued",
      target: { kind: "workspace", workspaceId: "queued-task", origin: "created" },
      launchPolicy: { kind: "agent_task", agentId: "exec", prompt: "queued-task prompt" },
    });
    expect(byAlias.get("running-task")).toMatchObject({ status: "running" });
    expect(byAlias.get("awaiting-task")).toMatchObject({
      status: "running",
      phase: "awaiting_report",
    });
    expect(byAlias.get("reported-task")).toMatchObject({
      status: "completed",
      result: {
        kind: "completed",
        reportMarkdown: "Completed report",
        structuredOutput: { ok: true },
        artifacts: { gitFormatPatch: { status: "ready", totalCommitCount: 1 } },
      },
    });
    expect(byAlias.get("interrupted-task")).toMatchObject({
      status: "interrupted",
      result: { kind: "interrupted" },
    });
    expect(byAlias.get("reported-task")?.executionId).not.toBe("reported-task");
  });

  test("golden adapts workspace turns with durable results and attach artifacts", async () => {
    const store = new TaskHandleStore(config);
    await store.upsertWorkspaceTurn({
      kind: "workspace_turn",
      handleId: "wst_golden",
      ownerWorkspaceId: OWNER,
      workspaceId: "target-workspace",
      turnId: "turn-1",
      status: "completed",
      createdAt: CREATED_AT,
      updatedAt: "2026-08-06T00:00:03.000Z",
      createdWorkspace: false,
      disposableWorkspace: false,
      title: "Review",
      prompt: "Review this",
      reportMarkdown: "Workspace turn complete",
      finalMessageRef: { messageId: "message-1", partCount: 2 },
      artifacts: {
        attachFiles: [
          {
            path: "/tmp/chart.png",
            filename: "chart.png",
            mediaType: "image/png",
            sourceToolCallId: "attach-1",
          },
        ],
      },
      attentionPolicy: "notify_on_terminal",
      terminalAttentionNotifiedAt: "2026-08-06T00:00:04.000Z",
    });

    expect(await registry.get(OWNER, "wst_golden")).toMatchObject({
      aliases: ["wst_golden"],
      target: { kind: "workspace", workspaceId: "target-workspace", origin: "existing" },
      launchPolicy: {
        kind: "workspace_turn",
        turnId: "turn-1",
        title: "Review",
        prompt: "Review this",
      },
      retentionPolicy: { kind: "retain_workspace" },
      attentionPolicy: "notify_on_terminal",
      status: "completed",
      terminalAttentionNotifiedAt: "2026-08-06T00:00:04.000Z",
      result: {
        kind: "completed",
        reportMarkdown: "Workspace turn complete",
        finalMessageRef: { messageId: "message-1", partCount: 2 },
        artifacts: {
          attachFiles: [{ path: "/tmp/chart.png", mediaType: "image/png" }],
        },
      },
    });
  });

  test("reads report and failure artifacts after legacy child workspace cleanup", async () => {
    const sessionDir = config.getSessionDir(OWNER);
    await upsertSubagentReportArtifact({
      workspaceId: OWNER,
      workspaceSessionDir: sessionDir,
      childTaskId: "cleaned-report-task",
      parentWorkspaceId: OWNER,
      ancestorWorkspaceIds: [OWNER],
      reportMarkdown: "Still durable",
      nowMs: Date.parse("2026-08-06T00:00:06.000Z"),
    });
    await upsertSubagentFailureArtifact({
      workspaceId: OWNER,
      workspaceSessionDir: sessionDir,
      childTaskId: "cleaned-failure-task",
      parentWorkspaceId: OWNER,
      ancestorWorkspaceIds: [OWNER],
      errorType: "model_refusal",
      errorMessage: "Model refused",
      nowMs: Date.parse("2026-08-06T00:00:07.000Z"),
    });

    expect(await registry.get(OWNER, "cleaned-report-task")).toMatchObject({
      status: "completed",
      result: { kind: "completed", reportMarkdown: "Still durable" },
    });
    expect(await registry.get(OWNER, "cleaned-failure-task")).toMatchObject({
      status: "error",
      result: { kind: "error", errorType: "model_refusal", error: "Model refused" },
    });
  });

  test("queries legacy chains and excludes transcript-only terminal workspaces from active results", async () => {
    await addAgentTask(config, "legacy-root", "running");
    await addAgentTask(config, "legacy-child", "starting", {
      parentWorkspaceId: "legacy-root",
      createdAt: "2026-08-06T00:00:01.000Z",
    });
    await addAgentTask(config, "legacy-terminal", "running", {
      parentWorkspaceId: "legacy-child",
      transcriptOnly: true,
      createdAt: "2026-08-06T00:00:02.000Z",
    });

    const handles = await registry.listAgentExecutions(OWNER);
    const root = handles.find((handle) => handle.aliases?.includes("legacy-root"));
    const child = handles.find((handle) => handle.aliases?.includes("legacy-child"));
    const terminal = handles.find((handle) => handle.aliases?.includes("legacy-terminal"));
    assert(root != null);
    assert(child != null);
    assert(terminal != null);

    expect(child.parentExecutionId).toBe(root.executionId);
    expect(terminal.parentExecutionId).toBe(child.executionId);
    expect(terminal).toMatchObject({
      status: "interrupted",
      result: { kind: "interrupted" },
    });
    expect(await registry.getAgentExecutionDepth(OWNER, "legacy-root")).toBe(1);
    expect(await registry.getAgentExecutionDepth(OWNER, "legacy-child")).toBe(2);
    expect(await registry.getAgentExecutionDepth(OWNER, "legacy-terminal")).toBe(3);
    expect(
      (await registry.listDescendantAgentExecutions(OWNER, root.executionId)).map(
        (handle) => handle.aliases?.[0]
      )
    ).toEqual(["legacy-child", "legacy-terminal"]);
    expect(
      (await registry.listActiveAgentExecutions(OWNER)).map((handle) => handle.aliases?.[0])
    ).toEqual(["legacy-root", "legacy-child"]);
  });

  test("maps a legacy child's parent to the canonical parent workspace execution", async () => {
    await addAgentTask(config, "canonical-parent-workspace", "running", {
      executionId: "exe_canonical_parent",
    });
    await addAgentTask(config, "legacy-child", "running", {
      parentWorkspaceId: "canonical-parent-workspace",
      createdAt: "2026-08-06T00:00:01.000Z",
    });
    const canonicalParent = canonicalHandle({
      executionId: "exe_canonical_parent",
      aliases: ["canonical-parent-workspace"],
      target: {
        kind: "workspace",
        workspaceId: "canonical-parent-workspace",
        origin: "created",
      },
      status: "running",
      startedAt: CREATED_AT,
    });
    await new ExecutionStore(config).upsert(canonicalParent);

    const child = await registry.get(OWNER, "legacy-child");
    expect(child).toMatchObject({
      requesterWorkspaceId: "canonical-parent-workspace",
      parentExecutionId: canonicalParent.executionId,
    });
    expect(await registry.getAgentExecutionDepth(OWNER, "legacy-child")).toBe(2);
    expect(
      (await registry.listDescendantAgentExecutions(OWNER, canonicalParent.executionId)).map(
        (handle) => handle.aliases?.[0]
      )
    ).toEqual(["legacy-child"]);
  });

  test("canonical records win over legacy aliases without rewriting legacy state", async () => {
    await addAgentTask(config, "running-task", "running");
    const canonical = {
      version: 1 as const,
      executionId: "exe_canonical",
      aliases: ["running-task"],
      ownerSessionId: OWNER,
      requesterWorkspaceId: OWNER,
      target: {
        kind: "workspace" as const,
        workspaceId: "running-task",
        origin: "created" as const,
      },
      launchPolicy: { kind: "agent_task" as const, agentId: "exec" },
      completionPolicy: { kind: "final_assistant_message" as const },
      retentionPolicy: { kind: "delete_workspace_on_completion" as const },
      attentionPolicy: "blocking_until_terminal" as const,
      status: "starting" as const,
      createdAt: CREATED_AT,
      updatedAt: CREATED_AT,
    };
    await new ExecutionStore(config).upsert(canonical);

    expect(await registry.get(OWNER, "running-task")).toEqual(canonical);
    expect(
      (await registry.listAgentExecutions(OWNER)).filter((item) =>
        item.aliases?.includes("running-task")
      )
    ).toEqual([canonical]);
  });
});
