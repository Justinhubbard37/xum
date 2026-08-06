import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import * as fsPromises from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";

import { DEFAULT_RUNTIME_CONFIG } from "@/common/constants/workspace";
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

async function addAgentTask(
  config: Config,
  taskId: string,
  taskStatus: Workspace["taskStatus"]
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
  });
}

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
      (await registry.list(OWNER)).filter((item) => item.aliases?.includes("running-task"))
    ).toEqual([canonical]);
  });
});
