import { createHash } from "node:crypto";

import {
  EXECUTION_HANDLE_VERSION,
  type ExecutionHandle,
  type ExecutionResult,
  type ExecutionStatus,
} from "@/common/types/execution";
import { resolveBackgroundWorkAttentionPolicy } from "@/common/types/backgroundWorkAttention";
import type { Workspace } from "@/common/types/project";
import type { Config } from "@/node/config";
import { ExecutionStore } from "@/node/services/executionStore";
import {
  TaskHandleStore,
  isWorkspaceTurnTaskId,
  type WorkspaceTurnTaskHandleRecord,
} from "@/node/services/taskHandleStore";
import {
  readSubagentFailureArtifact,
  readSubagentFailureArtifactsFile,
  type SubagentFailureArtifact,
} from "@/node/services/subagentFailureArtifacts";
import { readSubagentGitPatchArtifact } from "@/node/services/subagentGitPatchArtifacts";
import {
  readSubagentReportArtifact,
  readSubagentReportArtifactsFile,
  type SubagentReportArtifact,
} from "@/node/services/subagentReportArtifacts";

const EPOCH_ISO = new Date(0).toISOString();

type LegacyExecutionKind = "agent_task" | "workspace_turn";

function legacyExecutionId(kind: LegacyExecutionKind, sourceId: string): `exe_${string}` {
  const digest = createHash("sha256").update(`${kind}\0${sourceId}`).digest("hex").slice(0, 24);
  return `exe_legacy_${kind}_${digest}`;
}

function validIso(value: string | undefined): string | undefined {
  if (value == null || !Number.isFinite(Date.parse(value))) return undefined;
  return new Date(value).toISOString();
}

function msToIso(value: number | undefined): string | undefined {
  return value != null && Number.isFinite(value) ? new Date(value).toISOString() : undefined;
}

function terminalAt(status: ExecutionStatus, value: string): string | undefined {
  return status === "completed" || status === "interrupted" || status === "error"
    ? value
    : undefined;
}

/**
 * Read-through registry for canonical handles plus legacy task persistence.
 * Legacy sources are adapted in memory and never eagerly rewritten.
 */
export class ExecutionRegistry {
  private readonly executionStore: ExecutionStore;
  private readonly taskHandleStore: TaskHandleStore;

  constructor(
    private readonly config: Config,
    dependencies: {
      executionStore?: ExecutionStore;
      taskHandleStore?: TaskHandleStore;
    } = {}
  ) {
    this.executionStore = dependencies.executionStore ?? new ExecutionStore(config);
    this.taskHandleStore = dependencies.taskHandleStore ?? new TaskHandleStore(config);
  }

  async get(ownerSessionId: string, executionIdOrAlias: string): Promise<ExecutionHandle | null> {
    const direct = await this.executionStore.get(ownerSessionId, executionIdOrAlias);
    if (direct != null) return direct;

    const canonical = await this.executionStore.list(ownerSessionId);
    const aliased = canonical.find((handle) => handle.aliases?.includes(executionIdOrAlias));
    if (aliased != null) return aliased;

    if (isWorkspaceTurnTaskId(executionIdOrAlias)) {
      const workspaceTurn = await this.taskHandleStore.getWorkspaceTurn(
        ownerSessionId,
        executionIdOrAlias
      );
      if (workspaceTurn != null) return this.adaptWorkspaceTurn(workspaceTurn);
    }

    const legacyAgent = await this.readLegacyAgentTask(ownerSessionId, executionIdOrAlias);
    if (legacyAgent != null) return legacyAgent;

    const legacy = await this.listLegacy(ownerSessionId);
    return legacy.find((handle) => handle.executionId === executionIdOrAlias) ?? null;
  }

  async list(ownerSessionId: string): Promise<ExecutionHandle[]> {
    const canonical = await this.executionStore.list(ownerSessionId);
    const claimedIds = new Set(
      canonical.flatMap((handle) => [handle.executionId, ...(handle.aliases ?? [])])
    );
    const legacy = (await this.listLegacy(ownerSessionId)).filter(
      (handle) =>
        !claimedIds.has(handle.executionId) &&
        !(handle.aliases ?? []).some((alias) => claimedIds.has(alias))
    );
    return [...canonical, ...legacy].sort(
      (a, b) => a.createdAt.localeCompare(b.createdAt) || a.executionId.localeCompare(b.executionId)
    );
  }

  private async listLegacy(ownerSessionId: string): Promise<ExecutionHandle[]> {
    const workspaceTurns = await this.taskHandleStore.listWorkspaceTurns(ownerSessionId);
    const agentTasks = await this.listLegacyAgentTasks(ownerSessionId);
    return [...workspaceTurns.map((record) => this.adaptWorkspaceTurn(record)), ...agentTasks];
  }

  private adaptWorkspaceTurn(record: WorkspaceTurnTaskHandleRecord): ExecutionHandle {
    const createdAt = validIso(record.createdAt) ?? EPOCH_ISO;
    const updatedAt = validIso(record.updatedAt) ?? createdAt;
    const status = record.status;
    let result: ExecutionResult | undefined;
    if (status === "completed") {
      result = {
        kind: "completed",
        reportMarkdown: record.reportMarkdown ?? "",
        ...(record.finalMessageRef != null ? { finalMessageRef: record.finalMessageRef } : {}),
        ...(record.artifacts != null ? { artifacts: record.artifacts } : {}),
      };
    } else if (status === "interrupted") {
      result = {
        kind: "interrupted",
        ...(record.error != null ? { message: record.error } : {}),
      };
    } else if (status === "error") {
      result = { kind: "error", error: record.error ?? "Workspace turn failed" };
    }

    return {
      version: EXECUTION_HANDLE_VERSION,
      executionId: legacyExecutionId("workspace_turn", record.handleId),
      aliases: [record.handleId],
      ownerSessionId: record.ownerWorkspaceId,
      requesterWorkspaceId: record.ownerWorkspaceId,
      target: {
        kind: "workspace",
        workspaceId: record.workspaceId,
        origin: record.createdWorkspace ? "created" : "existing",
      },
      launchPolicy: {
        kind: "workspace_turn",
        turnId: record.turnId,
        ...(record.title != null ? { title: record.title } : {}),
        ...(record.prompt != null ? { prompt: record.prompt } : {}),
      },
      completionPolicy: { kind: "final_assistant_message" },
      retentionPolicy: {
        kind: record.disposableWorkspace ? "delete_workspace_on_completion" : "retain_workspace",
      },
      attentionPolicy: resolveBackgroundWorkAttentionPolicy(record.attentionPolicy),
      status,
      ...(result != null ? { result } : {}),
      createdAt,
      updatedAt,
      ...(status === "running" ? { startedAt: createdAt } : {}),
      ...(terminalAt(status, updatedAt) != null ? { terminalAt: updatedAt } : {}),
      ...(validIso(record.terminalAttentionNotifiedAt) != null
        ? { terminalAttentionNotifiedAt: validIso(record.terminalAttentionNotifiedAt) }
        : {}),
    };
  }

  private getLegacyAgentWorkspaces(ownerSessionId: string): Map<string, Workspace> {
    const byId = new Map<string, Workspace>();
    const config = this.config.loadConfigOrDefault();
    for (const project of config.projects.values()) {
      for (const workspace of project.workspaces) {
        if (workspace.parentWorkspaceId === ownerSessionId && workspace.id != null) {
          byId.set(workspace.id, workspace);
        }
      }
    }
    return byId;
  }

  private async listLegacyAgentTasks(ownerSessionId: string): Promise<ExecutionHandle[]> {
    const sessionDir = this.config.getSessionDir(ownerSessionId);
    const [reports, failures] = await Promise.all([
      readSubagentReportArtifactsFile(sessionDir),
      readSubagentFailureArtifactsFile(sessionDir),
    ]);
    const workspaces = this.getLegacyAgentWorkspaces(ownerSessionId);
    const taskIds = new Set([
      ...workspaces.keys(),
      ...Object.keys(reports.artifactsByChildTaskId),
      ...Object.keys(failures.failuresByChildTaskId),
    ]);
    const records = await Promise.all(
      [...taskIds].map((taskId) => this.readLegacyAgentTask(ownerSessionId, taskId, workspaces))
    );
    return records.filter((record): record is ExecutionHandle => record != null);
  }

  private async readLegacyAgentTask(
    ownerSessionId: string,
    taskId: string,
    knownWorkspaces = this.getLegacyAgentWorkspaces(ownerSessionId)
  ): Promise<ExecutionHandle | null> {
    const sessionDir = this.config.getSessionDir(ownerSessionId);
    const workspace = knownWorkspaces.get(taskId);
    const [report, failure] = await Promise.all([
      readSubagentReportArtifact(sessionDir, taskId),
      readSubagentFailureArtifact(sessionDir, taskId),
    ]);
    if (
      workspace == null &&
      report?.parentWorkspaceId !== ownerSessionId &&
      failure?.parentWorkspaceId !== ownerSessionId
    ) {
      return null;
    }
    const patch = await readSubagentGitPatchArtifact(sessionDir, taskId);
    return this.adaptAgentTask(ownerSessionId, taskId, workspace, report, failure, patch);
  }

  private adaptAgentTask(
    ownerSessionId: string,
    taskId: string,
    workspace: Workspace | undefined,
    report: SubagentReportArtifact | null,
    failure: SubagentFailureArtifact | null,
    patch: Awaited<ReturnType<typeof readSubagentGitPatchArtifact>>
  ): ExecutionHandle {
    let status: ExecutionStatus;
    let phase: "awaiting_report" | undefined;
    let result: ExecutionResult | undefined;
    if (report != null) {
      status = "completed";
      result = {
        kind: "completed",
        reportMarkdown: report.reportMarkdown,
        ...(report.structuredOutput !== undefined
          ? { structuredOutput: report.structuredOutput }
          : {}),
        ...(patch != null ? { artifacts: { gitFormatPatch: patch } } : {}),
      };
    } else if (failure != null || workspace?.taskLaunchError != null) {
      status = "error";
      result = {
        kind: "error",
        error: failure?.errorMessage ?? workspace?.taskLaunchError ?? "Agent task failed",
        ...(failure?.errorType != null ? { errorType: failure.errorType } : {}),
      };
    } else if (workspace?.taskStatus === "reported") {
      status = "completed";
      result = { kind: "completed", reportMarkdown: "" };
    } else if (workspace?.taskStatus === "interrupted") {
      status = "interrupted";
      result = { kind: "interrupted" };
    } else if (workspace?.taskStatus === "queued" || workspace?.taskStatus === "starting") {
      status = workspace.taskStatus;
    } else {
      status = "running";
      if (workspace?.taskStatus === "awaiting_report") phase = "awaiting_report";
    }

    const createdAt =
      validIso(workspace?.createdAt) ??
      msToIso(report?.createdAtMs) ??
      msToIso(failure?.createdAtMs) ??
      EPOCH_ISO;
    const updatedAt =
      validIso(workspace?.reportedAt) ??
      msToIso(report?.updatedAtMs) ??
      msToIso(failure?.updatedAtMs) ??
      createdAt;
    const title = workspace?.title ?? report?.title;
    const agentId = workspace?.agentId ?? workspace?.agentType;

    return {
      version: EXECUTION_HANDLE_VERSION,
      executionId: legacyExecutionId("agent_task", taskId),
      aliases: [taskId],
      ownerSessionId,
      requesterWorkspaceId: ownerSessionId,
      target: { kind: "workspace", workspaceId: taskId, origin: "created" },
      launchPolicy: {
        kind: "agent_task",
        ...(agentId != null ? { agentId } : {}),
        ...(title != null ? { title } : {}),
        ...(workspace?.taskPrompt != null ? { prompt: workspace.taskPrompt } : {}),
      },
      completionPolicy: { kind: "final_assistant_message" },
      retentionPolicy: { kind: "delete_workspace_on_completion" },
      attentionPolicy: resolveBackgroundWorkAttentionPolicy(workspace?.taskAttentionPolicy),
      status,
      ...(phase != null ? { phase } : {}),
      ...(result != null ? { result } : {}),
      createdAt,
      updatedAt,
      ...(status === "running" ? { startedAt: createdAt } : {}),
      ...(terminalAt(status, updatedAt) != null ? { terminalAt: updatedAt } : {}),
    };
  }
}
