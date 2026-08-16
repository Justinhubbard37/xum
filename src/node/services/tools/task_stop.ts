import { tool } from "ai";

import { getErrorMessage } from "@/common/utils/errors";
import type { ToolConfiguration, ToolFactory } from "@/common/utils/tools/tools";
import { WorkflowRunRecordSchema } from "@/common/orpc/schemas";
import {
  isActiveWorkflowRunStatus,
  isNestedWorkflowRun,
  isTerminalWorkflowRunStatus,
} from "@/common/types/workflow";
import { TaskStopToolResultSchema, TOOL_DEFINITIONS } from "@/common/utils/tools/toolDefinitions";

import { TASK_TERMINATION_TOOL_TIMEOUT_MS } from "@/constants/terminationTimeouts";
import { log } from "@/node/services/log";
import type { TaskService } from "@/node/services/taskService";
import { raceWithAbortAndTimeout } from "@/node/utils/concurrency/withTimeout";
import { isWorkspaceTurnTaskId } from "@/node/services/taskHandleStore";
import { fromBashTaskId, isWorkflowRunTaskId } from "./taskId";
import {
  dedupeStrings,
  parseToolResult,
  requireTaskService,
  requireWorkspaceId,
} from "./toolUtils";

const WORKFLOW_STOPPED_NOTE =
  "Workflow run stopped. Durable state is preserved; resume it later with workflow_resume.";

/**
 * Workflow runs are interrupted (resumable) rather than terminated: the durable event log is
 * preserved, which is why this reports a distinct "interrupted" status instead of "terminated"
 * (whose contract says in-progress work is discarded).
 */
async function interruptWorkflowRun(
  config: ToolConfiguration,
  workspaceId: string,
  taskId: string,
  options?: {
    deferTaskSweep?: boolean;
    lockAlreadyHeld?: boolean;
    onRunInterrupted?: (runId: string) => void;
  }
) {
  const workflowService = config.workflowService;
  if (workflowService?.getRun == null || workflowService.interruptRun == null) {
    return {
      status: "error" as const,
      taskId,
      error: "Workflow service not available for workflow run interrupts",
    };
  }

  // getRun is workspace-scoped: runs owned by other workspaces are reported as not found.
  const rawRun = await workflowService.getRun({ workspaceId, runId: taskId });
  if (rawRun == null) {
    return { status: "not_found" as const, taskId };
  }
  // safeParse keeps batch entries isolated: one unreadable record must not collapse the
  // whole Promise.all into a single opaque tool error (self-healing doctrine).
  const parsedRun = WorkflowRunRecordSchema.safeParse(rawRun);
  if (!parsedRun.success) {
    return {
      status: "error" as const,
      taskId,
      error: "Workflow run record is unreadable and cannot be interrupted.",
    };
  }
  const run = parsedRun.data;

  if (run.status === "completed" || run.status === "failed") {
    return { status: "already_inactive" as const, taskId };
  }

  if (run.status === "interrupted") {
    try {
      await workflowService.interruptRun({
        workspaceId,
        runId: taskId,
        retryTaskCleanup: true,
        ...(options?.deferTaskSweep === true ? { deferTaskSweep: true } : {}),
        ...(options?.lockAlreadyHeld === true ? { lockAlreadyHeld: true } : {}),
        onRunInterrupted: options?.onRunInterrupted,
      });
      return { status: "already_inactive" as const, taskId };
    } catch (error: unknown) {
      return { status: "error" as const, taskId, error: getErrorMessage(error) };
    }
  }

  let persistedByThisCall = false;
  try {
    await workflowService.interruptRun({
      workspaceId,
      runId: taskId,
      ...(options?.deferTaskSweep === true ? { deferTaskSweep: true } : {}),
      ...(options?.lockAlreadyHeld === true ? { lockAlreadyHeld: true } : {}),
      onRunInterrupted: (runId) => {
        persistedByThisCall ||= runId === taskId;
        options?.onRunInterrupted?.(runId);
      },
    });
  } catch (error: unknown) {
    // A terminal re-read means idempotent success only when another actor won the transition race.
    // If this invocation persisted interrupted and cleanup then failed, surface that failure so the
    // caller does not report success while workflow-owned workers may still be active.
    if (!persistedByThisCall) {
      const latestRawRun = await workflowService
        .getRun({ workspaceId, runId: taskId })
        .catch(() => null);
      const latestRun = WorkflowRunRecordSchema.safeParse(latestRawRun);
      if (latestRun.success && isTerminalWorkflowRunStatus(latestRun.data.status)) {
        return { status: "already_inactive" as const, taskId };
      }
    }
    return { status: "error" as const, taskId, error: getErrorMessage(error) };
  }
  return { status: "stopped" as const, taskId, note: WORKFLOW_STOPPED_NOTE };
}

async function interruptWorkflowRunsOwnedByAgentTaskTree(
  config: ToolConfiguration,
  taskService: TaskService,
  taskId: string,
  deferredWorkflowRunIds: string[]
): Promise<string | null> {
  const listDescendants = taskService.listDescendantAgentTasks?.bind(taskService);
  if (listDescendants == null) {
    return null;
  }

  const descendants = listDescendants(taskId);
  const userOwnedDescendants = listDescendants(taskId, { excludeWorkflowTasks: true });
  const userOwnedTaskIds = new Set(userOwnedDescendants.map((task) => task.taskId));
  const activeWorkflowOwnedDescendants = descendants.filter(
    (task) =>
      !userOwnedTaskIds.has(task.taskId) &&
      (task.status === "queued" ||
        task.status === "starting" ||
        task.status === "running" ||
        task.status === "awaiting_report")
  );

  const workflowService = config.workflowService;
  if (workflowService?.listRuns == null || workflowService.interruptRun == null) {
    return activeWorkflowOwnedDescendants.length > 0
      ? "Workflow service not available to stop workflow-owned descendants"
      : null;
  }

  let activeRunCount = 0;
  for (const ownerWorkspaceId of [taskId, ...userOwnedTaskIds]) {
    const rawRuns = await workflowService.listRuns({ workspaceId: ownerWorkspaceId });
    for (const rawRun of rawRuns) {
      const parsedRun = WorkflowRunRecordSchema.safeParse(rawRun);
      if (
        !parsedRun.success ||
        (!isActiveWorkflowRunStatus(parsedRun.data.status) &&
          parsedRun.data.status !== "interrupted") ||
        isNestedWorkflowRun(parsedRun.data)
      ) {
        continue;
      }

      activeRunCount += 1;
      const outcome = await interruptWorkflowRun(config, ownerWorkspaceId, parsedRun.data.id, {
        deferTaskSweep: true,
        lockAlreadyHeld: true,
        onRunInterrupted: (runId) => deferredWorkflowRunIds.push(runId),
      });
      if (outcome.status === "stopped") {
        deferredWorkflowRunIds.push(parsedRun.data.id);
      }
      if (outcome.status === "error") {
        return outcome.error;
      }
      if (outcome.status === "not_found") {
        return `Workflow run ${parsedRun.data.id} disappeared before it could be stopped`;
      }
    }
  }

  if (activeWorkflowOwnedDescendants.length > 0 && activeRunCount === 0) {
    return "Active workflow-owned descendants have no active owning workflow run";
  }
  return null;
}

async function sweepDeferredWorkflowRuns(
  taskService: TaskService,
  taskId: string,
  workflowRunIds: string[]
): Promise<void> {
  for (const workflowRunId of dedupeStrings(workflowRunIds)) {
    try {
      await taskService.markWorkflowRunEnded(workflowRunId);
    } catch (error: unknown) {
      log.warn("task_stop deferred workflow sweep failed", {
        taskId,
        workflowRunId,
        error: getErrorMessage(error),
      });
    }
  }
}

export const createTaskStopTool: ToolFactory = (config: ToolConfiguration) => {
  return tool({
    description: TOOL_DEFINITIONS.task_stop.description,
    inputSchema: TOOL_DEFINITIONS.task_stop.schema,
    execute: async (args, { abortSignal }): Promise<unknown> => {
      const workspaceId = requireWorkspaceId(config, "task_stop");
      const taskService = requireTaskService(config, "task_stop");

      const uniqueTaskIds = dedupeStrings(args.task_ids);

      const results = await Promise.all(
        uniqueTaskIds.map(async (taskId) => {
          // A pre-aborted call must not start stop work at all.
          if (abortSignal?.aborted) {
            return {
              status: "error" as const,
              taskId,
              error: "Termination interrupted before it started",
            };
          }
          const terminationPromise = (async () => {
            try {
              if (isWorkflowRunTaskId(taskId)) {
                return await interruptWorkflowRun(config, workspaceId, taskId);
              }

              if (isWorkspaceTurnTaskId(taskId)) {
                const interruptResult = await taskService.interruptWorkspaceTurn(
                  workspaceId,
                  taskId
                );
                if (!interruptResult.success) {
                  const msg = interruptResult.error;
                  if (/not found/i.test(msg) || /scope/i.test(msg)) {
                    return { status: "invalid_scope" as const, taskId };
                  }
                  return { status: "error" as const, taskId, error: msg };
                }
                if (interruptResult.data.alreadyInactive === true) {
                  return { status: "already_inactive" as const, taskId };
                }
                return {
                  status: "stopped" as const,
                  taskId,
                  note: "Workspace turn stopped. The workspace is preserved for future messages.",
                };
              }

              const maybeProcessId = fromBashTaskId(taskId);
              if (taskId.startsWith("bash:") && !maybeProcessId) {
                return { status: "error" as const, taskId, error: "Invalid bash taskId." };
              }

              if (maybeProcessId) {
                if (!config.backgroundProcessManager) {
                  return {
                    status: "error" as const,
                    taskId,
                    error: "Background process manager not available",
                  };
                }

                const proc = await config.backgroundProcessManager.getProcess(maybeProcessId);
                if (!proc) {
                  return { status: "not_found" as const, taskId };
                }

                const inScope =
                  proc.workspaceId === workspaceId ||
                  (await taskService.isDescendantAgentTask(workspaceId, proc.workspaceId));
                const workflowOwned =
                  proc.workspaceId !== workspaceId &&
                  ((await taskService.isWorkflowOwnedDescendantAgentTask?.(
                    workspaceId,
                    proc.workspaceId
                  )) ??
                    false);
                if (!inScope || workflowOwned) {
                  return { status: "invalid_scope" as const, taskId };
                }

                const terminateResult =
                  await config.backgroundProcessManager.terminate(maybeProcessId);
                if (!terminateResult.success) {
                  return { status: "error" as const, taskId, error: terminateResult.error };
                }

                return {
                  status: "stopped" as const,
                  taskId,
                  stoppedTaskIds: [taskId],
                };
              }

              const deferredWorkflowRunIds: string[] = [];
              const stopResult = await taskService.stopDescendantAgentTask(workspaceId, taskId, {
                // Run workflow interruption while TaskService holds the task-tree lifecycle lock.
                // Workflow worker creation uses the same lock, so no new workflow-owned branch can
                // appear between discovery and the direct user-owned subtree stop. Archive sweeps
                // are deferred because WorkspaceService.archive reacquires this non-reentrant lock.
                beforeStop:
                  taskService.listDescendantAgentTasks != null
                    ? async () =>
                        await interruptWorkflowRunsOwnedByAgentTaskTree(
                          config,
                          taskService,
                          taskId,
                          deferredWorkflowRunIds
                        )
                    : undefined,
              });
              await sweepDeferredWorkflowRuns(taskService, taskId, deferredWorkflowRunIds);
              if (!stopResult.success) {
                const msg = stopResult.error;
                // Exact-match the canonical scope errors: aggregated cleanup failures
                // may mention "descendant" or "not found" and must stay actionable errors.
                if (msg === "Task not found") {
                  return { status: "not_found" as const, taskId };
                }
                if (msg === "Task is not a descendant of this workspace") {
                  return { status: "invalid_scope" as const, taskId };
                }
                return { status: "error" as const, taskId, error: msg };
              }

              return stopResult.data.stoppedTaskIds.length === 0
                ? { status: "already_inactive" as const, taskId }
                : {
                    status: "stopped" as const,
                    taskId,
                    stoppedTaskIds: stopResult.data.stoppedTaskIds,
                  };
            } catch (error: unknown) {
              return { status: "error" as const, taskId, error: getErrorMessage(error) };
            }
          })();

          const outcome = await raceWithAbortAndTimeout(terminationPromise, {
            signal: abortSignal,
            timeoutMs: TASK_TERMINATION_TOOL_TIMEOUT_MS,
          });
          if (outcome.kind === "ok") {
            return outcome.value;
          }

          void terminationPromise.catch((error: unknown) => {
            log.debug("task_stop cleanup failed after tool returned", { taskId, error });
          });
          return {
            status: "error" as const,
            taskId,
            error:
              outcome.kind === "aborted"
                ? "Termination interrupted; cleanup continues in the background"
                : "Termination timed out; cleanup continues in the background",
          };
        })
      );

      return parseToolResult(TaskStopToolResultSchema, { results }, "task_stop");
    },
  });
};
