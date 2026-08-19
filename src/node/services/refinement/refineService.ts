/**
 * /refine orchestration (RLM track, phase r11): user-invokable trajectory
 * distillation with a paper trail.
 *
 * Owns everything around the runner (refineRunner.ts): RLM experiment gating
 * (backend refuses when off), one-run-at-a-time-per-workspace locking
 * (concurrent invocations are REJECTED, not queued — an explicit /refine has
 * nothing to gain from running twice over the same trajectory), trajectory
 * assembly (recent chat.jsonl + timeline events when the Timeline experiment
 * is on), model resolution, journal-row correlation, and the completion chat
 * message.
 *
 * v1 tradeoff (intentional, no proposal/approval UI): edits are auto-applied
 * and the summary row points at the r6 rollback paths ("bun run debug
 * refinements" / the refinement_rollback tool). Approval UX would double the
 * surface of an experimental feature whose every edit is already journaled
 * with a byte-exact inverse — cheap rollback is the safety mechanism.
 *
 * Failure posture: best-effort everywhere below the run result. Summary-row
 * append or emission failures log and continue (self-healing doctrine); a
 * stream failure returns an error so the user knows the pass did not finish.
 */
import * as os from "node:os";
import type { Tool } from "ai";

import { EXPERIMENT_IDS, type ExperimentId } from "@/common/constants/experiments";
import type { RefineAppliedEditPayload, RefineRecordPayload } from "@/common/orpc/schemas/api";
import { createMuxMessage, type MuxMessage } from "@/common/types/message";
import {
  MemoryRefinementActionSchema,
  RefinementEvidenceSchema,
  SkillRefinementActionSchema,
} from "@/common/types/refinement";
import { Err, Ok, type Result } from "@/common/types/result";
import { getErrorMessage } from "@/common/utils/errors";
import type { ToolConfiguration } from "@/common/utils/tools/tools";
import {
  REFINE_MAX_MESSAGES,
  REFINE_SUMMARY_LABEL,
  REFINE_TIMELINE_EVENT_LIMIT,
  REFINE_TIMEOUT_MS,
} from "@/constants/refine";
import type { Config } from "@/node/config";
import { LocalRuntime } from "@/node/runtime/LocalRuntime";
import type { AIService } from "@/node/services/aiService";
import { buildAbandonedBranchTranscript, isRlmModeEnabled } from "@/node/services/branchSummary";
import type { HistoryService } from "@/node/services/historyService";
import { log } from "@/node/services/log";
import {
  resolveConsolidationProjectPath,
  resolveDreamModelString,
} from "@/node/services/memoryConsolidationService";
import type { MemoryMetaService } from "@/node/services/memoryMeta";
import type { MemoryScopeContext, MemoryService } from "@/node/services/memoryService";
import { modelCostsIncluded } from "@/node/services/providerModelFactory";
import {
  listRefinements,
  type RefinementEvent,
} from "@/node/services/refinement/refinementRollback";
import { runRefinePass } from "@/node/services/refinement/refineRunner";
import type { SessionUsageService } from "@/node/services/sessionUsageService";
import type { TimelineService } from "@/node/services/timelineService";
import { createAgentSkillWriteTool } from "@/node/services/tools/agent_skill_write";
import { sharedDurableEventJournal } from "@/node/utils/journal/durableEventJournal";
import { createRefineSummaryMessageId } from "@/node/services/utils/messageIds";

// Types derive from the oRPC schemas (z.infer single source) so node-side
// fields can never silently be stripped by output validation.
export type RefineAppliedEdit = RefineAppliedEditPayload;
export type RefineRecord = RefineRecordPayload;

interface ExperimentsCheck {
  isExperimentEnabled(experimentId: ExperimentId): boolean;
}

/** Structural AIService subset (model creation + runtime metadata). */
type RefineAiService = Pick<AIService, "createModelWithPinnedMetadata" | "getWorkspaceMetadata">;

interface RefineServiceOptions {
  timelineService?: Pick<TimelineService, "list">;
  sessionUsageService?: SessionUsageService;
  /** Live-session emission hook so the appended summary row renders immediately. */
  emitChatMessage?: (workspaceId: string, message: MuxMessage) => void;
}

/** Human-readable action line for a refinement journal row. */
export function describeRefinementRow(row: RefinementEvent): string {
  if (row.data.kind === "memory") {
    const action = MemoryRefinementActionSchema.safeParse(row.data.action);
    if (action.success) {
      const rename = action.data.newPath !== undefined ? ` -> ${action.data.newPath}` : "";
      return `memory ${action.data.op} ${action.data.path}${rename}`;
    }
  }
  if (row.data.kind === "skill") {
    const action = SkillRefinementActionSchema.safeParse(row.data.action);
    if (action.success) {
      const file = action.data.filePath !== undefined ? `/${action.data.filePath}` : "";
      return `skill ${action.data.op} ${action.data.skillName}${file}`;
    }
  }
  return `${row.data.kind} edit`;
}

/** Build the durable, clearly-labeled summary row for an applied refine pass. */
export function createRefineSummaryMessage(record: RefineRecord): MuxMessage {
  const lines = [
    REFINE_SUMMARY_LABEL,
    "",
    ...record.applied.map((edit) => `- ${edit.description} (refinement ${edit.refinementId})`),
  ];
  if (record.summary.length > 0) {
    lines.push("", record.summary);
  }
  lines.push(
    "",
    "Rollback with: /debug refinements (bun run debug refinements <workspace-id> --rollback <id>) or the refinement_rollback tool."
  );
  return createMuxMessage(createRefineSummaryMessageId(), "user", lines.join("\n"), {
    timestamp: Date.now(),
    // Synthetic system-style row: provider-visible durable history (never
    // request-time injection), uiVisible so users see what was self-applied.
    synthetic: true,
    uiVisible: true,
    muxMetadata: { type: "refine-summary" },
  });
}

export class RefineService {
  /**
   * Per-workspace run lock. Reserved SYNCHRONOUSLY in run() before any await
   * so two near-simultaneous invocations can never both start; the loser is
   * rejected outright (see module doc).
   */
  private readonly inFlight = new Map<string, Promise<Result<RefineRecord, string>>>();

  constructor(
    private readonly config: Config,
    private readonly memoryService: MemoryService,
    private readonly metaService: MemoryMetaService,
    private readonly historyService: HistoryService,
    private readonly aiService: RefineAiService,
    private readonly experiments: ExperimentsCheck,
    private readonly options: RefineServiceOptions = {}
  ) {}

  private enabled(): boolean {
    // RLM is a sub-experiment of Programmatic Tool Calling; both machine
    // overrides must be on (same fallback path as backend-initiated branch
    // summaries — /refine has no send options to ride on).
    return isRlmModeEnabled(undefined, (id) => this.experiments.isExperimentEnabled(id));
  }

  async run(workspaceId: string): Promise<Result<RefineRecord, string>> {
    if (!this.enabled()) {
      return Err("rlm-mode experiment is disabled (enable Programmatic Tool Calling + RLM Mode)");
    }
    if (this.inFlight.has(workspaceId)) {
      return Err("a refine pass is already running for this workspace");
    }
    // runLocked executes synchronously up to its first await, so the map is
    // populated before any other caller can observe it.
    const run = this.runLocked(workspaceId);
    this.inFlight.set(workspaceId, run);
    try {
      return await run;
    } finally {
      this.inFlight.delete(workspaceId);
    }
  }

  private async runLocked(workspaceId: string): Promise<Result<RefineRecord, string>> {
    const workspace = this.config.findWorkspace(workspaceId);
    if (!workspace) return Err(`workspace not found: ${workspaceId}`);

    const messagesResult = await this.historyService.getLastMessages(
      workspaceId,
      REFINE_MAX_MESSAGES
    );
    if (!messagesResult.success) {
      return Err(`could not read workspace history: ${messagesResult.error}`);
    }
    // Reuse the branch-summary transcript builder: role-labeled,
    // thinking-stripped, char-bounded — exactly the evidence shape a
    // distillation pass needs.
    const transcript = buildAbandonedBranchTranscript(messagesResult.data);
    if (transcript.length === 0) {
      // Empty trajectory: a clean first-class no-op without spending a model call.
      return Ok({ applied: [], summary: "Nothing worth distilling.", noOp: true });
    }

    const timelineText = await this.buildTimelineText(workspaceId);

    // Model: reuse the dream-agent inherit cascade — refine is the same class
    // of background self-maintenance agent, so a per-workspace dream override
    // intentionally covers both.
    const modelString = resolveDreamModelString(this.config, workspaceId);
    const modelResult = await this.aiService.createModelWithPinnedMetadata(modelString, {
      agentInitiated: true,
      workspaceId,
    });
    if (!modelResult.success) {
      return Err(`could not create model ${modelString}: ${modelResult.error.type}`);
    }

    const projectPath = resolveConsolidationProjectPath(workspace);
    const ctx: MemoryScopeContext = {
      runtime: null,
      checkoutCwd: "",
      workspaceId,
      projectPath,
    };

    const sessionDir = this.config.getSessionDir(workspaceId);
    // Baseline BEFORE the pass: rows appended by this run have seq > baseline.
    // Correlation additionally requires the row's evidence.toolCallId to be
    // one of this pass's tool calls, so concurrent main-agent self-edits in
    // the same journal can never be misattributed to the refine pass.
    const baselineSeq = await this.readMaxJournalSeq(sessionDir);

    const skillWriteTool = await this.buildSkillWriteTool(workspaceId, sessionDir);

    const result = await runRefinePass({
      model: modelResult.data.model,
      memoryService: this.memoryService,
      metaService: this.metaService,
      ctx,
      transcript,
      timelineText,
      skillWriteTool,
      // Hard timeout: a wedged provider stream must not hold the run lock forever.
      abortSignal: AbortSignal.timeout(REFINE_TIMEOUT_MS),
      recordUsage: async (usage, providerMetadata) => {
        await this.options.sessionUsageService?.recordHeadlessUsage(
          workspaceId,
          modelString,
          usage,
          providerMetadata,
          {
            costsIncluded: modelCostsIncluded(modelResult.data.model),
            analyticsSource: "refine",
            metadataModel: modelResult.data.metadataModel,
          }
        );
      },
    });
    if (result.streamError !== undefined) {
      // Edits applied before the failure remain journaled + rollbackable;
      // point the user at the audit trail instead of hiding them.
      return Err(
        `refine stream failed: ${result.streamError} (any applied edits are listed by 'bun run debug refinements ${workspaceId}')`
      );
    }

    const applied = await this.collectAppliedEdits(
      sessionDir,
      workspaceId,
      baselineSeq,
      result.toolCallIds
    );
    const record: RefineRecord = {
      applied,
      summary: result.summary.length > 0 ? result.summary : "Nothing worth distilling.",
      noOp: applied.length === 0,
      usage: result.usage,
    };

    log.debug("[Refine] pass complete", {
      workspaceId,
      applied: applied.length,
      budgetExhausted: result.budgetExhausted,
      usage: result.usage,
    });

    // Completion UX: post the labeled summary row ONLY when edits were
    // applied — a no-op stays out of chat (the invoking toast reports it).
    if (!record.noOp) {
      await this.appendSummaryMessage(workspaceId, record);
    }
    return Ok(record);
  }

  /** Newest journal seq, or -1 for a fresh/absent journal. */
  private async readMaxJournalSeq(sessionDir: string): Promise<number> {
    const events = await sharedDurableEventJournal(sessionDir).read();
    return events.reduce((max, event) => Math.max(max, event.seq), -1);
  }

  private async collectAppliedEdits(
    sessionDir: string,
    workspaceId: string,
    baselineSeq: number,
    toolCallIds: string[]
  ): Promise<RefineAppliedEdit[]> {
    if (toolCallIds.length === 0) return [];
    const callIds = new Set(toolCallIds);
    const rows = await listRefinements(sessionDir);
    const applied: RefineAppliedEdit[] = [];
    for (const row of rows) {
      if (row.seq <= baselineSeq || row.workspaceId !== workspaceId) continue;
      const evidence = RefinementEvidenceSchema.safeParse(row.data.evidence);
      if (!evidence.success) continue;
      if (evidence.data.toolCallId === undefined || !callIds.has(evidence.data.toolCallId)) {
        continue;
      }
      applied.push({ refinementId: row.id, description: describeRefinementRow(row) });
    }
    return applied;
  }

  /**
   * Standard agent_skill_write tool confined to the workspace checkout's
   * .mux/skills (project scope). Only for host-local single-project
   * workspaces: remote runtimes would need a live runtime connection and
   * multi-project workspaces have no single skills root. Memory scopes remain
   * available either way. Returns undefined (memory-only pass) on any
   * resolution failure — never fails the run.
   */
  private async buildSkillWriteTool(
    workspaceId: string,
    sessionDir: string
  ): Promise<Tool | undefined> {
    try {
      const metadataResult = await this.aiService.getWorkspaceMetadata(workspaceId);
      if (!metadataResult.success) return undefined;
      const metadata = metadataResult.data;
      const runtimeType = metadata.runtimeConfig.type;
      if (runtimeType === "ssh" || runtimeType === "docker") return undefined;
      if ((metadata.projects?.length ?? 0) > 1) return undefined;
      const workspace = this.config.findWorkspace(workspaceId);
      if (!workspace) return undefined;
      const projectRoot = workspace.workspacePath;

      // Minimal host-local ToolConfiguration: the project-local skill path
      // only touches fs/promises under muxScope roots; workspaceSessionDir +
      // workspaceId make the tool's r2 refinement journaling land in this
      // session's durable journal.
      const toolConfig: ToolConfiguration = {
        cwd: projectRoot,
        runtime: new LocalRuntime(projectRoot),
        runtimeTempDir: os.tmpdir(),
        workspaceSessionDir: sessionDir,
        workspaceId,
        muxScope: {
          type: "project",
          muxHome: this.config.rootDir,
          projectRoot,
          projectStorageAuthority: "host-local",
        },
      };
      return createAgentSkillWriteTool(toolConfig);
    } catch (error) {
      log.debug("[Refine] skill tool unavailable; running memory-only", {
        workspaceId,
        error: getErrorMessage(error),
      });
      return undefined;
    }
  }

  /** Timeline digest when the Timeline experiment is on; undefined otherwise. */
  private async buildTimelineText(workspaceId: string): Promise<string | undefined> {
    if (!this.experiments.isExperimentEnabled(EXPERIMENT_IDS.TIMELINE)) return undefined;
    if (this.options.timelineService === undefined) return undefined;
    try {
      const page = await this.options.timelineService.list(workspaceId, {
        limit: REFINE_TIMELINE_EVENT_LIMIT,
      });
      if (page.events.length === 0) return undefined;
      // list() returns newest-first; present oldest-first for the model.
      return [...page.events]
        .reverse()
        .map((event) => {
          const description = event.data?.description ?? event.data?.digest ?? "";
          return `${new Date(event.ts).toISOString()} ${event.kind}${
            description.length > 0 ? `: ${description}` : ""
          }`;
        })
        .join("\n");
    } catch (error) {
      log.debug("[Refine] timeline read failed; continuing without it", {
        workspaceId,
        error: getErrorMessage(error),
      });
      return undefined;
    }
  }

  /** Best-effort: append + emit the summary row; failures log and continue. */
  private async appendSummaryMessage(workspaceId: string, record: RefineRecord): Promise<void> {
    try {
      const message = createRefineSummaryMessage(record);
      const appendResult = await this.historyService.appendToHistory(workspaceId, message);
      if (!appendResult.success) {
        log.warn("[Refine] failed to append summary row", {
          workspaceId,
          error: appendResult.error,
        });
        return;
      }
      this.options.emitChatMessage?.(workspaceId, message);
    } catch (error) {
      log.warn("[Refine] summary emission failed", {
        workspaceId,
        error: getErrorMessage(error),
      });
    }
  }
}
