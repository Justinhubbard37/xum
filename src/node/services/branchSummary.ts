/**
 * Branch summarization on fork/truncate (rlm-mode experiment).
 *
 * When RLM mode is on and history branches — a workspace forked from an
 * earlier message, or history truncated by an edit-resend — the abandoned
 * tail would otherwise vanish silently. This module summarizes that tail via
 * a cheap side-channel model call (thinking-stripped transcript, bounded
 * output tokens) and appends the summary as a durable, clearly-labeled user
 * row on the new branch BEFORE any subsequent provider request is built, so
 * log purity holds by construction: the row is ordinary durable history and
 * requests never inject live state.
 *
 * Failure posture: strictly best-effort. Model/key unavailability, timeouts,
 * or append failures skip the summary silently (log.debug) and never fail or
 * outlast the user-facing fork/edit operation beyond the hard deadline.
 */

import { streamText } from "ai";

import { EXPERIMENT_IDS, type ExperimentId } from "@/common/constants/experiments";
import { NAME_GEN_PREFERRED_MODELS } from "@/common/constants/nameGeneration";
import { WORDS_TO_TOKENS_RATIO, buildCompactionPrompt } from "@/common/constants/ui";
import { createMuxMessage, type MuxMessage } from "@/common/types/message";
import assert from "@/common/utils/assert";
import { getErrorMessage } from "@/common/utils/errors";
import { estimateMuxMessageTokens } from "@/common/utils/messages/keepRecentTail";
import {
  BRANCH_SUMMARY_MAX_OUTPUT_TOKENS,
  BRANCH_SUMMARY_MAX_TRANSCRIPT_CHARS,
  BRANCH_SUMMARY_MIN_SEGMENT_TOKENS,
  BRANCH_SUMMARY_TIMEOUT_MS,
} from "@/constants/branchSummary";

import type { AIService } from "./aiService";
import type { HistoryService } from "./historyService";
import { runLanguageModelCleanup } from "./languageModelCleanup";
import { log } from "./log";
import { createBranchSummaryMessageId } from "./utils/messageIds";

/** Human-readable marker prefixed to the durable summary row's text. */
export const BRANCH_SUMMARY_LABEL = "Summary of the abandoned branch:";

/** Structural subset of AIService so tests can pass lightweight fakes. */
export type BranchSummaryAiService = Pick<AIService, "createModel" | "getWorkspaceMetadata">;

/** Send-option experiment flags relevant to RLM gating (subset of ExperimentsSchema). */
export interface RlmExperimentFlags {
  rlm?: boolean;
  programmaticToolCalling?: boolean;
  programmaticToolCallingExclusive?: boolean;
}

/**
 * True when RLM mode applies. RLM is a sub-experiment of Programmatic Tool
 * Calling: without a PTC parent flag it stays inert (matching the experiments
 * registry). Send-option experiments are AUTHORITATIVE when present — the
 * frontend always sends the full boolean set (useSendMessageOptions /
 * sendOptions.ts), so an explicit `rlm: false` must win over machine
 * overrides, never fall through to them. Only backend-initiated operations
 * without send options (fork IPC) fall back to the persisted machine
 * overrides the renderer syncs into Settings.
 */
export function isRlmModeEnabled(
  experiments: RlmExperimentFlags | undefined,
  isExperimentEnabled: ((experimentId: ExperimentId) => boolean) | undefined
): boolean {
  if (experiments !== undefined) {
    return (
      experiments.rlm === true &&
      (experiments.programmaticToolCalling === true ||
        experiments.programmaticToolCallingExclusive === true)
    );
  }
  // Guard for test mocks that may not implement isExperimentEnabled.
  if (typeof isExperimentEnabled !== "function") {
    return false;
  }
  return (
    isExperimentEnabled(EXPERIMENT_IDS.RLM) &&
    (isExperimentEnabled(EXPERIMENT_IDS.PROGRAMMATIC_TOOL_CALLING) ||
      isExperimentEnabled(EXPERIMENT_IDS.PROGRAMMATIC_TOOL_CALLING_EXCLUSIVE))
  );
}

function extractTextForTranscript(message: MuxMessage): string {
  return (message.parts ?? [])
    .filter((part): part is { type: "text"; text: string } => part.type === "text")
    .map((part) => part.text.trim())
    .filter((text) => text.length > 0)
    .join("\n");
}

function summarizeToolMarker(part: unknown): string | null {
  if (typeof part !== "object" || part === null) return null;
  const record = part as { type?: unknown; toolName?: unknown };
  const type = typeof record.type === "string" ? record.type : null;
  if (!type) return null;
  const toolName =
    typeof record.toolName === "string"
      ? record.toolName
      : type.startsWith("tool-")
        ? type.slice(5)
        : null;
  return toolName ? `[tool ${toolName}]` : null;
}

/**
 * Format one abandoned message for the summarizer. Thinking-stripped by
 * construction: only text parts and compact tool markers survive — reasoning
 * parts are transient signal that inflates side-channel cost without adding
 * durable context worth preserving.
 */
function formatMessageForBranchTranscript(message: MuxMessage): string {
  const role = message.role === "user" ? "User" : message.role === "assistant" ? "Assistant" : null;
  if (!role) return "";

  const segments: string[] = [];
  const text = extractTextForTranscript(message);
  if (text) segments.push(text);
  for (const part of message.parts ?? []) {
    const marker = summarizeToolMarker(part);
    if (marker) segments.push(marker);
  }
  if (segments.length === 0) return "";
  return `${role}: ${segments.join("\n")}`;
}

/**
 * Build the thinking-stripped transcript of the abandoned segment, trimming
 * oldest messages first when over the input cap (the newest abandoned work
 * carries the most context worth preserving).
 */
export function buildAbandonedBranchTranscript(messages: MuxMessage[]): string {
  assert(Array.isArray(messages), "buildAbandonedBranchTranscript requires a message array");
  const formatted = messages.map(formatMessageForBranchTranscript).filter((s) => s.length > 0);

  let totalChars = formatted.reduce((sum, s) => sum + s.length, 0);
  let drop = 0;
  while (totalChars > BRANCH_SUMMARY_MAX_TRANSCRIPT_CHARS && drop < formatted.length - 1) {
    totalChars -= formatted[drop].length;
    drop += 1;
  }
  // A single oversized message can still exceed the cap after dropping all
  // older ones; hard-clamp from the end (newest content carries the most
  // context) so the transcript never blows a small side-channel model's window.
  return formatted.slice(drop).join("\n\n").slice(-BRANCH_SUMMARY_MAX_TRANSCRIPT_CHARS);
}

/**
 * Build the summarization prompt. Reuses the compaction prompt machinery
 * (include/exclude lists, word target) so summary style stays consistent with
 * epoch compaction, plus an abandoned-branch framing and explicit transcript
 * delimiters (prompt-injection guard: arbitrary chat history must not read as
 * instructions).
 */
export function buildAbandonedBranchSummaryPrompt(transcript: string): string {
  const targetWords = Math.round(BRANCH_SUMMARY_MAX_OUTPUT_TOKENS / WORDS_TO_TOKENS_RATIO);
  return [
    buildCompactionPrompt(targetWords),
    "",
    "Special case: the transcript below is an ABANDONED branch of the conversation — the user rewound to an earlier message, so these turns were removed from the active history. Summarize what was attempted, decided, and learned on that branch so the continuing assistant retains the context.",
    "",
    "<abandoned_branch>",
    transcript,
    "</abandoned_branch>",
  ].join("\n");
}

/**
 * Cheap side-channel model candidates: preferred small models first, then the
 * workspace's configured models as fallbacks (mirrors
 * WorkspaceService.getWorkspaceTitleModelCandidates, which is not reachable
 * from AgentSession).
 */
async function getSideChannelModelCandidates(
  aiService: BranchSummaryAiService,
  workspaceId: string
): Promise<string[]> {
  const candidates: string[] = [...NAME_GEN_PREFERRED_MODELS];
  const metadataResult = await aiService.getWorkspaceMetadata(workspaceId);
  if (!metadataResult.success) {
    return candidates;
  }
  const fallbackModels = [
    metadataResult.data.aiSettings?.model,
    ...Object.values(metadataResult.data.aiSettingsByAgent ?? {}).map((settings) => settings.model),
  ];
  for (const model of fallbackModels) {
    if (model && !candidates.includes(model)) {
      candidates.push(model);
    }
  }
  return candidates;
}

async function generateAbandonedBranchSummaryText(input: {
  aiService: BranchSummaryAiService;
  candidates: string[];
  prompt: string;
  timeoutMs: number;
}): Promise<string | null> {
  // One shared deadline across all candidates: the caller blocks on this, so
  // the total wait must stay bounded regardless of how many models fail over.
  const abortSignal = AbortSignal.timeout(input.timeoutMs);
  // Defensive double-bound: abortSignal cancels well-behaved providers, but a
  // provider that ignores abort must not hold the fork/edit operation hostage,
  // so every await below also races against this deadline promise.
  const deadline = new Promise<null>((resolve) => {
    if (abortSignal.aborted) {
      resolve(null);
      return;
    }
    abortSignal.addEventListener("abort", () => resolve(null), { once: true });
  });
  const maxAttempts = Math.min(input.candidates.length, 3);

  for (let i = 0; i < maxAttempts; i++) {
    if (abortSignal.aborted) break;
    const modelString = input.candidates[i];
    const modelResult = await input.aiService.createModel(modelString, undefined, {
      agentInitiated: true,
    });
    if (!modelResult.success) {
      log.debug("Branch summary: skipping model candidate", {
        modelString,
        error: modelResult.error.type,
      });
      continue;
    }
    try {
      // streamText (not generateText): Codex OAuth endpoints require
      // stream:true in the request body (same rationale as workspaceTitleGenerator).
      // No thinking provider options are passed, so the call itself stays
      // thinking-free on top of the thinking-stripped transcript.
      const stream = streamText({
        model: modelResult.data,
        prompt: input.prompt,
        maxOutputTokens: BRANCH_SUMMARY_MAX_OUTPUT_TOKENS,
        abortSignal,
      });
      // stream.text is a PromiseLike; wrap it so the race below can abandon it
      // while keeping its eventual rejection handled.
      const textPromise = Promise.resolve(stream.text);
      textPromise.catch(() => undefined);
      const racedText = await Promise.race([textPromise, deadline]);
      if (racedText === null) {
        log.debug("Branch summary: generation deadline reached", { modelString });
        break;
      }
      const text = racedText.trim();
      if (text.length > 0) {
        return text;
      }
      log.debug("Branch summary: model produced empty summary", { modelString });
    } catch (error) {
      log.debug("Branch summary generation failed", {
        modelString,
        error: getErrorMessage(error),
      });
    } finally {
      runLanguageModelCleanup(modelResult.data);
    }
  }
  return null;
}

/** Build the durable labeled summary row appended to the new branch. */
export function createBranchSummaryMessage(summaryText: string): MuxMessage {
  assert(summaryText.trim().length > 0, "branch summary text must be non-empty");
  return createMuxMessage(
    createBranchSummaryMessageId(),
    // A synthetic user row: provider-visible like other synthetic notices
    // (restart/wake messages), never mistaken for a streamed assistant turn
    // (no turn envelope/usage), and uiVisible so users see what was preserved.
    "user",
    `${BRANCH_SUMMARY_LABEL}\n\n${summaryText.trim()}`,
    {
      timestamp: Date.now(),
      synthetic: true,
      uiVisible: true,
      muxMetadata: { type: "branch-summary" },
    }
  );
}

/**
 * Summarize an abandoned history segment and append the labeled row to the
 * new branch's chat.jsonl. Returns the appended row (so live sessions can
 * emit it to the renderer) or null when no summary was produced.
 *
 * Runs SYNCHRONOUSLY (bounded by timeoutMs) inside fork/edit rather than
 * appending asynchronously on completion: an async append cannot be proven
 * race-free against the first turn on the new branch — it could land after
 * that turn's request was built (row invisible to the model for a turn) or
 * interleave with stream placeholder appends mid-turn. The hard deadline
 * keeps the user-facing operation responsive instead.
 *
 * Never throws; every failure path degrades to "no summary row".
 */
export async function maybeAppendAbandonedBranchSummary(input: {
  historyService: Pick<HistoryService, "appendToHistory">;
  aiService: BranchSummaryAiService;
  /** The NEW branch: fork target workspace, or the edited workspace post-truncation. */
  workspaceId: string;
  /** The removed tail, as returned by HistoryService.truncateAfterMessage. */
  abandonedMessages: MuxMessage[];
  /** Send-option experiments when available (edit path); omit for IPC ops without send options (fork). */
  experiments?: RlmExperimentFlags;
  /** Machine-override fallback (ExperimentsService/AIService.isExperimentEnabled). */
  isExperimentEnabled?: (experimentId: ExperimentId) => boolean;
  timeoutMs?: number;
}): Promise<MuxMessage | null> {
  try {
    // RLM off => byte-identical behavior to today: no model call, no row.
    if (!isRlmModeEnabled(input.experiments, input.isExperimentEnabled)) {
      return null;
    }
    if (input.abandonedMessages.length === 0) {
      return null;
    }

    // Tiny abandoned segments are not worth a model call.
    const estimatedTokens = input.abandonedMessages.reduce(
      (sum, message) => sum + estimateMuxMessageTokens(message),
      0
    );
    if (estimatedTokens < BRANCH_SUMMARY_MIN_SEGMENT_TOKENS) {
      return null;
    }

    const transcript = buildAbandonedBranchTranscript(input.abandonedMessages);
    if (transcript.length === 0) {
      return null;
    }

    const candidates = await getSideChannelModelCandidates(input.aiService, input.workspaceId);
    if (candidates.length === 0) {
      return null;
    }

    const summaryText = await generateAbandonedBranchSummaryText({
      aiService: input.aiService,
      candidates,
      prompt: buildAbandonedBranchSummaryPrompt(transcript),
      timeoutMs: input.timeoutMs ?? BRANCH_SUMMARY_TIMEOUT_MS,
    });
    if (summaryText === null) {
      return null;
    }

    const summaryMessage = createBranchSummaryMessage(summaryText);
    const appendResult = await input.historyService.appendToHistory(
      input.workspaceId,
      summaryMessage
    );
    if (!appendResult.success) {
      log.debug("Branch summary: failed to append summary row", {
        workspaceId: input.workspaceId,
        error: appendResult.error,
      });
      return null;
    }
    return summaryMessage;
  } catch (error) {
    // Self-healing doctrine: the summary is best-effort and must never fail
    // the fork/edit operation that triggered it.
    log.debug("Branch summary: unexpected failure", {
      workspaceId: input.workspaceId,
      error: getErrorMessage(error),
    });
    return null;
  }
}
