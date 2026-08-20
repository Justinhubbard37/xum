/**
 * RLM lever-eval metrics extraction.
 *
 * Extracts mechanical, judgment-free metrics from a workspace session dir
 * (chat.jsonl, durable-events.jsonl, devtools.jsonl, session-usage.json) so
 * A/B comparisons between prompting/tool-description/flag levers rest on
 * durable artifacts rather than anecdotes. Used by scripts/rlm-eval/run.ts.
 */

import * as fs from "node:fs";
import * as path from "node:path";

export interface CellMetrics {
  /** Any sandbox-vars-snapshot row with a non-empty vars object ({} serializes to 2 bytes). */
  varsAdopted: boolean;
  /** Largest vars snapshot in bytes (proxy for how much state was offloaded). */
  maxVarsSnapshotBytes: number;
  /** result-handle durable rows (oversized results offloaded to the kernel). */
  resultHandleCount: number;
  /** code_execution tool calls across all turns. */
  codeExecutionCalls: number;
  /** Non-code_execution tool calls (flat tool usage). */
  flatToolCalls: number;
  /** Provider round-trips (devtools step entries). */
  providerRequests: number;
  /** Token totals summed across models from session-usage.json. */
  inputTokens: number;
  cachedTokens: number;
  cacheCreateTokens: number;
  outputTokens: number;
  costUsd: number;
  /** Concatenated assistant text per user turn, for scenario verifiers. */
  assistantTextPerTurn: string[];
}

interface ChatPart {
  type?: string;
  text?: string;
  /** Tool parts persist as type "dynamic-tool" with the tool name here. */
  toolName?: string;
}

interface ChatMessage {
  role?: string;
  parts?: ChatPart[];
}

function readJsonl(filePath: string): unknown[] {
  if (!fs.existsSync(filePath)) return [];
  const rows: unknown[] = [];
  for (const line of fs.readFileSync(filePath, "utf-8").split("\n")) {
    const trimmed = line.trim();
    if (trimmed === "") continue;
    try {
      rows.push(JSON.parse(trimmed));
    } catch {
      // Self-healing read: skip torn/corrupt lines like the journal kit does.
    }
  }
  return rows;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

export function extractMetrics(sessionDir: string): CellMetrics {
  const metrics: CellMetrics = {
    varsAdopted: false,
    maxVarsSnapshotBytes: 0,
    resultHandleCount: 0,
    codeExecutionCalls: 0,
    flatToolCalls: 0,
    providerRequests: 0,
    inputTokens: 0,
    cachedTokens: 0,
    cacheCreateTokens: 0,
    outputTokens: 0,
    costUsd: 0,
    assistantTextPerTurn: [],
  };

  // durable-events.jsonl: vars snapshots + result handles
  for (const row of readJsonl(path.join(sessionDir, "durable-events.jsonl"))) {
    if (!isRecord(row)) continue;
    const data = isRecord(row.data) ? row.data : {};
    if (row.kind === "sandbox-vars-snapshot") {
      const size = typeof data.size === "number" ? data.size : 0;
      // "{}" is 2 bytes; anything larger means the guest actually stored state.
      if (size > 2) metrics.varsAdopted = true;
      metrics.maxVarsSnapshotBytes = Math.max(metrics.maxVarsSnapshotBytes, size);
    } else if (row.kind === "result-handle") {
      metrics.resultHandleCount += 1;
    }
  }

  // chat.jsonl: tool-call counts + assistant text grouped by user turn
  let currentTurnText: string[] | null = null;
  for (const row of readJsonl(path.join(sessionDir, "chat.jsonl"))) {
    if (!isRecord(row)) continue;
    const msg = row as ChatMessage;
    if (msg.role === "user") {
      currentTurnText = [];
      metrics.assistantTextPerTurn.push("");
      continue;
    }
    if (msg.role !== "assistant") continue;
    for (const part of msg.parts ?? []) {
      const type = part.type ?? "";
      if (type === "text" && typeof part.text === "string") {
        if (currentTurnText !== null) {
          currentTurnText.push(part.text);
          metrics.assistantTextPerTurn[metrics.assistantTextPerTurn.length - 1] += part.text;
        }
      } else if (type === "dynamic-tool" || type.startsWith("tool-")) {
        const toolName =
          typeof part.toolName === "string" ? part.toolName : type.replace(/^tool-/, "");
        if (toolName === "code_execution") metrics.codeExecutionCalls += 1;
        else metrics.flatToolCalls += 1;
      }
    }
  }

  // devtools.jsonl: provider round-trips
  for (const row of readJsonl(path.join(sessionDir, "devtools.jsonl"))) {
    if (isRecord(row) && row.type === "step") metrics.providerRequests += 1;
  }

  // session-usage.json: token + cost totals across models
  const usagePath = path.join(sessionDir, "session-usage.json");
  if (fs.existsSync(usagePath)) {
    try {
      const parsed: unknown = JSON.parse(fs.readFileSync(usagePath, "utf-8"));
      const byModel = isRecord(parsed) && isRecord(parsed.byModel) ? parsed.byModel : {};
      for (const modelUsage of Object.values(byModel)) {
        if (!isRecord(modelUsage)) continue;
        const bucket = (name: string): { tokens: number; cost: number } => {
          const b = isRecord(modelUsage[name]) ? (modelUsage[name] as Record<string, unknown>) : {};
          return {
            tokens: typeof b.tokens === "number" ? b.tokens : 0,
            cost: typeof b.cost_usd === "number" ? b.cost_usd : 0,
          };
        };
        const input = bucket("input");
        const cached = bucket("cached");
        const cacheCreate = bucket("cacheCreate");
        const output = bucket("output");
        const reasoning = bucket("reasoning");
        metrics.inputTokens += input.tokens;
        metrics.cachedTokens += cached.tokens;
        metrics.cacheCreateTokens += cacheCreate.tokens;
        metrics.outputTokens += output.tokens + reasoning.tokens;
        metrics.costUsd +=
          input.cost + cached.cost + cacheCreate.cost + output.cost + reasoning.cost;
      }
    } catch {
      // Missing/corrupt usage file leaves token metrics at zero rather than failing the cell.
    }
  }

  return metrics;
}
