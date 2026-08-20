/**
 * RLM lever-eval runner.
 *
 * Drives scenario x config x seed cells against a RUNNING dev-server sandbox
 * (`make dev-server-sandbox`) over its HTTP API, then extracts mechanical
 * metrics from each cell's session dir. Purpose: measure whether prompting /
 * flag levers actually change model behavior in RLM mode (vars adoption,
 * result-handle usage, token cost, task success) instead of relying on
 * single-run anecdotes.
 *
 * Usage:
 *   make dev-server-sandbox   # note MUX_ROOT + backend port from its output
 *   bun run scripts/rlm-eval/run.ts \
 *     --base-url http://127.0.0.1:<port> --root <MUX_ROOT> \
 *     [--model anthropic:claude-haiku-4-5] [--seeds 2] \
 *     [--scenarios bigfile-stats,control-quick] [--configs ptc-only,rlm-base,rlm-nudge] \
 *     [--out /tmp/rlm-eval-results.jsonl]
 *
 * Each cell gets a fresh scratch workspace; experiment flags ride the send
 * options (they win over machine overrides), so no Settings mutation is
 * needed. Results append to the --out JSONL (git SHA recorded per row for
 * cross-build tool-description comparisons) and an aggregate table prints at
 * the end.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { execSync } from "node:child_process";

import { extractMetrics } from "./metrics";
import { CONFIGS, SCENARIOS } from "./scenarios";
import type { CellMetrics } from "./metrics";

interface CliArgs {
  baseUrl: string;
  root: string;
  model: string;
  thinking: string;
  seeds: number;
  scenarios: string[];
  configs: string[];
  out: string;
  turnTimeoutMs: number;
}

function parseArgs(argv: string[]): CliArgs {
  const get = (flag: string): string | undefined => {
    const i = argv.indexOf(flag);
    return i >= 0 && i + 1 < argv.length ? argv[i + 1] : undefined;
  };
  const baseUrl = get("--base-url");
  const root = get("--root");
  if (!baseUrl || !root) {
    console.error("Required: --base-url <sandbox backend url> --root <sandbox MUX_ROOT>");
    process.exit(1);
  }
  return {
    baseUrl: baseUrl.replace(/\/$/, ""),
    root,
    model: get("--model") ?? "anthropic:claude-haiku-4-5",
    thinking: get("--thinking") ?? "off",
    seeds: Number(get("--seeds") ?? "2"),
    scenarios: (get("--scenarios") ?? SCENARIOS.map((s) => s.id).join(",")).split(","),
    configs: (get("--configs") ?? CONFIGS.map((c) => c.id).join(",")).split(","),
    out: get("--out") ?? "/tmp/rlm-eval-results.jsonl",
    turnTimeoutMs: Number(get("--turn-timeout-ms") ?? "180000"),
  };
}

async function post(baseUrl: string, route: string, body: unknown): Promise<unknown> {
  const res = await fetch(`${baseUrl}/api${route}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const json: unknown = await res.json();
  if (!res.ok) {
    throw new Error(`${route} -> HTTP ${res.status}: ${JSON.stringify(json).slice(0, 300)}`);
  }
  return json;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

/**
 * Wait for the turn to finish: the last chat.jsonl row is an assistant message,
 * assistant turns >= expected count, and no partial.json (streaming) remains.
 */
async function waitForTurn(
  sessionDir: string,
  expectedUserTurns: number,
  timeoutMs: number
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  let stableTicks = 0;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 3000));
    const chatPath = path.join(sessionDir, "chat.jsonl");
    if (!fs.existsSync(chatPath)) continue;
    const lines = fs.readFileSync(chatPath, "utf-8").trim().split("\n");
    let users = 0;
    let lastRole = "";
    let lastAssistantHasText = false;
    for (const line of lines) {
      try {
        const row: unknown = JSON.parse(line);
        if (isRecord(row) && typeof row.role === "string") {
          if (row.role === "user") users += 1;
          lastRole = row.role;
          if (row.role === "assistant") {
            // Mid-turn tool-call steps commit assistant rows without the final
            // text; treating those as settled races the extractor against the
            // closing text part (observed with Opus 5 @ medium thinking).
            const parts = Array.isArray(row.parts) ? row.parts : [];
            lastAssistantHasText = parts.some(
              (p: unknown) =>
                isRecord(p) &&
                p.type === "text" &&
                typeof p.text === "string" &&
                p.text.trim() !== ""
            );
          }
        }
      } catch {
        // skip torn line
      }
    }
    const streaming = fs.existsSync(path.join(sessionDir, "partial.json"));
    if (
      users >= expectedUserTurns &&
      lastRole === "assistant" &&
      lastAssistantHasText &&
      !streaming
    ) {
      // Two consecutive stable polls guard against mid-write reads.
      stableTicks += 1;
      if (stableTicks >= 2) return;
    } else {
      stableTicks = 0;
    }
  }
  throw new Error(`turn ${expectedUserTurns} did not settle within ${timeoutMs}ms`);
}

interface CellResult {
  scenario: string;
  config: string;
  seed: number;
  workspaceId: string;
  pass: boolean;
  verifyDetail: string;
  gitSha: string;
  model: string;
  thinking: string;
  metrics: CellMetrics;
}

async function runCell(
  args: CliArgs,
  scenarioId: string,
  configId: string,
  seed: number,
  gitSha: string
): Promise<CellResult> {
  const scenario = SCENARIOS.find((s) => s.id === scenarioId);
  const config = CONFIGS.find((c) => c.id === configId);
  if (!scenario || !config) throw new Error(`unknown scenario/config: ${scenarioId}/${configId}`);

  const fixtureDir = `/tmp/rlm-eval-fixtures/${scenario.id}`;
  const truth = scenario.setup(fixtureDir);
  const turns = scenario.turns(truth, fixtureDir);

  const created = await post(args.baseUrl, "/workspace/createScratch", {
    title: `rlm-eval ${scenario.id} ${config.id} s${seed}`,
  });
  const metadata = isRecord(created) && isRecord(created.metadata) ? created.metadata : {};
  const workspaceId = typeof metadata.id === "string" ? metadata.id : "";
  if (workspaceId === "") throw new Error("createScratch returned no workspace id");
  const sessionDir = path.join(args.root, "sessions", workspaceId);

  for (let i = 0; i < turns.length; i++) {
    await post(args.baseUrl, "/workspace/sendMessage", {
      workspaceId,
      message: turns[i],
      options: {
        model: args.model,
        thinkingLevel: args.thinking,
        agentId: "exec",
        experiments: config.experiments,
        ...(config.nudge !== undefined ? { additionalSystemInstructions: config.nudge } : {}),
      },
    });
    await waitForTurn(sessionDir, i + 1, args.turnTimeoutMs);
  }

  const metrics = extractMetrics(sessionDir);
  const verdict = scenario.verify(truth, metrics.assistantTextPerTurn);
  return {
    scenario: scenario.id,
    config: config.id,
    seed,
    workspaceId,
    pass: verdict.pass,
    verifyDetail: verdict.detail,
    gitSha,
    model: args.model,
    thinking: args.thinking,
    metrics,
  };
}

function printAggregate(results: CellResult[]): void {
  const byKey = new Map<string, CellResult[]>();
  for (const r of results) {
    const key = `${r.scenario} | ${r.config}`;
    const list = byKey.get(key) ?? [];
    list.push(r);
    byKey.set(key, list);
  }
  const header = [
    "scenario | config".padEnd(34),
    "pass".padEnd(6),
    "vars".padEnd(6),
    "handles".padEnd(8),
    "inTok".padEnd(8),
    "outTok".padEnd(8),
    "reqs".padEnd(6),
    "kernel".padEnd(8),
    "flat".padEnd(6),
  ].join("");
  console.log("\n" + header);
  console.log("-".repeat(header.length));
  for (const [key, cells] of byKey) {
    const n = cells.length;
    const mean = (f: (c: CellResult) => number): string =>
      (cells.reduce((a, c) => a + f(c), 0) / n).toFixed(0);
    const rate = (f: (c: CellResult) => boolean): string => `${cells.filter(f).length}/${n}`;
    console.log(
      [
        key.padEnd(34),
        rate((c) => c.pass).padEnd(6),
        rate((c) => c.metrics.varsAdopted).padEnd(6),
        mean((c) => c.metrics.resultHandleCount).padEnd(8),
        mean(
          (c) => c.metrics.inputTokens + c.metrics.cacheCreateTokens + c.metrics.cachedTokens
        ).padEnd(8),
        mean((c) => c.metrics.outputTokens).padEnd(8),
        mean((c) => c.metrics.providerRequests).padEnd(6),
        mean((c) => c.metrics.codeExecutionCalls).padEnd(8),
        mean((c) => c.metrics.flatToolCalls).padEnd(6),
      ].join("")
    );
  }
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const gitSha = execSync("git rev-parse --short HEAD", { encoding: "utf-8" }).trim();
  // devtools.jsonl (providerRequests metric) only exists when debug logs are on.
  await post(args.baseUrl, "/config/updateLlmDebugLogs", { enabled: true });
  const results: CellResult[] = [];
  for (const scenarioId of args.scenarios) {
    for (const configId of args.configs) {
      for (let seed = 0; seed < args.seeds; seed++) {
        const label = `${scenarioId}/${configId}/s${seed}`;
        try {
          const result = await runCell(args, scenarioId, configId, seed, gitSha);
          results.push(result);
          fs.appendFileSync(args.out, JSON.stringify(result) + "\n");
          console.log(
            `${label}: pass=${result.pass} vars=${result.metrics.varsAdopted} ` +
              `handles=${result.metrics.resultHandleCount} ws=${result.workspaceId} (${result.verifyDetail})`
          );
        } catch (err) {
          console.error(`${label}: ERROR ${String(err)}`);
        }
      }
    }
  }
  printAggregate(results);
  console.log(`\nResults appended to ${args.out} (gitSha ${gitSha})`);
}

void main();
