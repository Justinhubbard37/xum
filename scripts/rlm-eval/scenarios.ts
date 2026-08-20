/**
 * RLM lever-eval scenarios and lever configs.
 *
 * Scenarios are deterministic tasks with mechanical verifiers: fixtures are
 * generated with a seeded PRNG so expected answers are computed, not judged.
 * Configs are the independent variables (experiment flags + system-prompt
 * nudges); tool-description levers require code edits, so runs record the git
 * SHA for cross-build comparisons instead.
 */

import * as fs from "node:fs";
import * as path from "node:path";

export interface EvalScenario {
  id: string;
  description: string;
  /** Creates fixture files; returns ground-truth values used by turns/verify. */
  setup: (fixtureDir: string) => Record<string, string>;
  /** User messages sent sequentially (each waits for the previous turn to finish). */
  turns: (truth: Record<string, string>, fixtureDir: string) => string[];
  /** Mechanical pass/fail against per-turn assistant text. */
  verify: (
    truth: Record<string, string>,
    assistantTextPerTurn: string[]
  ) => { pass: boolean; detail: string };
}

export interface EvalConfig {
  id: string;
  experiments: {
    programmaticToolCalling: boolean;
    programmaticToolCallingExclusive?: boolean;
    rlm: boolean;
  };
  /** Optional prompting lever, sent as additionalSystemInstructions. */
  nudge?: string;
}

/** Deterministic PRNG (mulberry32) so fixture data and ground truth are reproducible. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const SCENARIOS: EvalScenario[] = [
  {
    id: "bigfile-stats",
    description:
      "Multi-turn analysis over a 1200-line data file: turn 2 rewards reusing state (vars) instead of re-reading.",
    setup: (fixtureDir) => {
      const rng = mulberry32(1337);
      const values: number[] = [];
      for (let i = 0; i < 1200; i++) values.push(Math.round((rng() * 100 + 50) * 1000) / 1000);
      fs.mkdirSync(fixtureDir, { recursive: true });
      fs.writeFileSync(path.join(fixtureDir, "values.txt"), values.join("\n") + "\n");
      const sorted = [...values].sort((a, b) => a - b);
      return {
        count: String(values.length),
        min: String(sorted[0]),
        max: String(sorted[sorted.length - 1]),
      };
    },
    turns: (_truth, fixtureDir) => [
      `Read the data file at ${fixtureDir}/values.txt (one number per line) and tell me exactly how many numbers it contains. End your reply with "COUNT=<n>".`,
      `Now tell me the minimum and maximum values in that same data. If you already have the data loaded, avoid re-reading the file. End your reply with "MIN=<v> MAX=<v>".`,
    ],
    verify: (truth, texts) => {
      const t1 = texts[0] ?? "";
      const t2 = texts[1] ?? "";
      const countOk = t1.includes(`COUNT=${truth.count}`);
      const minMaxOk = t2.includes(`MIN=${truth.min}`) && t2.includes(`MAX=${truth.max}`);
      return {
        pass: countOk && minMaxOk,
        detail: `count:${countOk ? "ok" : "FAIL"} minmax:${minMaxOk ? "ok" : "FAIL"}`,
      };
    },
  },
  {
    id: "control-quick",
    description:
      "Trivial task where kernel features are unnecessary: detects over-adoption overhead and prompt-cost regressions.",
    setup: () => ({ answer: "391" }),
    turns: () => [`What is 17 * 23? Reply with just the number.`],
    verify: (truth, texts) => {
      const pass = (texts[0] ?? "").includes(truth.answer);
      return { pass, detail: pass ? "answer:ok" : "answer:FAIL" };
    },
  },
];

export const CONFIGS: EvalConfig[] = [
  {
    id: "ptc-only",
    experiments: { programmaticToolCalling: true, rlm: false },
  },
  {
    id: "rlm-base",
    experiments: { programmaticToolCalling: true, rlm: true },
  },
  {
    id: "rlm-nudge",
    experiments: { programmaticToolCalling: true, rlm: true },
    nudge:
      "When you use code_execution, persist any data you might need in later turns in `vars` " +
      "(for example `vars.data = ...`) instead of re-reading files, and answer follow-up " +
      "questions from `vars` when the data is already there.",
  },
  // Kernel-first posture (r10): with flat tools removed, does the model adopt
  // vars organically, and does the nudge still add anything on top?
  {
    id: "rlm-excl",
    experiments: {
      programmaticToolCalling: true,
      programmaticToolCallingExclusive: true,
      rlm: true,
    },
  },
  {
    id: "rlm-excl-nudge",
    experiments: {
      programmaticToolCalling: true,
      programmaticToolCallingExclusive: true,
      rlm: true,
    },
    nudge:
      "When you use code_execution, persist any data you might need in later turns in `vars` " +
      "(for example `vars.data = ...`) instead of re-reading files, and answer follow-up " +
      "questions from `vars` when the data is already there.",
  },
];
