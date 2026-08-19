// Fixture-driven tests for scripts/gate_fingerprint.sh: each test spawns the
// real script against a throwaway git repo and asserts the memoization
// contract (check hits only while the worktree fingerprint is unchanged).
//
// Not part of the `bun test src` CI lane (like other scripts/ tooling tests);
// run explicitly: bun test ./scripts/gate_fingerprint.test.ts
import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdtemp, rm, writeFile, appendFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import * as path from "node:path";

const SCRIPT = path.resolve(import.meta.dir, "gate_fingerprint.sh");

// Hermetic git environment: host GIT_* vars and global config (hooks, commit
// trailers, diff drivers) must not leak into fixture repos or fingerprints.
function gitEnv(): Record<string, string> {
  const env: Record<string, string> = {};
  for (const [key, value] of Object.entries(process.env)) {
    if (value === undefined || key.startsWith("GIT_")) {
      continue;
    }
    env[key] = value;
  }
  env.GIT_CONFIG_GLOBAL = "/dev/null";
  env.GIT_CONFIG_SYSTEM = "/dev/null";
  env.GIT_AUTHOR_NAME = "Gate Test";
  env.GIT_AUTHOR_EMAIL = "gate-test@example.com";
  env.GIT_COMMITTER_NAME = "Gate Test";
  env.GIT_COMMITTER_EMAIL = "gate-test@example.com";
  return env;
}

interface RunResult {
  exitCode: number;
  stdout: string;
  stderr: string;
}

async function run(cwd: string, cmd: string[]): Promise<RunResult> {
  const proc = Bun.spawn(cmd, { cwd, env: gitEnv(), stdout: "pipe", stderr: "pipe" });
  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);
  return { exitCode, stdout: stdout.trim(), stderr: stderr.trim() };
}

async function git(cwd: string, ...args: string[]): Promise<void> {
  const result = await run(cwd, ["git", ...args]);
  if (result.exitCode !== 0) {
    throw new Error(`git ${args.join(" ")} failed (${result.exitCode}): ${result.stderr}`);
  }
}

async function gate(cwd: string, ...args: string[]): Promise<RunResult> {
  return run(cwd, ["bash", SCRIPT, ...args]);
}

async function fingerprint(cwd: string): Promise<string> {
  const result = await gate(cwd, "fingerprint");
  expect(result.exitCode).toBe(0);
  expect(result.stdout).toMatch(/^[0-9a-f]{64}$/);
  return result.stdout;
}

let repo: string;

beforeEach(async () => {
  repo = await mkdtemp(path.join(tmpdir(), "gate-fingerprint-test-"));
  await git(repo, "init", "-q");
  await writeFile(path.join(repo, "tracked.txt"), "hello\n");
  await git(repo, "add", "tracked.txt");
  await git(repo, "commit", "-q", "-m", "initial");
});

afterEach(async () => {
  await rm(repo, { recursive: true, force: true });
});

test("fingerprint is stable across runs and unperturbed by record", async () => {
  const before = await fingerprint(repo);
  expect(await fingerprint(repo)).toBe(before);

  const record = await gate(repo, "record", "static-check", "pass");
  expect(record.exitCode).toBe(0);
  // The store lives inside the git dir, so recording must not change the
  // fingerprint (a self-invalidating cache would never hit).
  expect(await fingerprint(repo)).toBe(before);
  // ...and the repo stays clean from git's perspective.
  const status = await run(repo, ["git", "status", "--porcelain"]);
  expect(status.stdout).toBe("");
});

test("check hits with unchanged tree; pass and fail both round-trip", async () => {
  // No record yet: miss.
  expect((await gate(repo, "check", "static-check")).exitCode).toBe(1);

  expect((await gate(repo, "record", "static-check", "pass")).exitCode).toBe(0);
  expect((await gate(repo, "record", "unit-tests", "fail")).exitCode).toBe(0);

  const pass = await gate(repo, "check", "static-check");
  expect(pass.exitCode).toBe(0);
  expect(pass.stdout).toBe("pass");

  const fail = await gate(repo, "check", "unit-tests");
  expect(fail.exitCode).toBe(0);
  expect(fail.stdout).toBe("fail");

  // A gate that was never recorded stays a miss even with a populated store.
  expect((await gate(repo, "check", "other-gate")).exitCode).toBe(1);
});

test("check misses after editing a tracked file", async () => {
  await gate(repo, "record", "static-check", "pass");
  await appendFile(path.join(repo, "tracked.txt"), "edited\n");

  expect((await gate(repo, "check", "static-check")).exitCode).toBe(1);

  // Re-recording against the changed tree makes check hit again.
  expect((await gate(repo, "record", "static-check", "fail")).exitCode).toBe(0);
  const rechecked = await gate(repo, "check", "static-check");
  expect(rechecked.exitCode).toBe(0);
  expect(rechecked.stdout).toBe("fail");
});

test("check misses when an untracked file appears or changes", async () => {
  await gate(repo, "record", "static-check", "pass");
  await writeFile(path.join(repo, "scratch.txt"), "one\n");
  expect((await gate(repo, "check", "static-check")).exitCode).toBe(1);

  // Content changes of an existing untracked file must also invalidate.
  await gate(repo, "record", "static-check", "pass");
  await writeFile(path.join(repo, "scratch.txt"), "two\n");
  expect((await gate(repo, "check", "static-check")).exitCode).toBe(1);

  // Fingerprint is content-based: deleting the file restores the original
  // fingerprint, so the very first record becomes fresh again.
  await rm(path.join(repo, "scratch.txt"));
  await gate(repo, "record", "static-check", "pass");
  expect((await gate(repo, "check", "static-check")).exitCode).toBe(0);
});

test("check misses after staging a change", async () => {
  await gate(repo, "record", "static-check", "pass");

  // Stage a brand-new file: it leaves the untracked list and must be caught
  // via the tracked diff instead.
  await writeFile(path.join(repo, "staged.txt"), "staged\n");
  await git(repo, "add", "staged.txt");
  expect((await gate(repo, "check", "static-check")).exitCode).toBe(1);
});

test("corrupt store self-heals instead of failing the caller", async () => {
  const storePath = await run(repo, [
    "git",
    "rev-parse",
    "--path-format=absolute",
    "--git-path",
    "gate_fingerprints.json",
  ]);
  expect(storePath.exitCode).toBe(0);
  await writeFile(storePath.stdout, "not json {{{");

  // check treats a corrupt store as a miss; record rewrites it cleanly.
  expect((await gate(repo, "check", "static-check")).exitCode).toBe(1);
  expect((await gate(repo, "record", "static-check", "pass")).exitCode).toBe(0);
  const rechecked = await gate(repo, "check", "static-check");
  expect(rechecked.exitCode).toBe(0);
  expect(rechecked.stdout).toBe("pass");
});
