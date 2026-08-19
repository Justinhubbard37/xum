import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import * as fsPromises from "node:fs/promises";
import * as path from "node:path";
import { z } from "zod";
import type { Tool } from "ai";

import { applyToolPolicyAndExperiments, reconcileHookReplacedCodeExecution } from "./toolAssembly";
import { sandboxHostService } from "@/node/services/sandbox/sandboxHostService";
import { DisposableTempDir } from "@/node/services/tempDir";
import { appendRefinementEvent } from "@/node/services/refinement/refinementJournal";
import { listRefinements } from "@/node/services/refinement/refinementRollback";

function executableTool(description: string): Tool {
  return {
    description,
    inputSchema: z.object({}),
    execute: () => Promise.resolve({ success: true }),
  } as unknown as Tool;
}

describe("applyToolPolicyAndExperiments", () => {
  test("exclusive PTC mode keeps mcp_prompt_get directly visible", async () => {
    const result = await applyToolPolicyAndExperiments({
      allTools: {
        bash: executableTool("Run a command"),
        mcp_prompt_get: executableTool("Fetch a prompt\n\nAvailable MCP prompts:\n- mcp__s__p"),
      },
      effectiveToolPolicy: undefined,
      experiments: { programmaticToolCallingExclusive: true },
      emitNestedToolEvent: () => undefined,
    });

    const names = Object.keys(result);
    expect(names).toContain("code_execution");
    expect(names).not.toContain("bash");
    // Sandbox declarations keep only the first description line, which would
    // hide the prompt catalog.
    expect(names).toContain("mcp_prompt_get");
    expect(result.mcp_prompt_get.description).toContain("mcp__s__p");
  });

  test("grant-denied tools are hidden from the model but stubbed in the sandbox", async () => {
    const result = await applyToolPolicyAndExperiments({
      allTools: {
        bash: executableTool("Run a command"),
        file_read: executableTool("Read a file"),
      },
      effectiveToolPolicy: undefined,
      experiments: { programmaticToolCalling: true },
      emitNestedToolEvent: () => undefined,
      capabilityGrants: {
        version: 1,
        bridgeTools: { allow: ["file_read"] },
        vars: false,
        hostEvents: false,
      },
    });

    // Grants are a ceiling on the model-visible set...
    expect(Object.keys(result)).not.toContain("bash");
    expect(Object.keys(result)).toContain("code_execution");

    // ...but the guest must still get the documented catchable stub error —
    // the bridge is built from the pre-grant set so denied tools are known,
    // not "mux.bash is not a function".
    const evalResult = (await result.code_execution.execute!(
      { code: "try { mux.bash({}); return 'no error'; } catch (e) { return e.message; }" },
      { toolCallId: "test-call-id", messages: [], context: undefined }
    )) as { success: boolean; result?: unknown };
    expect(evalResult.success).toBe(true);
    expect(evalResult.result).toBe("Capability denied: xum.bash is not granted for this sandbox");
  });
});

describe("persistent kernel graduation (RLM mode)", () => {
  const originalEnv = process.env.MUX_SANDBOX_PERSISTENT_MOUNTS;

  beforeEach(() => {
    // Pin the env override off so each test controls persistence explicitly.
    delete process.env.MUX_SANDBOX_PERSISTENT_MOUNTS;
  });

  afterEach(() => {
    if (originalEnv === undefined) {
      delete process.env.MUX_SANDBOX_PERSISTENT_MOUNTS;
    } else {
      process.env.MUX_SANDBOX_PERSISTENT_MOUNTS = originalEnv;
    }
  });

  async function assembleCodeExecution(opts: {
    rlm?: boolean;
    sandbox?: { workspaceId: string; sessionDir: string };
  }): Promise<Tool> {
    const tools = await applyToolPolicyAndExperiments({
      allTools: { file_read: executableTool("Read a file") },
      effectiveToolPolicy: undefined,
      experiments: { programmaticToolCalling: true, rlm: opts.rlm },
      emitNestedToolEvent: () => undefined,
      sandbox: opts.sandbox,
    });
    expect(tools.code_execution).toBeDefined();
    return tools.code_execution;
  }

  async function run(tool: Tool, code: string): Promise<{ success: boolean; result?: unknown }> {
    return (await tool.execute!(
      { code },
      { toolCallId: "test-call-id", messages: [], context: undefined }
    )) as { success: boolean; result?: unknown };
  }

  test("rlm on: persistent mount is used — vars survive across two invocations in one session", async () => {
    using tmp = new DisposableTempDir("tool-assembly-rlm-on");
    const scopeKey = "ws-tool-assembly-rlm-on";
    try {
      const codeExecution = await assembleCodeExecution({
        rlm: true,
        sandbox: { workspaceId: scopeKey, sessionDir: tmp.path },
      });
      expect(codeExecution.description).toContain("Persistent kernel");

      const first = await run(codeExecution, "vars.total = 40; return vars.total;");
      expect(first.success).toBe(true);
      expect(first.result).toBe(40);

      const second = await run(codeExecution, "vars.total += 2; return vars.total;");
      expect(second.success).toBe(true);
      expect(second.result).toBe(42);
    } finally {
      await sandboxHostService.disposeScope(scopeKey);
    }
  });

  test("rlm off: ephemeral per-call runtime and unchanged description", async () => {
    using tmp = new DisposableTempDir("tool-assembly-rlm-off");
    const withSandbox = await assembleCodeExecution({
      sandbox: { workspaceId: "ws-tool-assembly-rlm-off", sessionDir: tmp.path },
    });
    const withoutSandbox = await assembleCodeExecution({});

    // With the experiment off, sandbox context alone must not change the
    // model-visible description (byte-identical to today's ephemeral tool).
    expect(withSandbox.description).toBe(withoutSandbox.description);
    expect(withSandbox.description).not.toContain("Persistent kernel");

    // Ephemeral runtimes have no kernel `vars` namespace...
    const first = await run(withSandbox, "return typeof vars;");
    expect(first.success).toBe(true);
    expect(first.result).toBe("undefined");

    // ...and state set in one call does not leak into the next (fresh runtime).
    const second = await run(withSandbox, "globalThis.leak = 1; return globalThis.leak;");
    expect(second.success).toBe(true);
    expect(second.result).toBe(1);
    const third = await run(withSandbox, "return typeof globalThis.leak;");
    expect(third.success).toBe(true);
    expect(third.result).toBe("undefined");
  });

  test("refinement_rollback is exposed only with rlm on (and works end-to-end)", async () => {
    using tmp = new DisposableTempDir("tool-assembly-rlm-rollback");
    const scopeKey = "ws-tool-assembly-rlm-rollback";
    const sessionDir = path.join(tmp.path, "sessions", scopeKey);
    const assemble = (experiments: {
      programmaticToolCalling?: boolean;
      rlm?: boolean;
    }): Promise<Record<string, Tool>> =>
      applyToolPolicyAndExperiments({
        allTools: { file_read: executableTool("Read a file") },
        effectiveToolPolicy: undefined,
        experiments,
        emitNestedToolEvent: () => undefined,
        sandbox: { workspaceId: scopeKey, sessionDir },
      });
    try {
      // RLM off (PTC on): no rollback surface, byte-identical to today.
      const rlmOff = await assemble({ programmaticToolCalling: true });
      expect(rlmOff.refinement_rollback).toBeUndefined();

      // rlm flag without the PTC parent: no PTC branch, so no surface either.
      const ptcOff = await assemble({ rlm: true });
      expect(ptcOff.refinement_rollback).toBeUndefined();
      expect(ptcOff.code_execution).toBeUndefined();

      const rlmOn = await assemble({ programmaticToolCalling: true, rlm: true });
      expect(rlmOn.refinement_rollback).toBeDefined();

      // The wired tool rolls back a seeded skill-write row in the sandbox's
      // session dir and reports what changed.
      const skillFile = path.join(tmp.path, "checkout", ".mux", "skills", "s", "SKILL.md");
      await fsPromises.mkdir(path.dirname(skillFile), { recursive: true });
      await fsPromises.writeFile(skillFile, "body", "utf-8");
      await appendRefinementEvent({
        sessionDir,
        workspaceId: scopeKey,
        kind: "skill",
        action: { op: "write", skillName: "s", filePath: "SKILL.md" },
        inverse: { op: "delete-files", paths: [skillFile] },
        evidence: { toolName: "agent_skill_write" },
      });
      const rows = await listRefinements(sessionDir);
      const result = (await rlmOn.refinement_rollback.execute!(
        { id: rows[0].id, reason: "test rollback" },
        { toolCallId: "test-call-id", messages: [], context: undefined }
      )) as { success: boolean; rollbackOf?: string; deleted?: string[] };
      expect(result.success).toBe(true);
      expect(result.rollbackOf).toBe(rows[0].id);
      expect(result.deleted).toEqual([skillFile]);
      await expect(fsPromises.access(skillFile)).rejects.toThrow();
    } finally {
      await sandboxHostService.disposeScope(scopeKey);
    }
  });

  test("MUX_SANDBOX_PERSISTENT_MOUNTS=1 still opts in without the rlm experiment", async () => {
    using tmp = new DisposableTempDir("tool-assembly-env-mounts");
    const scopeKey = "ws-tool-assembly-env-mounts";
    process.env.MUX_SANDBOX_PERSISTENT_MOUNTS = "1";
    try {
      const codeExecution = await assembleCodeExecution({
        sandbox: { workspaceId: scopeKey, sessionDir: tmp.path },
      });
      expect(codeExecution.description).toContain("Persistent kernel");

      const first = await run(codeExecution, "vars.count = 1; return vars.count;");
      expect(first.success).toBe(true);
      expect(first.result).toBe(1);

      const second = await run(codeExecution, "vars.count += 1; return vars.count;");
      expect(second.success).toBe(true);
      expect(second.result).toBe(2);
    } finally {
      await sandboxHostService.disposeScope(scopeKey);
    }
  });
});

describe("reconcileHookReplacedCodeExecution", () => {
  test("spread-style wrapper gets the rebuilt description but keeps its execute", () => {
    const preHook = executableTool("defs: function bash; function file_read");
    // Middleware wrapped by spreading the pre-hook tool: same description,
    // new execute.
    const wrappedExecute = () => Promise.resolve({ success: true, audited: true });
    const hookReplacement: Tool = { ...preHook, execute: wrappedExecute };
    const rebuilt = executableTool("defs: function file_read");

    const result = reconcileHookReplacedCodeExecution(preHook, hookReplacement, rebuilt);

    // Model-facing metadata follows the rebuilt toolset (bash removed)...
    expect(result.description).toBe("defs: function file_read");
    // ...while the middleware's execution wrapper is preserved.
    expect(result.execute).toBe(wrappedExecute);
  });

  test("middleware-authored description is preserved", () => {
    const preHook = executableTool("defs: function bash; function file_read");
    const hookReplacement = executableTool("audited code execution");
    const rebuilt = executableTool("defs: function file_read");

    const result = reconcileHookReplacedCodeExecution(preHook, hookReplacement, rebuilt);

    // Middleware took ownership of the model-facing contract; return it as-is.
    expect(result).toBe(hookReplacement);
  });
});
