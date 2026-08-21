/**
 * QuickJS-heavy suite: keep out of broad Bun filters (runs isolated in CI,
 * see .github/workflows: isolated_unit_tests).
 */
import { describe, expect, test } from "bun:test";
import { readdirSync, statSync, writeFileSync } from "fs";
import { join } from "path";
import { tool } from "ai";
import { z } from "zod";
import { DisposableTempDir } from "@/node/services/tempDir";
import { QuickJSRuntimeFactory } from "@/node/services/ptc/quickjsRuntime";
import { ToolBridge } from "@/node/services/ptc/toolBridge";
import { FULL_GRANTS, LEAST_PRIVILEGE_GRANTS } from "@/common/types/capabilityGrants";
import { DurableEventJournal } from "@/node/utils/journal/durableEventJournal";
import { SandboxHostService, VarsSnapshotBudgetError } from "./sandboxHostService";
import { VARS_SNAPSHOT_MAX_BYTES } from "@/constants/resultHandles";

const runtimeFactory = new QuickJSRuntimeFactory();

describe("SandboxHostService", () => {
  test("ephemeral mounts are fresh per acquire and dispose on release", async () => {
    const host = new SandboxHostService();
    const first = await host.acquireMount({ lifetime: "ephemeral", runtimeFactory });
    const result = await first.runtime.eval("return 1 + 1;");
    expect(result.success).toBe(true);
    expect(result.result).toBe(2);
    first.release();
    expect(first.isDisposed).toBe(true);

    const second = await host.acquireMount({ lifetime: "ephemeral", runtimeFactory });
    expect(second).not.toBe(first);
    second.release();
  });

  test("persistent mount shares vars across separate evals and acquires", async () => {
    using tmp = new DisposableTempDir("sandbox-host-test");
    const host = new SandboxHostService();
    const mount = await host.acquireMount({
      lifetime: "persistent",
      runtimeFactory,
      scopeKey: "ws-persist",
      sessionDir: tmp.path,
    });

    const write = await mount.runtime.eval("vars.counter = 41; return vars.counter;");
    expect(write.success).toBe(true);
    expect(write.result).toBe(41);

    // Re-acquire: same scope returns the same live mount.
    const again = await host.acquireMount({
      lifetime: "persistent",
      runtimeFactory,
      scopeKey: "ws-persist",
      sessionDir: tmp.path,
    });
    expect(again).toBe(mount);

    const read = await again.runtime.eval("vars.counter += 1; return vars.counter;");
    expect(read.success).toBe(true);
    expect(read.result).toBe(42);

    mount.release(); // no-op for persistent mounts
    expect(mount.isDisposed).toBe(false);
    await host.disposeScope("ws-persist");
    expect(mount.isDisposed).toBe(true);
  });

  test("vars snapshot/restore survives a simulated restart", async () => {
    using tmp = new DisposableTempDir("sandbox-host-test");

    // "Process 1": write vars, snapshot via journal kit, dispose.
    const host1 = new SandboxHostService();
    const mount1 = await host1.acquireMount({
      lifetime: "persistent",
      runtimeFactory,
      scopeKey: "ws-restart",
      sessionDir: tmp.path,
    });
    const write = await mount1.runtime.eval(
      'vars.searchResults = { hits: [1, 2, 3], query: "foo" }; return true;'
    );
    expect(write.success).toBe(true);
    await host1.disposeScope("ws-restart"); // snapshots before disposing

    // The snapshot is a durable event referencing a blob.
    const journal = new DurableEventJournal(tmp.path);
    const events = await journal.read();
    const snapshot = events.find((e) => e.kind === "sandbox-vars-snapshot");
    expect(snapshot).toBeDefined();

    // "Process 2": fresh service (simulated restart) restores latest snapshot.
    const host2 = new SandboxHostService();
    const mount2 = await host2.acquireMount({
      lifetime: "persistent",
      runtimeFactory,
      scopeKey: "ws-restart",
      sessionDir: tmp.path,
    });
    expect(mount2).not.toBe(mount1);
    const read = await mount2.runtime.eval("return vars.searchResults;");
    expect(read.success).toBe(true);
    expect(read.result).toEqual({ hits: [1, 2, 3], query: "foo" });
    await host2.disposeScope("ws-restart");
  });

  test("persistVars rejects an over-budget vars namespace (nothing reaches disk)", async () => {
    using tmp = new DisposableTempDir("sandbox-host-test");
    const host = new SandboxHostService();
    const mount = await host.acquireMount({
      lifetime: "persistent",
      runtimeFactory,
      scopeKey: "ws-budget",
      sessionDir: tmp.path,
    });

    // All vars count against the budget — not just managed handle/load keys.
    const oversize = VARS_SNAPSHOT_MAX_BYTES + 16;
    const write = await mount.runtime.eval(`vars.big = "x".repeat(${oversize}); return true;`);
    expect(write.success).toBe(true);

    let thrown: unknown;
    try {
      await mount.persistVars();
      expect.unreachable("persistVars must reject an over-budget snapshot");
    } catch (error) {
      thrown = error;
    }
    expect(thrown).toBeInstanceOf(VarsSnapshotBudgetError);

    // The rejected snapshot must not have been journaled or blobbed.
    const journal = new DurableEventJournal(tmp.path);
    const events = await journal.read();
    expect(events.filter((e) => e.kind === "sandbox-vars-snapshot")).toHaveLength(0);
    await host.dropScope("ws-budget");
  });

  test("superseded snapshot blobs are reclaimed; referenced blobs survive", async () => {
    using tmp = new DisposableTempDir("sandbox-host-test");
    const host = new SandboxHostService();
    const mount = await host.acquireMount({
      lifetime: "persistent",
      runtimeFactory,
      scopeKey: "ws-reclaim",
      sessionDir: tmp.path,
    });
    const journal = new DurableEventJournal(tmp.path);
    const snapshotRefs = async () => {
      const events = await journal.read();
      return events
        .filter((e) => e.kind === "sandbox-vars-snapshot")
        .map((e) => (e.data as { blobHash: string }).blobHash);
    };

    await mount.runtime.eval('vars.state = "one"; return true;');
    await mount.persistVars();
    const [firstRef] = await snapshotRefs();
    expect(await journal.blobs.has(firstRef as never)).toBe(true);

    // A second, different snapshot supersedes the first: per-call
    // persistence must not retain every historical vars version on disk.
    await mount.runtime.eval('vars.state = "two"; return true;');
    await mount.persistVars();
    const refs = await snapshotRefs();
    expect(refs).toHaveLength(2);
    expect(await journal.blobs.has(firstRef as never)).toBe(false);
    expect(await journal.blobs.has(refs[1] as never)).toBe(true);

    // A superseded hash referenced by ANOTHER event kind must survive
    // (content addressing can share payloads across events): reference the
    // CURRENT latest snapshot, then supersede it — reclamation must skip it.
    const secondRef = refs[1];
    await journal.append({
      workspaceId: "ws-reclaim",
      kind: "result-handle",
      data: { handle: "vars.__h1", preview: "shared", blobHash: secondRef, size: 1 },
    });
    await mount.runtime.eval('vars.state = "three"; return true;');
    await mount.persistVars();
    expect(await journal.blobs.has(secondRef as never)).toBe(true);

    await host.disposeScope("ws-reclaim");
  });

  test("host→guest events: queue + drain via drainHostEvents()", async () => {
    using tmp = new DisposableTempDir("sandbox-host-test");
    const host = new SandboxHostService();
    const mount = await host.acquireMount({
      lifetime: "persistent",
      runtimeFactory,
      scopeKey: "ws-events",
      sessionDir: tmp.path,
    });

    mount.postHostEvent({ type: "task-complete", taskId: "t1" });
    mount.postHostEvent({ type: "task-complete", taskId: "t2" });

    const drained = await mount.runtime.eval("return drainHostEvents();");
    expect(drained.success).toBe(true);
    expect(drained.result).toEqual([
      { type: "task-complete", taskId: "t1" },
      { type: "task-complete", taskId: "t2" },
    ]);

    // Queue is empty after draining.
    const empty = await mount.runtime.eval("return drainHostEvents();");
    expect(empty.result).toEqual([]);
    await host.disposeScope("ws-events");
  });

  test("async capability + host event: promise resolves in-guest and completion is delivered", async () => {
    using tmp = new DisposableTempDir("sandbox-host-test");
    const host = new SandboxHostService();
    const mount = await host.acquireMount({
      lifetime: "persistent",
      runtimeFactory,
      scopeKey: "ws-async",
      sessionDir: tmp.path,
    });

    // Demo capability: resolves asynchronously AND posts a host event on
    // completion (the mux.task({background:true}) delivery pattern).
    mount.runtime.registerPromiseFunction("startTask", async (name) => {
      await new Promise((resolve) => setTimeout(resolve, 10));
      mount.postHostEvent({ type: "task-started", name });
      return { taskId: "task-123" };
    });

    const result = await mount.runtime.eval(`
      return (async () => {
        const handle = await startTask("demo");
        const events = drainHostEvents();
        return { handle, events };
      })();
    `);
    expect(result.success).toBe(true);
    expect(result.result).toEqual({
      handle: { taskId: "task-123" },
      events: [{ type: "task-started", name: "demo" }],
    });
    await host.disposeScope("ws-async");
  });

  test("postTaskTerminalEvent: sub-threshold report is queued inline and drained by the guest", async () => {
    using tmp = new DisposableTempDir("sandbox-host-test");
    const host = new SandboxHostService();
    await host.acquireMount({
      lifetime: "persistent",
      runtimeFactory,
      scopeKey: "ws-terminal",
      sessionDir: tmp.path,
    });

    await host.postTaskTerminalEvent("ws-terminal", {
      taskId: "child-1",
      status: "completed",
      reportMarkdown: "All done.",
    });

    const mount = await host.acquireMount({
      lifetime: "persistent",
      runtimeFactory,
      scopeKey: "ws-terminal",
      sessionDir: tmp.path,
    });
    const drained = await mount.runtime.eval("return drainHostEvents();");
    expect(drained.success).toBe(true);
    expect(drained.result).toEqual([
      {
        type: "task-terminal",
        taskId: "child-1",
        status: "completed",
        reportMarkdown: "All done.",
      },
    ]);
    await host.disposeScope("ws-terminal");
  });

  test("postTaskTerminalEvent: no live mount for the scope is a harmless no-op", async () => {
    const host = new SandboxHostService();
    // Must not throw or create any mount — the durable wake is the fallback.
    await host.postTaskTerminalEvent("ws-nobody", {
      taskId: "child-1",
      status: "completed",
      reportMarkdown: "report",
    });
    expect(host.hasScope("ws-nobody")).toBe(false);
  });

  test("postTaskTerminalEvent: dropped without the hostEvents grant", async () => {
    using tmp = new DisposableTempDir("sandbox-host-test");
    const host = new SandboxHostService();
    const mount = await host.acquireMount({
      lifetime: "persistent",
      runtimeFactory,
      scopeKey: "ws-terminal-denied",
      sessionDir: tmp.path,
      grants: LEAST_PRIVILEGE_GRANTS,
    });
    await host.postTaskTerminalEvent("ws-terminal-denied", {
      taskId: "child-1",
      status: "completed",
      reportMarkdown: "report",
    });
    expect(mount.drainHostEvents()).toEqual([]);
    await host.disposeScope("ws-terminal-denied");
  });

  test("postTaskTerminalEvent: oversized report is offloaded to an r4 handle + blob + durable event", async () => {
    using tmp = new DisposableTempDir("sandbox-host-test");
    const host = new SandboxHostService();
    const mount = await host.acquireMount({
      lifetime: "persistent",
      runtimeFactory,
      scopeKey: "ws-terminal-big",
      sessionDir: tmp.path,
    });

    const bigReport = "R".repeat(20_000); // over the 16KB offload threshold
    await host.postTaskTerminalEvent("ws-terminal-big", {
      taskId: "child-big",
      status: "completed",
      reportMarkdown: bigReport,
    });

    const drained = await mount.runtime.eval("return drainHostEvents();");
    expect(drained.success).toBe(true);
    const events = drained.result as Array<{
      type: string;
      taskId: string;
      status: string;
      reportMarkdown?: string;
      reportHandle?: { handle: string; preview: string; size: number };
    }>;
    expect(events).toHaveLength(1);
    const event = events[0];
    expect(event.type).toBe("task-terminal");
    expect(event.taskId).toBe("child-big");
    expect(event.reportMarkdown).toBeUndefined();
    expect(event.reportHandle?.handle).toBe("vars.__h1");
    expect(event.reportHandle?.size).toBe(20_000);
    expect(event.reportHandle?.preview).toContain("middle truncated");

    // The full report is readable at the handle in a later eval.
    const followUp = await mount.runtime.eval("return vars.__h1.length;");
    expect(followUp.result).toBe(20_000);

    // Blob + result-handle durable event mirror the guest-visible record.
    const journal = new DurableEventJournal(tmp.path);
    const journaled = await journal.read();
    const handleEvents = journaled.filter((e) => e.kind === "result-handle");
    expect(handleEvents).toHaveLength(1);
    const handleEvent = handleEvents[0];
    if (handleEvent.kind !== "result-handle") throw new Error("unreachable");
    expect(handleEvent.data.handle).toBe("vars.__h1");
    expect(await journal.blobs.getText(handleEvent.data.blobHash)).toBe(JSON.stringify(bigReport));
    // The vars mutation was snapshotted (handle numbering must stay monotonic
    // on disk even though no eval ran).
    expect(journaled.some((e) => e.kind === "sandbox-vars-snapshot")).toBe(true);
    await host.disposeScope("ws-terminal-big");
  });

  test("postHostEvent drops oldest events beyond the queue cap", async () => {
    using tmp = new DisposableTempDir("sandbox-host-test");
    const host = new SandboxHostService();
    const mount = await host.acquireMount({
      lifetime: "persistent",
      runtimeFactory,
      scopeKey: "ws-cap",
      sessionDir: tmp.path,
    });
    for (let i = 0; i < 260; i++) {
      mount.postHostEvent({ n: i });
    }
    const drained = mount.drainHostEvents() as Array<{ n: number }>;
    expect(drained).toHaveLength(256);
    expect(drained[0]).toEqual({ n: 4 }); // 0-3 dropped oldest-first
    expect(drained[255]).toEqual({ n: 259 });
    await host.disposeScope("ws-cap");
  });

  test("least-privilege grants disable vars and host events on the mount", async () => {
    using tmp = new DisposableTempDir("sandbox-host-test");
    const host = new SandboxHostService();
    const mount = await host.acquireMount({
      lifetime: "persistent",
      runtimeFactory,
      scopeKey: "ws-denied",
      sessionDir: tmp.path,
      grants: LEAST_PRIVILEGE_GRANTS,
    });

    // vars namespace was never initialized...
    const varsProbe = await mount.runtime.eval("return typeof globalThis.vars;");
    expect(varsProbe.result).toBe("undefined");
    // ...and the drain bridge is not exposed.
    const drainProbe = await mount.runtime.eval("return typeof globalThis.drainHostEvents;");
    expect(drainProbe.result).toBe("undefined");
    // Host-side APIs refuse too (clear errors, not crashes).
    expect(() => mount.postHostEvent({})).toThrow(/hostEvents grant/);
    try {
      await mount.snapshotVars();
      expect.unreachable("Should have thrown");
    } catch (e) {
      expect(String(e)).toContain("vars grant");
    }

    // disposeScope must not fail even though snapshotting is not granted.
    await host.disposeScope("ws-denied");
    expect(mount.isDisposed).toBe(true);
  });

  test("denied bridge capability produces a clear catchable guest error, not a crash", async () => {
    const host = new SandboxHostService();
    const mount = await host.acquireMount({ lifetime: "ephemeral", runtimeFactory });

    const tools = {
      file_read: tool({
        description: "read",
        inputSchema: z.object({}),
        execute: () => Promise.resolve({ content: "granted!" }),
      }),
      bash: tool({
        description: "run",
        inputSchema: z.object({}),
        execute: () => Promise.resolve({ output: "should never run" }),
      }),
    };
    const bridge = new ToolBridge(tools, {
      version: 1,
      bridgeTools: { allow: ["file_read"] },
      vars: false,
      hostEvents: false,
    });
    bridge.register(mount.runtime);

    const result = await mount.runtime.eval(`
      const granted = mux.file_read({});
      let denied;
      try {
        mux.bash({});
        denied = "no error";
      } catch (e) {
        denied = e.message;
      }
      return { granted, denied, sandboxStillWorks: 1 + 1 };
    `);
    expect(result.success).toBe(true);
    expect(result.result).toEqual({
      granted: { content: "granted!" },
      denied: "Capability denied: xum.bash is not granted for this sandbox",
      sandboxStillWorks: 2,
    });
    mount.release();
  });

  test("concurrent first acquisitions share one mount (no double runtime creation)", async () => {
    using tmp = new DisposableTempDir("sandbox-host-test");
    const host = new SandboxHostService();
    const [a, b] = await Promise.all([
      host.acquireMount({
        lifetime: "persistent",
        runtimeFactory,
        scopeKey: "ws-race",
        sessionDir: tmp.path,
      }),
      host.acquireMount({
        lifetime: "persistent",
        runtimeFactory,
        scopeKey: "ws-race",
        sessionDir: tmp.path,
      }),
    ]);
    expect(b).toBe(a);
    await host.disposeScope("ws-race");
  });

  test("withPersistentMount holds the lease: concurrent disposal waits for fn to finish", async () => {
    using tmp = new DisposableTempDir("sandbox-host-test");
    const host = new SandboxHostService();
    const order: string[] = [];
    let releaseFn: (() => void) | undefined;
    const gate = new Promise<void>((resolve) => {
      releaseFn = resolve;
    });

    const run = host.withPersistentMount(
      {
        lifetime: "persistent",
        runtimeFactory,
        scopeKey: "ws-lease",
        sessionDir: tmp.path,
      },
      async (mount) => {
        order.push("fn-start");
        await gate;
        // The mount must still be live: disposeScope started while fn held
        // the lease and must be queued behind it, not race it.
        expect(mount.isDisposed).toBe(false);
        const result = await mount.runtime.eval("return 7;");
        order.push("fn-end");
        return result;
      }
    );
    // Give fn time to enter the lease, then attempt disposal concurrently.
    await new Promise((resolve) => setTimeout(resolve, 20));
    const disposal = host.disposeScope("ws-lease").then(() => order.push("disposed"));
    releaseFn?.();
    const result = await run;
    await disposal;
    expect(result.success).toBe(true);
    expect(order).toEqual(["fn-start", "fn-end", "disposed"]);
  });

  test("exclusive() serializes concurrent runs on a shared mount", async () => {
    using tmp = new DisposableTempDir("sandbox-host-test");
    const host = new SandboxHostService();
    const mount = await host.acquireMount({
      lifetime: "persistent",
      runtimeFactory,
      scopeKey: "ws-serial",
      sessionDir: tmp.path,
    });
    const order: string[] = [];
    await Promise.all([
      mount.exclusive(async () => {
        order.push("a-start");
        // Yield long enough that an unserialized implementation interleaves b.
        await new Promise((resolve) => setTimeout(resolve, 20));
        order.push("a-end");
      }),
      mount.exclusive(async () => {
        order.push("b-start");
        await Promise.resolve();
        order.push("b-end");
      }),
    ]);
    expect(order).toEqual(["a-start", "a-end", "b-start", "b-end"]);
    await host.disposeScope("ws-serial");
  });

  test("dropScope: workspace removal disposes the mount without writing to disk", async () => {
    using tmp = new DisposableTempDir("sandbox-host-test");
    const host = new SandboxHostService();
    const mount = await host.acquireMount({
      lifetime: "persistent",
      runtimeFactory,
      scopeKey: "ws-drop",
      sessionDir: tmp.path,
    });
    const write = await mount.runtime.eval("vars.x = 1; return vars.x;");
    expect(write.success).toBe(true);

    // Record disk state, then drop: the mount must be disposed and NO new
    // files may appear (the caller is deleting the session directory).
    const before = readdirSync(tmp.path, { recursive: true }).length;
    await host.dropScope("ws-drop");
    expect(mount.isDisposed).toBe(true);
    expect(host.hasScope("ws-drop")).toBe(false);
    const after = readdirSync(tmp.path, { recursive: true }).length;
    expect(after).toBe(before);
  });

  test("discardScope: context reset discards vars instead of restoring the last snapshot", async () => {
    using tmp = new DisposableTempDir("sandbox-host-test");
    const host = new SandboxHostService();
    const mount = await host.acquireMount({
      lifetime: "persistent",
      runtimeFactory,
      scopeKey: "ws-reset",
      sessionDir: tmp.path,
    });
    const write = await mount.runtime.eval("vars.secret = 'pre-reset'; return vars.secret;");
    expect(write.success).toBe(true);
    await mount.persistVars();

    await host.discardScope("ws-reset", tmp.path);
    expect(mount.isDisposed).toBe(true);

    // The next mount must start fresh, NOT restore pre-reset state.
    const fresh = await host.acquireMount({
      lifetime: "persistent",
      runtimeFactory,
      scopeKey: "ws-reset",
      sessionDir: tmp.path,
    });
    expect(fresh).not.toBe(mount);
    const probe = await fresh.runtime.eval("return Object.keys(vars).length;");
    expect(probe.success).toBe(true);
    expect(probe.result).toBe(0);
    await host.disposeScope("ws-reset");
  });

  test("reacquiring with changed grants rebuilds the mount under the new grants", async () => {
    using tmp = new DisposableTempDir("sandbox-host-test");
    const host = new SandboxHostService();
    const full = await host.acquireMount({
      lifetime: "persistent",
      runtimeFactory,
      scopeKey: "ws-grant-change",
      sessionDir: tmp.path,
    });
    const write = await full.runtime.eval("vars.x = 1; return vars.x;");
    expect(write.success).toBe(true);

    // Same scope, narrowed grants: the full-grants mount must not be reused.
    const narrowed = await host.acquireMount({
      lifetime: "persistent",
      runtimeFactory,
      scopeKey: "ws-grant-change",
      sessionDir: tmp.path,
      grants: LEAST_PRIVILEGE_GRANTS,
    });
    expect(narrowed).not.toBe(full);
    expect(full.isDisposed).toBe(true);
    // The rebuilt mount enforces the new boundary: no vars, no drain bridge.
    const varsProbe = await narrowed.runtime.eval("return typeof globalThis.vars;");
    expect(varsProbe.result).toBe("undefined");
    const drainProbe = await narrowed.runtime.eval("return typeof globalThis.drainHostEvents;");
    expect(drainProbe.result).toBe("undefined");
    await host.disposeScope("ws-grant-change");
  });

  test("bridge narrowing rebuilds the mount, revoking guest-saved bridge references", async () => {
    using tmp = new DisposableTempDir("sandbox-host-test");
    const host = new SandboxHostService();
    const tools = {
      bash: tool({
        description: "run",
        inputSchema: z.object({}),
        execute: () => Promise.resolve({ output: "ran" }),
      }),
    };

    const broadMount = await host.acquireMount({
      lifetime: "persistent",
      runtimeFactory,
      scopeKey: "ws-saved-ref",
      sessionDir: tmp.path,
      bridgeKey: "bash",
    });
    new ToolBridge(tools, FULL_GRANTS).register(broadMount.runtime);
    // Guest saves a bridge reference in a global — re-registering `mux` can
    // never revoke this closure; only destroying the runtime can.
    const saved = await broadMount.runtime.eval(
      "globalThis.savedBash = mux.bash; vars.keep = 1; return typeof savedBash;"
    );
    expect(saved.result).toBe("function");

    // Policy narrowed: the effective bridge no longer includes bash.
    const narrowedMount = await host.acquireMount({
      lifetime: "persistent",
      runtimeFactory,
      scopeKey: "ws-saved-ref",
      sessionDir: tmp.path,
      bridgeKey: "",
    });
    expect(narrowedMount).not.toBe(broadMount);
    expect(broadMount.isDisposed).toBe(true);
    // Saved closure is gone with the old runtime; vars survived the rebuild.
    const probe = await narrowedMount.runtime.eval(
      "return { saved: typeof globalThis.savedBash, kept: vars.keep };"
    );
    expect(probe.result).toEqual({ saved: "undefined", kept: 1 });
    await host.disposeScope("ws-saved-ref");
  });

  test("re-registering a narrower bridge on a reused runtime revokes previously exposed tools", async () => {
    using tmp = new DisposableTempDir("sandbox-host-test");
    const host = new SandboxHostService();
    const mount = await host.acquireMount({
      lifetime: "persistent",
      runtimeFactory,
      scopeKey: "ws-rebridge",
      sessionDir: tmp.path,
    });
    const tools = {
      bash: tool({
        description: "run",
        inputSchema: z.object({}),
        execute: () => Promise.resolve({ output: "ran" }),
      }),
    };

    const broad = new ToolBridge(tools, {
      version: 1,
      bridgeTools: { allow: "all" },
      vars: true,
      hostEvents: true,
    });
    broad.register(mount.runtime);
    const allowed = await mount.runtime.eval("return mux.bash({});");
    expect(allowed.success).toBe(true);
    expect(allowed.result).toEqual({ output: "ran" });

    // Next request narrowed the policy: code_execution re-registers its fresh
    // bridge on the reused runtime, which must fully replace the old one.
    const narrow = new ToolBridge(tools, {
      version: 1,
      bridgeTools: { allow: [] },
      vars: true,
      hostEvents: true,
    });
    narrow.register(mount.runtime);
    const denied = await mount.runtime.eval(`
      try {
        mux.bash({});
        return "no error";
      } catch (e) {
        return e.message;
      }
    `);
    expect(denied.success).toBe(true);
    expect(denied.result).toBe("Capability denied: xum.bash is not granted for this sandbox");
    await host.disposeScope("ws-rebridge");
  });

  test("corrupt snapshot blob self-heals to empty vars", async () => {
    using tmp = new DisposableTempDir("sandbox-host-test");
    const host1 = new SandboxHostService();
    const mount1 = await host1.acquireMount({
      lifetime: "persistent",
      runtimeFactory,
      scopeKey: "ws-heal",
      sessionDir: tmp.path,
    });
    await mount1.runtime.eval("vars.x = 1; return true;");
    await host1.disposeScope("ws-heal");

    // Corrupt every blob under the session dir.
    const corruptDir = (dir: string) => {
      for (const entry of readdirSync(dir)) {
        const full = join(dir, entry);
        if (statSync(full).isDirectory()) corruptDir(full);
        else writeFileSync(full, "corrupted!");
      }
    };
    corruptDir(join(tmp.path, "blobs"));

    const host2 = new SandboxHostService();
    const mount2 = await host2.acquireMount({
      lifetime: "persistent",
      runtimeFactory,
      scopeKey: "ws-heal",
      sessionDir: tmp.path,
    });
    const read = await mount2.runtime.eval("return vars;");
    expect(read.success).toBe(true);
    expect(read.result).toEqual({});
    await host2.disposeScope("ws-heal");
  });

  test("storeResultHandle assigns monotonic vars handles and persistResultHandle journals blob + event", async () => {
    using tmp = new DisposableTempDir("sandbox-host-test");
    const host = new SandboxHostService();
    const mount = await host.acquireMount({
      lifetime: "persistent",
      runtimeFactory,
      scopeKey: "ws-handles",
      sessionDir: tmp.path,
    });

    const big = JSON.stringify({ data: "x".repeat(100) });
    expect(await mount.storeResultHandle(big, 10_000)).toBe("__h1");
    expect(await mount.storeResultHandle(JSON.stringify({ n: 2 }), 10_000)).toBe("__h2");

    // The full value is guest-accessible under the handle var.
    const read = await mount.runtime.eval("return vars.__h1.data.length;");
    expect(read.success).toBe(true);
    expect(read.result).toBe(100);

    await mount.persistResultHandle({ handle: "vars.__h1", preview: "head…tail", serialized: big });
    const journal = new DurableEventJournal(tmp.path);
    const events = await journal.read();
    const handleEvent = events.find((e) => e.kind === "result-handle");
    expect(handleEvent).toBeDefined();
    if (handleEvent?.kind !== "result-handle") throw new Error("unreachable");
    expect(handleEvent.data.handle).toBe("vars.__h1");
    expect(handleEvent.data.preview).toBe("head…tail");
    expect(handleEvent.data.size).toBe(big.length);
    // The blob is the durable full value.
    expect(await journal.blobs.getText(handleEvent.data.blobHash)).toBe(big);
    await host.disposeScope("ws-handles");
  });

  test("handle sequence survives a simulated restart via the vars snapshot", async () => {
    using tmp = new DisposableTempDir("sandbox-host-test");
    const host1 = new SandboxHostService();
    const mount1 = await host1.acquireMount({
      lifetime: "persistent",
      runtimeFactory,
      scopeKey: "ws-handle-seq",
      sessionDir: tmp.path,
    });
    expect(await mount1.storeResultHandle(JSON.stringify({ a: 1 }), 10_000)).toBe("__h1");
    await host1.disposeScope("ws-handle-seq"); // snapshots vars incl. __handleSeq

    const host2 = new SandboxHostService();
    const mount2 = await host2.acquireMount({
      lifetime: "persistent",
      runtimeFactory,
      scopeKey: "ws-handle-seq",
      sessionDir: tmp.path,
    });
    // Monotonic across the restart: a fresh handle must not clobber __h1.
    expect(await mount2.storeResultHandle(JSON.stringify({ b: 2 }), 10_000)).toBe("__h2");
    const read = await mount2.runtime.eval("return [vars.__h1.a, vars.__h2.b];");
    expect(read.result).toEqual([1, 2]);
    await host2.disposeScope("ws-handle-seq");
  });

  test("storeResultHandle evicts oldest handles beyond the cap but never the newest", async () => {
    using tmp = new DisposableTempDir("sandbox-host-test");
    const host = new SandboxHostService();
    const mount = await host.acquireMount({
      lifetime: "persistent",
      runtimeFactory,
      scopeKey: "ws-evict",
      sessionDir: tmp.path,
    });

    // Each entry serializes to 402 chars; cap 1000 holds two.
    const entry = (c: string) => JSON.stringify(c.repeat(400));
    await mount.storeResultHandle(entry("a"), 1000); // __h1
    await mount.storeResultHandle(entry("b"), 1000); // __h2 (804 total, fits)
    await mount.storeResultHandle(entry("c"), 1000); // __h3 → evicts __h1
    const afterThird = await mount.runtime.eval(
      "return [typeof vars.__h1, typeof vars.__h2, typeof vars.__h3];"
    );
    expect(afterThird.result).toEqual(["undefined", "string", "string"]);

    // A single value larger than the cap is still retained (never evict the
    // newest: the model was just told the handle exists) while all older
    // handles are dropped.
    await mount.storeResultHandle(entry("d".repeat(13)), 1000); // __h4, ~5202 chars
    const afterFourth = await mount.runtime.eval(
      "return [typeof vars.__h2, typeof vars.__h3, vars.__h4.length];"
    );
    expect(afterFourth.result).toEqual(["undefined", "undefined", 5200]);
    await host.disposeScope("ws-evict");
  });

  test("enforceVarsRetention counts loads with handles and evicts oldest-first, protecting new keys", async () => {
    using tmp = new DisposableTempDir("sandbox-host-test");
    const host = new SandboxHostService();
    const mount = await host.acquireMount({
      lifetime: "persistent",
      runtimeFactory,
      scopeKey: "ws-load-evict",
      sessionDir: tmp.path,
    });

    // Age order: __h1 (seq 1), then load "big" (seq 2), then __h3 (seq 3).
    await mount.storeResultHandle(JSON.stringify("a".repeat(400)), 10_000); // __h1, 402
    const seed = await mount.runtime.eval('vars.big = "x".repeat(398); return true;'); // 400 serialized
    expect(seed.success).toBe(true);
    await mount.enforceVarsRetention({
      newLoadKeys: ["big"],
      protectedKeys: ["big"],
      capBytes: 10_000,
    });
    await mount.storeResultHandle(JSON.stringify("c".repeat(400)), 10_000); // __h3 (seq skips: load took 2)

    // Total ~1204 > 900: the OLDEST managed entry (__h1) evicts first even
    // though the load is not a handle; the load itself and __h3 survive.
    await mount.enforceVarsRetention({ newLoadKeys: [], protectedKeys: [], capBytes: 900 });
    const afterFirst = await mount.runtime.eval(
      "return [typeof vars.__h1, typeof vars.big, typeof vars.__h3, vars.__loadMeta];"
    );
    expect(afterFirst.result).toEqual(["undefined", "string", "string", { big: 2 }]);

    // Tighter cap: the load (now oldest) evicts too, and its registry entry
    // goes with it — unless it is protected as a NEW key this call.
    await mount.enforceVarsRetention({ newLoadKeys: [], protectedKeys: ["big"], capBytes: 300 });
    const stillProtected = await mount.runtime.eval("return [typeof vars.big, typeof vars.__h3];");
    expect(stillProtected.result).toEqual(["string", "undefined"]);

    await mount.enforceVarsRetention({ newLoadKeys: [], protectedKeys: [], capBytes: 300 });
    const afterSecond = await mount.runtime.eval("return [typeof vars.big, vars.__loadMeta];");
    expect(afterSecond.result).toEqual(["undefined", {}]);
    await host.disposeScope("ws-load-evict");
  });
});
