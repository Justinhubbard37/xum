/**
 * Code Execution Tool for Programmatic Tool Calling (PTC)
 *
 * Executes JavaScript code in a sandboxed QuickJS environment with access to all
 * Xum tools via the `xum.*` namespace (`mux.*` remains a compatibility alias).
 * Enables multi-tool workflows in a single inference instead of multiple round-trips.
 */

import { tool } from "ai";
import { z } from "zod";
import type { Tool } from "ai";
import type { ToolBridge } from "@/node/services/ptc/toolBridge";
import type { IJSRuntime, IJSRuntimeFactory } from "@/node/services/ptc/runtime";
import type { PTCConsoleRecord, PTCEvent, PTCExecutionResult } from "@/node/services/ptc/types";
import type { SandboxMount } from "@/node/services/sandbox/sandboxHostService";
import type { KernelFileLoader } from "@/node/services/tools/kernelFileLoad";

import { analyzeCode } from "@/node/services/ptc/staticAnalysis";
import { log } from "@/node/services/log";
import { getCachedXumTypes, clearTypeCache } from "@/node/services/ptc/typeGenerator";
import {
  buildHandlePreview,
  RESULT_HANDLE_OFFLOAD_THRESHOLD_BYTES,
  RESULT_HANDLE_VARS_CAP_BYTES,
} from "@/constants/resultHandles";
import { KERNEL_CONSOLE_CAP_BYTES } from "@/constants/kernelOutput";

// Default limits
const DEFAULT_MEMORY_BYTES = 64 * 1024 * 1024; // 64MB
const DEFAULT_TIMEOUT_SECS = 5 * 60; // 5 minutes
const MAX_TIMEOUT_SECS = 60 * 60; // 1 hour

/**
 * Clear all type caches. Call for test isolation or when tool schemas might have changed.
 */
export function clearTypeCaches(): void {
  clearTypeCache();
}

/** PTC event with parentToolCallId attached by code_execution */
export type PTCEventWithParent = PTCEvent & { parentToolCallId: string };

/**
 * Create the code_execution tool.
 *
 * This function is async because it generates TypeScript type definitions
 * from the tool schemas, which requires async JSON Schema to TypeScript conversion.
 *
 * @param runtimeFactory Factory for creating QuickJS runtime instances
 * @param toolBridge Bridge containing tools to expose in sandbox
 * @param emitNestedEvent Callback for streaming nested tool events (includes parentToolCallId)
 * @param withMount Optional SandboxHostService lease runner
 *   (withPersistentMount bound to this workspace's scope). When absent,
 *   behavior is the classic ephemeral per-call flow (create → eval → dispose).
 *   The runner holds the scope lock from mount acquisition through fn's
 *   completion, so the register→eval→persist sequence cannot race concurrent
 *   grant changes or scope disposal; the persistent runtime is not disposed
 *   here.
 */
export type MountRunner = (
  fn: (mount: SandboxMount) => Promise<PTCExecutionResult>
) => Promise<PTCExecutionResult>;

/**
 * Late-bound dispatch state for a created code_execution instance. execute()
 * reads bridge + mount runner from here at CALL time (not closure-capture
 * time) so retargetCodeExecutionTool can swing an already-created instance —
 * and any middleware wrapper delegating to it, even through a captured
 * `execute` function reference — onto a fresh bridge/mount.
 */
interface RetargetableState {
  toolBridge: ToolBridge;
  withMount: MountRunner | undefined;
  /** Host file loader backing mux.load (kernel mode only); see KernelBridgeOptions. */
  loadFile: KernelFileLoader | undefined;
}

const retargetableStates = new WeakMap<object, RetargetableState>();

/**
 * Point `target` (an instance returned by createCodeExecutionTool) at the
 * bridge + mount runner of `donor` (another such instance). Used when a
 * request.assemble hook wrapped/replaced code_execution while also editing
 * other bridgeable tools: the wrapper delegates to the PRE-hook instance,
 * which must dispatch through the rebuilt post-hook bridge instead of the
 * stale one. Returns false when either tool was not created by this factory.
 */
export function retargetCodeExecutionTool(target: Tool, donor: Tool): boolean {
  const targetState = retargetableStates.get(target);
  const donorState = retargetableStates.get(donor);
  if (targetState === undefined || donorState === undefined) {
    return false;
  }
  targetState.toolBridge = donorState.toolBridge;
  targetState.withMount = donorState.withMount;
  targetState.loadFile = donorState.loadFile;
  return true;
}

/** Model-visible replacement for an offloaded oversized value. */
export interface OffloadedValueRecord {
  /** Guest expression holding the full value, e.g. "vars.__h3". */
  handle: string;
  /** Bounded head/tail excerpt of the serialized value. */
  preview: string;
  /** Full serialized size in bytes. */
  size: number;
  /** One-line follow-up hint (offloaded top-level return values only). */
  hint?: string;
}

/**
 * Offload one oversized value to the persistent kernel. Returns the
 * model-visible replacement record, or null when the value is sub-threshold
 * or could not be offloaded (in which case it must stay inline).
 */
async function offloadValue(
  mount: SandboxMount,
  value: unknown
): Promise<OffloadedValueRecord | null> {
  let serialized: string | undefined;
  try {
    serialized = JSON.stringify(value);
  } catch {
    // Non-JSON values cannot live in vars (data-only contract); keep inline.
    return null;
  }
  if (typeof serialized !== "string") return null;
  const size = Buffer.byteLength(serialized, "utf8");
  if (size <= RESULT_HANDLE_OFFLOAD_THRESHOLD_BYTES) return null;

  // Store in vars FIRST: if the guest assignment fails, the model record must
  // keep the full inline value — never point the model at a missing handle.
  let handleKey: string;
  try {
    handleKey = await mount.storeResultHandle(serialized, RESULT_HANDLE_VARS_CAP_BYTES);
  } catch (error) {
    log.warn("code_execution: result-handle vars assignment failed; keeping full value inline", {
      error,
    });
    return null;
  }
  const handle = `vars.${handleKey}`;
  const preview = buildHandlePreview(serialized, size);
  try {
    await mount.persistResultHandle({ handle, preview, serialized });
  } catch (error) {
    // The model-visible preview is durably logged with the tool result in
    // chat.jsonl either way; a journaling failure only degrades durability of
    // the FULL value and must never fail the call (self-healing doctrine).
    log.warn("code_execution: result-handle journaling failed; continuing", { error });
  }
  return { handle, preview, size };
}

/**
 * RLM context offloading for the TOP-LEVEL return value: values above the
 * threshold stop entering the model context. The model-visible result is
 * replaced by { handle, preview, size } while the full value lands in
 * vars.__hN (guest), the blob store, and one result-handle durable event.
 * Nested records need no offload machinery in kernel mode — they carry no
 * payload at all (see compactKernelToolCallRecords). Mutates `result` in
 * place.
 */
async function offloadOversizedReturnValue(
  mount: SandboxMount,
  result: PTCExecutionResult
): Promise<string | null> {
  if (result.result !== undefined) {
    const offloaded = await offloadValue(mount, result.result);
    if (offloaded !== null) {
      result.result = {
        ...offloaded,
        hint: `Return value exceeded the inline limit; the full value is stored in the kernel — access or slice ${offloaded.handle} in a follow-up code_execution call.`,
      } satisfies OffloadedValueRecord;
      // "vars.__hN" → "__hN": the bare vars key, for retention protection.
      return offloaded.handle.replace(/^vars\./, "");
    }
  }
  return null;
}

/**
 * Keys successfully loaded by mux.load THIS call (r12): their compact records
 * carry the {key, ...} summary result, failed loads carry only an error.
 */
function collectNewLoadKeys(result: PTCExecutionResult, loadActive: boolean): string[] {
  if (!loadActive) return [];
  const keys = new Set<string>();
  for (const record of result.toolCalls) {
    if (record.toolName !== "load" || record.error !== undefined) continue;
    const key =
      typeof record.result === "object" && record.result !== null
        ? (record.result as { key?: unknown }).key
        : undefined;
    if (typeof key === "string" && key.length > 0) keys.add(key);
  }
  return [...keys];
}

/**
 * Kernel-mode record suppression (r12): the point of the persistent kernel is
 * that in-kernel data does NOT transit the model context. Every nested
 * mux.* record becomes a compact {toolName, args, ok, bytes, error?} summary —
 * never an inline result, regardless of size. The running guest already
 * received the full value; return value / console / vars are the model's
 * deliberate channels for surfacing data. On failure the error message stays
 * visible (bounded — message only) so the model can retry intelligently.
 * Mutates `result` in place; nested UI events already streamed the full
 * values live.
 *
 * Exception: mux.load records stay as-is when the kernel load is active —
 * their result is a bounded {key, bytes, lines, preview} summary by
 * construction (the file content goes host-side straight into vars and never
 * touches the record), and the model needs the key/shape it just created.
 * When the kernel load is inactive, a bridged tool that happens to be named
 * "load" gets no exception (its records are ordinary and must not leak).
 */
function compactKernelToolCallRecords(result: PTCExecutionResult, loadActive: boolean): void {
  result.toolCalls = result.toolCalls.map((record) => {
    if (loadActive && record.toolName === "load") return record;
    let bytes = 0;
    if (record.result !== undefined) {
      try {
        bytes = Buffer.byteLength(JSON.stringify(record.result) ?? "", "utf8");
      } catch {
        // Bridged results are JSON round-tripped, so this is unreachable in
        // practice; size 0 is an honest fallback (nothing model-visible).
        bytes = 0;
      }
    }
    return {
      toolName: record.toolName,
      args: record.args,
      ok: record.error === undefined,
      bytes,
      ...(record.error !== undefined ? { error: record.error } : {}),
      duration_ms: record.duration_ms,
    };
  });
}

/**
 * Kernel-mode console bound (r12): console output is the model's deliberate
 * debug/print channel and stays visible, but it must not become a suppression
 * bypass. Total console bytes per execution are capped; the crossing record
 * keeps a bounded head and a final warn record reports what was dropped —
 * never a silent drop. Byte accounting uses the JSON serialization of each
 * record's args (what the model would see). Mutates `result` in place.
 */
function capKernelConsoleOutput(result: PTCExecutionResult): void {
  let total = 0;
  let droppedRecords = 0;
  let droppedBytes = 0;
  const kept: PTCConsoleRecord[] = [];
  for (const record of result.consoleOutput) {
    let serialized: string;
    try {
      serialized = JSON.stringify(record.args) ?? "";
    } catch {
      serialized = "";
    }
    const size = Buffer.byteLength(serialized, "utf8");
    if (droppedRecords === 0 && total + size <= KERNEL_CONSOLE_CAP_BYTES) {
      kept.push(record);
      total += size;
      continue;
    }
    droppedRecords += 1;
    if (droppedRecords === 1 && total < KERNEL_CONSOLE_CAP_BYTES) {
      // Crossing record: keep a bounded head (char-sliced — close enough to
      // bytes for a soft cap) instead of dropping it whole.
      const remaining = KERNEL_CONSOLE_CAP_BYTES - total;
      kept.push({
        level: record.level,
        args: [`${serialized.slice(0, remaining)}…[truncated]`],
        timestamp: record.timestamp,
      });
      droppedBytes += Math.max(0, size - remaining);
      total = KERNEL_CONSOLE_CAP_BYTES;
      continue;
    }
    droppedBytes += size;
  }
  if (droppedRecords === 0) return;
  kept.push({
    level: "warn",
    args: [
      `[console output truncated: ${KERNEL_CONSOLE_CAP_BYTES}-byte kernel cap reached; ${droppedRecords} record(s) / ~${droppedBytes} bytes dropped]`,
    ],
    timestamp: result.consoleOutput[result.consoleOutput.length - 1]?.timestamp ?? 0,
  });
  result.consoleOutput = kept;
}

/** Model-facing description options for createCodeExecutionTool. */
export interface CodeExecutionToolOptions {
  /**
   * RLM + PTC-exclusive posture: code_execution is the single kernel tool, so
   * its description leads with a short preamble tying the kernel features
   * (persistent vars, result handles + slicing, task_spawn/events) together.
   * Only honored when a persistent mount exists — advertising kernel features
   * without a kernel would instruct the model to use APIs that don't exist.
   */
  kernelFirst?: boolean;
  /**
   * Host file loader backing mux.load (r12 bulk ingestion). Only honored in
   * kernel mode with file_read bridged — same "never advertise a missing
   * API" rule as kernelFirst.
   */
  loadFile?: KernelFileLoader;
}

export async function createCodeExecutionTool(
  runtimeFactory: IJSRuntimeFactory,
  toolBridge: ToolBridge,
  emitNestedEvent?: (event: PTCEventWithParent) => void,
  withMount?: MountRunner,
  options?: CodeExecutionToolOptions
): Promise<Tool> {
  const bridgeableTools = toolBridge.getBridgeableTools();
  const state: RetargetableState = { toolBridge, withMount, loadFile: options?.loadFile };

  // Kernel mode = persistent mount available (RLM experiment, or the
  // XUM_SANDBOX_PERSISTENT_MOUNTS dev override that rides the same path).
  // Gates every model-visible kernel surface below so RLM-off requests stay
  // byte-identical to today.
  const kernel = withMount !== undefined;

  // xum.load availability: kernel mode + a host file loader + file_read
  // bridged (load rides file_read's grant). Must match
  // ToolBridge.addKernelMethods so types/description never advertise a
  // missing member.
  const loadEnabled = kernel && options?.loadFile !== undefined && "file_read" in bridgeableTools;

  // Generate xum types for type validation and documentation (cached by tool set hash)
  const xumTypes = await getCachedXumTypes(bridgeableTools, { kernel, load: loadEnabled });

  // Persistent-kernel addendum: only advertised when this instance runs on a
  // persistent mount (RLM mode or XUM_SANDBOX_PERSISTENT_MOUNTS). Ephemeral
  // instances must keep today's description byte-identical so RLM-off
  // provider requests are unchanged.
  const persistentKernelNotes = !kernel
    ? ""
    : `

**Persistent kernel:** the global \`vars\` object persists across code_execution calls and turns (JSON-serializable values only) and survives restarts via snapshots. Nested tool results do NOT enter your context: each mux.* call's visible record is a compact {tool, ok, bytes} summary (plus the error message on failure). Data reaches you only through your \`return\` value (offloaded to a {handle, preview, size} vars handle like \`vars.__h1\` when >${Math.floor(RESULT_HANDLE_OFFLOAD_THRESHOLD_BYTES / 1024)}KB serialized — read or slice it in a follow-up call), \`console\` output (capped at ${Math.floor(KERNEL_CONSOLE_CAP_BYTES / 1024)}KB per execution), and \`vars\`. Keep working data in \`vars\` and return only what you need to see. Note \`mux.file_read\` errors beyond its ~16KB/1000-line per-call cap (it does not offload).${
        loadEnabled
          ? `
**Bulk file ingestion:** \`mux.load({path, key})\` reads a whole file host-side into \`vars[key]\` (string) and shows you only {key, bytes, lines, preview}. Use it instead of paginated \`mux.file_read\` for large files.`
          : ""
      }${
        "task" in bridgeableTools
          ? `
**Fire-and-forget sub-agents:** \`xum.task_spawn(args)\` (same args as \`xum.task\`) returns immediately with {taskId, status:"spawned"} once the child is admitted. Terminal reports are queued in the kernel — drain with \`xum.events()\` in a later call. The queue is best-effort (an app restart may drop it); every report still reaches you via the normal task wake.`
          : ""
      }`;

  // Kernel-first preamble: only for the RLM + exclusive posture (see
  // CodeExecutionToolOptions). Exclusive mode without RLM and the env-var
  // mount override keep their current descriptions byte-identical.
  const kernelFirstPreamble =
    kernel && options?.kernelFirst === true
      ? `**Kernel-first workflow:** this is your primary tool — other tools are \`mux.*\` calls inside it. Persist state in \`vars\` across calls and turns; nested results stay in the kernel (you see compact {tool, ok, bytes} summaries), and an oversized return value comes back as {handle, preview, size} — read or slice the full value at its handle in a follow-up call${
          "task" in bridgeableTools
            ? "; spawn sub-agents with `mux.task_spawn(...)` and collect their reports with `mux.events()`"
            : ""
        }.

`
      : "";

  const codeExecutionTool = tool({
    description: `${kernelFirstPreamble}Execute sandboxed JavaScript to batch tools and transform outputs.

**When to use:** Prefer this tool when making 2+ tool calls, especially when later calls depend on earlier results. Reduces round-trip latency.

**Available tools (TypeScript definitions):**
\`\`\`typescript
${xumTypes}
\`\`\`

**Usage notes:**
- \`xum.*\` functions are synchronous—do not use \`await\`. \`mux.*\` is a compatibility alias.
- Use \`return\` to provide a final result to the model
- Use \`console.log/warn/error\` for debugging - output is captured
- Results are JSON-serialized; non-serializable values return \`{ error: "..." }\`
- On failure, partial results (completed tool calls) are returned for debugging${persistentKernelNotes}

**Security:** The sandbox has no access to \`require\`, \`import\`, \`process\`, \`fetch\`, or filesystem outside of \`xum.*\` tools.`,

    inputSchema: z.object({
      code: z
        .string()
        .min(1)
        .describe(
          "JavaScript code to execute. xum.* calls are synchronous—do not use await. mux.* is a compatibility alias. Use 'return' for final result."
        ),
      timeout_secs: z
        .number()
        .int()
        .positive()
        .nullish()
        .describe(
          "Execution timeout in seconds (default: 300, max: 3600). " +
            "Increase when spawning subagents that may take 5-15+ minutes."
        ),
    }),

    execute: async (
      { code, timeout_secs },
      { abortSignal, toolCallId }
    ): Promise<PTCExecutionResult> => {
      const execStartTime = Date.now();

      // Late-bound dispatch: snapshot the CURRENT bridge + mount runner as a
      // pair so a retarget (see retargetCodeExecutionTool) lands atomically —
      // the whole call uses either the old pair or the new pair, never a mix.
      const { toolBridge: activeBridge, withMount: activeMount, loadFile: activeLoadFile } = state;

      // Mirrors the creation-time loadEnabled gate against the ACTIVE bridge
      // (a retarget may have narrowed file_read away).
      const loadActive =
        activeLoadFile !== undefined && activeBridge.getBridgeableToolNames().includes("file_read");

      // Static analysis before execution - catch syntax errors and sandbox-forbidden patterns.
      // TypeScript typing issues are intentionally non-blocking for one-off runtime scripts.
      const analysis = await analyzeCode(code);
      if (!analysis.valid) {
        const errorMessages = analysis.errors.map((e) => {
          const location =
            e.line && e.column
              ? ` (line ${e.line}, col ${e.column})`
              : e.line
                ? ` (line ${e.line})`
                : "";
          return `- ${e.message}${location}`;
        });
        return {
          success: false,
          error: `Code analysis failed:\n${errorMessages.join("\n")}`,
          toolCalls: [],
          consoleOutput: [],
          duration_ms: Date.now() - execStartTime,
        };
      }

      const runWithRuntime = async (
        mount: SandboxMount | null,
        runtime: IJSRuntime
      ): Promise<PTCExecutionResult> => {
        const onAbort = () => runtime.abort();
        try {
          // Set resource limits (clamp timeout to max)
          const timeoutSecs = Math.min(timeout_secs ?? DEFAULT_TIMEOUT_SECS, MAX_TIMEOUT_SECS);
          runtime.setLimits({
            memoryBytes: DEFAULT_MEMORY_BYTES,
            timeoutMs: timeoutSecs * 1000,
          });

          // Subscribe to events for UI streaming
          // Wrap callback to include parentToolCallId from AI SDK context
          if (emitNestedEvent) {
            runtime.onEvent((event: PTCEvent) => {
              emitNestedEvent({ ...event, parentToolCallId: toolCallId });
            });
          }

          // Register tools - they'll use runtime.getAbortSignal() for cancellation.
          // Always re-register, even on reused persistent mounts: each request
          // builds a fresh ToolBridge from the CURRENT policy + grants, and a
          // stale bridge would keep exposing tools after permissions narrowed.
          // Registration just overwrites the guest's `xum`/`mux` globals, so this is
          // cheap and idempotent. Persistent mounts get the kernel extras
          // (xum.task_spawn / xum.events) bound to this mount's event queue.
          activeBridge.register(
            runtime,
            mount?.lifetime === "persistent"
              ? {
                  drainHostEvents: () => mount.drainHostEvents(),
                  ...(activeLoadFile !== undefined ? { loadFile: activeLoadFile } : {}),
                }
              : undefined
          );

          // Handle abort signal - interrupt sandbox and cancel nested tools
          if (abortSignal) {
            // If already aborted, abort runtime immediately
            if (abortSignal.aborted) {
              runtime.abort();
            } else {
              abortSignal.addEventListener("abort", onAbort, { once: true });
            }
          }

          // Execute the code
          const result = await runtime.eval(code);

          // Kernel-mode context isolation (r12): nested records become compact
          // summaries and console output is bounded, regardless of grants —
          // suppression only drops data, it stores nothing. Runs even for
          // failed evals: partial toolCalls records are model-visible too and
          // must not leak either (their error messages stay visible).
          if (mount?.lifetime === "persistent") {
            compactKernelToolCallRecords(result, loadActive);
            capKernelConsoleOutput(result);
          }

          // RLM return-value offloading BEFORE the vars snapshot below, so the
          // handle vars land in the same durable snapshot the model's
          // {handle, preview, size} record relies on.
          if (mount?.lifetime === "persistent" && mount.grants.vars) {
            const returnHandleKey = await offloadOversizedReturnValue(mount, result);

            // r12: loads count toward the r4 vars retention cap — register
            // this call's loaded keys and evict oldest managed entries
            // (handles + loads) beyond the cap. Keys the model was JUST told
            // about (new loads + the fresh return handle) are protected.
            // Retention failure must never fail the call (self-healing).
            const newLoadKeys = collectNewLoadKeys(result, loadActive);
            if (newLoadKeys.length > 0 || returnHandleKey !== null) {
              try {
                await mount.enforceVarsRetention({
                  newLoadKeys,
                  protectedKeys:
                    returnHandleKey !== null ? [...newLoadKeys, returnHandleKey] : newLoadKeys,
                  capBytes: RESULT_HANDLE_VARS_CAP_BYTES,
                });
              } catch (error) {
                log.warn("code_execution: vars retention enforcement failed; continuing", {
                  error,
                });
              }
            }
          }

          // Persist the shared vars namespace after each call on persistent
          // mounts so state survives crashes/restarts (turn-boundary snapshots
          // are the Track 2 refinement; per-call is the safe foundation).
          // Failed/timed-out/aborted evals may still have mutated vars before
          // failing and the live guest keeps those mutations, so persist after
          // failures too — memory and disk must agree.
          if (mount?.lifetime === "persistent" && mount.grants.vars) {
            try {
              await mount.persistVars();
            } catch (persistError) {
              // Vars became unsnapshottable (e.g. guest created a cycle then
              // threw). Leaving the live mount would make memory and disk
              // permanently disagree; dispose it so the next acquire rebuilds
              // from the last durable snapshot. Never mask the eval result
              // with a snapshot error.
              log.warn(
                "code_execution: vars snapshot failed; disposing mount so the next call restores the last durable snapshot",
                { persistError }
              );
              mount.dispose();
            }
          }
          return result;
        } finally {
          // A late abort of THIS call's signal must not poison a reused runtime.
          abortSignal?.removeEventListener("abort", onAbort);
        }
      };

      // Persistent mounts can be handed to concurrent code_execution calls
      // for the same workspace, but eval() mutates runtime-wide state (abort
      // controller, tool-call attribution, event handler). The lease runner
      // holds the scope lock from acquisition through the whole
      // register→eval→persist sequence, so concurrent calls, grant changes,
      // and scope disposal are all serialized against this execution.
      if (activeMount) {
        return await activeMount((mount) => runWithRuntime(mount, mount.runtime));
      }
      // Classic ephemeral flow: per-call runtime, no serialization needed.
      const runtime = await runtimeFactory.create();
      try {
        return await runWithRuntime(null, runtime);
      } finally {
        runtime.dispose();
      }
    },
  });
  retargetableStates.set(codeExecutionTool, state);
  return codeExecutionTool;
}
