/**
 * Shared per-target mutation locks (RLM rollback hardening).
 *
 * The rollback engine's divergence check and its inverse apply are two steps;
 * without a lock shared with ORDINARY writers, a normal MemoryService write
 * or agent_skill_write/delete to the same root can land between them and be
 * silently overwritten by the rollback (the rollback session mutex + lockfile
 * only serialize other rollbacks). Every mutation path therefore acquires a
 * process-wide mutex keyed by the canonical mutation root, and the rollback
 * re-verifies divergence INSIDE that lock immediately before applying.
 *
 * Keys (must be identical strings on the writer and rollback sides):
 * - memory, global/project scopes: `<muxRoot>/memory` (one coarse key — the
 *   rollback confinement root; per-scope granularity is not worth divergent
 *   key derivations, and memory writes are ms-range local I/O);
 * - memory, workspace scope: `<sessionDir>/memory` (the store root, which is
 *   also the rollback confinement root);
 * - skills: the resolved skills root (`.../.mux/skills`, `.../.agents/skills`
 *   or `<muxRoot>/skills`), as returned by the rollback confinement resolver
 *   and known to the local skill tools. Runtime-backed (SSH/Docker) skill
 *   writers are excluded: their rows are stamped `runtime: "remote"` and are
 *   never rollbackable, so there is nothing to serialize against.
 *
 * Lock ordering (deadlock safety): the rollback acquires its per-session
 * mutex, then the cross-process lockfile, then these target locks (sorted);
 * writers acquire ONLY a target lock (and may take the journal blob lock
 * inside it). Nothing acquires the session mutex or lockfile while holding a
 * target lock, so no cycle exists.
 *
 * Cross-process scope: this gives the strong guarantee in-process only. The
 * debug-CLI rollback runs in a separate process where ordinary writers do not
 * consult the rollback lockfile (a per-write existence probe would tax every
 * memory write); its window is narrowed by the same in-lock re-verification
 * running immediately before each write while the rollback lockfile is held.
 */

import * as path from "node:path";

import { MutexMap } from "@/node/utils/concurrency/mutexMap";

/** Process-wide registry; see module doc for key derivation and ordering. */
export const targetMutationLocks = new MutexMap<string>();

/** Canonical lock key for a memory store root (see module doc). */
export function memoryMutationLockKey(muxRoot: string, physicalRoot: string): string {
  const memoryRoot = path.resolve(muxRoot, "memory");
  const resolved = path.resolve(physicalRoot);
  return resolved === memoryRoot || resolved.startsWith(memoryRoot + path.sep)
    ? memoryRoot
    : resolved;
}

/**
 * Acquire several target locks (deduped, sorted for a deterministic global
 * order so overlapping multi-root rollbacks cannot ABBA-deadlock), then run.
 */
export async function withTargetMutationLocks<T>(keys: string[], fn: () => Promise<T>): Promise<T> {
  const sorted = [...new Set(keys.map((key) => path.resolve(key)))].sort();
  const run = (index: number): Promise<T> =>
    index >= sorted.length
      ? fn()
      : targetMutationLocks.withLock(sorted[index], () => run(index + 1));
  return run(0);
}
