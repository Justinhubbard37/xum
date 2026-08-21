/**
 * Cross-process filesystem lock (extracted from the journal kit's append
 * lock so the durable-event blob lock can share one proven protocol).
 *
 * Protocol:
 * - Lock birth is atomic-with-content: the token (`pid:nonce`) is fully
 *   written to a temp file first and hard-linked into place (link fails
 *   EEXIST when held), so a reader can never observe a token-less lock.
 * - Waiting is a bounded jittered poll — there is no portable cross-process
 *   wake primitive available here.
 * - Crash remnants are reclaimed when the recorded owner pid is provably
 *   dead; claim-by-rename makes reclamation atomic (of two concurrent
 *   reclaimers only one rename succeeds), and reading the claimed file AFTER
 *   the rename verifies we claimed the token we judged dead — a raced fresh
 *   lock is restored via link (atomic, loses gracefully to an even newer
 *   lock, whose holder's release tolerates the loss).
 * - Release is ownership-verified: a mismatched token means the lock was
 *   reclaimed and re-acquired by someone else; leave it alone.
 */

import assert from "node:assert";
import crypto from "node:crypto";
import * as fs from "fs/promises";
import * as path from "path";
import { log } from "@/node/services/log";

/** Poll interval while another live process holds the lock. */
const FILE_LOCK_RETRY_MS = 10;

export interface ProcessFileLockOptions {
  /** Absolute or relative lockfile path; the parent directory is created. */
  lockPath: string;
  /** Max milliseconds to wait before acquisition fails. */
  timeoutMs: number;
  /** Human label for error/log messages (e.g. "append lock", "blob lock"). */
  label: string;
}

/** True when a signal-0 probe reaches the pid (EPERM = alive, not ours). */
function isPidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === "EPERM";
  }
}

export async function acquireProcessFileLock(
  options: ProcessFileLockOptions
): Promise<AsyncDisposable> {
  const { lockPath, timeoutMs, label } = options;
  assert(lockPath.length > 0, "acquireProcessFileLock requires a lock path");
  assert(timeoutMs > 0, "acquireProcessFileLock timeoutMs must be positive");
  const token = `${process.pid}:${crypto.randomBytes(8).toString("hex")}`;
  const tempPath = `${lockPath}.tmp-${token.replace(":", "-")}`;
  const deadline = Date.now() + timeoutMs;
  await fs.mkdir(path.dirname(lockPath), { recursive: true });
  await fs.writeFile(tempPath, token, "utf-8");
  try {
    for (;;) {
      try {
        await fs.link(tempPath, lockPath);
        return { [Symbol.asyncDispose]: () => releaseFileLock(lockPath, token, label) };
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "EEXIST") {
          throw error;
        }
      }
      await reclaimStaleFileLock(lockPath, label);
      if (Date.now() >= deadline) {
        throw new Error(`Timed out acquiring ${label} ${lockPath} after ${timeoutMs}ms`);
      }
      await new Promise((resolve) =>
        setTimeout(resolve, FILE_LOCK_RETRY_MS + Math.random() * FILE_LOCK_RETRY_MS)
      );
    }
  } finally {
    await fs.unlink(tempPath).catch(() => undefined);
  }
}

/** Reclaim the lock if its recorded owner is provably dead (see module doc). */
async function reclaimStaleFileLock(lockPath: string, label: string): Promise<void> {
  let observed: string;
  try {
    observed = await fs.readFile(lockPath, "utf-8");
  } catch {
    return; // Already released or reclaimed; retry acquisition.
  }
  const pid = Number.parseInt(observed.split(":")[0], 10);
  if (!Number.isSafeInteger(pid) || pid <= 0 || isPidAlive(pid)) {
    return;
  }
  const graveyard = `${lockPath}.stale-${crypto.randomBytes(4).toString("hex")}`;
  try {
    await fs.rename(lockPath, graveyard);
  } catch {
    return; // Another reclaimer won the rename; retry acquisition.
  }
  const claimed = await fs.readFile(graveyard, "utf-8").catch(() => null);
  if (claimed !== null && claimed !== observed) {
    log.warn(`FileLock: reclaim raced a fresh ${label} on ${lockPath}; restoring it`);
    await fs.link(graveyard, lockPath).catch(() => undefined);
  }
  await fs.unlink(graveyard).catch(() => undefined);
}

/** Release only if we still own the lock (a raced reclaim may have replaced it). */
async function releaseFileLock(lockPath: string, token: string, label: string): Promise<void> {
  try {
    const content = await fs.readFile(lockPath, "utf-8");
    if (content !== token) {
      log.warn(`FileLock: ${label} ${lockPath} changed owners before release; leaving it`);
      return;
    }
    await fs.unlink(lockPath);
  } catch (error) {
    log.debug(`FileLock: failed to release ${label} ${lockPath}`, { error });
  }
}
