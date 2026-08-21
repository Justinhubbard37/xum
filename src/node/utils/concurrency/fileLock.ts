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
 * - Crash remnants are reclaimed when the recorded owner is provably gone:
 *   its pid is dead, OR the pid is alive but belongs to a DIFFERENT process
 *   (PID reuse, detected via a process-birth identity recorded in the
 *   token), OR staleness cannot be proven either way and the lock's mtime
 *   exceeds a generous lease. Claim-by-rename makes reclamation atomic (of
 *   two concurrent reclaimers only one rename succeeds), and reading the
 *   claimed file AFTER the rename verifies we claimed the token we judged
 *   stale — a raced fresh lock is restored via link (atomic, loses
 *   gracefully to an even newer lock, whose holder's release tolerates the
 *   loss).
 * - Release is ownership-verified: a mismatched token means the lock was
 *   reclaimed and re-acquired by someone else; leave it alone.
 */

import assert from "node:assert";
import { spawnSync } from "node:child_process";
import crypto from "node:crypto";
import { readFileSync } from "node:fs";
import * as fs from "fs/promises";
import * as path from "path";
import { log } from "@/node/services/log";

/** Poll interval while another live process holds the lock. */
const FILE_LOCK_RETRY_MS = 10;

/**
 * Lease for locks whose staleness cannot be proven via pid + birth identity
 * (platforms without a birth probe, or pre-birth-format tokens). Every
 * legitimate hold is ms (appends) to seconds (blob-lock recovery sweeps), so
 * minutes-old means the owner crashed — while a lease this generous can
 * never displace a merely slow live holder. Recovery from PID reuse on
 * birth-less platforms is thus bounded by this lease instead of requiring
 * manual lockfile cleanup.
 */
const FILE_LOCK_LEASE_MS = 5 * 60_000;

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

/**
 * Birth-probe memo: probing can spawn `ps` on non-Linux platforms, and the
 * reclaim path polls every ~15ms during contention. A short TTL bounds the
 * spawn rate; process birth is immutable, so a cached LIVE answer only goes
 * stale via pid reuse, which cannot happen within the TTL while the pid is
 * still alive.
 */
const birthCache = new Map<number, { birth: string | null; at: number }>();
const BIRTH_CACHE_TTL_MS = 1_000;

/**
 * Stable identity of the process currently occupying `pid`, or null when the
 * platform offers no probe (or the process vanished mid-probe). Recorded in
 * lock tokens and compared at reclamation: a live pid whose CURRENT birth
 * differs from the token's proves the OS reused the pid for an unrelated
 * process — without this, kill(pid, 0) alone would judge a crashed owner's
 * reused pid live forever, wedging every append until manual cleanup.
 * Token creation and verification run the same probe order on the same
 * machine (session dirs are host-local), so formats always align.
 * Exported for tests (constructing a verified-live token needs the format).
 */
export function getProcessBirth(pid: number): string | null {
  const cached = birthCache.get(pid);
  if (cached !== undefined && Date.now() - cached.at < BIRTH_CACHE_TTL_MS) {
    return cached.birth;
  }
  const birth = probeProcessBirth(pid);
  birthCache.set(pid, { birth, at: Date.now() });
  return birth;
}

function probeProcessBirth(pid: number): string | null {
  // Linux: /proc/<pid>/stat field 22 (starttime, clock ticks since boot) is
  // unique per pid incarnation. The comm field can embed spaces/parens, so
  // fields are parsed after the LAST ')' where the format is well-defined
  // (state is field 3 → starttime is offset 19).
  try {
    const stat = readFileSync(`/proc/${pid}/stat`, "utf-8");
    const rest = stat.slice(stat.lastIndexOf(")") + 2).split(" ");
    const starttime = rest[19];
    if (starttime !== undefined && /^\d+$/.test(starttime)) {
      return `linux-ticks:${starttime}`;
    }
  } catch {
    // Not Linux (or the process vanished); try the portable fallback.
  }
  // macOS/BSD: full start timestamp, stable per process incarnation.
  try {
    const out = spawnSync("ps", ["-o", "lstart=", "-p", String(pid)], { encoding: "utf-8" });
    const line = out.stdout?.trim();
    if (out.status === 0 && line !== undefined && line.length > 0) {
      return `ps-lstart:${line}`;
    }
  } catch {
    // ps unavailable (e.g. Windows): undeterminable, lease policy governs.
  }
  return null;
}

/** Parsed lock token. Legacy `pid:nonce` tokens have no birth (null). */
function parseLockToken(raw: string): { pid: number | null; birth: string | null } {
  const parts = raw.split(":");
  const pid = Number.parseInt(parts[0], 10);
  if (!Number.isSafeInteger(pid) || pid <= 0) {
    return { pid: null, birth: null };
  }
  const birthHex = parts[2];
  if (birthHex === undefined || !/^[0-9a-f]+$/.test(birthHex)) {
    return { pid, birth: null };
  }
  return { pid, birth: Buffer.from(birthHex, "hex").toString("utf-8") };
}

export async function acquireProcessFileLock(
  options: ProcessFileLockOptions
): Promise<AsyncDisposable> {
  const { lockPath, timeoutMs, label } = options;
  assert(lockPath.length > 0, "acquireProcessFileLock requires a lock path");
  assert(timeoutMs > 0, "acquireProcessFileLock timeoutMs must be positive");
  const nonce = crypto.randomBytes(8).toString("hex");
  // Record our birth identity so a future reclaimer can distinguish "this
  // pid is alive" from "this pid now belongs to someone else" (hex-encoded:
  // ps-derived birth strings contain spaces and colons).
  const ownBirth = getProcessBirth(process.pid);
  const token =
    ownBirth === null
      ? `${process.pid}:${nonce}`
      : `${process.pid}:${nonce}:${Buffer.from(ownBirth).toString("hex")}`;
  const tempPath = `${lockPath}.tmp-${process.pid}-${nonce}`;
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

/**
 * True when the lock is provably or presumptively stale (see module doc):
 * dead pid; live pid with a mismatched birth identity (PID reuse); or
 * undeterminable liveness past the lease. A live pid whose birth VERIFIABLY
 * matches the token is never stale, regardless of age — displacing a live
 * holder risks double-entry, which no lease can justify.
 */
async function isLockStale(lockPath: string, observed: string): Promise<boolean> {
  const { pid, birth } = parseLockToken(observed);
  if (pid === null) {
    // Malformed token: no owner to probe; only the lease bounds it.
    return await lockLeaseExpired(lockPath);
  }
  if (!isPidAlive(pid)) {
    return true;
  }
  const currentBirth = getProcessBirth(pid);
  if (birth !== null && currentBirth !== null) {
    return currentBirth !== birth;
  }
  return await lockLeaseExpired(lockPath);
}

/** True when the lockfile's mtime is older than the stale-lock lease. */
async function lockLeaseExpired(lockPath: string): Promise<boolean> {
  try {
    const stats = await fs.stat(lockPath);
    return Date.now() - stats.mtimeMs > FILE_LOCK_LEASE_MS;
  } catch {
    return false; // Vanished (released/reclaimed): retry acquisition instead.
  }
}

/** Reclaim the lock if its recorded owner is provably gone (see module doc). */
async function reclaimStaleFileLock(lockPath: string, label: string): Promise<void> {
  let observed: string;
  try {
    observed = await fs.readFile(lockPath, "utf-8");
  } catch {
    return; // Already released or reclaimed; retry acquisition.
  }
  if (!(await isLockStale(lockPath, observed))) {
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
