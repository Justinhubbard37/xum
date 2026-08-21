/**
 * Journal kit: append-only JSONL journal with monotonic sequence assignment,
 * stable-ID dedupe, and self-healing reads (substrate 2 of the shared agent
 * foundation).
 *
 * Doctrine (matches HistoryService/TimelineService): one malformed or
 * duplicated line must never brick the log. Appends are single-write JSONL
 * lines; torn tails from crashes are healed by prepending a separator on the
 * next append and by skipping unparseable lines on read.
 *
 * Writer serialization is two-level:
 * - within one instance, appends run through an internal promise queue;
 * - across instances AND processes (the debug CLI appending while the app is
 *   live), each append holds a cross-process lockfile while it derives the
 *   next sequence and writes, revalidating the cached counter against the
 *   file size so a foreign append can never lead to a duplicated seq.
 */

import assert from "node:assert";
import crypto from "node:crypto";
import * as fs from "fs/promises";
import * as path from "path";
import { log } from "@/node/services/log";

/** Poll interval while another live process holds the append lock. */
const APPEND_LOCK_RETRY_MS = 10;
/** Default bound on waiting for the append lock (see JournalOptions). */
const APPEND_LOCK_TIMEOUT_MS = 5_000;

/** True when a signal-0 probe reaches the pid (EPERM = alive, not ours). */
function isPidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === "EPERM";
  }
}

/** Minimal schema contract (zod-compatible) so the kit stays dependency-light. */
export interface JournalRowSchema<T> {
  safeParse(value: unknown): { success: true; data: T } | { success: false; error?: unknown };
}

export interface JournalOptions<T> {
  filePath: string;
  schema: JournalRowSchema<T>;
  /** Extract the monotonic sequence from a row. */
  getSeq: (row: T) => number;
  /** Extract the stable unique id from a row (dedupe key on read). */
  getId: (row: T) => string;
  /**
   * Max milliseconds to wait for the cross-process append lock before the
   * append fails. Appends normally hold the lock for well under a
   * millisecond, so hitting this means another process is wedged mid-append;
   * failing (callers already tolerate append failures per the self-healing
   * doctrine) beats writing an unserialized — possibly seq-colliding — row.
   */
  appendLockTimeoutMs?: number;
}

export class Journal<T> {
  private readonly filePath: string;
  private readonly lockPath: string;
  private readonly schema: JournalRowSchema<T>;
  private readonly getSeq: (row: T) => number;
  private readonly getId: (row: T) => string;
  private readonly appendLockTimeoutMs: number;
  /** Next sequence to assign; null until the file has been scanned once. */
  private nextSeq: number | null = null;
  /**
   * File size in bytes right after OUR last locked append; null until then.
   * A different size at the next append means another instance or process
   * appended in between, so the cached nextSeq must be re-derived.
   */
  private lastKnownSize: number | null = null;

  /** Serializes appends so seq assignment and tail-healing are race-free. */
  private writeQueue: Promise<unknown> = Promise.resolve();

  constructor(options: JournalOptions<T>) {
    assert(options.filePath.length > 0, "Journal requires a file path");
    this.filePath = options.filePath;
    this.lockPath = `${options.filePath}.lock`;
    this.schema = options.schema;
    this.getSeq = options.getSeq;
    this.getId = options.getId;
    this.appendLockTimeoutMs = options.appendLockTimeoutMs ?? APPEND_LOCK_TIMEOUT_MS;
    assert(this.appendLockTimeoutMs > 0, "Journal appendLockTimeoutMs must be positive");
  }

  /**
   * Append one row built from the next monotonic sequence number. The build
   * result is validated against the schema before hitting disk (crash-fast on
   * programmer error rather than persisting garbage).
   */
  async append(build: (seq: number) => T): Promise<T> {
    const task = this.writeQueue.then(async () => {
      await fs.mkdir(path.dirname(this.filePath), { recursive: true });
      // Cross-process serialization: seq derivation and the write must be one
      // exclusive unit, or a concurrent writer in another process (debug CLI
      // vs live backend) could assign the same sequence number.
      await using _lock = await this.acquireAppendLock();
      const { seq, fileSize } = await this.nextSeqLocked();
      const row = build(seq);
      assert(
        this.getSeq(row) === seq,
        `Journal append: row seq ${this.getSeq(row)} must equal assigned seq ${seq}`
      );
      const parsed = this.schema.safeParse(row);
      assert(
        parsed.success,
        `Journal append: row failed schema validation: ${JSON.stringify(row)}`
      );

      // Heal a torn tail (crash mid-append): start on a fresh line so this row
      // stays parseable even if the previous write was truncated.
      const separator = (await this.hasUnterminatedTail()) ? "\n" : "";
      const line = JSON.stringify(row);
      assert(!line.includes("\n"), "Journal rows must serialize to a single line");
      const payload = `${separator}${line}\n`;
      await fs.appendFile(this.filePath, payload, "utf-8");
      this.nextSeq = seq + 1;
      this.lastKnownSize = fileSize + Buffer.byteLength(payload, "utf-8");
      return row;
    });
    // Keep the queue alive even if this append fails.
    this.writeQueue = task.catch(() => undefined);
    return task;
  }

  /**
   * Read all rows, self-healing as we go:
   * - unparseable / schema-invalid lines are skipped (warn-logged),
   * - duplicate ids are dropped (first occurrence wins),
   * - rows are stable-sorted by seq (ties keep file order).
   */
  async read(): Promise<T[]> {
    let raw: string;
    try {
      raw = await fs.readFile(this.filePath, "utf-8");
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") {
        return [];
      }
      throw error;
    }

    const rows: T[] = [];
    const seenIds = new Set<string>();
    const lines = raw.split("\n");
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue;
      let value: unknown;
      try {
        value = JSON.parse(line);
      } catch {
        log.warn(`Journal: skipping malformed JSON at line ${i + 1} in ${this.filePath}`);
        continue;
      }
      const parsed = this.schema.safeParse(value);
      if (!parsed.success) {
        log.warn(`Journal: skipping schema-invalid row at line ${i + 1} in ${this.filePath}`);
        continue;
      }
      const id = this.getId(parsed.data);
      if (seenIds.has(id)) {
        log.warn(`Journal: dropping duplicate row id '${id}' at line ${i + 1} in ${this.filePath}`);
        continue;
      }
      seenIds.add(id);
      rows.push(parsed.data);
    }

    // Stable sort by seq; JS Array.prototype.sort is stable, so file order is
    // preserved for equal sequence numbers.
    rows.sort((a, b) => this.getSeq(a) - this.getSeq(b));
    return rows;
  }

  /**
   * Derive the next sequence under the append lock. The cached counter is
   * trusted only while the file size still matches what we observed after our
   * own last append; any other size means a foreign writer appended (or the
   * file was replaced) and the counter is re-derived from a full scan. Foreign
   * appends are rare (debug CLI rollbacks), so the rescan cost is incidental.
   */
  private async nextSeqLocked(): Promise<{ seq: number; fileSize: number }> {
    let fileSize = 0;
    try {
      fileSize = (await fs.stat(this.filePath)).size;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
        throw error;
      }
    }
    if (this.nextSeq !== null && this.lastKnownSize === fileSize) {
      return { seq: this.nextSeq, fileSize };
    }
    const rows = await this.read();
    const maxSeq = rows.reduce((max, row) => Math.max(max, this.getSeq(row)), -1);
    return { seq: maxSeq + 1, fileSize };
  }

  /**
   * Acquire the cross-process append lockfile (`<file>.lock`). Lock birth is
   * atomic-with-content: the token (`pid:nonce`) is fully written to a temp
   * file first and hard-linked into place (link fails EEXIST when held), so a
   * reader can never observe a token-less lock. Waiting is a bounded jittered
   * poll — there is no portable cross-process wake primitive available here.
   */
  private async acquireAppendLock(): Promise<AsyncDisposable> {
    const token = `${process.pid}:${crypto.randomBytes(8).toString("hex")}`;
    const tempPath = `${this.lockPath}.tmp-${token.replace(":", "-")}`;
    const deadline = Date.now() + this.appendLockTimeoutMs;
    await fs.writeFile(tempPath, token, "utf-8");
    try {
      for (;;) {
        try {
          await fs.link(tempPath, this.lockPath);
          return { [Symbol.asyncDispose]: () => this.releaseAppendLock(token) };
        } catch (error) {
          if ((error as NodeJS.ErrnoException).code !== "EEXIST") {
            throw error;
          }
        }
        await this.reclaimStaleAppendLock();
        if (Date.now() >= deadline) {
          throw new Error(
            `Journal: timed out acquiring append lock ${this.lockPath} after ${this.appendLockTimeoutMs}ms`
          );
        }
        await new Promise((resolve) =>
          setTimeout(resolve, APPEND_LOCK_RETRY_MS + Math.random() * APPEND_LOCK_RETRY_MS)
        );
      }
    } finally {
      await fs.unlink(tempPath).catch(() => undefined);
    }
  }

  /**
   * Reclaim the append lock if its recorded owner is provably dead (crash
   * remnant). Claim-by-rename makes reclamation atomic: of two concurrent
   * reclaimers only one rename succeeds (the loser gets ENOENT and simply
   * retries acquisition). Reading the claimed file AFTER the rename verifies
   * we claimed the token we judged dead; in the narrow release-then-relock
   * window we could have grabbed a fresh live lock instead, which is restored
   * via link (atomic, loses gracefully to an even newer lock — that victim's
   * release tolerates the loss, see releaseAppendLock).
   */
  private async reclaimStaleAppendLock(): Promise<void> {
    let observed: string;
    try {
      observed = await fs.readFile(this.lockPath, "utf-8");
    } catch {
      return; // Already released or reclaimed; retry acquisition.
    }
    const pid = Number.parseInt(observed.split(":")[0], 10);
    if (!Number.isSafeInteger(pid) || pid <= 0 || isPidAlive(pid)) {
      return;
    }
    const graveyard = `${this.lockPath}.stale-${crypto.randomBytes(4).toString("hex")}`;
    try {
      await fs.rename(this.lockPath, graveyard);
    } catch {
      return; // Another reclaimer won the rename; retry acquisition.
    }
    const claimed = await fs.readFile(graveyard, "utf-8").catch(() => null);
    if (claimed !== null && claimed !== observed) {
      log.warn(`Journal: reclaim raced a fresh append lock on ${this.lockPath}; restoring it`);
      await fs.link(graveyard, this.lockPath).catch(() => undefined);
    }
    await fs.unlink(graveyard).catch(() => undefined);
  }

  /** Release only if we still own the lock (a raced reclaim may have replaced it). */
  private async releaseAppendLock(token: string): Promise<void> {
    try {
      const content = await fs.readFile(this.lockPath, "utf-8");
      if (content !== token) {
        log.warn(`Journal: append lock ${this.lockPath} changed owners before release; leaving it`);
        return;
      }
      await fs.unlink(this.lockPath);
    } catch (error) {
      log.debug(`Journal: failed to release append lock ${this.lockPath}`, { error });
    }
  }

  /** True when the file exists, is non-empty, and does not end with "\n". */
  private async hasUnterminatedTail(): Promise<boolean> {
    let handle: fs.FileHandle;
    try {
      handle = await fs.open(this.filePath, "r");
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") {
        return false;
      }
      throw error;
    }
    try {
      const { size } = await handle.stat();
      if (size === 0) {
        return false;
      }
      const buffer = Buffer.alloc(1);
      await handle.read(buffer, 0, 1, size - 1);
      return buffer.toString("utf-8") !== "\n";
    } finally {
      await handle.close();
    }
  }
}
