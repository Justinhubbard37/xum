import { describe, expect, test } from "bun:test";
import { spawnSync } from "node:child_process";
import * as fs from "fs/promises";
import * as path from "path";
import { DisposableTempDir } from "@/node/services/tempDir";
import { acquireProcessFileLock, getProcessBirth } from "./fileLock";

async function lockExists(lockPath: string): Promise<boolean> {
  return fs.access(lockPath).then(
    () => true,
    () => false
  );
}

describe("acquireProcessFileLock", () => {
  test("acquire/release round-trip installs and removes the lockfile", async () => {
    using tmp = new DisposableTempDir("file-lock-test");
    const lockPath = path.join(tmp.path, "x.lock");
    {
      await using _lock = await acquireProcessFileLock({ lockPath, timeoutMs: 500, label: "test" });
      expect(await lockExists(lockPath)).toBe(true);
    }
    expect(await lockExists(lockPath)).toBe(false);
  });

  test("reclaims a lock whose recorded owner pid is provably dead", async () => {
    using tmp = new DisposableTempDir("file-lock-test");
    const lockPath = path.join(tmp.path, "x.lock");
    const child = spawnSync(process.execPath, ["--version"]);
    expect(child.pid).toBeGreaterThan(0);
    await fs.writeFile(lockPath, `${child.pid}:deadbeef`, { encoding: "utf-8", flag: "wx" });

    await using _lock = await acquireProcessFileLock({ lockPath, timeoutMs: 2_000, label: "test" });
    expect(await lockExists(lockPath)).toBe(true);
  });

  test("reclaims a live-pid lock whose recorded process birth does not match (PID reuse)", async () => {
    using tmp = new DisposableTempDir("file-lock-test");
    const lockPath = path.join(tmp.path, "x.lock");
    // Our own pid is definitely alive, but the recorded birth identity is a
    // different (crashed) process's: the OS handed its PID to us. Without
    // birth verification this lock is judged live forever.
    const bogusBirth = Buffer.from("crashed-process-birth").toString("hex");
    await fs.writeFile(lockPath, `${process.pid}:cafe:${bogusBirth}`, {
      encoding: "utf-8",
      flag: "wx",
    });

    await using _lock = await acquireProcessFileLock({ lockPath, timeoutMs: 2_000, label: "test" });
    expect(await lockExists(lockPath)).toBe(true);
  });

  test("reclaims an undetermined-birth live-pid lock once its lease expires", async () => {
    using tmp = new DisposableTempDir("file-lock-test");
    const lockPath = path.join(tmp.path, "x.lock");
    // Old-format token (no birth recorded): staleness cannot be proven via
    // birth, so the bounded mtime lease governs. An hours-old lock cannot be
    // a legitimate hold (all holds are ms-to-seconds).
    await fs.writeFile(lockPath, `${process.pid}:cafe`, { encoding: "utf-8", flag: "wx" });
    const ancient = new Date(Date.now() - 60 * 60 * 1000);
    await fs.utimes(lockPath, ancient, ancient);

    await using _lock = await acquireProcessFileLock({ lockPath, timeoutMs: 2_000, label: "test" });
    expect(await lockExists(lockPath)).toBe(true);
  });

  test("retains a fresh undetermined-birth live-pid lock (lease not expired)", async () => {
    using tmp = new DisposableTempDir("file-lock-test");
    const lockPath = path.join(tmp.path, "x.lock");
    await fs.writeFile(lockPath, `${process.pid}:cafe`, { encoding: "utf-8", flag: "wx" });
    try {
      await acquireProcessFileLock({ lockPath, timeoutMs: 150, label: "test" });
      expect.unreachable("a fresh live-pid lock must not be reclaimed");
    } catch (error) {
      expect(String(error)).toContain("Timed out");
    }
  });

  test("never lease-breaks a verified-live holder, no matter how old the lock is", async () => {
    using tmp = new DisposableTempDir("file-lock-test");
    const lockPath = path.join(tmp.path, "x.lock");
    const realBirth = getProcessBirth(process.pid);
    if (realBirth === null) {
      // Platform without a birth probe: the lease governs instead; the
      // "retains a fresh lock" test covers the conservative path.
      return;
    }
    // Same pid AND same birth = provably the original holder, still alive: a
    // wedged-but-live holder must never be displaced (double-entry risk),
    // even past the lease age.
    await fs.writeFile(lockPath, `${process.pid}:cafe:${Buffer.from(realBirth).toString("hex")}`, {
      encoding: "utf-8",
      flag: "wx",
    });
    const ancient = new Date(Date.now() - 60 * 60 * 1000);
    await fs.utimes(lockPath, ancient, ancient);
    try {
      await acquireProcessFileLock({ lockPath, timeoutMs: 150, label: "test" });
      expect.unreachable("a verified-live holder must never be reclaimed");
    } catch (error) {
      expect(String(error)).toContain("Timed out");
    }
  });
});
