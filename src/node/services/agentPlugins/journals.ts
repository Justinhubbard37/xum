/**
 * Shared crash-recovery journal vocabulary for managed Agent Plugin installs.
 *
 * AgentPluginInstallService writes a journal file into the staging root
 * (`<muxHome>/plugin-staging`, a SIBLING of the managed `plugins` container)
 * before every directory move of an install/update/uninstall, and consumes it
 * only when the mutation's cleanup fully lands. A surviving journal therefore
 * means the managed container may hold unreconciled state (an orphaned
 * promotion, a half-swapped update, a staged-away uninstall).
 *
 * This lives outside installService.ts so discovery.ts can derive
 * journal-based suppression for processes that never construct the install
 * service (headless `mux workflow` resolving plugin:// scripts) without an
 * import cycle: installService imports discovery for container scans.
 */
import * as fsPromises from "node:fs/promises";
import * as path from "node:path";

/** Staging dir name under the mux home dir — NOT under ~/.mux/plugins, which discovery scans. */
export const STAGING_DIR_NAME = "plugin-staging";

export const PROMOTION_JOURNAL_PREFIX = "promotion-";
export const UPDATE_JOURNAL_PREFIX = "update-";
export const UNINSTALL_JOURNAL_PREFIX = "uninstall-";

export const JOURNAL_PREFIXES = [
  PROMOTION_JOURNAL_PREFIX,
  UPDATE_JOURNAL_PREFIX,
  UNINSTALL_JOURNAL_PREFIX,
] as const;

export function isJournalName(entry: string): boolean {
  return JOURNAL_PREFIXES.some((prefix) => entry.startsWith(prefix));
}

/**
 * Whether the staging root SIBLING of the given container holds any recovery
 * journals. Fail-closed: an unreadable staging root (non-ENOENT) reports
 * true, because "cannot tell" must not release discovery over a container
 * that may hold unreconciled trees.
 */
export async function containerHasUnreconciledJournals(containerPath: string): Promise<boolean> {
  const stagingRoot = path.join(path.dirname(containerPath), STAGING_DIR_NAME);
  try {
    return (await fsPromises.readdir(stagingRoot)).some(
      (entry) => isJournalName(entry) && entry.endsWith(".json")
    );
  } catch (error) {
    return !(error instanceof Error && "code" in error && error.code === "ENOENT");
  }
}
