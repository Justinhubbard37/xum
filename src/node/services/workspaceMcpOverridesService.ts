import { createHash } from "node:crypto";
import * as path from "path";
import * as jsonc from "jsonc-parser";
import assert from "@/common/utils/assert";
import type { WorkspaceMCPOverrides } from "@/common/types/mcp";
import type { RuntimeConfig } from "@/common/types/runtime";
import type { FrontendWorkspaceMetadata } from "@/common/types/workspace";
import type { Config } from "@/node/config";
import { type createRuntime } from "@/node/runtime/runtimeFactory";
import { createRuntimeForWorkspace } from "@/node/runtime/runtimeHelpers";
import { execBuffered, readFileString, writeFileString } from "@/node/utils/runtime/helpers";
import { hasErrorCode } from "@/node/services/tools/skillFileUtils";
import { log } from "@/node/services/log";
import { getErrorMessage } from "@/common/utils/errors";

const MCP_OVERRIDES_DIR = ".mux";
const MCP_OVERRIDES_JSONC = "mcp.local.jsonc";
const MCP_OVERRIDES_JSON = "mcp.local.json";

const MCP_OVERRIDES_GITIGNORE_PATTERNS = [
  `${MCP_OVERRIDES_DIR}/${MCP_OVERRIDES_JSONC}`,
  `${MCP_OVERRIDES_DIR}/${MCP_OVERRIDES_JSON}`,
];

function joinForRuntime(runtimeConfig: RuntimeConfig | undefined, ...parts: string[]): string {
  assert(parts.length > 0, "joinForRuntime requires at least one path segment");

  // Remote runtimes run inside a POSIX shell (SSH host, Docker container), even if the user is
  // running mux on Windows. Use POSIX joins so we don't accidentally introduce backslashes.
  const usePosix = runtimeConfig?.type === "ssh" || runtimeConfig?.type === "docker";
  return usePosix ? path.posix.join(...parts) : path.join(...parts);
}

function isAbsoluteForRuntime(runtimeConfig: RuntimeConfig | undefined, filePath: string): boolean {
  const usePosix = runtimeConfig?.type === "ssh" || runtimeConfig?.type === "docker";
  return usePosix ? path.posix.isAbsolute(filePath) : path.isAbsolute(filePath);
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((v) => typeof v === "string");
}

function normalizeWorkspaceMcpOverrides(raw: unknown): WorkspaceMCPOverrides {
  if (!raw || typeof raw !== "object") {
    return {};
  }

  const obj = raw as {
    disabledServers?: unknown;
    enabledServers?: unknown;
    toolAllowlist?: unknown;
  };

  const disabledServers = isStringArray(obj.disabledServers)
    ? [...new Set(obj.disabledServers.map((s) => s.trim()).filter(Boolean))]
    : undefined;

  const enabledServers = isStringArray(obj.enabledServers)
    ? [...new Set(obj.enabledServers.map((s) => s.trim()).filter(Boolean))]
    : undefined;

  let toolAllowlist: Record<string, string[]> | undefined;
  if (
    obj.toolAllowlist &&
    typeof obj.toolAllowlist === "object" &&
    !Array.isArray(obj.toolAllowlist)
  ) {
    const next: Record<string, string[]> = {};
    for (const [serverName, value] of Object.entries(
      obj.toolAllowlist as Record<string, unknown>
    )) {
      if (!serverName || typeof serverName !== "string") continue;
      if (!isStringArray(value)) continue;

      // Empty array is meaningful ("expose no tools"), so keep it.
      next[serverName] = [...new Set(value.map((t) => t.trim()).filter((t) => t.length > 0))];
    }

    if (Object.keys(next).length > 0) {
      toolAllowlist = next;
    }
  }

  const normalized: WorkspaceMCPOverrides = {
    disabledServers: disabledServers && disabledServers.length > 0 ? disabledServers : undefined,
    enabledServers: enabledServers && enabledServers.length > 0 ? enabledServers : undefined,
    toolAllowlist,
  };

  // Drop empty object to keep persistence clean.
  if (!normalized.disabledServers && !normalized.enabledServers && !normalized.toolAllowlist) {
    return {};
  }

  return normalized;
}

/**
 * Opaque revision token for optimistic-concurrency saves. Derived from the
 * normalized overrides content, so any successful write (including the Agent
 * Plugin uninstaller pruning `plugin:` keys) changes the revision and stale
 * snapshots held by an open Workspace MCP dialog are rejected instead of
 * silently restoring removed entries.
 */
function computeOverridesRevision(overrides: WorkspaceMCPOverrides): string {
  return createHash("sha256").update(JSON.stringify(overrides)).digest("hex").slice(0, 16);
}

/** Thrown when a save's expectedRevision no longer matches the stored overrides. */
export class WorkspaceMcpOverridesConflictError extends Error {
  constructor() {
    super(
      "Workspace MCP settings changed while this dialog was open. " +
        "Close and reopen it to load the latest values, then reapply your changes."
    );
    this.name = "WorkspaceMcpOverridesConflictError";
  }
}

function isEmptyOverrides(overrides: WorkspaceMCPOverrides): boolean {
  return (
    (!overrides.disabledServers || overrides.disabledServers.length === 0) &&
    (!overrides.enabledServers || overrides.enabledServers.length === 0) &&
    (!overrides.toolAllowlist || Object.keys(overrides.toolAllowlist).length === 0)
  );
}

/** True when the error (or its RuntimeError-wrapped cause) carries the fs code. */
function hasFsCode(error: unknown, code: string): boolean {
  if (hasErrorCode(error, code)) {
    return true;
  }
  const cause = error instanceof Error ? error.cause : undefined;
  return hasErrorCode(cause, code);
}

async function statIsFile(
  runtime: ReturnType<typeof createRuntime>,
  filePath: string,
  mode: "lenient" | "strict"
): Promise<boolean> {
  try {
    const stat = await runtime.stat(filePath);
    return !stat.isDirectory;
  } catch (error) {
    // Strict callers must distinguish "file genuinely absent" (fine: no
    // overrides) from "cannot tell" (EACCES, I/O error): treating the latter
    // as absent would let the plugin uninstaller retire a prune tombstone
    // against a file it never actually read. Strict reads only run against
    // local/worktree runtimes, so node fs error codes are reliable here
    // (RuntimeError wraps them as `cause`).
    if (mode === "strict" && !hasFsCode(error, "ENOENT") && !hasFsCode(error, "ENOTDIR")) {
      throw error;
    }
    return false;
  }
}

export class WorkspaceMcpOverridesService {
  constructor(private readonly config: Config) {
    assert(config, "WorkspaceMcpOverridesService requires a Config instance");
  }

  private async getWorkspaceMetadata(workspaceId: string): Promise<FrontendWorkspaceMetadata> {
    assert(typeof workspaceId === "string", "workspaceId must be a string");
    const trimmed = workspaceId.trim();
    assert(trimmed.length > 0, "workspaceId must not be empty");

    const all = await this.config.getAllWorkspaceMetadata();
    const metadata = all.find((m) => m.id === trimmed);
    if (!metadata) {
      throw new Error(`Workspace metadata not found for ${trimmed}`);
    }

    return metadata;
  }

  private getLegacyOverridesFromConfig(workspaceId: string): WorkspaceMCPOverrides | undefined {
    const config = this.config.loadConfigOrDefault();

    for (const [_projectPath, projectConfig] of config.projects) {
      const workspace = projectConfig.workspaces.find((w) => w.id === workspaceId);
      if (workspace) {
        // NOTE: Legacy storage (PR #1180) wrote overrides into ~/.mux/config.json.
        // We keep reading it here only to migrate into the workspace-local file.
        return workspace.mcp;
      }
    }

    return undefined;
  }

  private async clearLegacyOverridesInConfig(workspaceId: string): Promise<void> {
    await this.config.editConfig((config) => {
      for (const [_projectPath, projectConfig] of config.projects) {
        const workspace = projectConfig.workspaces.find((w) => w.id === workspaceId);
        if (workspace) {
          delete workspace.mcp;
          return config;
        }
      }
      return config;
    });
  }

  private async getRuntimeAndWorkspacePath(workspaceId: string): Promise<{
    metadata: FrontendWorkspaceMetadata;
    runtime: ReturnType<typeof createRuntime>;
    workspacePath: string;
  }> {
    const metadata = await this.getWorkspaceMetadata(workspaceId);

    const runtime = createRuntimeForWorkspace(metadata);

    // In-place workspaces (CLI/benchmarks) store the workspace path directly by setting
    // metadata.projectPath === metadata.name.
    const isInPlace = metadata.projectPath === metadata.name;
    const workspacePath = isInPlace
      ? metadata.projectPath
      : runtime.getWorkspacePath(metadata.projectPath, metadata.name);

    assert(
      typeof workspacePath === "string" && workspacePath.length > 0,
      "workspacePath is required"
    );

    return { metadata, runtime, workspacePath };
  }

  private getOverridesFilePaths(
    workspacePath: string,
    runtimeConfig: RuntimeConfig | undefined
  ): {
    jsoncPath: string;
    jsonPath: string;
  } {
    assert(typeof workspacePath === "string", "workspacePath must be a string");

    return {
      jsoncPath: joinForRuntime(
        runtimeConfig,
        workspacePath,
        MCP_OVERRIDES_DIR,
        MCP_OVERRIDES_JSONC
      ),
      jsonPath: joinForRuntime(runtimeConfig, workspacePath, MCP_OVERRIDES_DIR, MCP_OVERRIDES_JSON),
    };
  }

  private async readOverridesFile(
    runtime: ReturnType<typeof createRuntime>,
    filePath: string,
    mode: "lenient" | "strict"
  ): Promise<unknown> {
    try {
      const raw = await readFileString(runtime, filePath);
      const errors: jsonc.ParseError[] = [];
      const parsed: unknown = jsonc.parse(raw, errors) as unknown;
      if (errors.length > 0) {
        // Strict callers (the plugin uninstaller's override prune) must not
        // see "{}" for a file whose real content is unreadable: retiring a
        // prune tombstone against that empty view would let the stale
        // enabledServers key silently re-enable a reinstalled plugin's
        // server once the file becomes readable again.
        if (mode === "strict") {
          throw new Error(`Workspace MCP overrides file has JSONC parse errors: ${filePath}`);
        }
        log.warn("[MCP] Failed to parse workspace MCP overrides (JSONC parse errors)", {
          filePath,
          errorCount: errors.length,
        });
        return {};
      }
      return parsed;
    } catch (error) {
      if (mode === "strict") {
        throw error;
      }
      // Treat any read failure as "no overrides".
      log.debug("[MCP] Failed to read workspace MCP overrides file", { filePath, error });
      return {};
    }
  }

  private async ensureOverridesDir(
    runtime: ReturnType<typeof createRuntime>,
    workspacePath: string,
    runtimeConfig: RuntimeConfig | undefined
  ): Promise<void> {
    const overridesDirPath = joinForRuntime(runtimeConfig, workspacePath, MCP_OVERRIDES_DIR);

    try {
      await runtime.ensureDir(overridesDirPath);
    } catch (err) {
      const msg = getErrorMessage(err);
      throw new Error(`Failed to create ${MCP_OVERRIDES_DIR} directory: ${msg}`);
    }
  }

  private async ensureOverridesGitignored(
    runtime: ReturnType<typeof createRuntime>,
    workspacePath: string,
    runtimeConfig: RuntimeConfig | undefined
  ): Promise<void> {
    try {
      const isInsideGitResult = await execBuffered(runtime, "git rev-parse --is-inside-work-tree", {
        cwd: workspacePath,
        timeout: 10,
      });
      if (isInsideGitResult.exitCode !== 0 || isInsideGitResult.stdout.trim() !== "true") {
        return;
      }

      const excludePathResult = await execBuffered(
        runtime,
        "git rev-parse --git-path info/exclude",
        {
          cwd: workspacePath,
          timeout: 10,
        }
      );
      if (excludePathResult.exitCode !== 0) {
        return;
      }

      const excludeFilePathRaw = excludePathResult.stdout.trim();
      if (excludeFilePathRaw.length === 0) {
        return;
      }

      const excludeFilePath = isAbsoluteForRuntime(runtimeConfig, excludeFilePathRaw)
        ? excludeFilePathRaw
        : joinForRuntime(runtimeConfig, workspacePath, excludeFilePathRaw);

      let existing = "";
      try {
        existing = await readFileString(runtime, excludeFilePath);
      } catch {
        // Missing exclude file is OK.
      }

      const existingPatterns = new Set(
        existing
          .split(/\r?\n/)
          .map((line) => line.trim())
          .filter((line) => line.length > 0)
      );
      const missingPatterns = MCP_OVERRIDES_GITIGNORE_PATTERNS.filter(
        (pattern) => !existingPatterns.has(pattern)
      );
      if (missingPatterns.length === 0) {
        return;
      }

      const needsNewline = existing.length > 0 && !existing.endsWith("\n");
      const updated = existing + (needsNewline ? "\n" : "") + missingPatterns.join("\n") + "\n";

      await writeFileString(runtime, excludeFilePath, updated);
    } catch (error) {
      // Best-effort only; never fail a workspace operation because git ignore couldn't be updated.
      log.debug("[MCP] Failed to add workspace MCP overrides file to git exclude", {
        workspacePath,
        error,
      });
    }
  }

  private async removeOverridesFile(
    runtime: ReturnType<typeof createRuntime>,
    workspacePath: string
  ): Promise<void> {
    // Remove both file names so we never leave conflicting sources behind.
    // The exit code MUST be checked: callers (e.g. the Agent Plugin
    // uninstaller retiring override-prune tombstones) rely on
    // setOverridesForWorkspace rejecting when clearing overrides failed —
    // a swallowed `rm` failure would leave a stale enabledServers key that
    // a plugin reinstall could silently reactivate.
    const result = await execBuffered(
      runtime,
      `rm -f "${MCP_OVERRIDES_DIR}/${MCP_OVERRIDES_JSONC}" "${MCP_OVERRIDES_DIR}/${MCP_OVERRIDES_JSON}"`,
      {
        cwd: workspacePath,
        timeout: 10,
      }
    );
    if (result.exitCode !== 0) {
      throw new Error(
        `Failed to remove workspace MCP overrides file: ${result.stderr.trim() || `rm exited with code ${result.exitCode}`}`
      );
    }
  }

  /**
   * Read workspace MCP overrides from <workspace>/.mux/mcp.local.jsonc.
   *
   * If the file doesn't exist, we fall back to legacy overrides stored in ~/.mux/config.json
   * and migrate them into the workspace-local file.
   *
   * The returned revision is an opaque token for setOverridesForWorkspace's
   * expectedRevision check.
   */
  async getOverridesForWorkspace(
    workspaceId: string,
    options?: { mode?: "lenient" | "strict" }
  ): Promise<{ overrides: WorkspaceMCPOverrides; revision: string }> {
    const overrides = await this.loadOverrides(workspaceId, options?.mode ?? "lenient");
    return { overrides, revision: computeOverridesRevision(overrides) };
  }

  private async loadOverrides(
    workspaceId: string,
    mode: "lenient" | "strict" = "lenient"
  ): Promise<WorkspaceMCPOverrides> {
    const { metadata, runtime, workspacePath } = await this.getRuntimeAndWorkspacePath(workspaceId);
    const { jsoncPath, jsonPath } = this.getOverridesFilePaths(
      workspacePath,
      metadata.runtimeConfig
    );

    // Prefer JSONC, then JSON.
    const jsoncExists = await statIsFile(runtime, jsoncPath, mode);
    if (jsoncExists) {
      const parsed = await this.readOverridesFile(runtime, jsoncPath, mode);
      return normalizeWorkspaceMcpOverrides(parsed);
    }

    const jsonExists = await statIsFile(runtime, jsonPath, mode);
    if (jsonExists) {
      const parsed = await this.readOverridesFile(runtime, jsonPath, mode);
      return normalizeWorkspaceMcpOverrides(parsed);
    }

    // No workspace-local file => try migrating legacy config.json storage.
    const legacy = this.getLegacyOverridesFromConfig(workspaceId);
    if (!legacy || isEmptyOverrides(legacy)) {
      return {};
    }

    const normalizedLegacy = normalizeWorkspaceMcpOverrides(legacy);
    if (isEmptyOverrides(normalizedLegacy)) {
      return {};
    }

    try {
      await this.ensureOverridesDir(runtime, workspacePath, metadata.runtimeConfig);
      await writeFileString(runtime, jsoncPath, JSON.stringify(normalizedLegacy, null, 2) + "\n");
      await this.ensureOverridesGitignored(runtime, workspacePath, metadata.runtimeConfig);
      await this.clearLegacyOverridesInConfig(workspaceId);
      log.info("[MCP] Migrated workspace MCP overrides from config.json", {
        workspaceId,
        filePath: jsoncPath,
      });
    } catch (error) {
      // Migration is best-effort; if it fails, still honor legacy overrides.
      log.warn("[MCP] Failed to migrate workspace MCP overrides; using legacy config.json values", {
        workspaceId,
        error,
      });
    }

    return normalizedLegacy;
  }

  /**
   * All writes flow through this queue so the expectedRevision check-and-set
   * in setOverridesForWorkspace is atomic within the main process (the only
   * writer of these files).
   */
  private writeQueue: Promise<unknown> = Promise.resolve();

  private runExclusive<T>(fn: () => Promise<T>): Promise<T> {
    const run = () => fn();
    const next = this.writeQueue.then(run, run);
    this.writeQueue = next.catch(() => undefined);
    return next;
  }

  /**
   * Persist workspace MCP overrides to <workspace>/.mux/mcp.local.jsonc.
   *
   * Empty overrides remove the workspace-local file.
   *
   * When options.expectedRevision is provided, the write is rejected with
   * WorkspaceMcpOverridesConflictError if the stored overrides changed since
   * that revision was read — a stale Workspace MCP dialog snapshot must not
   * silently restore entries removed by a concurrent writer (e.g. the Agent
   * Plugin uninstaller pruning `plugin:<instanceId>:` keys).
   */
  async setOverridesForWorkspace(
    workspaceId: string,
    overrides: WorkspaceMCPOverrides,
    options?: { expectedRevision?: string }
  ): Promise<void> {
    assert(overrides && typeof overrides === "object", "overrides must be an object");

    return this.runExclusive(async () => {
      if (options?.expectedRevision !== undefined) {
        const current = await this.loadOverrides(workspaceId);
        if (computeOverridesRevision(current) !== options.expectedRevision) {
          throw new WorkspaceMcpOverridesConflictError();
        }
      }

      const { metadata, runtime, workspacePath } =
        await this.getRuntimeAndWorkspacePath(workspaceId);
      const { jsoncPath } = this.getOverridesFilePaths(workspacePath, metadata.runtimeConfig);

      const normalized = normalizeWorkspaceMcpOverrides(overrides);

      // Always clear any legacy storage so we converge on the workspace-local file.
      await this.clearLegacyOverridesInConfig(workspaceId);

      if (isEmptyOverrides(normalized)) {
        await this.removeOverridesFile(runtime, workspacePath);
        return;
      }

      await this.ensureOverridesDir(runtime, workspacePath, metadata.runtimeConfig);
      await writeFileString(runtime, jsoncPath, JSON.stringify(normalized, null, 2) + "\n");
      await this.ensureOverridesGitignored(runtime, workspacePath, metadata.runtimeConfig);
    });
  }
}
