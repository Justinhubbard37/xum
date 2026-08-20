import * as fsPromises from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";

import writeFileAtomic from "write-file-atomic";

import {
  AgentPluginInstallEntrySchema,
  type AgentPluginGitSource,
  type AgentPluginInstallEntry,
} from "@/common/config/schemas/agentPluginInstalls";
import { isValidAgentPluginName } from "@/common/utils/agentPluginName";
import type {
  AgentPluginInstallPreview,
  AgentPluginListItem,
  AgentPluginManifestSummary,
  AgentPluginPreviewHook,
  AgentPluginPreviewMcpServer,
  AgentPluginPreviewSkill,
  AgentPluginUpdateCheck,
} from "@/common/orpc/schemas/agentPlugins";
import { resolvePluginHookGrants } from "@/node/services/agentPlugins/hookSandbox";
import assert from "@/common/utils/assert";
import { getErrorMessage } from "@/common/utils/errors";
import type { Config } from "@/node/config";
import { parseSkillMarkdown } from "@/node/services/agentSkills/parseSkillMarkdown";
import { log } from "@/node/services/log";
import type { MCPServerManager } from "@/node/services/mcpServerManager";
import {
  WorkspaceMcpOverridesConflictError,
  type WorkspaceMcpOverridesService,
} from "@/node/services/workspaceMcpOverridesService";
import { ensurePathContained, hasErrorCode } from "@/node/services/tools/skillFileUtils";
import { execFileAsync } from "@/node/utils/disposableExec";
import {
  discoverAgentPluginAt,
  discoverAgentPlugins,
  type AgentPluginContainer,
  type AgentPluginInfo,
} from "./discovery";
import type { AgentPluginManifest } from "./manifest";
import {
  buildPluginServerKey,
  computePluginInstanceId,
  getPluginDataPath,
  loadPluginMcpServers,
} from "./mcpConfig";
import { isFullCommitSha, parseAgentPluginSourceInput } from "./sourceInput";

/**
 * Managed Agent Plugin installer (agent-plugins experiment; global scope only).
 *
 * Flow: parse input → shallow clone to a staging dir under ~/.mux →
 * validate the STAGED clone with the same manifest/component discovery used
 * at runtime → return a consent preview → on confirm, re-clone the exact SHA,
 * promote into ~/.mux/plugins/<name>, and record a registry entry
 * ({source, ref, lockedSha}) in ~/.mux/plugins.json.
 *
 * The registry is a standalone file (NOT a config.json section): older builds
 * rebuild config.json from known fields on save, so a downgrade would drop an
 * embedded registry — and owning the file lets writes THROW on failure so
 * install/update/uninstall can roll back instead of silently succeeding with
 * an unpersisted registry.
 *
 * Invariants:
 * - The installer NEVER writes into a project checkout (v1 is global-only).
 * - `lockedSha` is what runs; branches are only a tracking channel for the
 *   update badge. Nothing auto-applies.
 * - Update = temp clone + wholesale directory swap (rename-old → promote-new
 *   → delete-old), never in-place `git pull` — local edits to a managed
 *   plugin dir are discarded on update.
 * - Applying an update or uninstalling recycles that plugin's running MCP
 *   servers: content can change behind an unchanged stdio command line, so
 *   the config-signature check cannot notice (correctness, not polish).
 * - Failure paths must leave no partial state: staging dirs are cleaned up,
 *   and promote + registry-write failures roll back.
 */

/** Registry file name under the mux home dir. */
const REGISTRY_FILE_NAME = "plugins.json";

/** Preview/staging clones live here — NOT under ~/.mux/plugins, which discovery scans. */
const STAGING_DIR_NAME = "plugin-staging";

/** Staging dirs left behind by crashes are reclaimed after this age. */
const STALE_STAGING_MAX_AGE_MS = 60 * 60 * 1000;

const LS_REMOTE_TIMEOUT_MS = 30_000;
const CLONE_TIMEOUT_MS = 120_000;

/** Result of resolving a user-supplied ref against the remote. */
interface ResolvedRemoteRef {
  ref: string;
  refType: "branch" | "tag" | "commit";
  /** Peeled commit SHA for branch/tag; the ref itself for commit. */
  sha: string;
}

function gitEnv(): Record<string, string> {
  // Fail fast instead of hanging on credential prompts: installs run from the
  // UI with no terminal attached (acceptance: "private repo without auth" must
  // fail cleanly).
  const env: Record<string, string> = { GIT_TERMINAL_PROMPT: "0" };
  if (process.env.GIT_SSH_COMMAND === undefined) {
    env.GIT_SSH_COMMAND = "ssh -oBatchMode=yes";
  }
  return env;
}

async function runGit(args: string[], opts?: { timeoutMs?: number }): Promise<string> {
  using proc = execFileAsync("git", args, {
    env: gitEnv(),
    timeoutMs: opts?.timeoutMs ?? CLONE_TIMEOUT_MS,
  });
  const { stdout } = await proc.result;
  return stdout;
}

async function pathExists(candidate: string): Promise<boolean> {
  try {
    await fsPromises.access(candidate);
    return true;
  } catch {
    return false;
  }
}

function shortenHome(absPath: string): string {
  const home = os.homedir();
  if (absPath === home) {
    return "~";
  }
  return absPath.startsWith(home + path.sep) ? `~${absPath.slice(home.length)}` : absPath;
}

function manifestSummary(manifest: AgentPluginManifest): AgentPluginManifestSummary {
  return {
    name: manifest.name,
    ...(manifest.version !== undefined ? { version: manifest.version } : {}),
    ...(manifest.description !== undefined ? { description: manifest.description } : {}),
    ...(manifest.author?.name !== undefined ? { authorName: manifest.author.name } : {}),
    ...(manifest.homepage !== undefined ? { homepage: manifest.homepage } : {}),
    ...(manifest.repository !== undefined ? { repository: manifest.repository } : {}),
    ...(manifest.license !== undefined ? { license: manifest.license } : {}),
  };
}

export class AgentPluginInstallService {
  private readonly containerDir: string;
  private readonly stagingRoot: string;
  private readonly registryFile: string;
  /** Serializes mutations (install/update/uninstall) so directory swaps and registry writes cannot interleave. */
  private mutationQueue: Promise<unknown> = Promise.resolve();

  constructor(
    private readonly config: Config,
    private readonly deps: {
      isEnabled: () => boolean;
      /** Recycles running MCP servers whose config key starts with the given prefix. */
      mcpServerManager?: MCPServerManager;
      /** Used to prune plugin server keys from per-workspace overrides on uninstall. */
      workspaceMcpOverridesService?: WorkspaceMcpOverridesService;
    }
  ) {
    assert(path.isAbsolute(config.rootDir), "AgentPluginInstallService: rootDir must be absolute");
    this.containerDir = path.join(config.rootDir, "plugins");
    this.stagingRoot = path.join(config.rootDir, STAGING_DIR_NAME);
    this.registryFile = path.join(config.rootDir, REGISTRY_FILE_NAME);
  }

  // ---------------------------------------------------------------------
  // Registry persistence (~/.mux/plugins.json)
  // ---------------------------------------------------------------------

  /**
   * The registry document as stored on disk: the top-level ENVELOPE (an
   * object that must hold a `plugins` array, and may hold future top-level
   * fields like a registry version) plus the raw entry list. Mutations
   * operate on the raw entries (matching by their `name` property) and write
   * the envelope back with only `plugins` replaced, so both unknown entry
   * fields and unknown top-level fields written by newer builds survive an
   * install/update/uninstall on this build (upgrade↔downgrade stays
   * lossless).
   *
   * A missing file is an empty registry; corrupted content — unparseable
   * JSON or a structurally invalid envelope like `{}` / `{"plugins": null}`
   * — is not. Reads ("lenient") degrade corruption to an empty list so the
   * section still renders (dirs show as unmanaged), but mutations ("strict")
   * must refuse: treating a corrupted file as empty would let the next
   * install rewrite it with a single entry, permanently orphaning every
   * previously managed install.
   */
  private async readRegistryDocument(mode: "lenient" | "strict"): Promise<{
    envelope: Record<string, unknown>;
    rawEntries: unknown[];
  }> {
    const corrupted = (detail: string): never => {
      throw new Error(
        `The plugin registry (${shortenHome(this.registryFile)}) is corrupted: ${detail}. Repair or remove the file, then retry.`
      );
    };

    let raw: string;
    try {
      raw = await fsPromises.readFile(this.registryFile, "utf8");
    } catch (error) {
      // Only a MISSING file is an empty registry. Any other read failure
      // (e.g. an unreadable mode-000 file in a writable ~/.mux) must block
      // mutations: the atomic write replaces the file wholesale, so treating
      // "unreadable" as "empty" would erase every existing entry.
      if (hasErrorCode(error, "ENOENT")) {
        return { envelope: {}, rawEntries: [] };
      }
      if (mode === "strict") {
        corrupted(`it cannot be read (${getErrorMessage(error)})`);
      }
      log.warn("Ignoring unreadable plugin registry file", {
        file: this.registryFile,
        error: getErrorMessage(error),
      });
      return { envelope: {}, rawEntries: [] };
    }

    let parsedJson: unknown;
    try {
      parsedJson = JSON.parse(raw);
    } catch (error) {
      if (mode === "strict") {
        corrupted(`it cannot be parsed (${getErrorMessage(error)})`);
      }
      log.warn("Ignoring unparseable plugin registry file", {
        file: this.registryFile,
        error: getErrorMessage(error),
      });
      return { envelope: {}, rawEntries: [] };
    }

    if (
      typeof parsedJson !== "object" ||
      parsedJson === null ||
      Array.isArray(parsedJson) ||
      !Array.isArray((parsedJson as { plugins?: unknown }).plugins)
    ) {
      if (mode === "strict") {
        corrupted("expected an object with a 'plugins' array");
      }
      log.warn("Ignoring structurally invalid plugin registry file", {
        file: this.registryFile,
      });
      return { envelope: {}, rawEntries: [] };
    }

    return {
      envelope: parsedJson as Record<string, unknown>,
      rawEntries: (parsedJson as { plugins: unknown[] }).plugins,
    };
  }

  /**
   * Lenient-on-read: entries this build does not recognize degrade to
   * "unmanaged dirs" rather than errors (discovery stays the source of truth
   * for what loads; the registry only annotates) — but they stay in the raw
   * file. Name validation in the schema doubles as a filesystem-safety gate:
   * a traversal name like `..` must never reach targetPathFor.
   */
  private parseRegistryEntries(rawEntries: unknown[]): AgentPluginInstallEntry[] {
    const entries: AgentPluginInstallEntry[] = [];
    for (const rawEntry of rawEntries) {
      const parsed = AgentPluginInstallEntrySchema.safeParse(rawEntry);
      if (parsed.success) {
        entries.push(parsed.data);
      } else {
        log.debug("Skipping unrecognized managed plugin registry entry (preserved on disk)", {
          entry: rawEntry,
          error: parsed.error.message,
        });
      }
    }
    return entries;
  }

  private async readRegistry(mode: "lenient" | "strict"): Promise<AgentPluginInstallEntry[]> {
    return this.parseRegistryEntries((await this.readRegistryDocument(mode)).rawEntries);
  }

  /** `name` of a raw registry entry, for identity matching during raw rewrites. */
  private rawEntryName(rawEntry: unknown): string | undefined {
    if (typeof rawEntry !== "object" || rawEntry === null) {
      return undefined;
    }
    const name = (rawEntry as { name?: unknown }).name;
    return typeof name === "string" ? name : undefined;
  }

  /**
   * Atomic write that THROWS on failure (unlike Config.saveConfig's
   * log-and-swallow) so callers can roll back filesystem changes instead of
   * reporting success with an unpersisted registry. Takes the RAW envelope
   * and entry list so unrecognized top-level fields and entries are written
   * back verbatim (only `plugins` is replaced).
   */
  private async writeRegistry(
    envelope: Record<string, unknown>,
    rawEntries: unknown[]
  ): Promise<void> {
    await writeFileAtomic(
      this.registryFile,
      JSON.stringify({ ...envelope, plugins: rawEntries }, null, 2),
      "utf-8"
    );
  }

  private assertEnabled(): void {
    if (!this.deps.isEnabled()) {
      throw new Error("Agent Plugins experiment is not enabled.");
    }
  }

  private runExclusive<T>(fn: () => Promise<T>): Promise<T> {
    const run = this.mutationQueue.then(fn, fn);
    this.mutationQueue = run.catch(() => undefined);
    return run;
  }

  /**
   * Lexical install location — the identity `computePluginInstanceId` hashes
   * for global plugins. The name grammar excludes `.`/`..`/separators, so a
   * malformed registry entry can never resolve outside the container (this
   * path is deleted recursively on uninstall).
   */
  private targetPathFor(name: string): string {
    assert(isValidAgentPluginName(name), `invalid plugin name: ${JSON.stringify(name)}`);
    const target = path.join(this.containerDir, name);
    assert(
      path.dirname(target) === this.containerDir,
      "targetPathFor: resolved path must be an immediate child of the container"
    );
    return target;
  }

  private instanceIdFor(name: string): string {
    return computePluginInstanceId(this.targetPathFor(name));
  }

  // ---------------------------------------------------------------------
  // Staging helpers
  // ---------------------------------------------------------------------

  /**
   * Staging lives under ~/.mux (same filesystem as the container) so promote
   * is a plain rename, and outside ~/.mux/plugins so a staged clone can never
   * be discovered as an installed plugin.
   */
  private async createStagingDir(): Promise<string> {
    await fsPromises.mkdir(this.stagingRoot, { recursive: true });
    await this.purgeStaleStaging();
    return fsPromises.mkdtemp(path.join(this.stagingRoot, "stage-"));
  }

  /** Best-effort reclaim of staging dirs orphaned by crashes. */
  private async purgeStaleStaging(): Promise<void> {
    try {
      const now = Date.now();
      for (const entry of await fsPromises.readdir(this.stagingRoot)) {
        const entryPath = path.join(this.stagingRoot, entry);
        try {
          const stat = await fsPromises.stat(entryPath);
          if (now - stat.mtimeMs > STALE_STAGING_MAX_AGE_MS) {
            await fsPromises.rm(entryPath, { recursive: true, force: true });
          }
        } catch {
          // Entry vanished or is unreadable — skip.
        }
      }
    } catch {
      // Missing staging root is fine.
    }
  }

  private async removeDir(dirPath: string): Promise<void> {
    await fsPromises.rm(dirPath, { recursive: true, force: true });
  }

  // ---------------------------------------------------------------------
  // Git plumbing
  // ---------------------------------------------------------------------

  /** Resolve what a preview/install/update should check out, via `git ls-remote` (no fetch). */
  private async resolveRemoteRef(url: string, ref: string | undefined): Promise<ResolvedRemoteRef> {
    if (ref !== undefined && isFullCommitSha(ref)) {
      return { ref: ref.toLowerCase(), refType: "commit", sha: ref.toLowerCase() };
    }
    if (ref === undefined) {
      // Remote default branch: `ls-remote --symref <url> HEAD` prints
      //   ref: refs/heads/<default>\tHEAD
      //   <sha>\tHEAD
      const output = await this.lsRemote(url, ["--symref", url, "HEAD"]);
      const symrefMatch = /^ref:\s+refs\/heads\/(\S+)\s+HEAD$/m.exec(output);
      const shaMatch = /^([0-9a-f]{40})\s+HEAD$/m.exec(output);
      if (!symrefMatch || !shaMatch) {
        throw new Error(`Could not determine the default branch of ${url}.`);
      }
      return { ref: symrefMatch[1], refType: "branch", sha: shaMatch[1] };
    }

    if (/^[0-9a-f]{7,39}$/i.test(ref)) {
      // A short SHA can't be fetched shallowly and can't be resolved by ls-remote.
      throw new Error(
        `'${ref}' looks like an abbreviated commit SHA. Use the full 40-character SHA, a branch, or a tag.`
      );
    }

    const output = await this.lsRemote(url, [
      url,
      `refs/heads/${ref}`,
      `refs/tags/${ref}`,
      `refs/tags/${ref}^{}`,
    ]);
    const lines = output
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.length > 0);
    let branchSha: string | undefined;
    let tagSha: string | undefined;
    let peeledTagSha: string | undefined;
    for (const line of lines) {
      const [sha, refName] = line.split(/\s+/);
      if (!sha || !refName) continue;
      if (refName === `refs/heads/${ref}`) branchSha = sha;
      else if (refName === `refs/tags/${ref}^{}`) peeledTagSha = sha;
      else if (refName === `refs/tags/${ref}`) tagSha = sha;
    }

    if (branchSha !== undefined) {
      return { ref, refType: "branch", sha: branchSha };
    }
    // Annotated tags list both the tag object and the peeled commit (^{});
    // lockedSha must be the commit so it can be compared against `rev-parse HEAD`.
    const resolvedTagSha = peeledTagSha ?? tagSha;
    if (resolvedTagSha !== undefined) {
      return { ref, refType: "tag", sha: resolvedTagSha };
    }
    throw new Error(`Ref '${ref}' was not found on the remote (no matching branch or tag).`);
  }

  private async lsRemote(url: string, args: string[]): Promise<string> {
    try {
      return await runGit(["ls-remote", ...args], { timeoutMs: LS_REMOTE_TIMEOUT_MS });
    } catch (error) {
      throw new Error(`Could not reach ${url}: ${getErrorMessage(error)}`);
    }
  }

  /** Shallow-clone `resolved` into a fresh staging dir; returns { dir, sha } with sha = HEAD. */
  private async cloneResolved(
    url: string,
    resolved: ResolvedRemoteRef
  ): Promise<{ dir: string; sha: string }> {
    const dir = await this.createStagingDir();
    try {
      if (resolved.refType === "commit") {
        await this.fetchExactSha(url, resolved.sha, dir);
      } else {
        await runGit([
          "clone",
          "--depth",
          "1",
          "--single-branch",
          "--branch",
          resolved.ref,
          "-c",
          "advice.detachedHead=false",
          url,
          dir,
        ]);
      }
      const sha = (await runGit(["-C", dir, "rev-parse", "HEAD"])).trim();
      assert(isFullCommitSha(sha), "cloneResolved: rev-parse HEAD must be a full SHA");
      return { dir, sha };
    } catch (error) {
      await this.removeDir(dir);
      throw new Error(`Failed to clone ${url}: ${getErrorMessage(error)}`);
    }
  }

  /**
   * Clone exactly `sha` (what the user consented to). Prefers a direct SHA
   * fetch (GitHub allows it); falls back to cloning the tracking ref and
   * verifying HEAD still matches, so a remote that moved between preview and
   * install fails loudly instead of installing unreviewed content.
   */
  private async cloneExactSha(source: AgentPluginGitSource, sha: string): Promise<string> {
    const dir = await this.createStagingDir();
    try {
      try {
        await this.fetchExactSha(source.url, sha, dir);
      } catch {
        if (source.refType === "commit") {
          throw new Error(`Could not fetch commit ${sha} from ${source.url}.`);
        }
        // fetchExactSha left an initialized repo behind; git clone refuses a
        // non-empty destination, so reset the staging dir before falling back.
        await this.removeDir(dir);
        await fsPromises.mkdir(dir, { recursive: true });
        await runGit([
          "clone",
          "--depth",
          "1",
          "--single-branch",
          "--branch",
          source.ref,
          "-c",
          "advice.detachedHead=false",
          source.url,
          dir,
        ]);
      }
      const head = (await runGit(["-C", dir, "rev-parse", "HEAD"])).trim();
      if (head !== sha) {
        throw new Error(
          `The remote moved since the preview (expected ${sha.slice(0, 12)}, got ${head.slice(0, 12)}). Run the preview again.`
        );
      }
      return dir;
    } catch (error) {
      await this.removeDir(dir);
      throw error instanceof Error ? error : new Error(getErrorMessage(error));
    }
  }

  private async fetchExactSha(url: string, sha: string, dir: string): Promise<void> {
    await runGit(["init", "--quiet", dir]);
    await runGit(["-C", dir, "remote", "add", "origin", url]);
    await runGit(["-C", dir, "fetch", "--depth", "1", "origin", sha]);
    await runGit([
      "-C",
      dir,
      "-c",
      "advice.detachedHead=false",
      "checkout",
      "--quiet",
      "FETCH_HEAD",
    ]);
  }

  // ---------------------------------------------------------------------
  // Staged-clone validation + preview assembly
  // ---------------------------------------------------------------------

  /**
   * Run the exact runtime validation (manifest + component discovery) against
   * a staged clone. Throws user-facing errors for non-plugins, including a
   * clear message for Claude Code plugin/marketplace repos (explicit non-goal).
   */
  private async validateStagedClone(stagedDir: string): Promise<{
    plugin: AgentPluginInfo;
    warnings: string[];
  }> {
    const hasManifest = await pathExists(path.join(stagedDir, "plugin.json"));
    if (!hasManifest) {
      if (
        (await pathExists(path.join(stagedDir, ".claude-plugin", "plugin.json"))) ||
        (await pathExists(path.join(stagedDir, ".claude-plugin", "marketplace.json")))
      ) {
        throw new Error(
          "This repository is a Claude Code plugin or marketplace (found .claude-plugin/). Mux implements the vendor-neutral Agent Plugins 1.0.0 format and cannot install Claude Code collections."
        );
      }
      throw new Error(
        "No plugin.json found at the repository root. The repo is not an Agent Plugin — if the plugin lives in a subdirectory, monorepo subpath installs land in v2."
      );
    }

    const { plugin, diagnostics } = await discoverAgentPluginAt({
      pluginDir: stagedDir,
      scope: "global",
    });
    if (!plugin) {
      const reasons = diagnostics.map((d) => d.message);
      throw new Error(
        reasons.length > 0 ? `Invalid plugin: ${reasons.join("; ")}` : "Invalid plugin manifest."
      );
    }
    return { plugin, warnings: diagnostics.map((d) => d.message) };
  }

  /**
   * Executable hooks.js disclosure for the consent preview: hooks load
   * automatically before request assembly and can observe/rewrite/block tool
   * calls, so installing one without disclosure would consent to less than
   * what activates. toolGrants mirrors resolvePluginHookGrants — the exact
   * grants the runtime will honor.
   */
  private collectHook(
    plugin: Pick<AgentPluginInfo, "rootPath" | "hooksPath" | "manifest">
  ): AgentPluginPreviewHook | undefined {
    if (plugin.hooksPath === undefined) {
      return undefined;
    }
    const grants = resolvePluginHookGrants(plugin.manifest);
    assert(grants.bridgeTools.allow !== "all", "plugin hook grants must enumerate tools");
    return {
      path: path.relative(plugin.rootPath, plugin.hooksPath),
      toolGrants: [...grants.bridgeTools.allow],
    };
  }

  private async collectSkills(
    plugin: Pick<AgentPluginInfo, "rootPath" | "skillsDir">,
    warnings: string[]
  ): Promise<AgentPluginPreviewSkill[]> {
    const skillsDir = plugin.skillsDir;
    if (skillsDir === undefined) {
      return [];
    }
    const skills: AgentPluginPreviewSkill[] = [];
    let entries: string[] = [];
    try {
      // Include symlinked skill dirs, matching runtime discovery
      // (listSkillDirectoriesFromLocalFs): a symlinked skill activates after
      // install, so it MUST appear in the consent preview.
      entries = (await fsPromises.readdir(skillsDir, { withFileTypes: true }))
        .filter((entry) => entry.isDirectory() || entry.isSymbolicLink())
        .map((entry) => entry.name)
        .sort((a, b) => a.localeCompare(b));
    } catch {
      return [];
    }
    for (const dirName of entries) {
      const skillPath = path.join(skillsDir, dirName, "SKILL.md");
      // Spec §4.1 containment anchored at the plugin root, mirroring runtime
      // component checks: a symlink escaping the plugin is surfaced as a
      // warning instead of silently ignored.
      let containedSkillPath: string;
      try {
        // allowMissing (matching runtime assertSkillDirValid): resolve through
        // the symlinked dir even when SKILL.md is absent, so an escaping
        // symlink fails containment instead of hiding behind ENOENT.
        containedSkillPath = await ensurePathContained(plugin.rootPath, skillPath, {
          allowMissing: true,
        });
      } catch (error) {
        if (!hasErrorCode(error, "ENOENT")) {
          warnings.push(`skills/${dirName}: resolves outside the plugin root; it will not load`);
        }
        // ENOENT (unresolvable path) → not a skill dir; skip silently.
        continue;
      }
      let stat;
      try {
        stat = await fsPromises.stat(containedSkillPath);
      } catch {
        continue;
      }
      if (!stat.isFile()) continue;
      try {
        const content = await fsPromises.readFile(containedSkillPath, "utf8");
        const parsed = parseSkillMarkdown({ content, byteSize: stat.size });
        skills.push({
          name: parsed.frontmatter.name,
          ...(parsed.frontmatter.description !== undefined
            ? { description: parsed.frontmatter.description }
            : {}),
        });
      } catch (error) {
        warnings.push(`skills/${dirName}: ${getErrorMessage(error)}`);
      }
    }
    return skills;
  }

  /**
   * Normalize the staged plugin's mcp.json into the preview list. Uses the
   * FINAL instance identity so `PLUGIN_DATA` paths shown to the user match
   * what will run; staged-root path fragments are rewritten to the final
   * install path for readability.
   */
  private async collectMcpServers(
    plugin: AgentPluginInfo,
    finalTargetPath: string,
    instanceId: string,
    warnings: string[]
  ): Promise<AgentPluginPreviewMcpServer[]> {
    if (plugin.mcpConfigPath === undefined) {
      return [];
    }
    const { servers, diagnostics } = await loadPluginMcpServers(plugin, {
      muxHome: this.config.rootDir,
      instanceId,
    });
    warnings.push(...diagnostics.map((d) => d.message));

    const rewrite = (value: string): string => value.split(plugin.rootPath).join(finalTargetPath);

    const result: AgentPluginPreviewMcpServer[] = [];
    for (const info of Object.values(servers)) {
      assert(info.plugin !== undefined, "plugin server info must carry provenance");
      if (info.transport === "stdio") {
        const commandLine = [info.command, ...(info.args ?? [])].map(rewrite).join(" ");
        const envKeys = Object.keys(info.env ?? {}).filter(
          (key) => key !== "PLUGIN_ROOT" && key !== "PLUGIN_DATA"
        );
        result.push({
          serverName: info.plugin.serverName,
          transport: "stdio",
          summary: envKeys.length > 0 ? `${commandLine} (env: ${envKeys.join(", ")})` : commandLine,
        });
      } else {
        result.push({
          serverName: info.plugin.serverName,
          transport: info.transport === "http" ? "http" : "sse",
          summary: info.url,
        });
      }
    }
    return result.sort((a, b) => a.serverName.localeCompare(b.serverName));
  }

  // ---------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------

  /**
   * Stage + validate an install without writing anything permanent. The
   * staged clone is deleted before returning (stateless preview): install
   * re-fetches the exact consented SHA, so cancelling leaves no state.
   */
  async preview(args: {
    input: string;
    ref?: string | undefined;
    subpath?: string | undefined;
  }): Promise<AgentPluginInstallPreview> {
    this.assertEnabled();

    const parsed = parseAgentPluginSourceInput(args.input);
    const explicitRef = args.ref?.trim() ?? "";
    if (explicitRef.length > 0 && parsed.ref !== undefined && parsed.ref !== explicitRef) {
      throw new Error(
        `Conflicting refs: '@${parsed.ref}' in the source and '${explicitRef}' in the ref field.`
      );
    }
    const ref = parsed.ref ?? (explicitRef.length > 0 ? explicitRef : undefined);
    const subpath = parsed.subpath ?? (args.subpath?.trim() ? args.subpath.trim() : undefined);
    if (subpath !== undefined) {
      // Approved v1 scope: the descriptor grammar knows subpaths, installs don't.
      throw new Error(
        "Monorepo subpath installs land in v2. Point at a repo whose root is the plugin."
      );
    }

    const resolved = await this.resolveRemoteRef(parsed.url, ref);
    const { dir: stagedDir, sha } = await this.cloneResolved(parsed.url, resolved);
    try {
      const { plugin, warnings } = await this.validateStagedClone(stagedDir);
      const targetPath = this.targetPathFor(plugin.name);
      await this.assertNoCollision(plugin.name);

      const skills = await this.collectSkills(plugin, warnings);
      const mcpServers = await this.collectMcpServers(
        plugin,
        targetPath,
        this.instanceIdFor(plugin.name),
        warnings
      );
      const hook = this.collectHook(plugin);

      if (resolved.refType === "tag" && sha !== resolved.sha) {
        warnings.push(
          `Tag '${resolved.ref}' moved between resolution and clone — installing ${sha.slice(0, 12)}.`
        );
      }

      const source: AgentPluginGitSource = {
        type: "git",
        url: parsed.url,
        ref: resolved.ref,
        refType: resolved.refType,
      };
      return {
        source,
        lockedSha: sha,
        manifest: manifestSummary(plugin.manifest),
        skills,
        mcpServers,
        ...(hook !== undefined ? { hook } : {}),
        warnings,
        targetPath: shortenHome(targetPath),
      };
    } finally {
      await this.removeDir(stagedDir);
    }
  }

  private async assertNoCollision(name: string): Promise<void> {
    // Strict: a corrupted registry must fail installs up front (with the
    // repair message) instead of letting a later strict read fail mid-flow.
    // Collide on RAW entry names, not just parsed ones: an entry this build
    // cannot parse (written by a newer build) still owns its name — the
    // install rewrite would otherwise filter it out and replace it.
    const { rawEntries } = await this.readRegistryDocument("strict");
    if (rawEntries.some((rawEntry) => this.rawEntryName(rawEntry) === name)) {
      throw new Error(`A managed plugin named '${name}' is already installed. Uninstall it first.`);
    }
    if (await pathExists(this.targetPathFor(name))) {
      // Never overwrite: an unmanaged dir may hold local work.
      throw new Error(
        `${shortenHome(this.targetPathFor(name))} already exists. Remove the directory first — the installer never overwrites.`
      );
    }
  }

  /** Fetch the consented SHA, validate again, promote into the container, and record the registry entry. */
  async install(args: {
    source: AgentPluginGitSource;
    expectedSha: string;
  }): Promise<AgentPluginInstallEntry> {
    this.assertEnabled();
    assert(isFullCommitSha(args.expectedSha), "install: expectedSha must be a full commit SHA");
    if (args.source.subpath !== undefined) {
      throw new Error("Monorepo subpath installs land in v2.");
    }

    return this.runExclusive(async () => {
      const stagedDir = await this.cloneExactSha(args.source, args.expectedSha);
      try {
        const { plugin } = await this.validateStagedClone(stagedDir);
        const name = plugin.name;
        await this.assertNoCollision(name);
        await this.assertNoPendingOverridePrune(name);
        const targetPath = this.targetPathFor(name);

        // The installed tree is a plain content snapshot: the registry holds
        // all provenance, and updates replace the directory wholesale, so a
        // .git dir would only invite in-place edits that updates discard.
        await this.removeDir(path.join(stagedDir, ".git"));

        await fsPromises.mkdir(this.containerDir, { recursive: true });
        await fsPromises.rename(stagedDir, targetPath);

        const entry: AgentPluginInstallEntry = {
          name,
          scope: "global",
          source: args.source,
          lockedSha: args.expectedSha,
          installedAt: new Date().toISOString(),
          manifest: {
            ...(plugin.manifest.version !== undefined ? { version: plugin.manifest.version } : {}),
            ...(plugin.manifest.description !== undefined
              ? { description: plugin.manifest.description }
              : {}),
          },
        };
        try {
          const { envelope, rawEntries } = await this.readRegistryDocument("strict");
          await this.writeRegistry(envelope, [
            ...rawEntries.filter((rawEntry) => this.rawEntryName(rawEntry) !== name),
            entry,
          ]);
        } catch (error) {
          // No partial state: a promote without a registry entry would look
          // like an unmanaged dir and block reinstall.
          await this.removeDir(targetPath);
          throw new Error(`Failed to persist the plugin registry: ${getErrorMessage(error)}`);
        }
        log.info(`Installed agent plugin '${name}' at ${args.expectedSha.slice(0, 12)}`);
        return entry;
      } finally {
        await this.removeDir(stagedDir);
      }
    });
  }

  /** Managed registry entries merged with unmanaged plugins found by global discovery. */
  async list(): Promise<AgentPluginListItem[]> {
    this.assertEnabled();

    // Section open is the natural retry moment for override-prune tombstones
    // left by uninstalls whose workspaces were temporarily unreachable.
    await this.retryPendingOverridePrunes().catch((error: unknown) => {
      log.warn("Failed to retry pending override prunes", { error: getErrorMessage(error) });
    });

    const registry = await this.readRegistry("lenient");
    const containers: AgentPluginContainer[] = [
      { path: this.containerDir, scope: "global" },
      { path: path.join(os.homedir(), ".agents", "plugins"), scope: "global" },
    ];
    const { plugins } = await discoverAgentPlugins(containers);

    const items: AgentPluginListItem[] = [];
    const managedByName = new Map(registry.map((entry) => [entry.name, entry]));

    for (const plugin of plugins) {
      const isManagedLocation =
        plugin.containerPath === this.containerDir && managedByName.has(plugin.dirName);
      const entry = isManagedLocation ? managedByName.get(plugin.dirName) : undefined;
      if (entry) {
        managedByName.delete(plugin.dirName);
      }

      const warnings: string[] = [];
      const skillCount = (await this.collectSkills(plugin, warnings)).length;
      let mcpServerCount = 0;
      if (plugin.mcpConfigPath !== undefined) {
        try {
          const { servers } = await loadPluginMcpServers(plugin, {
            muxHome: this.config.rootDir,
            instanceId: computePluginInstanceId(path.join(plugin.containerPath, plugin.dirName)),
          });
          mcpServerCount = Object.keys(servers).length;
        } catch (error) {
          log.warn(`Agent plugin ${plugin.rootPath}: failed to count MCP servers`, { error });
        }
      }

      // Managed rows keep their REGISTRY identity: update/uninstall look
      // entries up by this name, so a locally edited/corrupted manifest name
      // must not make the row unrepairable from Settings. The drift is still
      // surfaced in the description.
      const manifestNameDrift =
        entry !== undefined && plugin.name !== entry.name
          ? `plugin.json names itself '${plugin.name}' — the installed name '${entry.name}' stays authoritative.`
          : undefined;
      const description = manifestNameDrift ?? plugin.manifest.description;
      items.push({
        name: entry?.name ?? plugin.name,
        managed: entry !== undefined,
        present: true,
        location: shortenHome(path.join(plugin.containerPath, plugin.dirName)),
        ...(plugin.manifest.version !== undefined ? { version: plugin.manifest.version } : {}),
        ...(description !== undefined ? { description } : {}),
        ...(entry !== undefined
          ? {
              source: entry.source,
              lockedSha: entry.lockedSha,
              installedAt: entry.installedAt,
              ...(entry.updatedAt !== undefined ? { updatedAt: entry.updatedAt } : {}),
            }
          : {}),
        skillCount,
        mcpServerCount,
      });
    }

    // Registry entries whose directory vanished (self-heal display; uninstall still works).
    for (const entry of managedByName.values()) {
      items.push({
        name: entry.name,
        managed: true,
        present: false,
        location: shortenHome(this.targetPathFor(entry.name)),
        ...(entry.manifest?.version !== undefined ? { version: entry.manifest.version } : {}),
        ...(entry.manifest?.description !== undefined
          ? { description: entry.manifest.description }
          : {}),
        source: entry.source,
        lockedSha: entry.lockedSha,
        installedAt: entry.installedAt,
        ...(entry.updatedAt !== undefined ? { updatedAt: entry.updatedAt } : {}),
        skillCount: 0,
        mcpServerCount: 0,
      });
    }

    return items.sort((a, b) => a.name.localeCompare(b.name));
  }

  /**
   * Uninstall: delete dir + registry entry + prune that plugin's per-workspace
   * MCP overrides (reinstall re-attaches the same instanceId, so stale
   * overrides would silently re-enable servers — violating default-disabled).
   * PLUGIN_DATA is preserved unless `deletePluginData` is set.
   */
  async uninstall(args: { name: string; deletePluginData: boolean }): Promise<void> {
    this.assertEnabled();

    return this.runExclusive(async () => {
      const { envelope, rawEntries: rawRegistry } = await this.readRegistryDocument("strict");
      const registry = this.parseRegistryEntries(rawRegistry);
      const entry = registry.find((e) => e.name === args.name);
      if (!entry) {
        throw new Error(`'${args.name}' is not a managed plugin install.`);
      }

      const targetPath = this.targetPathFor(entry.name);
      const instanceId = this.instanceIdFor(entry.name);
      const serverKeyPrefix = buildPluginServerKey(instanceId, "");

      // Enumerate pruning targets BEFORE committing anything: if this fails,
      // the uninstall aborts with the install fully intact (retryable from
      // Settings) instead of leaving stale overrides behind post-commit.
      const workspaceIdsToPrune = await this.listWorkspaceIdsForOverridePruning();

      // Stop running servers before deleting the tree out from under them.
      await this.deps.mcpServerManager?.stopServersWithKeyPrefix(serverKeyPrefix);

      // Stage the tree — and, when requested, the plugin-data dir — out
      // BEFORE touching the registry so every step can fail without partial
      // state: a failed rename (e.g. a locked file on Windows) leaves the
      // install fully intact, and a failed registry write renames everything
      // back. Deleting the staged dirs afterwards is best-effort — they sit
      // under the staging root, where stale-dir reclamation cleans up
      // leftovers, so a locked dir cannot strand the user in a state where
      // the Settings row is gone but their requested cleanup never happens.
      await fsPromises.mkdir(this.stagingRoot, { recursive: true });
      const trashDir = path.join(this.stagingRoot, `trash-${Date.now()}-${entry.name}`);
      let stagedTree = false;
      try {
        await fsPromises.rename(targetPath, trashDir);
        stagedTree = true;
      } catch (error) {
        if (!hasErrorCode(error, "ENOENT")) {
          throw new Error(`Failed to remove the plugin directory: ${getErrorMessage(error)}`);
        }
        // Missing tree (present:false row): registry-only uninstall.
      }

      const restoreTree = async (context: string): Promise<void> => {
        if (!stagedTree) {
          return;
        }
        await fsPromises.rename(trashDir, targetPath).catch((rollbackError: unknown) => {
          log.error(`Failed to restore plugin dir after ${context}`, {
            targetPath,
            rollbackError,
          });
        });
      };

      const dataPath = getPluginDataPath(this.config.rootDir, instanceId);
      const dataTrashDir = path.join(this.stagingRoot, `trash-data-${Date.now()}-${entry.name}`);
      let stagedData = false;
      if (args.deletePluginData) {
        try {
          await fsPromises.rename(dataPath, dataTrashDir);
          stagedData = true;
        } catch (error) {
          if (!hasErrorCode(error, "ENOENT")) {
            // Fail BEFORE the registry commit so the row stays and the user
            // can retry the requested cleanup.
            await restoreTree("failed plugin-data staging");
            throw new Error(`Failed to remove the plugin data: ${getErrorMessage(error)}`);
          }
          // No data dir: nothing to delete.
        }
      }

      // The commit write carries a PESSIMISTIC tombstone for every workspace
      // that needs pruning: if a prune later fails — or the best-effort
      // shrink write below fails — the durable record already exists.
      // Over-blocking a reinstall until cleanup is confirmed is safe;
      // silently losing the record (stale enabledServers reactivating a
      // reinstalled server) is not.
      const commitEnvelope = { ...envelope };
      const pendingForCommit = this.updateRawPendingPrunes(
        this.rawPendingPrunes(envelope),
        serverKeyPrefix,
        workspaceIdsToPrune
      );
      if (pendingForCommit.length > 0) {
        commitEnvelope.pendingOverridePrunes = pendingForCommit;
      } else {
        delete commitEnvelope.pendingOverridePrunes;
      }
      try {
        await this.writeRegistry(
          commitEnvelope,
          rawRegistry.filter((rawEntry) => this.rawEntryName(rawEntry) !== entry.name)
        );
      } catch (error) {
        await restoreTree("failed registry write");
        if (stagedData) {
          await fsPromises.rename(dataTrashDir, dataPath).catch((rollbackError: unknown) => {
            log.error("Failed to restore plugin data after failed registry write", {
              dataPath,
              rollbackError,
            });
          });
        }
        throw new Error(`Failed to persist the plugin registry: ${getErrorMessage(error)}`);
      }

      // The uninstall is committed; everything below is best-effort cleanup
      // that must not abort the remaining steps.
      if (stagedTree) {
        await this.removeDir(trashDir).catch((error: unknown) => {
          log.warn("Failed to delete uninstalled plugin tree; leaving it for staging reclamation", {
            trashDir,
            error: getErrorMessage(error),
          });
        });
      }
      if (stagedData) {
        await this.removeDir(dataTrashDir).catch((error: unknown) => {
          log.warn("Failed to delete plugin data; leaving it for staging reclamation", {
            dataTrashDir,
            error: getErrorMessage(error),
          });
        });
      }

      // Re-invalidate AFTER the tree is gone: a getToolsForWorkspace call
      // that started right after the pre-rename stop snapshots the new epoch,
      // and can still have discovered the plugin before the rename — its
      // freshly started server would otherwise publish validly and keep
      // running from the removed tree. This runs BEFORE override pruning so
      // pruning problems cannot skip the correctness-critical invalidation.
      await this.deps.mcpServerManager?.stopServersWithKeyPrefix(serverKeyPrefix);

      // Per-workspace failures are caught inside; the failure-prone
      // enumeration already happened pre-commit and the pessimistic
      // tombstone is already durable (commit write above). Shrink it to what
      // actually failed — best-effort: a failed shrink leaves the over-broad
      // tombstone, which self-heals on the next retry (section open or the
      // reinstall gate).
      const failedPruneIds = await this.pruneWorkspaceOverrides(
        serverKeyPrefix,
        workspaceIdsToPrune
      );
      if (workspaceIdsToPrune.length > 0) {
        // STRICT re-read for the shrink: a lenient read degrading a transient
        // I/O error or corruption to an empty document would make this write
        // rewrite plugins.json with an empty plugin list, orphaning every
        // other managed install. On any failure the pessimistic tombstone
        // from the commit write simply stays (safe, self-heals on retry).
        try {
          const { envelope: envelopeAfter, rawEntries: entriesAfter } =
            await this.readRegistryDocument("strict");
          const pendingAfter = this.updateRawPendingPrunes(
            this.rawPendingPrunes(envelopeAfter),
            serverKeyPrefix,
            failedPruneIds
          );
          await this.writePendingOverridePrunes(envelopeAfter, entriesAfter, pendingAfter);
        } catch (error) {
          log.warn("Failed to shrink pending override prune tombstone (kept pessimistic)", {
            serverKeyPrefix,
            failedPruneIds,
            error: getErrorMessage(error),
          });
        }
      }

      log.info(`Uninstalled agent plugin '${entry.name}'`);
    });
  }

  /**
   * Enumerate the local/worktree workspace IDs whose MCP overrides an
   * uninstall must prune. Called BEFORE the uninstall commits anything:
   * enumeration is the only pruning step that can fail wholesale (outside
   * the per-workspace catch), and a post-commit failure would leave stale
   * overrides with no Settings row left to retry from — a reinstall reuses
   * the same instance ID and would silently re-enable those servers.
   * Remote runtimes are skipped — they never see plugin servers
   * (resolveAgentPluginsMcpContext returns null off-host).
   */
  private async listWorkspaceIdsForOverridePruning(): Promise<string[]> {
    if (!this.deps.workspaceMcpOverridesService) {
      return [];
    }
    const allMetadata = await this.config.getAllWorkspaceMetadata();
    return allMetadata
      .filter((metadata) => {
        const runtimeType = metadata.runtimeConfig.type;
        return runtimeType === "local" || runtimeType === "worktree";
      })
      .map((metadata) => metadata.id);
  }

  /**
   * Remove `plugin:<instanceId>:*` keys from the given workspaces' MCP
   * overrides. Best-effort per workspace: a missing checkout must not block
   * uninstall. Returns the workspace IDs whose prune FAILED so callers can
   * persist a retryable tombstone — silently discarding a failure would let
   * a reinstall (same instance ID) pick up the stale override and re-enable
   * the server without consent.
   */
  private async pruneWorkspaceOverrides(
    serverKeyPrefix: string,
    workspaceIds: string[]
  ): Promise<string[]> {
    const overridesService = this.deps.workspaceMcpOverridesService;
    if (!overridesService) {
      return [];
    }
    // A concurrent Workspace MCP dialog save can land between our read and
    // write; expectedRevision detects that, and we re-read + re-filter.
    const MAX_CAS_ATTEMPTS = 3;
    const failedWorkspaceIds: string[] = [];
    for (const workspaceId of workspaceIds) {
      try {
        for (let attempt = 1; ; attempt++) {
          // Strict read: a temporarily unreadable overrides file must FAIL
          // this workspace's prune (preserving its tombstone for retry), not
          // read as "{}" and retire the tombstone against keys never seen.
          const { overrides, revision } = await overridesService.getOverridesForWorkspace(
            workspaceId,
            { mode: "strict" }
          );
          const dropKey = (key: string) => key.startsWith(serverKeyPrefix);
          const enabledServers = overrides.enabledServers?.filter((key) => !dropKey(key));
          const disabledServers = overrides.disabledServers?.filter((key) => !dropKey(key));
          const toolAllowlist = overrides.toolAllowlist
            ? Object.fromEntries(
                Object.entries(overrides.toolAllowlist).filter(([key]) => !dropKey(key))
              )
            : undefined;

          const changed =
            (overrides.enabledServers?.length ?? 0) !== (enabledServers?.length ?? 0) ||
            (overrides.disabledServers?.length ?? 0) !== (disabledServers?.length ?? 0) ||
            Object.keys(overrides.toolAllowlist ?? {}).length !==
              Object.keys(toolAllowlist ?? {}).length;
          if (!changed) {
            break;
          }
          try {
            await overridesService.setOverridesForWorkspace(
              workspaceId,
              {
                ...(enabledServers !== undefined ? { enabledServers } : {}),
                ...(disabledServers !== undefined ? { disabledServers } : {}),
                ...(toolAllowlist !== undefined ? { toolAllowlist } : {}),
              },
              { expectedRevision: revision }
            );
            break;
          } catch (error) {
            if (error instanceof WorkspaceMcpOverridesConflictError && attempt < MAX_CAS_ATTEMPTS) {
              continue;
            }
            throw error;
          }
        }
      } catch (error) {
        failedWorkspaceIds.push(workspaceId);
        log.warn("Failed to prune plugin MCP overrides for workspace", {
          workspaceId,
          error: getErrorMessage(error),
        });
      }
    }
    return failedWorkspaceIds;
  }

  /**
   * Pending override prunes ("tombstones") persisted in the registry
   * envelope under `pendingOverridePrunes`: uninstalls whose per-workspace
   * override cleanup failed (checkout temporarily unavailable, unwritable
   * override file). They are retried on section open (list) and gate a
   * reinstall of the same instance ID, so a stale `enabledServers` key can
   * never silently re-enable a reinstalled plugin's server.
   *
   * Rewrites operate on the RAW item list, mirroring the registry-entry
   * rules: items this build cannot parse (a newer release's tombstone
   * variant) pass through untouched, and recognized items keep their unknown
   * fields when their `workspaceIds` shrink.
   */
  private isRecognizedPrune(
    item: unknown
  ): item is { prefix: string; workspaceIds: string[] } & Record<string, unknown> {
    if (typeof item !== "object" || item === null) {
      return false;
    }
    const prefix = (item as { prefix?: unknown }).prefix;
    const workspaceIds = (item as { workspaceIds?: unknown }).workspaceIds;
    return (
      typeof prefix === "string" &&
      prefix.length > 0 &&
      Array.isArray(workspaceIds) &&
      workspaceIds.every((id): id is string => typeof id === "string")
    );
  }

  /** The raw `pendingOverridePrunes` array as stored (unknown variants included). */
  private rawPendingPrunes(envelope: Record<string, unknown>): unknown[] {
    const raw = envelope.pendingOverridePrunes;
    return Array.isArray(raw) ? raw : [];
  }

  /** Recognized tombstones only (for matching/retrying). */
  private parsePendingOverridePrunes(
    envelope: Record<string, unknown>
  ): Array<{ prefix: string; workspaceIds: string[] }> {
    return this.rawPendingPrunes(envelope)
      .filter((item) => this.isRecognizedPrune(item))
      .map((item) => ({ prefix: item.prefix, workspaceIds: item.workspaceIds }));
  }

  /**
   * Set this build's tombstone for `prefix` within the raw item list:
   * removes the recognized item for that prefix (merging its unknown fields
   * into the replacement) and appends the new one when `workspaceIds` is
   * non-empty. Unrecognized items are preserved verbatim.
   */
  private updateRawPendingPrunes(
    rawPending: unknown[],
    prefix: string,
    workspaceIds: string[]
  ): unknown[] {
    const existing = rawPending.find(
      (item) => this.isRecognizedPrune(item) && item.prefix === prefix
    );
    const next = rawPending.filter(
      (item) => !(this.isRecognizedPrune(item) && item.prefix === prefix)
    );
    if (workspaceIds.length > 0) {
      next.push({
        ...((existing as Record<string, unknown> | undefined) ?? {}),
        prefix,
        workspaceIds,
      });
    }
    return next;
  }

  /** Persist the raw tombstone list into the envelope (removing the key when empty). */
  private async writePendingOverridePrunes(
    envelope: Record<string, unknown>,
    rawEntries: unknown[],
    rawPending: unknown[]
  ): Promise<void> {
    const nextEnvelope = { ...envelope };
    if (rawPending.length > 0) {
      nextEnvelope.pendingOverridePrunes = rawPending;
    } else {
      delete nextEnvelope.pendingOverridePrunes;
    }
    await this.writeRegistry(nextEnvelope, rawEntries);
  }

  /**
   * Retry one tombstone's pruning. Workspaces that no longer exist in the
   * config are dropped first — a deleted workspace's overrides can never
   * reactivate anything, so keeping its ID would block reinstall forever.
   * Returns the IDs that still need pruning (existing workspaces whose
   * prune failed, or everything when metadata enumeration itself failed).
   */
  private async retryPrune(prune: { prefix: string; workspaceIds: string[] }): Promise<string[]> {
    let liveWorkspaceIds = prune.workspaceIds;
    try {
      const allMetadata = await this.config.getAllWorkspaceMetadata();
      const knownIds = new Set(allMetadata.map((metadata) => metadata.id));
      liveWorkspaceIds = prune.workspaceIds.filter((workspaceId) => knownIds.has(workspaceId));
    } catch (error) {
      // Enumeration failed: keep the full list (over-blocking is safe).
      log.warn("Failed to reconcile pending override prune against workspaces", {
        error: getErrorMessage(error),
      });
    }
    return this.pruneWorkspaceOverrides(prune.prefix, liveWorkspaceIds);
  }

  /**
   * Reinstall gate: a plugin name maps to the same instance ID, so a pending
   * prune for its prefix means stale workspace overrides could re-enable the
   * reinstalled plugin's servers without consent. Retry the prune now; only
   * a fully successful cleanup unblocks the install. Runs under the caller's
   * exclusive mutation lock (install's runExclusive).
   */
  private async assertNoPendingOverridePrune(name: string): Promise<void> {
    const serverKeyPrefix = buildPluginServerKey(this.instanceIdFor(name), "");
    const { envelope, rawEntries } = await this.readRegistryDocument("strict");
    const pending = this.parsePendingOverridePrunes(envelope);
    const match = pending.find((prune) => prune.prefix === serverKeyPrefix);
    if (!match) {
      return;
    }

    const failed = await this.retryPrune(match);
    const remaining = this.updateRawPendingPrunes(
      this.rawPendingPrunes(envelope),
      serverKeyPrefix,
      failed
    );
    await this.writePendingOverridePrunes(envelope, rawEntries, remaining);
    if (failed.length > 0) {
      throw new Error(
        `A previous uninstall of '${name}' could not clean up its workspace MCP overrides yet (workspaces: ${failed.join(", ")}). Retry once those workspaces are accessible.`
      );
    }
  }

  /**
   * Retry all pending override prunes; persists progress. Best-effort: runs
   * on section open (list), so transient failures self-heal the next time
   * the affected checkout is reachable. The read-modify-write runs under the
   * exclusive mutation queue — an install/update/uninstall committing while
   * the workspace I/O is in flight would otherwise be clobbered by this
   * write's stale registry snapshot.
   */
  private async retryPendingOverridePrunes(): Promise<void> {
    return this.runExclusive(async () => {
      const { envelope, rawEntries } = await this.readRegistryDocument("lenient");
      const pending = this.parsePendingOverridePrunes(envelope);
      if (pending.length === 0) {
        return;
      }

      let rawPending = this.rawPendingPrunes(envelope);
      let progressed = false;
      for (const prune of pending) {
        const failed = await this.retryPrune(prune);
        if (failed.length !== prune.workspaceIds.length) {
          progressed = true;
          rawPending = this.updateRawPendingPrunes(rawPending, prune.prefix, failed);
        }
      }

      if (progressed) {
        await this.writePendingOverridePrunes(envelope, rawEntries, rawPending).catch(
          (error: unknown) => {
            log.warn("Failed to persist pending override prune progress", {
              error: getErrorMessage(error),
            });
          }
        );
      }
    });
  }

  /**
   * Compare each managed entry's tracking ref against `lockedSha` via
   * `git ls-remote` (no fetch). Runs on Settings-section open and on the
   * explicit "Check for updates" action only — no background timers.
   */
  async checkUpdates(): Promise<AgentPluginUpdateCheck[]> {
    this.assertEnabled();

    const registry = await this.readRegistry("lenient");
    return Promise.all(
      registry.map(async (entry): Promise<AgentPluginUpdateCheck> => {
        if (entry.source.refType === "commit") {
          return { name: entry.name, status: "pinned" };
        }
        try {
          const resolved = await this.resolveRemoteRef(entry.source.url, entry.source.ref);
          if (resolved.refType !== entry.source.refType) {
            // e.g. a tracked branch was deleted and a tag with the same name exists now.
            return {
              name: entry.name,
              status: "error",
              message: `Tracked ${entry.source.refType} '${entry.source.ref}' is now a ${resolved.refType} on the remote.`,
            };
          }
          if (resolved.sha === entry.lockedSha) {
            return { name: entry.name, status: "up-to-date" };
          }
          return {
            name: entry.name,
            // A moved tag is suspicious (tags are supposed to be immutable) — warn, don't just offer.
            status: entry.source.refType === "tag" ? "tag-moved" : "update-available",
            remoteSha: resolved.sha,
          };
        } catch (error) {
          return { name: entry.name, status: "error", message: getErrorMessage(error) };
        }
      })
    );
  }

  /**
   * Apply an update: temp clone at the new SHA → re-validate → wholesale
   * directory swap (rename-old → promote-new → delete-old) → bump lockedSha →
   * recycle that plugin's MCP servers. Never an in-place `git pull`; local
   * edits to the managed dir are discarded.
   */
  async update(args: { name: string }): Promise<AgentPluginInstallEntry> {
    this.assertEnabled();

    return this.runExclusive(async () => {
      const { envelope, rawEntries: rawRegistry } = await this.readRegistryDocument("strict");
      const registry = this.parseRegistryEntries(rawRegistry);
      const entry = registry.find((e) => e.name === args.name);
      if (!entry) {
        throw new Error(`'${args.name}' is not a managed plugin install.`);
      }
      if (entry.source.refType === "commit") {
        throw new Error(
          `'${entry.name}' is pinned to commit ${entry.lockedSha.slice(0, 12)}; uninstall and reinstall to change it.`
        );
      }

      const resolved = await this.resolveRemoteRef(entry.source.url, entry.source.ref);
      if (resolved.refType !== entry.source.refType) {
        // The ref name now resolves to a different kind on the remote (e.g. a
        // tracked branch was deleted and a tag of the same name exists). The
        // update check flags this as an error; a stale Update click must not
        // silently install content from a different ref kind while the
        // registry keeps claiming the old one.
        throw new Error(
          `Tracked ${entry.source.refType} '${entry.source.ref}' is now a ${resolved.refType} on the remote. Uninstall and reinstall to track it.`
        );
      }
      if (resolved.sha === entry.lockedSha) {
        return entry; // Already current.
      }

      const stagedDir = await this.cloneExactSha(entry.source, resolved.sha);
      try {
        const { plugin } = await this.validateStagedClone(stagedDir);
        if (plugin.name !== entry.name) {
          // Container-entry names are identity (instanceId, PLUGIN_DATA,
          // workspace overrides hash the path) — never rename on update.
          throw new Error(
            `The plugin renamed itself upstream ('${entry.name}' → '${plugin.name}'). Uninstall and reinstall to adopt the new name.`
          );
        }
        await this.removeDir(path.join(stagedDir, ".git"));

        const targetPath = this.targetPathFor(entry.name);
        const serverKeyPrefix = buildPluginServerKey(this.instanceIdFor(entry.name), "");
        const trashDir = path.join(this.stagingRoot, `trash-${Date.now()}-${entry.name}`);
        const hadOldTree = await pathExists(targetPath);

        // Stop this plugin's running MCP servers BEFORE the old tree moves:
        // a live server can lose its files mid-swap on POSIX, and open
        // handles can make the rename itself fail on Windows.
        await this.deps.mcpServerManager?.stopServersWithKeyPrefix(serverKeyPrefix);

        if (hadOldTree) {
          await fsPromises.rename(targetPath, trashDir);
        }
        try {
          await fsPromises.mkdir(this.containerDir, { recursive: true });
          await fsPromises.rename(stagedDir, targetPath);
        } catch (error) {
          if (hadOldTree) {
            // Roll the old tree back so a failed swap never leaves the plugin missing.
            await fsPromises.rename(trashDir, targetPath).catch((rollbackError: unknown) => {
              log.error("Failed to roll back plugin dir after failed update swap", {
                targetPath,
                rollbackError,
              });
            });
          }
          throw error;
        }
        if (hadOldTree) {
          // Best-effort: the trash dir sits under the staging root, where
          // stale-dir reclamation cleans up leftovers.
          await this.removeDir(trashDir).catch((error: unknown) => {
            log.warn("Failed to delete replaced plugin tree; leaving it for staging reclamation", {
              trashDir,
              error: getErrorMessage(error),
            });
          });
        }

        const updated: AgentPluginInstallEntry = {
          ...entry,
          lockedSha: resolved.sha,
          updatedAt: new Date().toISOString(),
          manifest: {
            ...(plugin.manifest.version !== undefined ? { version: plugin.manifest.version } : {}),
            ...(plugin.manifest.description !== undefined
              ? { description: plugin.manifest.description }
              : {}),
          },
        };
        // The new tree is already promoted; a failed write surfaces as an
        // error and the stale lockedSha keeps the update badge visible, so
        // retrying the update self-heals the mismatch.
        //
        // Patch ONLY the fields this update owns (lockedSha, updatedAt, and
        // the manifest's version/description) into the RAW entry: spreading
        // the Zod-parsed entry would replace `source`/`manifest` wholesale
        // with their stripped counterparts, deleting nested metadata a newer
        // build may have stored there (breaking downgrade round-trips).
        try {
          await this.writeRegistry(
            envelope,
            rawRegistry.map((rawEntry) => {
              if (this.rawEntryName(rawEntry) !== entry.name) {
                return rawEntry;
              }
              const rawRecord = rawEntry as Record<string, unknown>;
              const rawManifest =
                typeof rawRecord.manifest === "object" &&
                rawRecord.manifest !== null &&
                !Array.isArray(rawRecord.manifest)
                  ? (rawRecord.manifest as Record<string, unknown>)
                  : {};
              // version/description are owned by the update (they mirror the
              // newly installed plugin.json), so stale values are dropped and
              // fresh ones written; unknown manifest keys pass through.
              const {
                version: _staleVersion,
                description: _staleDescription,
                ...preservedManifest
              } = rawManifest;
              return {
                ...rawRecord,
                lockedSha: updated.lockedSha,
                updatedAt: updated.updatedAt,
                manifest: { ...preservedManifest, ...updated.manifest },
              };
            })
          );
        } finally {
          // Recycle post-promote even when the registry write fails: the tree
          // already swapped, so (1) content changed behind a stable path —
          // possibly an unchanged stdio command line — which the config
          // signature cannot see, and (2) a concurrent getToolsForWorkspace
          // that began after the pre-swap invalidation but discovered the
          // plugin before the rename may have published a server from the
          // replaced tree. Servers restart on next use; default-disabled
          // state and workspace overrides are untouched (identity is the
          // lexical path).
          await this.deps.mcpServerManager?.stopServersWithKeyPrefix(serverKeyPrefix);
        }

        log.info(
          `Updated agent plugin '${entry.name}' ${entry.lockedSha.slice(0, 12)} → ${resolved.sha.slice(0, 12)}`
        );
        return updated;
      } finally {
        await this.removeDir(stagedDir);
      }
    });
  }
}
