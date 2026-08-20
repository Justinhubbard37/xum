/* eslint-disable @typescript-eslint/await-thenable -- bun:test types `await expect(...).rejects.toThrow()` as void */
import { afterEach, beforeEach, describe, expect, spyOn, test } from "bun:test";
import * as fsPromises from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";

import { Config } from "@/node/config";
import type { MCPServerManager } from "@/node/services/mcpServerManager";
import type { WorkspaceMcpOverridesService } from "@/node/services/workspaceMcpOverridesService";
import { execFileAsync } from "@/node/utils/disposableExec";
import { AgentPluginInstallService } from "./installService";
import {
  AGENT_PLUGIN_MCP_SCHEMA_ID_1_0_0,
  buildAddedPluginKeyValidator,
  computePluginInstanceId,
  getPluginDataPath,
} from "./mcpConfig";
import { AGENT_PLUGIN_SCHEMA_ID_1_0_0 } from "./manifest";

/**
 * Lifecycle tests against a real local git "remote". Local-path remotes go
 * through the same clone/ls-remote plumbing as network URLs, so the full
 * preview → install → check → update → uninstall loop runs hermetically.
 */

async function git(cwd: string, ...args: string[]): Promise<string> {
  using proc = execFileAsync("git", ["-C", cwd, ...args]);
  return (await proc.result).stdout;
}

async function initRemote(dir: string): Promise<void> {
  using proc = execFileAsync("git", ["init", "--quiet", "-b", "main", dir]);
  await proc.result;
  await git(dir, "config", "user.email", "test@example.com");
  await git(dir, "config", "user.name", "Test");
}

async function commitAll(dir: string, message: string): Promise<string> {
  await git(dir, "add", "-A");
  await git(dir, "commit", "--quiet", "-m", message);
  return (await git(dir, "rev-parse", "HEAD")).trim();
}

async function writePluginFixture(dir: string, opts?: { version?: string }): Promise<void> {
  await fsPromises.writeFile(
    path.join(dir, "plugin.json"),
    JSON.stringify({
      $schema: AGENT_PLUGIN_SCHEMA_ID_1_0_0,
      name: "demo-plugin",
      version: opts?.version ?? "1.0.0",
      description: "Demo plugin",
    })
  );
  await fsPromises.mkdir(path.join(dir, "skills", "greet"), { recursive: true });
  await fsPromises.writeFile(
    path.join(dir, "skills", "greet", "SKILL.md"),
    "---\nname: greet\ndescription: Greets people\n---\n\nSay hi.\n"
  );
  await fsPromises.writeFile(
    path.join(dir, "mcp.json"),
    JSON.stringify({
      $schema: AGENT_PLUGIN_MCP_SCHEMA_ID_1_0_0,
      mcpServers: {
        echo: { type: "stdio", command: "node", args: ["${PLUGIN_ROOT}/server.js"] },
      },
    })
  );
}

describe("AgentPluginInstallService", () => {
  let muxRoot: string;
  let remoteDir: string;
  let config: Config;
  let service: AgentPluginInstallService;
  let enabled = true;

  const pluginsDir = () => path.join(muxRoot, "plugins");
  const stagingDir = () => path.join(muxRoot, "plugin-staging");
  const registryFile = () => path.join(muxRoot, "plugins.json");
  const registry = async (): Promise<unknown[]> => {
    try {
      const raw = await fsPromises.readFile(registryFile(), "utf8");
      return (JSON.parse(raw) as { plugins: unknown[] }).plugins;
    } catch {
      return [];
    }
  };
  const pathExists = async (p: string) =>
    fsPromises.access(p).then(
      () => true,
      () => false
    );
  const stagingLeftovers = async () =>
    (await pathExists(stagingDir())) ? fsPromises.readdir(stagingDir()) : [];

  beforeEach(async () => {
    muxRoot = await fsPromises.mkdtemp(path.join(os.tmpdir(), "mux-plugin-test-"));
    remoteDir = await fsPromises.mkdtemp(path.join(os.tmpdir(), "mux-plugin-remote-"));
    config = new Config(muxRoot);
    enabled = true;
    service = new AgentPluginInstallService(config, { isEnabled: () => enabled });
    await initRemote(remoteDir);
    await writePluginFixture(remoteDir);
    await commitAll(remoteDir, "init");
  });

  afterEach(async () => {
    await fsPromises.rm(muxRoot, { recursive: true, force: true });
    await fsPromises.rm(remoteDir, { recursive: true, force: true });
  });

  test("consent preview discloses symlinked skills and warns on escaping symlinks", async () => {
    // Runtime discovery loads symlinked skill dirs, so the preview must
    // disclose them; symlinks escaping the plugin root are warned about.
    await fsPromises.mkdir(path.join(remoteDir, "shared", "linked-skill"), { recursive: true });
    await fsPromises.writeFile(
      path.join(remoteDir, "shared", "linked-skill", "SKILL.md"),
      "---\nname: linked-skill\ndescription: Lives outside skills/, reached via symlink\n---\n\nBody.\n"
    );
    await fsPromises.symlink(
      "../shared/linked-skill",
      path.join(remoteDir, "skills", "linked-skill")
    );
    await fsPromises.symlink("/etc", path.join(remoteDir, "skills", "escaping"));
    await commitAll(remoteDir, "symlinked skills");

    const preview = await service.preview({ input: remoteDir });
    expect(preview.skills.map((skill) => skill.name)).toEqual(["greet", "linked-skill"]);
    expect(preview.warnings.some((warning) => warning.includes("skills/escaping"))).toBe(true);
  });

  test("consent preview discloses executable hooks with their tool grants", async () => {
    // hooks.js loads automatically after install and can rewrite/block tool
    // calls — consent must surface it (with the grants the runtime honors).
    expect((await service.preview({ input: remoteDir })).hook).toBeUndefined();

    await fsPromises.writeFile(
      path.join(remoteDir, "hooks.js"),
      "({ 'tool.execute.before': () => undefined })\n"
    );
    await commitAll(remoteDir, "least-privilege hook");
    const leastPrivilege = await service.preview({ input: remoteDir });
    expect(leastPrivilege.hook).toEqual({ path: "hooks.js", toolGrants: [] });

    const manifestPath = path.join(remoteDir, "plugin.json");
    const manifest = JSON.parse(await fsPromises.readFile(manifestPath, "utf8")) as Record<
      string,
      unknown
    >;
    manifest.extensions = { mux: { hooks: { tools: ["bash", "file_read"] } } };
    await fsPromises.writeFile(manifestPath, JSON.stringify(manifest, null, 2));
    await commitAll(remoteDir, "hook with tool grants");
    const granted = await service.preview({ input: remoteDir });
    expect(granted.hook).toEqual({ path: "hooks.js", toolGrants: ["bash", "file_read"] });
  });

  test("preview stages+validates without writing; install promotes and records the registry", async () => {
    const head = (await git(remoteDir, "rev-parse", "HEAD")).trim();

    const preview = await service.preview({ input: remoteDir });
    expect(preview.source).toEqual({
      type: "git",
      url: remoteDir,
      ref: "main",
      refType: "branch",
    });
    expect(preview.lockedSha).toBe(head);
    expect(preview.manifest).toMatchObject({ name: "demo-plugin", version: "1.0.0" });
    expect(preview.skills).toEqual([{ name: "greet", description: "Greets people" }]);
    expect(preview.mcpServers).toHaveLength(1);
    expect(preview.mcpServers[0].serverName).toBe("echo");
    expect(preview.mcpServers[0].transport).toBe("stdio");
    // Command line shows the FINAL install path, not the staging clone path.
    expect(preview.mcpServers[0].summary).toBe(
      `node ${path.join(pluginsDir(), "demo-plugin", "server.js")}`
    );

    // Cancelling after preview = nothing written anywhere.
    expect(await pathExists(path.join(pluginsDir(), "demo-plugin"))).toBe(false);
    expect(await registry()).toEqual([]);
    expect(await stagingLeftovers()).toEqual([]);

    const entry = await service.install({ source: preview.source, expectedSha: preview.lockedSha });
    expect(entry.name).toBe("demo-plugin");
    expect(entry.lockedSha).toBe(head);

    const installedDir = path.join(pluginsDir(), "demo-plugin");
    expect(await pathExists(path.join(installedDir, "plugin.json"))).toBe(true);
    // Plain content snapshot: provenance lives in the registry, not .git.
    expect(await pathExists(path.join(installedDir, ".git"))).toBe(false);
    expect(await registry()).toHaveLength(1);
    expect((await registry())[0]).toMatchObject({
      name: "demo-plugin",
      lockedSha: head,
      scope: "global",
    });
    expect(await stagingLeftovers()).toEqual([]);

    const items = await service.list();
    expect(items).toHaveLength(1);
    expect(items[0]).toMatchObject({
      name: "demo-plugin",
      managed: true,
      present: true,
      skillCount: 1,
      mcpServerCount: 1,
      lockedSha: head,
    });
  });

  test("never overwrites: registry and directory collisions are clear errors", async () => {
    const preview = await service.preview({ input: remoteDir });
    await service.install({ source: preview.source, expectedSha: preview.lockedSha });

    // Managed entry with the same name.
    await expect(service.preview({ input: remoteDir })).rejects.toThrow(/already installed/);
    await expect(
      service.install({ source: preview.source, expectedSha: preview.lockedSha })
    ).rejects.toThrow(/already installed/);

    // Unmanaged directory at the target path (registry entry removed, dir kept).
    await fsPromises.rm(registryFile(), { force: true });
    await expect(service.preview({ input: remoteDir })).rejects.toThrow(/already exists/);
  });

  test("update: badge on branch movement, atomic swap, lockedSha bump, local edits discarded", async () => {
    const preview = await service.preview({ input: remoteDir });
    await service.install({ source: preview.source, expectedSha: preview.lockedSha });

    expect(await service.checkUpdates()).toEqual([{ name: "demo-plugin", status: "up-to-date" }]);

    await writePluginFixture(remoteDir, { version: "2.0.0" });
    const newHead = await commitAll(remoteDir, "v2");

    expect(await service.checkUpdates()).toEqual([
      { name: "demo-plugin", status: "update-available", remoteSha: newHead },
    ]);

    // Local edits to a managed dir are discarded on update (documented behavior).
    const installedDir = path.join(pluginsDir(), "demo-plugin");
    await fsPromises.writeFile(path.join(installedDir, "local-edit.txt"), "scratch");

    const updated = await service.update({ name: "demo-plugin" });
    expect(updated.lockedSha).toBe(newHead);
    expect(updated.updatedAt).toBeDefined();
    expect(updated.manifest?.version).toBe("2.0.0");
    expect(await pathExists(path.join(installedDir, "local-edit.txt"))).toBe(false);
    expect(((await registry())[0] as { lockedSha: string }).lockedSha).toBe(newHead);
    expect(await stagingLeftovers()).toEqual([]);
  });

  test("tag refs pin; a moved tag reports tag-moved; commit refs report pinned", async () => {
    const firstSha = (await git(remoteDir, "rev-parse", "HEAD")).trim();
    await git(remoteDir, "tag", "v1");

    const tagPreview = await service.preview({ input: remoteDir, ref: "v1" });
    expect(tagPreview.source.refType).toBe("tag");
    expect(tagPreview.lockedSha).toBe(firstSha);
    await service.install({ source: tagPreview.source, expectedSha: tagPreview.lockedSha });

    await writePluginFixture(remoteDir, { version: "2.0.0" });
    await commitAll(remoteDir, "v2");
    await git(remoteDir, "tag", "-f", "v1");

    const checks = await service.checkUpdates();
    expect(checks[0].status).toBe("tag-moved");

    await service.uninstall({ name: "demo-plugin", deletePluginData: false });
    expect(await registry()).toEqual([]);
    expect(await pathExists(path.join(pluginsDir(), "demo-plugin"))).toBe(false);

    // Full-SHA install pins hard: no update checks apply.
    const shaPreview = await service.preview({ input: remoteDir, ref: firstSha });
    expect(shaPreview.source.refType).toBe("commit");
    await service.install({ source: shaPreview.source, expectedSha: firstSha });
    expect(await service.checkUpdates()).toEqual([{ name: "demo-plugin", status: "pinned" }]);
    await expect(service.update({ name: "demo-plugin" })).rejects.toThrow(/pinned/);
  });

  test("update stops the plugin's MCP servers before the old tree moves", async () => {
    // Snapshot which tree is installed at each recycle: the pre-swap stop
    // must observe the OLD tree still intact (a live server losing its files
    // mid-swap on POSIX / holding locks on Windows is the failure mode).
    const observedVersions: Array<string | null> = [];
    const mcpStub = {
      stopServersWithKeyPrefix: async () => {
        try {
          const manifest = JSON.parse(
            await fsPromises.readFile(path.join(pluginsDir(), "demo-plugin", "plugin.json"), "utf8")
          ) as { version: string };
          observedVersions.push(manifest.version);
        } catch {
          observedVersions.push(null);
        }
      },
    } as unknown as MCPServerManager;
    const serviceWithMcp = new AgentPluginInstallService(config, {
      isEnabled: () => true,
      mcpServerManager: mcpStub,
    });

    const preview = await serviceWithMcp.preview({ input: remoteDir });
    await serviceWithMcp.install({ source: preview.source, expectedSha: preview.lockedSha });
    await writePluginFixture(remoteDir, { version: "2.0.0" });
    await commitAll(remoteDir, "v2");

    await serviceWithMcp.update({ name: "demo-plugin" });

    // Two recycles: pre-swap (old tree, servers stopped while their files
    // still exist) and post-promote (new content behind the stable path).
    expect(observedVersions.length).toBe(2);
    expect(observedVersions[0]).toBe("1.0.0");
    expect(observedVersions[1]).toBe("2.0.0");
  });

  test("uninstall completes even when deleting the staged tree fails", async () => {
    const preview = await service.preview({ input: remoteDir });
    await service.install({ source: preview.source, expectedSha: preview.lockedSha });

    // Force the best-effort trash deletion to fail (e.g. a Windows file
    // lock). It must not abort uninstall before override pruning runs.
    const internals = service as unknown as { removeDir: (dir: string) => Promise<void> };
    const removeDirSpy = spyOn(internals, "removeDir").mockImplementationOnce(() =>
      Promise.reject(new Error("EBUSY: resource busy or locked"))
    );
    try {
      await service.uninstall({ name: "demo-plugin", deletePluginData: false });
    } finally {
      removeDirSpy.mockRestore();
    }

    // Uninstall completed: registry entry + container dir gone; the staged
    // tree remains under staging for stale-dir reclamation.
    expect(await registry()).toEqual([]);
    expect(await pathExists(path.join(pluginsDir(), "demo-plugin"))).toBe(false);
    expect((await stagingLeftovers()).some((name) => name.startsWith("trash-"))).toBe(true);

    // And reinstall is not blocked by leftover state.
    const preview2 = await service.preview({ input: remoteDir });
    const entry = await service.install({
      source: preview2.source,
      expectedSha: preview2.lockedSha,
    });
    expect(entry.name).toBe("demo-plugin");
  });

  test("uninstall re-invalidates MCP servers after the tree is removed", async () => {
    // A getToolsForWorkspace that starts right after the pre-rename stop can
    // discover the plugin before the rename and start a server from the
    // removed tree; the post-removal invalidation must catch it. Snapshot
    // the tree state at each recycle: first stop sees the tree, second stop
    // must run after it is gone.
    const treeStates: boolean[] = [];
    const mcpStub = {
      stopServersWithKeyPrefix: async () => {
        treeStates.push(await pathExists(path.join(pluginsDir(), "demo-plugin")));
      },
    } as unknown as MCPServerManager;
    const serviceWithMcp = new AgentPluginInstallService(config, {
      isEnabled: () => true,
      mcpServerManager: mcpStub,
    });

    const preview = await serviceWithMcp.preview({ input: remoteDir });
    await serviceWithMcp.install({ source: preview.source, expectedSha: preview.lockedSha });
    await serviceWithMcp.uninstall({ name: "demo-plugin", deletePluginData: false });

    expect(treeStates).toEqual([true, false]);
  });

  test("uninstall aborts intact when pruning enumeration fails (pre-commit)", async () => {
    let stops = 0;
    const mcpStub = {
      stopServersWithKeyPrefix: () => {
        stops += 1;
        return Promise.resolve();
      },
    } as unknown as MCPServerManager;
    // An overrides service makes uninstall enumerate workspace metadata (the
    // only pruning step that can fail wholesale, outside the per-workspace
    // catch). That enumeration must happen BEFORE anything commits: a
    // post-commit failure would strand stale enabled-server overrides with
    // no Settings row left to retry from, and a reinstall (same instance ID)
    // would silently re-enable those servers.
    const overridesStub = {
      prunePluginOverrideKeys: () => Promise.resolve(),
    };
    const serviceWithMcp = new AgentPluginInstallService(config, {
      isEnabled: () => true,
      mcpServerManager: mcpStub,
      workspaceMcpOverridesService: overridesStub as unknown as WorkspaceMcpOverridesService,
    });

    const preview = await serviceWithMcp.preview({ input: remoteDir });
    await serviceWithMcp.install({ source: preview.source, expectedSha: preview.lockedSha });

    stops = 0;
    const metadataSpy = spyOn(config, "getAllWorkspaceMetadata").mockImplementationOnce(() =>
      Promise.reject(new Error("metadata enumeration failed"))
    );
    try {
      await expect(
        serviceWithMcp.uninstall({ name: "demo-plugin", deletePluginData: false })
      ).rejects.toThrow(/metadata enumeration failed/);
    } finally {
      metadataSpy.mockRestore();
    }

    // Nothing was committed and no servers were stopped: the install is fully
    // intact and the row remains, so the user can simply retry.
    expect(stops).toBe(0);
    expect(await registry()).toHaveLength(1);
    expect(await pathExists(path.join(pluginsDir(), "demo-plugin", "plugin.json"))).toBe(true);

    // The retry completes the uninstall, including both invalidations.
    await serviceWithMcp.uninstall({ name: "demo-plugin", deletePluginData: false });
    expect(stops).toBe(2);
    expect(await registry()).toEqual([]);
  });

  test("failed per-workspace prunes persist a tombstone that gates reinstall and self-heals", async () => {
    const instanceId = computePluginInstanceId(path.join(pluginsDir(), "demo-plugin"));
    const serverKey = `plugin:${instanceId}:echo`;

    // One local workspace with the plugin's server enabled; its override
    // file is temporarily unwritable.
    let overridesBroken = true;
    let storedOverrides: Record<string, unknown> = { enabledServers: [serverKey] };
    const overridesStub = {
      prunePluginOverrideKeys: (_id: string, keyPrefix: string) => {
        if (overridesBroken) {
          return Promise.reject(new Error("checkout unavailable"));
        }
        storedOverrides = {
          enabledServers: (storedOverrides.enabledServers as string[]).filter(
            (key) => !key.startsWith(keyPrefix)
          ),
        };
        return Promise.resolve();
      },
    };
    const serviceWithOverrides = new AgentPluginInstallService(config, {
      isEnabled: () => true,
      workspaceMcpOverridesService: overridesStub as unknown as WorkspaceMcpOverridesService,
    });
    const metadataSpy = spyOn(config, "getAllWorkspaceMetadata").mockImplementation(() =>
      Promise.resolve([{ id: "ws-1", runtimeConfig: { type: "local" } }] as unknown as Awaited<
        ReturnType<Config["getAllWorkspaceMetadata"]>
      >)
    );

    try {
      const preview = await serviceWithOverrides.preview({ input: remoteDir });
      await serviceWithOverrides.install({
        source: preview.source,
        expectedSha: preview.lockedSha,
      });
      await serviceWithOverrides.uninstall({ name: "demo-plugin", deletePluginData: false });

      // Uninstall committed, but the failed prune left a persisted tombstone.
      expect(await registry()).toEqual([]);
      const doc = JSON.parse(await fsPromises.readFile(registryFile(), "utf8")) as {
        pendingOverridePrunes?: Array<{ prefix: string; workspaceIds: string[] }>;
      };
      expect(doc.pendingOverridePrunes).toEqual([
        { prefix: `plugin:${instanceId}:`, workspaceIds: ["ws-1"] },
      ]);

      // Reinstalling the same name is gated while the stale override remains:
      // the same instance ID would silently re-enable the server.
      const preview2 = await serviceWithOverrides.preview({ input: remoteDir });
      await expect(
        serviceWithOverrides.install({ source: preview2.source, expectedSha: preview2.lockedSha })
      ).rejects.toThrow(/could not clean up its workspace MCP overrides/);

      // Once the workspace is reachable again, the retry (section open or the
      // install gate itself) prunes the override and unblocks reinstall.
      overridesBroken = false;
      const entry = await serviceWithOverrides.install({
        source: preview2.source,
        expectedSha: preview2.lockedSha,
      });
      expect(entry.name).toBe("demo-plugin");
      expect(storedOverrides.enabledServers ?? []).toEqual([]);
      const docAfter = JSON.parse(await fsPromises.readFile(registryFile(), "utf8")) as {
        pendingOverridePrunes?: unknown;
      };
      expect(docAfter.pendingOverridePrunes).toBeUndefined();
    } finally {
      metadataSpy.mockRestore();
    }
  });

  test("tombstone survives even when both the prune and the shrink write fail", async () => {
    const instanceId = computePluginInstanceId(path.join(pluginsDir(), "demo-plugin"));
    const overridesStub = {
      prunePluginOverrideKeys: () => Promise.reject(new Error("checkout unavailable")),
    };
    const serviceWithOverrides = new AgentPluginInstallService(config, {
      isEnabled: () => true,
      workspaceMcpOverridesService: overridesStub as unknown as WorkspaceMcpOverridesService,
    });
    const metadataSpy = spyOn(config, "getAllWorkspaceMetadata").mockImplementation(() =>
      Promise.resolve([{ id: "ws-1", runtimeConfig: { type: "local" } }] as unknown as Awaited<
        ReturnType<Config["getAllWorkspaceMetadata"]>
      >)
    );

    try {
      const preview = await serviceWithOverrides.preview({ input: remoteDir });
      await serviceWithOverrides.install({
        source: preview.source,
        expectedSha: preview.lockedSha,
      });

      // The commit write (which must carry the pessimistic tombstone) runs
      // for real; the post-prune shrink write fails.
      const internals = serviceWithOverrides as unknown as {
        writeRegistry: (envelope: Record<string, unknown>, entries: unknown[]) => Promise<void>;
      };
      const originalWrite = internals.writeRegistry.bind(serviceWithOverrides);
      let writeCalls = 0;
      const writeSpy = spyOn(internals, "writeRegistry").mockImplementation(
        (envelope: Record<string, unknown>, entries: unknown[]) => {
          writeCalls += 1;
          if (writeCalls === 2) {
            return Promise.reject(new Error("ENOSPC: no space left on device"));
          }
          return originalWrite(envelope, entries);
        }
      );
      try {
        await serviceWithOverrides.uninstall({ name: "demo-plugin", deletePluginData: false });
      } finally {
        writeSpy.mockRestore();
      }

      // The durable record is the COMMIT write's pessimistic tombstone: even
      // with the shrink write lost, reinstall stays gated.
      const doc = JSON.parse(await fsPromises.readFile(registryFile(), "utf8")) as {
        pendingOverridePrunes?: Array<{ prefix: string; workspaceIds: string[] }>;
      };
      expect(doc.pendingOverridePrunes).toEqual([
        { prefix: `plugin:${instanceId}:`, workspaceIds: ["ws-1"] },
      ]);
      const preview2 = await serviceWithOverrides.preview({ input: remoteDir });
      await expect(
        serviceWithOverrides.install({ source: preview2.source, expectedSha: preview2.lockedSha })
      ).rejects.toThrow(/could not clean up its workspace MCP overrides/);
    } finally {
      metadataSpy.mockRestore();
    }
  });

  test("tombstones for deleted workspaces retire instead of blocking reinstall forever", async () => {
    const instanceId = computePluginInstanceId(path.join(pluginsDir(), "demo-plugin"));
    // Overrides service that permanently throws (as it would for a workspace
    // that no longer exists in config).
    const overridesStub = {
      prunePluginOverrideKeys: () => Promise.reject(new Error("Workspace metadata not found")),
    };
    const serviceWithOverrides = new AgentPluginInstallService(config, {
      isEnabled: () => true,
      workspaceMcpOverridesService: overridesStub as unknown as WorkspaceMcpOverridesService,
    });

    // Seed a tombstone naming a workspace that is not in config anymore.
    await fsPromises.writeFile(
      registryFile(),
      JSON.stringify({
        plugins: [],
        pendingOverridePrunes: [{ prefix: `plugin:${instanceId}:`, workspaceIds: ["ws-deleted"] }],
      })
    );

    // The deleted workspace can never reactivate anything, so the reinstall
    // gate drops it instead of blocking forever on its permanent failure.
    const preview = await serviceWithOverrides.preview({ input: remoteDir });
    const entry = await serviceWithOverrides.install({
      source: preview.source,
      expectedSha: preview.lockedSha,
    });
    expect(entry.name).toBe("demo-plugin");
    const doc = JSON.parse(await fsPromises.readFile(registryFile(), "utf8")) as {
      pendingOverridePrunes?: unknown;
    };
    expect(doc.pendingOverridePrunes).toBeUndefined();
  });

  test("tombstone rewrites preserve unknown variants and fields from newer builds", async () => {
    // A newer build's tombstone variant (unrecognized shape) plus a
    // recognized tombstone carrying an unknown field, for an unrelated
    // prefix whose workspace no longer exists (so it retires by itself).
    const futureVariant = { kind: "future-cleanup", payload: { x: 1 } };
    const foreignPrune = {
      prefix: "plugin:0000000000000000:",
      workspaceIds: ["ws-gone"],
      reason: "future-field",
    };
    await fsPromises.writeFile(
      registryFile(),
      JSON.stringify({ plugins: [], pendingOverridePrunes: [futureVariant, foreignPrune] })
    );

    // A full uninstall cycle rewrites pendingOverridePrunes twice (commit +
    // shrink); the unknown variant must ride through verbatim.
    const preview = await service.preview({ input: remoteDir });
    await service.install({ source: preview.source, expectedSha: preview.lockedSha });
    await service.uninstall({ name: "demo-plugin", deletePluginData: false });

    const doc = JSON.parse(await fsPromises.readFile(registryFile(), "utf8")) as {
      pendingOverridePrunes: unknown[];
    };
    expect(doc.pendingOverridePrunes).toContainEqual(futureVariant);
    // The recognized foreign tombstone kept its unknown field (ws-gone is not
    // in this config, so a retry would retire it — but no retry ran for it
    // during uninstall, which only touches its own prefix).
    expect(doc.pendingOverridePrunes).toContainEqual(foreignPrune);
  });

  test("tombstone retries on list are serialized with registry mutations", async () => {
    const instanceId = computePluginInstanceId(path.join(pluginsDir(), "other-name"));
    // A tombstone whose prune blocks until released, so a mutation can be
    // issued while the retry's read-modify-write is in flight.
    let releasePrune!: () => void;
    const pruneGate = new Promise<void>((resolve) => {
      releasePrune = resolve;
    });
    const overridesStub = {
      prunePluginOverrideKeys: async () => {
        await pruneGate;
      },
    };
    const serviceWithOverrides = new AgentPluginInstallService(config, {
      isEnabled: () => true,
      workspaceMcpOverridesService: overridesStub as unknown as WorkspaceMcpOverridesService,
    });
    const metadataSpy = spyOn(config, "getAllWorkspaceMetadata").mockImplementation(() =>
      Promise.resolve([{ id: "ws-1", runtimeConfig: { type: "local" } }] as unknown as Awaited<
        ReturnType<Config["getAllWorkspaceMetadata"]>
      >)
    );
    await fsPromises.writeFile(
      registryFile(),
      JSON.stringify({
        plugins: [],
        pendingOverridePrunes: [{ prefix: `plugin:${instanceId}:`, workspaceIds: ["ws-1"] }],
      })
    );

    try {
      // list() starts the retry, which parks inside the (locked) prune.
      const listPromise = serviceWithOverrides.list();
      await new Promise((resolve) => setTimeout(resolve, 10));

      // A concurrent install must serialize AFTER the retry's write: without
      // the shared mutation lock, the retry's stale snapshot would erase the
      // newly installed entry.
      const preview = await serviceWithOverrides.preview({ input: remoteDir });
      const installPromise = serviceWithOverrides.install({
        source: preview.source,
        expectedSha: preview.lockedSha,
      });
      await new Promise((resolve) => setTimeout(resolve, 10));
      releasePrune();

      await listPromise;
      await installPromise;

      // The installed entry survived the retry's write, and the tombstone cleared.
      const doc = JSON.parse(await fsPromises.readFile(registryFile(), "utf8")) as {
        plugins: Array<{ name: string }>;
        pendingOverridePrunes?: unknown;
      };
      expect(doc.plugins.map((entry) => entry.name)).toEqual(["demo-plugin"]);
      expect(doc.pendingOverridePrunes).toBeUndefined();
    } finally {
      metadataSpy.mockRestore();
    }
  });

  test("uninstall stages plugin-data before committing when deletion is requested", async () => {
    const preview = await service.preview({ input: remoteDir });
    await service.install({ source: preview.source, expectedSha: preview.lockedSha });

    const instanceId = computePluginInstanceId(path.join(pluginsDir(), "demo-plugin"));
    const dataPath = getPluginDataPath(muxRoot, instanceId);
    await fsPromises.mkdir(dataPath, { recursive: true });
    await fsPromises.writeFile(path.join(dataPath, "state.json"), "{}");

    // Make the data dir unstageable: rename mutates the parent (plugin-data/).
    await fsPromises.chmod(path.join(muxRoot, "plugin-data"), 0o555);
    try {
      await expect(
        service.uninstall({ name: "demo-plugin", deletePluginData: true })
      ).rejects.toThrow(/Failed to remove the plugin data/);
    } finally {
      await fsPromises.chmod(path.join(muxRoot, "plugin-data"), 0o755);
    }

    // The uninstall did not commit: the Settings row survives so the user can
    // retry the requested cleanup, and nothing was half-removed.
    expect(await registry()).toHaveLength(1);
    expect(await pathExists(path.join(pluginsDir(), "demo-plugin", "plugin.json"))).toBe(true);
    expect(await pathExists(path.join(dataPath, "state.json"))).toBe(true);

    // Retry succeeds and honors the data-deletion request.
    await service.uninstall({ name: "demo-plugin", deletePluginData: true });
    expect(await registry()).toEqual([]);
    expect(await pathExists(dataPath)).toBe(false);
  });

  test("uninstall preserves plugin-data by default and deletes it when asked", async () => {
    const preview = await service.preview({ input: remoteDir });
    await service.install({ source: preview.source, expectedSha: preview.lockedSha });

    const instanceId = computePluginInstanceId(path.join(pluginsDir(), "demo-plugin"));
    const dataPath = getPluginDataPath(muxRoot, instanceId);
    await fsPromises.mkdir(dataPath, { recursive: true });
    await fsPromises.writeFile(path.join(dataPath, "state.json"), "{}");

    await service.uninstall({ name: "demo-plugin", deletePluginData: false });
    expect(await pathExists(dataPath)).toBe(true);

    const preview2 = await service.preview({ input: remoteDir });
    await service.install({ source: preview2.source, expectedSha: preview2.lockedSha });
    await service.uninstall({ name: "demo-plugin", deletePluginData: true });
    expect(await pathExists(dataPath)).toBe(false);
  });

  test("failure paths leave no partial state", async () => {
    // Unreachable remote.
    await expect(service.preview({ input: "/nonexistent/repo/path" })).rejects.toThrow(
      /Could not reach/
    );

    // Repo that is not a plugin.
    const notPlugin = await fsPromises.mkdtemp(path.join(os.tmpdir(), "mux-not-plugin-"));
    try {
      await initRemote(notPlugin);
      await fsPromises.writeFile(path.join(notPlugin, "README.md"), "hi");
      await commitAll(notPlugin, "init");
      await expect(service.preview({ input: notPlugin })).rejects.toThrow(/No plugin\.json/);

      // Claude Code collection → clear message naming the limitation.
      await fsPromises.mkdir(path.join(notPlugin, ".claude-plugin"), { recursive: true });
      await fsPromises.writeFile(path.join(notPlugin, ".claude-plugin", "plugin.json"), "{}");
      await commitAll(notPlugin, "claude");
      await expect(service.preview({ input: notPlugin })).rejects.toThrow(/Claude Code/);
    } finally {
      await fsPromises.rm(notPlugin, { recursive: true, force: true });
    }

    // Subpath installs are parsed but rejected in v1.
    await expect(service.preview({ input: remoteDir, subpath: "sub" })).rejects.toThrow(/v2/);

    // Unknown ref.
    await expect(service.preview({ input: remoteDir, ref: "does-not-exist" })).rejects.toThrow(
      /not found on the remote/
    );

    // Nothing was written by any of the failures above.
    expect(await pathExists(path.join(pluginsDir(), "demo-plugin"))).toBe(false);
    expect(await registry()).toEqual([]);
    expect(await stagingLeftovers()).toEqual([]);

    // Disabled experiment gates every method.
    enabled = false;
    await expect(service.preview({ input: remoteDir })).rejects.toThrow(/not enabled/);
    await expect(service.list()).rejects.toThrow(/not enabled/);
    enabled = true;

    // Remote moved between preview and install: the exact consented SHA is
    // installed (never the newer unreviewed tip). If the SHA became
    // unfetchable, install fails with "moved since the preview" instead.
    const preview = await service.preview({ input: remoteDir });
    await writePluginFixture(remoteDir, { version: "9.9.9" });
    await commitAll(remoteDir, "moved");
    const entry = await service.install({
      source: preview.source,
      expectedSha: preview.lockedSha,
    });
    expect(entry.lockedSha).toBe(preview.lockedSha);
    expect(entry.manifest?.version).toBe("1.0.0");
    const installedManifest = JSON.parse(
      await fsPromises.readFile(path.join(pluginsDir(), "demo-plugin", "plugin.json"), "utf8")
    ) as { version: string };
    expect(installedManifest.version).toBe("1.0.0");
  });

  test("registry rewrites preserve entries and fields from newer builds", async () => {
    // Simulate a newer build's registry content: an unknown source kind and
    // an extra per-entry field this build's schemas do not know about.
    const futureEntry = {
      name: "future-plugin",
      scope: "global",
      source: { type: "archive", url: "https://example.com/p.tgz", sha256: "ab" },
      lockedSha: "b".repeat(40),
      installedAt: "2026-09-01T00:00:00.000Z",
      futureField: { nested: true },
    };
    await fsPromises.writeFile(registryFile(), JSON.stringify({ plugins: [futureEntry] }));

    // Full lifecycle on this build: install, update, uninstall of a git plugin.
    const preview = await service.preview({ input: remoteDir });
    await service.install({ source: preview.source, expectedSha: preview.lockedSha });
    await writePluginFixture(remoteDir, { version: "2.0.0" });
    await commitAll(remoteDir, "v2");
    await service.update({ name: "demo-plugin" });
    await service.uninstall({ name: "demo-plugin", deletePluginData: false });

    // The unrecognized entry survived every rewrite verbatim.
    expect(await registry()).toEqual([futureEntry]);
    // And it never surfaced as a managed row this build could mutate.
    expect((await service.list()).map((item) => item.name)).not.toContain("future-plugin");
  });

  test("update preserves unknown nested fields inside the entry's source and manifest", async () => {
    const preview = await service.preview({ input: remoteDir });
    await service.install({ source: preview.source, expectedSha: preview.lockedSha });

    // A newer build stored extra metadata INSIDE the git source and manifest
    // of this entry; a shallow merge of the Zod-parsed entry would strip it.
    const onDisk = JSON.parse(await fsPromises.readFile(registryFile(), "utf8")) as {
      plugins: Array<Record<string, unknown>>;
    };
    (onDisk.plugins[0].source as Record<string, unknown>).integrity = "sha256-future";
    onDisk.plugins[0].manifest = {
      ...(onDisk.plugins[0].manifest as Record<string, unknown>),
      icon: "sparkles",
    };
    await fsPromises.writeFile(registryFile(), JSON.stringify(onDisk));

    await writePluginFixture(remoteDir, { version: "2.0.0" });
    const newHead = await commitAll(remoteDir, "v2");
    await service.update({ name: "demo-plugin" });

    const after = JSON.parse(await fsPromises.readFile(registryFile(), "utf8")) as {
      plugins: Array<{
        lockedSha: string;
        source: Record<string, unknown>;
        manifest: Record<string, unknown>;
      }>;
    };
    expect(after.plugins[0].lockedSha).toBe(newHead);
    // Owned fields updated…
    expect(after.plugins[0].manifest.version).toBe("2.0.0");
    // …unknown nested metadata untouched.
    expect(after.plugins[0].source.integrity).toBe("sha256-future");
    expect(after.plugins[0].manifest.icon).toBe("sparkles");
  });

  test("managed list rows keep registry identity when the manifest name drifts", async () => {
    const preview = await service.preview({ input: remoteDir });
    await service.install({ source: preview.source, expectedSha: preview.lockedSha });

    // Local edit renames the manifest to another VALID plugin name.
    const manifestPath = path.join(pluginsDir(), "demo-plugin", "plugin.json");
    const manifest = JSON.parse(await fsPromises.readFile(manifestPath, "utf8")) as {
      name: string;
    };
    manifest.name = "impostor";
    await fsPromises.writeFile(manifestPath, JSON.stringify(manifest));

    // The row keeps the registry name (update/uninstall look up by it) and
    // surfaces the drift; the operations remain usable.
    const items = await service.list();
    const row = items.find((item) => item.managed);
    expect(row?.name).toBe("demo-plugin");
    expect(row?.description).toContain("impostor");
    await service.uninstall({ name: "demo-plugin", deletePluginData: false });
    expect(await registry()).toEqual([]);
  });

  test("mutations refuse a corrupted registry file instead of orphaning entries", async () => {
    const preview = await service.preview({ input: remoteDir });
    await service.install({ source: preview.source, expectedSha: preview.lockedSha });

    // Corrupt the registry file (invalid JSON, not just an invalid entry).
    await fsPromises.writeFile(registryFile(), "{ not json");

    // Reads stay lenient: the section still renders, dirs show unmanaged.
    const items = await service.list();
    expect(items).toHaveLength(1);
    expect(items[0]).toMatchObject({ name: "demo-plugin", managed: false });

    // Mutations refuse with a repair message — treating the corrupt file as
    // empty would let this install rewrite it with one entry, permanently
    // orphaning everything previously managed.
    const remote2 = await fsPromises.mkdtemp(path.join(os.tmpdir(), "mux-plugin-remote2-"));
    try {
      await initRemote(remote2);
      await writePluginFixture(remote2);
      await fsPromises.writeFile(
        path.join(remote2, "plugin.json"),
        JSON.stringify({
          $schema: AGENT_PLUGIN_SCHEMA_ID_1_0_0,
          name: "other-plugin",
          version: "1.0.0",
        })
      );
      await commitAll(remote2, "init");
      await expect(service.preview({ input: remote2 })).rejects.toThrow(/corrupted/);
    } finally {
      await fsPromises.rm(remote2, { recursive: true, force: true });
    }
    await expect(
      service.uninstall({ name: "demo-plugin", deletePluginData: false })
    ).rejects.toThrow(/corrupted/);
    await expect(service.update({ name: "demo-plugin" })).rejects.toThrow(/corrupted/);

    // The corrupt file was never rewritten.
    expect(await fsPromises.readFile(registryFile(), "utf8")).toBe("{ not json");

    // Structurally invalid envelopes (parseable JSON without a plugins
    // array) are corruption too — {} or {"plugins": null} must not let a
    // mutation rewrite the registry down to a single entry.
    for (const invalidEnvelope of ["{}", '{ "plugins": null }', "[]"]) {
      await fsPromises.writeFile(registryFile(), invalidEnvelope);
      await expect(
        service.uninstall({ name: "demo-plugin", deletePluginData: false })
      ).rejects.toThrow(/corrupted/);
      expect(await fsPromises.readFile(registryFile(), "utf8")).toBe(invalidEnvelope);
    }
  });

  test("install refuses names owned by entries this build cannot parse", async () => {
    // A newer build's entry (unknown source kind) named demo-plugin, with no
    // directory on disk: this build must still treat the name as taken —
    // installing over it would filter the raw entry out and replace it.
    await fsPromises.writeFile(
      registryFile(),
      JSON.stringify({
        plugins: [
          {
            name: "demo-plugin",
            scope: "global",
            source: { type: "archive", url: "https://example.com/p.tgz" },
            lockedSha: "c".repeat(40),
            installedAt: "2026-09-01T00:00:00.000Z",
          },
        ],
      })
    );

    await expect(service.preview({ input: remoteDir })).rejects.toThrow(/already installed/);
    // The unrecognized entry is untouched.
    expect(await registry()).toHaveLength(1);
  });

  test("mutations refuse an unreadable registry file (non-ENOENT read failure)", async () => {
    const preview = await service.preview({ input: remoteDir });
    await service.install({ source: preview.source, expectedSha: preview.lockedSha });

    await fsPromises.chmod(registryFile(), 0o000);
    try {
      // Reads degrade to unmanaged; mutations refuse instead of letting the
      // atomic write replace the unreadable file and erase its entries.
      const items = await service.list();
      expect(items[0]).toMatchObject({ name: "demo-plugin", managed: false });
      await expect(
        service.uninstall({ name: "demo-plugin", deletePluginData: false })
      ).rejects.toThrow(/cannot be read/);
      await expect(service.update({ name: "demo-plugin" })).rejects.toThrow(/cannot be read/);
    } finally {
      await fsPromises.chmod(registryFile(), 0o644);
    }

    // Registry intact once readable again.
    expect(await registry()).toHaveLength(1);
    expect((await service.list())[0]).toMatchObject({ name: "demo-plugin", managed: true });
  });

  test("registry rewrites preserve unknown top-level envelope fields", async () => {
    // A newer build added top-level registry metadata alongside `plugins`.
    await fsPromises.writeFile(
      registryFile(),
      JSON.stringify({ registryVersion: 2, migrationState: { seeded: true }, plugins: [] })
    );

    const preview = await service.preview({ input: remoteDir });
    await service.install({ source: preview.source, expectedSha: preview.lockedSha });
    await writePluginFixture(remoteDir, { version: "2.0.0" });
    await commitAll(remoteDir, "v2");
    await service.update({ name: "demo-plugin" });
    await service.uninstall({ name: "demo-plugin", deletePluginData: false });

    // Every mutation rewrote only `plugins`; the envelope survived verbatim.
    const after = JSON.parse(await fsPromises.readFile(registryFile(), "utf8")) as Record<
      string,
      unknown
    >;
    expect(after.registryVersion).toBe(2);
    expect(after.migrationState).toEqual({ seeded: true });
    expect(after.plugins).toEqual([]);
  });

  test("update recycles MCP servers even when the registry write fails post-promote", async () => {
    let stops = 0;
    const mcpStub = {
      stopServersWithKeyPrefix: () => {
        stops += 1;
        return Promise.resolve();
      },
    } as unknown as MCPServerManager;
    const serviceWithMcp = new AgentPluginInstallService(config, {
      isEnabled: () => true,
      mcpServerManager: mcpStub,
    });

    const preview = await serviceWithMcp.preview({ input: remoteDir });
    await serviceWithMcp.install({ source: preview.source, expectedSha: preview.lockedSha });
    await writePluginFixture(remoteDir, { version: "2.0.0" });
    await commitAll(remoteDir, "v2");

    stops = 0;
    const internals = serviceWithMcp as unknown as {
      writeRegistry: (envelope: Record<string, unknown>, entries: unknown[]) => Promise<void>;
    };
    const writeSpy = spyOn(internals, "writeRegistry").mockImplementationOnce(() =>
      Promise.reject(new Error("ENOSPC: no space left on device"))
    );
    try {
      await expect(serviceWithMcp.update({ name: "demo-plugin" })).rejects.toThrow(/ENOSPC/);
    } finally {
      writeSpy.mockRestore();
    }

    // Both recycles ran (pre-swap + post-promote) despite the failed write:
    // the tree already swapped, so a server started from the replaced tree
    // must not be retained.
    expect(stops).toBe(2);
    // Stale lockedSha keeps the badge; a retry self-heals.
    expect(((await registry())[0] as { lockedSha: string }).lockedSha).toBe(preview.lockedSha);
    const retried = await serviceWithMcp.update({ name: "demo-plugin" });
    expect(retried.manifest?.version).toBe("2.0.0");
  });

  test("update rejects a tracked ref whose kind changed on the remote", async () => {
    await git(remoteDir, "branch", "track");
    const preview = await service.preview({ input: remoteDir, ref: "track" });
    expect(preview.source.refType).toBe("branch");
    await service.install({ source: preview.source, expectedSha: preview.lockedSha });

    // The tracked branch is deleted and a tag with the same name appears,
    // pointing at newer content.
    await writePluginFixture(remoteDir, { version: "2.0.0" });
    const newHead = await commitAll(remoteDir, "v2");
    await git(remoteDir, "branch", "-D", "track");
    await git(remoteDir, "tag", "track", newHead);

    // A stale Update click must not install tag content while the registry
    // still claims a branch.
    await expect(service.update({ name: "demo-plugin" })).rejects.toThrow(/now a tag/);
    expect(((await registry())[0] as { lockedSha: string }).lockedSha).toBe(preview.lockedSha);
  });

  test("registry survives config.json rewrites and drops traversal names on read", async () => {
    const preview = await service.preview({ input: remoteDir });
    await service.install({ source: preview.source, expectedSha: preview.lockedSha });

    // Registry is a standalone file: rebuilding config.json (what older
    // builds do on every save) cannot drop it.
    await config.editConfig((cfg) => {
      cfg.defaultModel = "openai:gpt-4o";
      return cfg;
    });
    expect(await registry()).toHaveLength(1);
    expect((await service.list())[0]).toMatchObject({ name: "demo-plugin", managed: true });

    // Malicious/corrupt entries with traversal names must never reach the
    // filesystem layer: uninstall of ".." would delete the entire mux root.
    const onDisk = JSON.parse(await fsPromises.readFile(registryFile(), "utf8")) as {
      plugins: unknown[];
    };
    const template = onDisk.plugins[0] as Record<string, unknown>;
    onDisk.plugins.push({ ...template, name: ".." }, { ...template, name: "a/../b" });
    await fsPromises.writeFile(registryFile(), JSON.stringify(onDisk));

    const items = await service.list();
    expect(items.map((item) => item.name)).toEqual(["demo-plugin"]);
    await expect(service.uninstall({ name: "..", deletePluginData: false })).rejects.toThrow(
      /not a managed plugin/
    );
  });

  test("uninstall restores the registry entry when the tree cannot be staged out", async () => {
    const preview = await service.preview({ input: remoteDir });
    await service.install({ source: preview.source, expectedSha: preview.lockedSha });

    // Force the stage-out rename to fail by making the container read-only
    // (rename mutates the parent directory).
    await fsPromises.chmod(pluginsDir(), 0o555);
    try {
      await expect(
        service.uninstall({ name: "demo-plugin", deletePluginData: false })
      ).rejects.toThrow(/Failed to remove the plugin directory/);
    } finally {
      await fsPromises.chmod(pluginsDir(), 0o755);
    }

    // No partial state: the install is fully intact and still managed.
    expect(await registry()).toHaveLength(1);
    expect(await pathExists(path.join(pluginsDir(), "demo-plugin", "plugin.json"))).toBe(true);
    expect((await service.list())[0]).toMatchObject({ name: "demo-plugin", managed: true });

    // And the retry succeeds once the obstruction is gone.
    await service.uninstall({ name: "demo-plugin", deletePluginData: false });
    expect(await registry()).toEqual([]);
  });

  test("install rolls back the promoted dir when the registry write fails", async () => {
    // A getToolsForWorkspace running during the promote↔rollback window can
    // have discovered the briefly-visible tree; the rollback must invalidate
    // the plugin prefix (like update/uninstall) so no server survives from
    // the deleted, unregistered tree.
    const stoppedPrefixes: string[] = [];
    const mcpStub = {
      stopServersWithKeyPrefix: (prefix: string) => {
        stoppedPrefixes.push(prefix);
        return Promise.resolve();
      },
    };
    const serviceWithMcp = new AgentPluginInstallService(config, {
      isEnabled: () => true,
      mcpServerManager: mcpStub as unknown as MCPServerManager,
    });
    const preview = await serviceWithMcp.preview({ input: remoteDir });

    const internals = serviceWithMcp as unknown as {
      writeRegistry: (envelope: Record<string, unknown>, entries: unknown[]) => Promise<void>;
    };
    const writeSpy = spyOn(internals, "writeRegistry").mockImplementationOnce(() =>
      Promise.reject(new Error("ENOSPC: no space left on device"))
    );
    try {
      await expect(
        serviceWithMcp.install({ source: preview.source, expectedSha: preview.lockedSha })
      ).rejects.toThrow(/persist the plugin registry/);
    } finally {
      writeSpy.mockRestore();
    }

    // No partial state: the promoted dir was rolled back and any server
    // started from the briefly-visible tree was invalidated.
    expect(await pathExists(path.join(pluginsDir(), "demo-plugin"))).toBe(false);
    expect(await stagingLeftovers()).toEqual([]);
    const instanceId = computePluginInstanceId(path.join(pluginsDir(), "demo-plugin"));
    expect(stoppedPrefixes).toEqual([`plugin:${instanceId}:`]);

    // The retry of the same consented install succeeds.
    const entry = await serviceWithMcp.install({
      source: preview.source,
      expectedSha: preview.lockedSha,
    });
    expect(entry.name).toBe("demo-plugin");
    expect(await registry()).toHaveLength(1);
  });

  test("added plugin override keys are validated against discovered servers", async () => {
    // The overrides revision is content-derived, so a dialog opened before an
    // uninstall (overrides {}) sees an unchanged revision after it — only a
    // discovery check at save time can reject the ghost row's new key. The
    // source is DISCOVERED server keys (managed + project + ~/.agents +
    // unmanaged containers), not the managed registry, so non-managed plugin
    // servers stay enableable.
    const discoveredKey = "plugin:abc123:echo";
    const validator = buildAddedPluginKeyValidator(() => Promise.resolve(new Set([discoveredKey])));

    // Discovered server (managed or not): addition accepted.
    await validator({}, { enabledServers: [discoveredKey] });

    // Undiscovered plugin key: NEW key rejected (enabled list, allowlist alike)…
    await expect(validator({}, { enabledServers: ["plugin:gone:echo"] })).rejects.toThrow(
      /does not match any available plugin server/
    );
    await expect(validator({}, { toolAllowlist: { "plugin:gone:echo": [] } })).rejects.toThrow(
      /does not match any available plugin server/
    );
    // …while round-tripping an EXISTING stale key and non-plugin keys stays allowed.
    await validator(
      { enabledServers: ["plugin:gone:echo"] },
      { enabledServers: ["plugin:gone:echo"] }
    );
    await validator({}, { enabledServers: ["ordinary-server"] });

    // Discovery failure → additions rejected (never accept unverifiable keys).
    const failingValidator = buildAddedPluginKeyValidator(() =>
      Promise.reject(new Error("discovery unavailable"))
    );
    await expect(failingValidator({}, { enabledServers: [discoveredKey] })).rejects.toThrow(
      /does not match any available plugin server/
    );
  });

  test("falls back to a branch clone when the remote refuses direct SHA fetches", async () => {
    // GitHub-style servers can reject fetching unadvertised objects; simulate
    // by pointing the exact-SHA fetch at a file:// remote with SHA-in-want
    // disabled, so only the advertised branch tip is fetchable.
    await git(remoteDir, "config", "uploadpack.allowAnySHA1InWant", "false");
    await git(remoteDir, "config", "uploadpack.allowReachableSHA1InWant", "false");
    const fileUrl = `file://${remoteDir}`;

    const preview = await service.preview({ input: fileUrl });
    const entry = await service.install({ source: preview.source, expectedSha: preview.lockedSha });
    expect(entry.lockedSha).toBe(preview.lockedSha);
    expect(await pathExists(path.join(pluginsDir(), "demo-plugin", "plugin.json"))).toBe(true);
    expect(await stagingLeftovers()).toEqual([]);
  });

  test("list surfaces unmanaged plugin dirs read-only and missing managed installs", async () => {
    // Unmanaged: a directory dropped into the container by hand.
    const unmanagedDir = path.join(pluginsDir(), "handmade");
    await fsPromises.mkdir(unmanagedDir, { recursive: true });
    await fsPromises.writeFile(
      path.join(unmanagedDir, "plugin.json"),
      JSON.stringify({ $schema: AGENT_PLUGIN_SCHEMA_ID_1_0_0, name: "handmade" })
    );

    // Missing managed install: registry entry without a directory.
    const preview = await service.preview({ input: remoteDir });
    await service.install({ source: preview.source, expectedSha: preview.lockedSha });
    await fsPromises.rm(path.join(pluginsDir(), "demo-plugin"), { recursive: true, force: true });

    const items = await service.list();
    expect(items).toHaveLength(2);
    const managed = items.find((item) => item.name === "demo-plugin");
    expect(managed).toMatchObject({ managed: true, present: false, version: "1.0.0" });
    const unmanaged = items.find((item) => item.name === "handmade");
    expect(unmanaged).toMatchObject({ managed: false, present: true });
  });
});
