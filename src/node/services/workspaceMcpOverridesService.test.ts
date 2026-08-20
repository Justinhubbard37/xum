import { describe, it, expect, beforeEach, afterEach } from "bun:test";
import * as fs from "fs/promises";
import * as os from "os";
import * as path from "path";
import { Config } from "@/node/config";
import { createRuntime } from "@/node/runtime/runtimeFactory";
import { execBuffered } from "@/node/utils/runtime/helpers";
import {
  WorkspaceMcpOverridesConflictError,
  WorkspaceMcpOverridesService,
} from "./workspaceMcpOverridesService";

function getWorkspacePath(args: {
  srcDir: string;
  projectName: string;
  workspaceName: string;
}): string {
  return path.join(args.srcDir, args.projectName, args.workspaceName);
}

async function pathExists(filePath: string): Promise<boolean> {
  try {
    await fs.stat(filePath);
    return true;
  } catch {
    return false;
  }
}

describe("WorkspaceMcpOverridesService", () => {
  let tempDir: string;
  let config: Config;

  beforeEach(async () => {
    tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "mux-mcp-overrides-test-"));
    config = new Config(tempDir);
  });

  afterEach(async () => {
    await fs.rm(tempDir, { recursive: true, force: true });
  });

  it("returns empty overrides when no file and no legacy config", async () => {
    const projectPath = "/fake/project";
    const workspaceId = "ws-id";
    const workspaceName = "branch";

    const workspacePath = getWorkspacePath({
      srcDir: config.srcDir,
      projectName: "project",
      workspaceName,
    });
    await fs.mkdir(workspacePath, { recursive: true });

    await config.editConfig((cfg) => {
      cfg.projects.set(projectPath, {
        workspaces: [
          {
            path: workspacePath,
            id: workspaceId,
            name: workspaceName,
            runtimeConfig: { type: "worktree", srcBaseDir: config.srcDir },
          },
        ],
      });
      return cfg;
    });

    const service = new WorkspaceMcpOverridesService(config);
    const { overrides } = await service.getOverridesForWorkspace(workspaceId);

    expect(overrides).toEqual({});
    expect(await pathExists(path.join(workspacePath, ".mux", "mcp.local.jsonc"))).toBe(false);
  });

  it("adds .mux/mcp.local.jsonc to git exclude when writing overrides", async () => {
    const projectPath = "/fake/project";
    const workspaceId = "ws-id";
    const workspaceName = "branch";

    const workspacePath = getWorkspacePath({
      srcDir: config.srcDir,
      projectName: "project",
      workspaceName,
    });
    await fs.mkdir(workspacePath, { recursive: true });

    const runtime = createRuntime({ type: "local" }, { projectPath: workspacePath });
    const gitInitResult = await execBuffered(runtime, "git init", {
      cwd: workspacePath,
      timeout: 10,
    });
    expect(gitInitResult.exitCode).toBe(0);

    await config.editConfig((cfg) => {
      cfg.projects.set(projectPath, {
        workspaces: [
          {
            path: workspacePath,
            id: workspaceId,
            name: workspaceName,
            runtimeConfig: { type: "worktree", srcBaseDir: config.srcDir },
          },
        ],
      });
      return cfg;
    });

    const service = new WorkspaceMcpOverridesService(config);

    const excludePathResult = await execBuffered(runtime, "git rev-parse --git-path info/exclude", {
      cwd: workspacePath,
      timeout: 10,
    });
    expect(excludePathResult.exitCode).toBe(0);

    const excludePathRaw = excludePathResult.stdout.trim();
    expect(excludePathRaw.length).toBeGreaterThan(0);

    const excludePath = path.isAbsolute(excludePathRaw)
      ? excludePathRaw
      : path.join(workspacePath, excludePathRaw);

    const before = (await pathExists(excludePath)) ? await fs.readFile(excludePath, "utf-8") : "";
    expect(before).not.toContain(".mux/mcp.local.jsonc");

    await service.setOverridesForWorkspace(workspaceId, {
      disabledServers: ["server-a"],
    });

    const after = await fs.readFile(excludePath, "utf-8");
    expect(after).toContain(".mux/mcp.local.jsonc");
  });
  it("persists overrides to .mux/mcp.local.jsonc and reads them back", async () => {
    const projectPath = "/fake/project";
    const workspaceId = "ws-id";
    const workspaceName = "branch";

    const workspacePath = getWorkspacePath({
      srcDir: config.srcDir,
      projectName: "project",
      workspaceName,
    });
    await fs.mkdir(workspacePath, { recursive: true });

    await config.editConfig((cfg) => {
      cfg.projects.set(projectPath, {
        workspaces: [
          {
            path: workspacePath,
            id: workspaceId,
            name: workspaceName,
            runtimeConfig: { type: "worktree", srcBaseDir: config.srcDir },
          },
        ],
      });
      return cfg;
    });

    const service = new WorkspaceMcpOverridesService(config);

    await service.setOverridesForWorkspace(workspaceId, {
      disabledServers: ["server-a", "server-a"],
      toolAllowlist: { "server-b": ["tool1", "tool1", ""] },
    });

    const filePath = path.join(workspacePath, ".mux", "mcp.local.jsonc");
    expect(await pathExists(filePath)).toBe(true);

    const roundTrip = await service.getOverridesForWorkspace(workspaceId);
    expect(roundTrip.overrides).toEqual({
      disabledServers: ["server-a"],
      toolAllowlist: { "server-b": ["tool1"] },
    });
  });

  it("rejects saves with a stale revision instead of clobbering newer overrides", async () => {
    const projectPath = "/fake/project";
    const workspaceId = "ws-id";
    const workspaceName = "branch";

    const workspacePath = getWorkspacePath({
      srcDir: config.srcDir,
      projectName: "project",
      workspaceName,
    });
    await fs.mkdir(workspacePath, { recursive: true });

    await config.editConfig((cfg) => {
      cfg.projects.set(projectPath, {
        workspaces: [
          {
            path: workspacePath,
            id: workspaceId,
            name: workspaceName,
            runtimeConfig: { type: "worktree", srcBaseDir: config.srcDir },
          },
        ],
      });
      return cfg;
    });

    const service = new WorkspaceMcpOverridesService(config);
    await service.setOverridesForWorkspace(workspaceId, {
      enabledServers: ["plugin:abc:server"],
    });

    // Dialog snapshot taken here...
    const snapshot = await service.getOverridesForWorkspace(workspaceId);

    // ...then a concurrent writer (e.g. plugin uninstall prune) removes the key.
    await service.setOverridesForWorkspace(
      workspaceId,
      {},
      { expectedRevision: snapshot.revision }
    );

    // Replaying the stale snapshot must fail, not restore the pruned key.
    // eslint-disable-next-line @typescript-eslint/await-thenable -- bun-types mistype .rejects.toThrow as void
    await expect(
      service.setOverridesForWorkspace(workspaceId, snapshot.overrides, {
        expectedRevision: snapshot.revision,
      })
    ).rejects.toThrow(WorkspaceMcpOverridesConflictError);

    const current = await service.getOverridesForWorkspace(workspaceId);
    expect(current.overrides).toEqual({});

    // A save with the CURRENT revision goes through.
    await service.setOverridesForWorkspace(
      workspaceId,
      { disabledServers: ["other"] },
      { expectedRevision: current.revision }
    );
    const after = await service.getOverridesForWorkspace(workspaceId);
    expect(after.overrides).toEqual({ disabledServers: ["other"] });
  });

  it("strict reads throw on unreadable content instead of reporting empty overrides", async () => {
    const projectPath = "/fake/project";
    const workspaceId = "ws-id";
    const workspaceName = "branch";

    const workspacePath = getWorkspacePath({
      srcDir: config.srcDir,
      projectName: "project",
      workspaceName,
    });
    await fs.mkdir(path.join(workspacePath, ".mux"), { recursive: true });
    // Content exists but is not parseable: the plugin uninstaller's prune
    // must NOT see "{}" here — it would retire its tombstone against keys it
    // never read, resurrecting stale enabledServers on reinstall.
    await fs.writeFile(
      path.join(workspacePath, ".mux", "mcp.local.jsonc"),
      '{ "enabledServers": ["plugin:abc:echo"'
    );

    await config.editConfig((cfg) => {
      cfg.projects.set(projectPath, {
        workspaces: [
          {
            path: workspacePath,
            id: workspaceId,
            name: workspaceName,
            runtimeConfig: { type: "worktree", srcBaseDir: config.srcDir },
          },
        ],
      });
      return cfg;
    });

    const service = new WorkspaceMcpOverridesService(config);
    // Lenient (UI/list paths): degrade to empty.
    const lenient = await service.getOverridesForWorkspace(workspaceId);
    expect(lenient.overrides).toEqual({});
    // Strict (prune path): fail loudly so the caller keeps its retry state.
    // eslint-disable-next-line @typescript-eslint/await-thenable -- bun-types mistype .rejects.toThrow as void
    await expect(service.getOverridesForWorkspace(workspaceId, { mode: "strict" })).rejects.toThrow(
      /parse errors/
    );
    // Strict on a genuinely absent file is still fine (no overrides).
    await fs.rm(path.join(workspacePath, ".mux", "mcp.local.jsonc"));
    const absent = await service.getOverridesForWorkspace(workspaceId, { mode: "strict" });
    expect(absent.overrides).toEqual({});
  });

  it("removes workspace-local file when overrides are set to empty", async () => {
    const projectPath = "/fake/project";
    const workspaceId = "ws-id";
    const workspaceName = "branch";

    const workspacePath = getWorkspacePath({
      srcDir: config.srcDir,
      projectName: "project",
      workspaceName,
    });
    await fs.mkdir(workspacePath, { recursive: true });

    await config.editConfig((cfg) => {
      cfg.projects.set(projectPath, {
        workspaces: [
          {
            path: workspacePath,
            id: workspaceId,
            name: workspaceName,
            runtimeConfig: { type: "worktree", srcBaseDir: config.srcDir },
          },
        ],
      });
      return cfg;
    });

    const service = new WorkspaceMcpOverridesService(config);

    await service.setOverridesForWorkspace(workspaceId, {
      disabledServers: ["server-a"],
    });

    const filePath = path.join(workspacePath, ".mux", "mcp.local.jsonc");
    expect(await pathExists(filePath)).toBe(true);

    await service.setOverridesForWorkspace(workspaceId, {});
    expect(await pathExists(filePath)).toBe(false);
  });

  it("migrates legacy config.json overrides into workspace-local file", async () => {
    const projectPath = "/fake/project";
    const workspaceId = "ws-id";
    const workspaceName = "branch";

    const workspacePath = getWorkspacePath({
      srcDir: config.srcDir,
      projectName: "project",
      workspaceName,
    });
    await fs.mkdir(workspacePath, { recursive: true });

    await config.editConfig((cfg) => {
      cfg.projects.set(projectPath, {
        workspaces: [
          {
            path: workspacePath,
            id: workspaceId,
            name: workspaceName,
            runtimeConfig: { type: "worktree", srcBaseDir: config.srcDir },
            mcp: {
              disabledServers: ["server-a"],
              toolAllowlist: { "server-b": ["tool1"] },
            },
          },
        ],
      });
      return cfg;
    });

    const service = new WorkspaceMcpOverridesService(config);
    const { overrides } = await service.getOverridesForWorkspace(workspaceId);

    expect(overrides).toEqual({
      disabledServers: ["server-a"],
      toolAllowlist: { "server-b": ["tool1"] },
    });

    // File written
    const filePath = path.join(workspacePath, ".mux", "mcp.local.jsonc");
    expect(await pathExists(filePath)).toBe(true);

    // Legacy config cleared
    const loaded = config.loadConfigOrDefault();
    const projectConfig = loaded.projects.get(projectPath);
    expect(projectConfig).toBeDefined();
    expect(projectConfig!.workspaces[0].mcp).toBeUndefined();
  });
});
