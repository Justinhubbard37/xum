import { describe, expect, it } from "bun:test";

import * as fsPromises from "node:fs/promises";
import * as path from "node:path";
import { REFINEMENT_INLINE_MAX_CHARS } from "@/common/types/refinement";
import { Config } from "@/node/config";
import { LocalRuntime } from "@/node/runtime/LocalRuntime";
import { MemoryMetaService } from "@/node/services/memoryMeta";
import { MemoryService, type MemoryScopeContext } from "@/node/services/memoryService";
import { TestTempDir } from "@/node/services/tools/testHelpers";
import { sharedDurableEventJournal } from "@/node/utils/journal/durableEventJournal";
import { appendRefinementEvent } from "./refinementJournal";
import { listRefinements, rollbackRefinement, type RefinementEvent } from "./refinementRollback";

function pathExists(target: string): Promise<boolean> {
  return fsPromises.access(target).then(
    () => true,
    () => false
  );
}

interface RollbackFixture extends Disposable {
  muxHome: string;
  checkout: string;
  sessionDir: string;
  service: MemoryService;
  ctx: MemoryScopeContext;
}

const WORKSPACE_ID = "ws-rollback";
const EVIDENCE = { toolName: "test" };

/** Real MemoryService against a temp mux home: rollbacks consume real r2 rows. */
async function createFixture(): Promise<RollbackFixture> {
  const tempDir = new TestTempDir("test-refinement-rollback");
  const muxHome = path.join(tempDir.path, "mux-home");
  const checkout = path.join(tempDir.path, "checkout");
  await fsPromises.mkdir(muxHome, { recursive: true });
  await fsPromises.mkdir(checkout, { recursive: true });
  const config = new Config(muxHome);
  const service = new MemoryService(config, new MemoryMetaService(muxHome));
  return {
    muxHome,
    checkout,
    sessionDir: config.getSessionDir(WORKSPACE_ID),
    service,
    ctx: {
      runtime: new LocalRuntime(checkout),
      checkoutCwd: checkout,
      workspaceId: WORKSPACE_ID,
      projectPath: "/stable/project-id",
    },
    [Symbol.dispose]() {
      tempDir[Symbol.dispose]();
    },
  };
}

async function lastRow(sessionDir: string): Promise<RefinementEvent> {
  const rows = await listRefinements(sessionDir);
  expect(rows.length).toBeGreaterThan(0);
  return rows[rows.length - 1];
}

describe("refinementRollback", () => {
  it("create → edit → rollback restores byte-identical prior content (inline)", async () => {
    using fixture = await createFixture();
    const prior = "# Notes\n\noriginal content with unicode: ünïcödé ✓\n";
    await fixture.service.create(fixture.ctx, "/memories/global/notes.md", prior, "agent");
    await fixture.service.strReplace(
      fixture.ctx,
      "/memories/global/notes.md",
      "original content",
      "edited content",
      "agent"
    );
    const editRow = await lastRow(fixture.sessionDir);
    const physicalPath = path.join(fixture.muxHome, "memory", "global", "notes.md");
    expect(await fsPromises.readFile(physicalPath, "utf-8")).toContain("edited content");

    const result = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: editRow.id,
      evidence: EVIDENCE,
    });
    expect(result.success).toBe(true);
    expect(await fsPromises.readFile(physicalPath, "utf-8")).toBe(prior);
  });

  it("restores blob-backed prior content byte-identically and journals rollbackOf", async () => {
    using fixture = await createFixture();
    // Above the inline cap → the r2 inverse offloads prior content to a blob.
    const prior = `start\n${"x".repeat(REFINEMENT_INLINE_MAX_CHARS + 100)}\nend\n`;
    await fixture.service.create(fixture.ctx, "/memories/global/big.md", prior, "agent");
    await fixture.service.strReplace(fixture.ctx, "/memories/global/big.md", "start", "s", "agent");
    const editRow = await lastRow(fixture.sessionDir);
    const inverse = editRow.data.inverse as { op: string; files: Array<{ blobRef?: string }> };
    expect(inverse.files[0].blobRef).toBeDefined();

    const result = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: editRow.id,
      evidence: EVIDENCE,
    });
    expect(result.success).toBe(true);
    const physicalPath = path.join(fixture.muxHome, "memory", "global", "big.md");
    expect(await fsPromises.readFile(physicalPath, "utf-8")).toBe(prior);

    const rollbackRow = await lastRow(fixture.sessionDir);
    expect(rollbackRow.data.rollbackOf).toBe(editRow.id);
    expect(rollbackRow.data.kind).toBe("memory");
    expect(rollbackRow.data.action).toMatchObject({ op: "rollback", of: editRow.id });
    if (!result.success) throw new Error("unreachable");
    expect(result.data.rollbackRowId).toBe(rollbackRow.id);
  });

  it("refuses a double rollback of the same id, but allows rolling back the rollback", async () => {
    using fixture = await createFixture();
    await fixture.service.create(fixture.ctx, "/memories/global/a.md", "v1\n", "agent");
    await fixture.service.strReplace(fixture.ctx, "/memories/global/a.md", "v1", "v2", "agent");
    const editRow = await lastRow(fixture.sessionDir);
    const first = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: editRow.id,
      evidence: EVIDENCE,
    });
    expect(first.success).toBe(true);

    const second = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: editRow.id,
      evidence: EVIDENCE,
    });
    expect(second.success).toBe(false);
    if (second.success) throw new Error("unreachable");
    expect(second.error).toContain("already rolled back");

    // Rolling back the rollback re-applies the edit (double inversion).
    const rollbackRow = await lastRow(fixture.sessionDir);
    const undo = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: rollbackRow.id,
      evidence: EVIDENCE,
    });
    expect(undo.success).toBe(true);
    const physicalPath = path.join(fixture.muxHome, "memory", "global", "a.md");
    expect(await fsPromises.readFile(physicalPath, "utf-8")).toBe("v2\n");
  });

  it("refuses on divergence (file deleted since the edit) and applies with force", async () => {
    using fixture = await createFixture();
    await fixture.service.create(fixture.ctx, "/memories/global/gone.md", "v1\n", "agent");
    await fixture.service.strReplace(fixture.ctx, "/memories/global/gone.md", "v1", "v2", "agent");
    const editRow = await lastRow(fixture.sessionDir);
    const physicalPath = path.join(fixture.muxHome, "memory", "global", "gone.md");
    // Out-of-band deletion: the inverse expects the edited file to exist.
    await fsPromises.rm(physicalPath);

    const refused = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: editRow.id,
      evidence: EVIDENCE,
    });
    expect(refused.success).toBe(false);
    if (refused.success) throw new Error("unreachable");
    expect(refused.error).toContain("diverges");
    expect(refused.error).toContain(physicalPath);
    expect(await pathExists(physicalPath)).toBe(false);

    const forced = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: editRow.id,
      force: true,
      evidence: EVIDENCE,
    });
    expect(forced.success).toBe(true);
    expect(await fsPromises.readFile(physicalPath, "utf-8")).toBe("v1\n");
  });

  it("refuses when a later refinement row touched the same path (roll back newest first)", async () => {
    using fixture = await createFixture();
    await fixture.service.create(fixture.ctx, "/memories/global/stack.md", "v1\n", "agent");
    const createRow = await lastRow(fixture.sessionDir);
    await fixture.service.strReplace(fixture.ctx, "/memories/global/stack.md", "v1", "v2", "agent");

    const refused = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: createRow.id,
      evidence: EVIDENCE,
    });
    expect(refused.success).toBe(false);
    if (refused.success) throw new Error("unreachable");
    expect(refused.error).toContain("later refinement row");
  });

  it("rolls back a skill write via the delete-files inverse and back again", async () => {
    using fixture = await createFixture();
    const skillFile = path.join(fixture.checkout, ".mux", "skills", "my-skill", "SKILL.md");
    const content = "---\nname: my-skill\n---\n\nbody\n";
    await fsPromises.mkdir(path.dirname(skillFile), { recursive: true });
    await fsPromises.writeFile(skillFile, content, "utf-8");
    // Same emitter the skill tools use: a write that created the file journals
    // a delete-files inverse.
    await appendRefinementEvent({
      sessionDir: fixture.sessionDir,
      workspaceId: WORKSPACE_ID,
      kind: "skill",
      action: { op: "write", skillName: "my-skill", filePath: "SKILL.md" },
      inverse: { op: "delete-files", paths: [skillFile] },
      evidence: { toolName: "agent_skill_write" },
    });
    const writeRow = await lastRow(fixture.sessionDir);

    const result = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: writeRow.id,
      evidence: EVIDENCE,
    });
    expect(result.success).toBe(true);
    expect(await pathExists(skillFile)).toBe(false);

    // The rollback row restores the deleted file byte-identically.
    const rollbackRow = await lastRow(fixture.sessionDir);
    expect(rollbackRow.data.rollbackOf).toBe(writeRow.id);
    const undo = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: rollbackRow.id,
      evidence: EVIDENCE,
    });
    expect(undo.success).toBe(true);
    expect(await fsPromises.readFile(skillFile, "utf-8")).toBe(content);
  });

  it("undoes a memory rename via the mirrored rename inverse", async () => {
    using fixture = await createFixture();
    await fixture.service.create(fixture.ctx, "/memories/global/old.md", "v1\n", "agent");
    await fixture.service.rename(
      fixture.ctx,
      "/memories/global/old.md",
      "/memories/global/new.md",
      "agent"
    );
    const renameRow = await lastRow(fixture.sessionDir);
    const result = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: renameRow.id,
      evidence: EVIDENCE,
    });
    expect(result.success).toBe(true);
    const oldPath = path.join(fixture.muxHome, "memory", "global", "old.md");
    expect(await fsPromises.readFile(oldPath, "utf-8")).toBe("v1\n");
    expect(await pathExists(path.join(fixture.muxHome, "memory", "global", "new.md"))).toBe(false);
  });

  describe("confinement guard rails", () => {
    it("refuses inverse paths outside every legal root, even with force", async () => {
      using fixture = await createFixture();
      // Corrupted row: a memory-kind inverse pointing at a repo AGENTS.md.
      const evilPath = path.join(fixture.checkout, "AGENTS.md");
      await appendRefinementEvent({
        sessionDir: fixture.sessionDir,
        workspaceId: WORKSPACE_ID,
        kind: "memory",
        action: { op: "str_replace", path: "/memories/global/x.md" },
        inverse: { op: "restore-files", files: [{ path: evilPath, content: "pwned" }] },
        evidence: { toolName: "memory" },
      });
      const row = await lastRow(fixture.sessionDir);
      const result = await rollbackRefinement({
        sessionDir: fixture.sessionDir,
        id: row.id,
        force: true,
        evidence: EVIDENCE,
      });
      expect(result.success).toBe(false);
      if (result.success) throw new Error("unreachable");
      expect(result.error).toContain("outside every memory scope root");
      expect(await pathExists(evilPath)).toBe(false);
    });

    it("refuses traversal that escapes the memory root lexically", async () => {
      using fixture = await createFixture();
      // Literal traversal in the stored path (path.join would pre-collapse it).
      const escapePath = `${fixture.muxHome}/memory/global/../../config.json`;
      await appendRefinementEvent({
        sessionDir: fixture.sessionDir,
        workspaceId: WORKSPACE_ID,
        kind: "memory",
        action: { op: "delete", path: "/memories/global/x.md" },
        inverse: { op: "restore-files", files: [{ path: escapePath, content: "pwned" }] },
        evidence: { toolName: "memory" },
      });
      const row = await lastRow(fixture.sessionDir);
      const result = await rollbackRefinement({
        sessionDir: fixture.sessionDir,
        id: row.id,
        force: true,
        evidence: EVIDENCE,
      });
      expect(result.success).toBe(false);
      if (result.success) throw new Error("unreachable");
      expect(result.error).toContain("Refusing rollback");
      expect(await pathExists(path.join(fixture.muxHome, "config.json"))).toBe(false);
    });

    it("refuses skill paths without a .mux/skills or .agents/skills root", async () => {
      using fixture = await createFixture();
      const evilPath = path.join(fixture.checkout, "src", "main.ts");
      await appendRefinementEvent({
        sessionDir: fixture.sessionDir,
        workspaceId: WORKSPACE_ID,
        kind: "skill",
        action: { op: "write", skillName: "x", filePath: "SKILL.md" },
        inverse: { op: "delete-files", paths: [evilPath] },
        evidence: { toolName: "agent_skill_write" },
      });
      await fsPromises.mkdir(path.dirname(evilPath), { recursive: true });
      await fsPromises.writeFile(evilPath, "code", "utf-8");
      const row = await lastRow(fixture.sessionDir);
      const result = await rollbackRefinement({
        sessionDir: fixture.sessionDir,
        id: row.id,
        force: true,
        evidence: EVIDENCE,
      });
      expect(result.success).toBe(false);
      if (result.success) throw new Error("unreachable");
      expect(result.error).toContain("outside every skills root");
      expect(await fsPromises.readFile(evilPath, "utf-8")).toBe("code");
    });

    it("refuses symlink escapes out of the skills root", async () => {
      using fixture = await createFixture();
      const skillsRoot = path.join(fixture.checkout, ".mux", "skills");
      const outside = path.join(fixture.checkout, "outside");
      await fsPromises.mkdir(outside, { recursive: true });
      await fsPromises.mkdir(skillsRoot, { recursive: true });
      // <skillsRoot>/evil → symlink to a directory outside the root.
      await fsPromises.symlink(outside, path.join(skillsRoot, "evil"));
      const target = path.join(skillsRoot, "evil", "SKILL.md");
      await appendRefinementEvent({
        sessionDir: fixture.sessionDir,
        workspaceId: WORKSPACE_ID,
        kind: "skill",
        action: { op: "write", skillName: "evil", filePath: "SKILL.md" },
        inverse: { op: "restore-files", files: [{ path: target, content: "pwned" }] },
        evidence: { toolName: "agent_skill_write" },
      });
      const row = await lastRow(fixture.sessionDir);
      const result = await rollbackRefinement({
        sessionDir: fixture.sessionDir,
        id: row.id,
        force: true,
        evidence: EVIDENCE,
      });
      expect(result.success).toBe(false);
      if (result.success) throw new Error("unreachable");
      expect(result.error).toContain("symlink");
      expect(await pathExists(path.join(outside, "SKILL.md"))).toBe(false);
    });

    it("refuses non-rollbackable refinement kinds", async () => {
      using fixture = await createFixture();
      await sharedDurableEventJournal(fixture.sessionDir).append({
        workspaceId: WORKSPACE_ID,
        kind: "refinement",
        data: { kind: "other", action: {}, inverse: { op: "delete-files", paths: ["/x"] } },
      });
      const row = await lastRow(fixture.sessionDir);
      const result = await rollbackRefinement({
        sessionDir: fixture.sessionDir,
        id: row.id,
        evidence: EVIDENCE,
      });
      expect(result.success).toBe(false);
      if (result.success) throw new Error("unreachable");
      expect(result.error).toContain("not rollbackable");
    });
  });

  it("refuses unknown ids", async () => {
    using fixture = await createFixture();
    const result = await rollbackRefinement({
      sessionDir: fixture.sessionDir,
      id: "does-not-exist",
      evidence: EVIDENCE,
    });
    expect(result.success).toBe(false);
    if (result.success) throw new Error("unreachable");
    expect(result.error).toContain("No refinement row");
  });
});
