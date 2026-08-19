/**
 * Refinement rollback engine (RLM track, phase r6): makes the r2 journal
 * actionable. `listRefinements` returns the byId-deduped refinement rows of a
 * session; `rollbackRefinement` applies a row's recorded inverse back to the
 * filesystem and journals the rollback as a refinement row of its own (with
 * `rollbackOf`), so rollbacks are themselves invertible — rolling back a
 * rollback just inverts again.
 *
 * Safety posture:
 * - Confinement (never overridable, not even with force): inverse paths must
 *   resolve inside legal self-modification roots — memory scope roots under
 *   the session's mux home, or `.mux/skills` / `.agents/skills` directories.
 *   r2 only instruments the memory + skill tools, so repo AGENTS.md files and
 *   built-in skills (embedded in the app bundle) never appear in the journal;
 *   the confinement check refuses them anyway in case of a corrupted row.
 * - Divergence (overridable with force, CLI-only): if the current file state
 *   no longer matches what the inverse expects — a later journaled row touched
 *   the same paths, or the files were deleted/recreated since — refuse with an
 *   error listing the divergence.
 *
 * Scope note: inverses are applied to the HOST filesystem. Skill rows written
 * by remote runtimes carry runtime-namespace paths; those either fail the
 * confinement/divergence checks or simply do not exist locally, and are not
 * translated here (same v1 scope as the r2 emitters' cross-workspace caveat).
 */

import * as fsPromises from "node:fs/promises";
import * as path from "node:path";
import writeFileAtomic from "write-file-atomic";
import assert from "@/common/utils/assert";
import type { DurableEvent } from "@/common/types/durableEvent";
import {
  MemoryRefinementActionSchema,
  RefinementInverseSchema,
  RollbackRefinementActionSchema,
  SkillRefinementActionSchema,
  type RefinementInverse,
  type RollbackRefinementAction,
} from "@/common/types/refinement";
import { getErrorMessage } from "@/common/utils/errors";
import { sharedDurableEventJournal } from "@/node/utils/journal/durableEventJournal";
import { log } from "@/node/services/log";
import {
  resolveRefinementInverse,
  type RefinementFileCapture,
  type RefinementInverseDraft,
} from "./refinementJournal";

export type RefinementEvent = Extract<DurableEvent, { kind: "refinement" }>;

/** All refinement rows in the session journal (byId-deduped, seq order). */
export async function listRefinements(sessionDir: string): Promise<RefinementEvent[]> {
  assert(sessionDir.length > 0, "listRefinements requires a session dir");
  const events = await sharedDurableEventJournal(sessionDir).read();
  return events.filter((event): event is RefinementEvent => event.kind === "refinement");
}

export interface RollbackRefinementOptions {
  sessionDir: string;
  /** Envelope `id` of the refinement row to roll back. */
  id: string;
  /** Apply despite detected divergence. Confinement is NEVER overridable. */
  force?: boolean;
  /** Attribution for the emitted rollback row. */
  evidence: { toolName: string; toolCallId?: string; actor?: string };
  /** Caller-supplied justification, recorded in the rollback row's action. */
  reason?: string;
}

export interface RollbackApplied {
  /** Envelope id of the emitted rollback row; null if journaling failed. */
  rollbackRowId: string | null;
  /** Files restored to their recorded prior contents. */
  restored: string[];
  /** Files deleted (the target row had created them). */
  deleted: string[];
  /** Rename that was undone. */
  renamed?: { from: string; to: string };
}

export type RollbackRefinementResult =
  | { success: true; data: RollbackApplied }
  | { success: false; error: string };

/** Expected, recoverable rollback refusals; converted to { success: false }. */
class RollbackError extends Error {}

// ---------------------------------------------------------------------------
// Confinement: legal self-modification roots
// ---------------------------------------------------------------------------

/**
 * Memory scope roots derivable from the session dir. MemoryService stores
 * global/project scopes under `<muxRoot>/memory/...` and workspace scope under
 * `<muxRoot>/sessions/<ws>/memory/...` (see MemoryService.getStore). Returns
 * null when the session dir does not sit in a `<muxRoot>/sessions/<ws>`
 * layout — memory rollbacks are refused then, because no root can be trusted.
 */
function inferMemoryLayout(sessionDir: string): { muxRoot: string; sessionsDir: string } | null {
  const sessionsDir = path.dirname(path.resolve(sessionDir));
  if (path.basename(sessionsDir) !== "sessions") {
    return null;
  }
  return { muxRoot: path.dirname(sessionsDir), sessionsDir };
}

/**
 * Resolve the legal root containing `filePath` for the row's kind, or throw.
 * Purely lexical (the path is normalized by path.resolve); symlink escapes are
 * caught separately by assertNoSymlinkEscape before any write/delete.
 */
function resolveConfinementRoot(
  sessionDir: string,
  kind: "memory" | "skill",
  filePath: string
): string {
  if (!path.isAbsolute(filePath)) {
    throw new RollbackError(`Refusing rollback: inverse path is not absolute: '${filePath}'`);
  }
  const resolved = path.resolve(filePath);
  const segments = resolved.split(path.sep);

  if (kind === "skill") {
    // Skill files live under a `.mux/skills` or `.agents/skills` directory
    // (project checkout or home). Require at least <skill>/<file> below the
    // skills root so the roots themselves can never be a rollback target.
    for (let i = 0; i + 1 < segments.length; i++) {
      const pair = `${segments[i]}/${segments[i + 1]}`;
      if ((pair === ".mux/skills" || pair === ".agents/skills") && segments.length >= i + 4) {
        return segments.slice(0, i + 2).join(path.sep);
      }
    }
    throw new RollbackError(
      `Refusing rollback: path is outside every skills root (.mux/skills, .agents/skills): '${filePath}'`
    );
  }

  const layout = inferMemoryLayout(sessionDir);
  if (layout === null) {
    throw new RollbackError(
      `Refusing rollback: cannot derive memory roots from session dir '${sessionDir}' (expected <muxRoot>/sessions/<workspace>)`
    );
  }
  // <muxRoot>/memory/<scope>/<file...> (global + project scopes).
  const memoryRoot = path.join(layout.muxRoot, "memory");
  const relToMemory = path.relative(memoryRoot, resolved);
  if (!relToMemory.startsWith("..") && !path.isAbsolute(relToMemory)) {
    if (relToMemory.split(path.sep).length >= 2) {
      return memoryRoot;
    }
    throw new RollbackError(
      `Refusing rollback: path targets a memory scope root, not a file inside it: '${filePath}'`
    );
  }
  // <muxRoot>/sessions/<ws>/memory/<file...> (workspace scope). Constrained to
  // exactly the per-workspace memory subdir so a corrupted inverse can never
  // touch session artifacts (chat.jsonl, journals) of any workspace.
  const relToSessions = path.relative(layout.sessionsDir, resolved);
  if (!relToSessions.startsWith("..") && !path.isAbsolute(relToSessions)) {
    const parts = relToSessions.split(path.sep);
    if (parts.length >= 3 && parts[1] === "memory") {
      return path.join(layout.sessionsDir, parts[0], "memory");
    }
  }
  throw new RollbackError(
    `Refusing rollback: path is outside every memory scope root: '${filePath}'`
  );
}

/**
 * Symlink-escape prevention (mirrors LocalMemoryStore.assertContained):
 * realpath the deepest existing ancestor of the target and require it to stay
 * inside the (realpathed) root. A missing root means nothing exists under it,
 * so there is nothing to escape through.
 */
async function assertNoSymlinkEscape(rootAbs: string, targetAbs: string): Promise<void> {
  let realRoot: string;
  try {
    realRoot = await fsPromises.realpath(rootAbs);
  } catch {
    return;
  }
  let candidate = targetAbs;
  for (;;) {
    try {
      const real = await fsPromises.realpath(candidate);
      const rel = path.relative(realRoot, real);
      if (rel !== "" && (rel.startsWith("..") || path.isAbsolute(rel))) {
        throw new RollbackError(
          `Refusing rollback: '${targetAbs}' escapes its root through a symlink`
        );
      }
      return;
    } catch (error) {
      if (error instanceof RollbackError) {
        throw error;
      }
      const parent = path.dirname(candidate);
      if (parent === candidate) {
        return; // No existing ancestor at all (unreachable in practice).
      }
      candidate = parent;
    }
  }
}

/** Every filesystem path a parsed inverse touches. */
function inversePaths(inverse: RefinementInverse): string[] {
  switch (inverse.op) {
    case "delete-files":
      return inverse.paths;
    case "restore-files":
      return inverse.files.map((file) => file.path);
    case "rename":
      return [inverse.from, inverse.to];
  }
}

// ---------------------------------------------------------------------------
// Divergence detection
// ---------------------------------------------------------------------------

/** Path overlap including prefix containment (a rename can move a whole dir). */
function pathsOverlap(a: string, b: string): boolean {
  const ra = path.resolve(a);
  const rb = path.resolve(b);
  return ra === rb || ra.startsWith(rb + path.sep) || rb.startsWith(ra + path.sep);
}

async function fileExists(target: string): Promise<boolean> {
  try {
    const stat = await fsPromises.stat(target);
    return stat.isFile();
  } catch {
    return false;
  }
}

/**
 * Presence the current filesystem must show for the target's restore-files
 * inverse to apply cleanly: rows whose action was a delete expect their files
 * to be ABSENT now (present = recreated since); edit rows expect them PRESENT
 * (absent = deleted since). Returns null when the action is unparseable — the
 * caller then requires force, because no expectation can be established.
 */
function expectedPresenceForRestore(target: RefinementEvent): "present" | "absent" | null {
  const rollback = RollbackRefinementActionSchema.safeParse(target.data.action);
  if (rollback.success) {
    // Handled content-exactly by the caller via the original row's inverse.
    return null;
  }
  if (target.data.kind === "memory") {
    const parsed = MemoryRefinementActionSchema.safeParse(target.data.action);
    if (!parsed.success) return null;
    return parsed.data.op === "delete" ? "absent" : "present";
  }
  const parsed = SkillRefinementActionSchema.safeParse(target.data.action);
  if (!parsed.success) return null;
  return parsed.data.op === "write" ? "present" : "absent";
}

interface InverseContentReader {
  read(file: { path: string; text?: string; blobRef?: string }): Promise<string>;
}

/**
 * Collect divergence complaints for rolling back `target` given the current
 * filesystem + journal state. Empty array = safe to apply.
 */
async function collectDivergence(
  rows: RefinementEvent[],
  target: RefinementEvent,
  inverse: RefinementInverse,
  readContent: InverseContentReader
): Promise<string[]> {
  const complaints: string[] = [];
  const targetPaths = inversePaths(inverse);

  // Later journaled rows touching the same paths: the state the inverse
  // expects has been superseded — roll the newest row back first.
  for (const row of rows) {
    if (row.seq <= target.seq) continue;
    const parsed = RefinementInverseSchema.safeParse(row.data.inverse);
    if (!parsed.success) continue;
    const overlap = inversePaths(parsed.data).some((p) =>
      targetPaths.some((t) => pathsOverlap(p, t))
    );
    if (overlap) {
      complaints.push(`later refinement row ${row.id} (seq ${row.seq}) touched the same paths`);
    }
  }

  switch (inverse.op) {
    case "delete-files": {
      // Inverse of a create: the created files must still exist.
      for (const p of inverse.paths) {
        if (!(await fileExists(p))) {
          complaints.push(`expected '${p}' to exist (created by the target row), but it is gone`);
        }
      }
      break;
    }
    case "rename": {
      if (!(await fileExists(inverse.from)) && !(await dirExists(inverse.from))) {
        complaints.push(`expected rename source '${inverse.from}' to exist`);
      }
      if ((await fileExists(inverse.to)) || (await dirExists(inverse.to))) {
        complaints.push(`expected rename destination '${inverse.to}' to be absent`);
      }
      break;
    }
    case "restore-files": {
      const rollbackAction = RollbackRefinementActionSchema.safeParse(target.data.action);
      if (rollbackAction.success) {
        // Target is itself a rollback: it applied the original row's inverse,
        // so the current state must still match that applied inverse —
        // content-exact where the original restored files.
        complaints.push(
          ...(await collectRollbackTargetDivergence(rows, rollbackAction.data, readContent))
        );
        break;
      }
      const presence = expectedPresenceForRestore(target);
      if (presence === null) {
        complaints.push("cannot determine the expected file state from the row's action payload");
        break;
      }
      for (const file of inverse.files) {
        const exists = await fileExists(file.path);
        if (presence === "present" && !exists) {
          complaints.push(
            `expected '${file.path}' to exist (edited by the target row), but it was deleted since`
          );
        }
        if (presence === "absent" && exists) {
          complaints.push(
            `expected '${file.path}' to be absent (deleted by the target row), but it was recreated since`
          );
        }
      }
      break;
    }
  }
  return complaints;
}

async function dirExists(target: string): Promise<boolean> {
  try {
    const stat = await fsPromises.stat(target);
    return stat.isDirectory();
  } catch {
    return false;
  }
}

/**
 * Divergence for rolling back a rollback row: the rollback applied the
 * ORIGINAL row's inverse, so the disk must still match that applied state.
 * This is the one case where content-exact comparison is possible, because
 * the applied contents are recorded in the original row.
 */
async function collectRollbackTargetDivergence(
  rows: RefinementEvent[],
  action: RollbackRefinementAction,
  readContent: InverseContentReader
): Promise<string[]> {
  const original = rows.find((row) => row.id === action.of);
  if (original === undefined) {
    return [`the original row '${action.of}' this rollback applied is missing from the journal`];
  }
  const applied = RefinementInverseSchema.safeParse(original.data.inverse);
  if (!applied.success) {
    return [`the original row '${action.of}' has an unparseable inverse`];
  }
  const complaints: string[] = [];
  switch (applied.data.op) {
    case "delete-files":
      for (const p of applied.data.paths) {
        if (await fileExists(p)) {
          complaints.push(
            `expected '${p}' to be absent (the rollback deleted it), but it was recreated since`
          );
        }
      }
      break;
    case "restore-files":
      for (const file of applied.data.files) {
        if (!(await fileExists(file.path))) {
          complaints.push(
            `expected '${file.path}' to exist (the rollback restored it), but it was deleted since`
          );
          continue;
        }
        const expected = await readContent.read(file);
        const current = await fsPromises.readFile(file.path, "utf-8");
        if (current !== expected) {
          complaints.push(`'${file.path}' was edited since the rollback restored it`);
        }
      }
      break;
    case "rename":
      // Structural rename expectations are already covered by the target's
      // own inverse (the mirrored rename) in collectDivergence.
      break;
  }
  return complaints;
}

// ---------------------------------------------------------------------------
// Rollback
// ---------------------------------------------------------------------------

/**
 * Roll back one refinement row: validate, capture the pre-rollback state as
 * the new row's inverse, apply the target's inverse to disk, and append the
 * rollback row with `rollbackOf`. Refusals return { success: false }.
 */
export async function rollbackRefinement(
  opts: RollbackRefinementOptions
): Promise<RollbackRefinementResult> {
  try {
    assert(opts.sessionDir.length > 0, "rollbackRefinement requires a session dir");
    assert(opts.id.length > 0, "rollbackRefinement requires a target row id");
    const journal = sharedDurableEventJournal(opts.sessionDir);
    const rows = await listRefinements(opts.sessionDir);

    const target = rows.find((row) => row.id === opts.id);
    if (target === undefined) {
      throw new RollbackError(`No refinement row with id '${opts.id}' in this session`);
    }
    const kind = target.data.kind;
    if (kind !== "memory" && kind !== "skill") {
      throw new RollbackError(
        `Refinement kind '${kind}' is not rollbackable (only memory and skill rows are)`
      );
    }
    const existingRollback = rows.find((row) => row.data.rollbackOf === opts.id);
    if (existingRollback !== undefined) {
      throw new RollbackError(
        `Row '${opts.id}' was already rolled back by row '${existingRollback.id}'. Roll back that row instead to re-apply.`
      );
    }

    const parsedInverse = RefinementInverseSchema.safeParse(target.data.inverse);
    if (!parsedInverse.success) {
      throw new RollbackError(
        `Row '${opts.id}' has an unparseable inverse payload: ${parsedInverse.error.message}`
      );
    }
    const inverse = parsedInverse.data;

    // Confinement first — never overridable. A corrupted inverse must never
    // write outside the memory/skill roots (repo AGENTS.md, built-in skills,
    // or anything else).
    const roots = new Map<string, string>();
    for (const p of inversePaths(inverse)) {
      roots.set(p, resolveConfinementRoot(opts.sessionDir, kind, p));
    }
    for (const [p, root] of roots) {
      await assertNoSymlinkEscape(root, path.resolve(p));
    }

    const readContent: InverseContentReader = {
      read: async (file) => {
        if (file.text !== undefined) return file.text;
        assert(file.blobRef !== undefined, "refinement file has neither text nor blobRef");
        const text = await journal.blobs.getText(file.blobRef);
        if (text === null) {
          throw new RollbackError(`Blob ${file.blobRef} for '${file.path}' is missing or corrupt`);
        }
        return text;
      },
    };

    const divergence = await collectDivergence(rows, target, inverse, readContent);
    if (divergence.length > 0 && opts.force !== true) {
      throw new RollbackError(
        `Refusing rollback of '${opts.id}': current state diverges from what the inverse expects:\n` +
          divergence.map((line) => `  - ${line}`).join("\n") +
          `\nRe-run with force to apply anyway.`
      );
    }

    // Capture the pre-rollback state (the new row's inverse) BEFORE mutating.
    const newInverse = await capturePreRollbackInverse(inverse);

    // Apply the target's inverse to disk.
    const applied: RollbackApplied = { rollbackRowId: null, restored: [], deleted: [] };
    switch (inverse.op) {
      case "delete-files":
        for (const p of inverse.paths) {
          await fsPromises.rm(p, { force: true });
          applied.deleted.push(p);
        }
        break;
      case "restore-files":
        for (const file of inverse.files) {
          const content = await readContent.read(file);
          await fsPromises.mkdir(path.dirname(file.path), { recursive: true });
          // Same atomic-write discipline as LocalMemoryStore.writeFile.
          await writeFileAtomic(file.path, content, { encoding: "utf-8" });
          applied.restored.push(file.path);
        }
        break;
      case "rename":
        await fsPromises.mkdir(path.dirname(inverse.to), { recursive: true });
        await fsPromises.rename(inverse.from, inverse.to);
        applied.renamed = { from: inverse.from, to: inverse.to };
        break;
    }

    // Journal the rollback row. The filesystem is already restored at this
    // point, so a journaling failure must not fail the operation (self-healing
    // doctrine) — but it is reported via rollbackRowId: null.
    try {
      const action: RollbackRefinementAction = {
        op: "rollback",
        of: opts.id,
        ...(opts.reason !== undefined ? { reason: opts.reason } : {}),
      };
      const row = await journal.append({
        workspaceId: target.workspaceId,
        kind: "refinement",
        data: {
          kind,
          action,
          inverse: await resolveRefinementInverse(journal.blobs, newInverse),
          evidence: {
            workspaceId: target.workspaceId,
            toolName: opts.evidence.toolName,
            ...(opts.evidence.toolCallId !== undefined
              ? { toolCallId: opts.evidence.toolCallId }
              : {}),
            ...(opts.evidence.actor !== undefined ? { actor: opts.evidence.actor } : {}),
          },
          rollbackOf: opts.id,
        },
      });
      applied.rollbackRowId = row.id;
    } catch (error) {
      log.error("[refinement] rollback applied but journaling the rollback row failed", {
        id: opts.id,
        error,
      });
    }

    return { success: true, data: applied };
  } catch (error) {
    if (error instanceof RollbackError) {
      return { success: false, error: error.message };
    }
    return { success: false, error: `Rollback failed: ${getErrorMessage(error)}` };
  }
}

/**
 * Build the inverse of applying `inverse` from the CURRENT filesystem state.
 * - delete-files → restore the current contents of the files it will delete.
 * - restore-files → restore current contents where files exist; where they do
 *   not (the restore will create them), delete them again. A mixed state is
 *   only reachable with force; the single-op inverse contract cannot express
 *   "restore some, delete others", so restoring existing files wins and the
 *   force-created files are left behind on a double rollback (logged).
 * - rename → the mirrored rename.
 */
async function capturePreRollbackInverse(
  inverse: RefinementInverse
): Promise<RefinementInverseDraft> {
  switch (inverse.op) {
    case "rename":
      return { op: "rename", from: inverse.to, to: inverse.from };
    case "delete-files": {
      const files: RefinementFileCapture[] = [];
      for (const p of inverse.paths) {
        if (await fileExists(p)) {
          files.push({ path: p, content: await fsPromises.readFile(p, "utf-8") });
        }
      }
      return { op: "restore-files", files };
    }
    case "restore-files": {
      const existing: RefinementFileCapture[] = [];
      const missing: string[] = [];
      for (const file of inverse.files) {
        if (await fileExists(file.path)) {
          existing.push({
            path: file.path,
            content: await fsPromises.readFile(file.path, "utf-8"),
          });
        } else {
          missing.push(file.path);
        }
      }
      if (existing.length === 0 && missing.length > 0) {
        return { op: "delete-files", paths: missing };
      }
      if (existing.length > 0 && missing.length > 0) {
        log.warn(
          "[refinement] mixed pre-rollback state (force apply): double rollback will not delete force-created files",
          { missing }
        );
      }
      return { op: "restore-files", files: existing };
    }
  }
}
