/**
 * Staged /refine edit persistence (RLM track, r11 security hardening).
 *
 * SECURITY RATIONALE — this module is the staging seam that keeps /refine
 * from auto-applying model output: the refine pass runs a model over
 * attacker-influenceable trajectory text (chat history, timeline events)
 * with memory/skill mutation tools. Budget, scope confinement, and r6
 * rollback all act AFTER execution, so a prompt-injected pass could persist
 * malicious instructions into memory/skills that later sessions trust.
 * Instead of executing, the pass STAGES its intended mutations here; nothing
 * is written until the user explicitly runs `/refine apply`, which replays
 * the staged inputs through the same journaled tool paths (so rollback keeps
 * working). One staged set exists per workspace at a time: a new /refine run
 * replaces it.
 *
 * Self-healing: a corrupt or unreadable staged file is treated as "nothing
 * staged" rather than failing the workspace.
 */
import * as fsPromises from "node:fs/promises";
import * as path from "node:path";
import { z } from "zod";

import { log } from "@/node/services/log";

const STAGED_REFINE_FILENAME = "refine-staged.json";

export const StagedRefineEditSchema = z.object({
  /** Which journaled tool path applies this edit. */
  tool: z.enum(["memory", "agent_skill_write"]),
  /**
   * Tool-call id from the staging pass. Reused at apply time so the r2
   * refinement journal rows correlate back to exactly this staged set.
   */
  toolCallId: z.string(),
  /** Human-readable action line shown in the staged-summary chat row. */
  description: z.string(),
  /**
   * Raw tool input captured at staging time. Validated against the target
   * tool's schema again at apply time — the file sits on disk and must be
   * treated as untrusted input.
   */
  input: z.unknown(),
});
export type StagedRefineEdit = z.infer<typeof StagedRefineEditSchema>;

export const StagedRefineSetSchema = z.object({
  version: z.literal(1),
  workspaceId: z.string(),
  createdAt: z.number(),
  /** The staging pass's closing model summary, reused in the apply record. */
  summary: z.string(),
  edits: z.array(StagedRefineEditSchema).min(1),
});
export type StagedRefineSet = z.infer<typeof StagedRefineSetSchema>;

function stagedFilePath(sessionDir: string): string {
  return path.join(sessionDir, STAGED_REFINE_FILENAME);
}

export async function saveStagedRefineSet(sessionDir: string, set: StagedRefineSet): Promise<void> {
  await fsPromises.mkdir(sessionDir, { recursive: true });
  await fsPromises.writeFile(stagedFilePath(sessionDir), JSON.stringify(set, null, 2));
}

export async function loadStagedRefineSet(sessionDir: string): Promise<StagedRefineSet | null> {
  try {
    const raw = await fsPromises.readFile(stagedFilePath(sessionDir), "utf8");
    const parsed = StagedRefineSetSchema.safeParse(JSON.parse(raw));
    if (!parsed.success) {
      log.debug("[Refine] ignoring corrupt staged set", { error: parsed.error.message });
      return null;
    }
    return parsed.data;
  } catch {
    return null;
  }
}

export async function clearStagedRefineSet(sessionDir: string): Promise<void> {
  await fsPromises.rm(stagedFilePath(sessionDir), { force: true });
}
