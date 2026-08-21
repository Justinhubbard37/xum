/**
 * Refinement journal emitters (RLM track, phase r2).
 *
 * Every harness self-modification (memory tool mutations, agent_skill_write,
 * agent_skill_delete) appends exactly one invertible `refinement` durable
 * event to the acting workspace's session journal. Journaling is purely
 * additive: it never changes tool behavior and a journaling failure must
 * never fail the user-facing mutation (self-healing doctrine — log.debug and
 * continue).
 *
 * Cross-workspace caveat (intended v1 scope): memory and skill files are
 * global- or project-scoped, but the durable journal is per-session. Rows
 * land in the journal of the workspace that made the edit, so concurrent
 * edits to one shared file from different workspaces are each attributed to
 * (and invertible from) their own acting workspace's log.
 */

import { createHash } from "node:crypto";

import assert from "@/common/utils/assert";
import {
  REFINEMENT_INLINE_MAX_CHARS,
  type MemoryRefinementAction,
  type RefinementEvidence,
  type RefinementInverse,
  type RefinementPostState,
  type SkillRefinementAction,
} from "@/common/types/refinement";
import type { BlobStore } from "@/node/utils/journal/blobStore";
import { sharedDurableEventJournal } from "@/node/utils/journal/durableEventJournal";
import { log } from "@/node/services/log";

/** Prior-content capture with inline content; the emitter offloads large contents to blobs. */
export interface RefinementFileCapture {
  path: string;
  content: string;
}

/** Inverse draft with captured contents inline; blob offload happens at append. */
export type RefinementInverseDraft =
  | { op: "delete-files"; paths: string[] }
  | { op: "restore-files"; files: RefinementFileCapture[] }
  | { op: "rename"; from: string; to: string };

export interface RefinementEmitArgs {
  /** Acting workspace's session dir (owns durable-events.jsonl + blobs). */
  sessionDir: string;
  workspaceId: string;
  kind: "memory" | "skill";
  action: MemoryRefinementAction | SkillRefinementAction;
  inverse: RefinementInverseDraft;
  evidence: { toolName: string; toolCallId?: string; actor?: string };
  /**
   * Contents the action left on disk (edit-type actions only: create,
   * str_replace, insert, skill write). Hashed at append into the row's
   * `postState` so rollback can detect out-of-band edits content-exactly.
   */
  postFiles?: RefinementFileCapture[];
  /**
   * "remote" when the mutation ran through a non-local runtime (SSH/Docker).
   * Such rows carry runtime-namespace paths and are refused by rollback,
   * which only applies inverses to the host filesystem.
   */
  runtime?: "remote";
}

/** Shared by the rollback engine to compare current files against `postState`. */
export function sha256Hex(text: string): string {
  return createHash("sha256").update(text, "utf-8").digest("hex");
}

/**
 * Offload large captured contents to the blob store; small ones stay inline.
 * Exported so the rollback service (refinementRollback.ts) resolves the
 * inverses of its own rollback rows through the identical offload policy.
 */
export async function resolveRefinementInverse(
  blobs: BlobStore,
  draft: RefinementInverseDraft
): Promise<RefinementInverse> {
  if (draft.op !== "restore-files") {
    return draft;
  }
  const files = await Promise.all(
    draft.files.map(async (file) => {
      if (file.content.length <= REFINEMENT_INLINE_MAX_CHARS) {
        return { path: file.path, text: file.content };
      }
      const { ref } = await blobs.put(file.content);
      return { path: file.path, blobRef: ref };
    })
  );
  return { op: "restore-files", files };
}

/**
 * Append one `refinement` durable event. Never throws — the mutation this row
 * describes must succeed even when the journal is unavailable.
 */
export async function appendRefinementEvent(args: RefinementEmitArgs): Promise<void> {
  try {
    assert(args.sessionDir.length > 0, "refinement journal requires a session dir");
    assert(args.workspaceId.length > 0, "refinement journal requires a workspace id");
    const journal = sharedDurableEventJournal(args.sessionDir);
    const inverse = await resolveRefinementInverse(journal.blobs, args.inverse);
    // Optional fields are spread conditionally: an explicit `undefined` value
    // would fail the JsonValue schema validation on append and drop the row.
    const evidence: RefinementEvidence = {
      workspaceId: args.workspaceId,
      toolName: args.evidence.toolName,
      ...(args.evidence.toolCallId !== undefined ? { toolCallId: args.evidence.toolCallId } : {}),
      ...(args.evidence.actor !== undefined ? { actor: args.evidence.actor } : {}),
    };
    const postState: RefinementPostState | undefined =
      args.postFiles !== undefined
        ? {
            files: args.postFiles.map((file) => ({
              path: file.path,
              sha256: sha256Hex(file.content),
            })),
          }
        : undefined;
    await journal.append({
      workspaceId: args.workspaceId,
      kind: "refinement",
      data: {
        kind: args.kind,
        action: args.action,
        inverse,
        evidence,
        ...(postState !== undefined ? { postState } : {}),
        ...(args.runtime !== undefined ? { runtime: args.runtime } : {}),
      },
    });
  } catch (error) {
    log.debug("[refinement] failed to journal refinement event; continuing", {
      kind: args.kind,
      workspaceId: args.workspaceId,
      error,
    });
  }
}

/**
 * Tool-side convenience wrapper: resolves the session journal from the tool
 * configuration. Skips (log-only) when the tool runs without a workspace
 * session — there is no journal to attribute the edit to.
 */
export async function appendRefinementEventFromTool(
  config: { workspaceSessionDir?: string; workspaceId?: string },
  args: Omit<RefinementEmitArgs, "sessionDir" | "workspaceId">
): Promise<void> {
  if (!config.workspaceSessionDir || !config.workspaceId) {
    log.debug("[refinement] skipping refinement journal: no workspace session", {
      kind: args.kind,
    });
    return;
  }
  await appendRefinementEvent({
    ...args,
    sessionDir: config.workspaceSessionDir,
    workspaceId: config.workspaceId,
  });
}
