/**
 * Refinement payload contracts (v1) — the concrete vocabulary carried inside
 * `refinement` durable events (src/common/types/durableEvent.ts).
 *
 * RefinementDataSchema deliberately keeps `action`/`inverse`/`evidence` as
 * opaque JSON so the envelope stays generic across future refinement kinds;
 * these schemas are the producer/consumer contract for the harness
 * self-modification emitters (memory tool + skill CRUD tools). Applying the
 * `inverse` must fully restore the file state that existed before the action.
 */

import { z } from "zod";
import { BlobRefSchema } from "./durableEvent";

/**
 * Inline cap for prior-content payloads in refinement inverses; larger
 * contents go to the session blob store and are referenced by BlobRef
 * (mirrors the hook-context inline cap).
 */
export const REFINEMENT_INLINE_MAX_CHARS = 4_096;

/** One file to restore: exactly one of `text` (small) or `blobRef` (large). */
export const RefinementFileSchema = z
  .object({
    /**
     * Absolute physical path: host-local for memory files, runtime-namespace
     * for skill files on remote runtimes (the inverse is applied through the
     * same filesystem that performed the action).
     */
    path: z.string().min(1),
    text: z.string().optional(),
    blobRef: BlobRefSchema.optional(),
  })
  .refine((file) => (file.text === undefined) !== (file.blobRef === undefined), {
    message: "refinement file requires exactly one of text or blobRef",
  });
export type RefinementFile = z.infer<typeof RefinementFileSchema>;

/**
 * Invertible file-level operations. File-level (rather than command-level)
 * payloads keep the applier trivial and byte-exact: no re-parsing of memory
 * commands or skill frontmatter is needed to roll an edit back.
 */
export const RefinementInverseSchema = z.discriminatedUnion("op", [
  z.object({ op: z.literal("delete-files"), paths: z.array(z.string().min(1)).min(1) }),
  z.object({ op: z.literal("restore-files"), files: z.array(RefinementFileSchema) }),
  z.object({ op: z.literal("rename"), from: z.string().min(1), to: z.string().min(1) }),
]);
export type RefinementInverse = z.infer<typeof RefinementInverseSchema>;

/** Action payload for `data.kind === "memory"` rows (memory tool commands). */
export const MemoryRefinementActionSchema = z.object({
  op: z.enum(["create", "str_replace", "insert", "delete", "rename"]),
  /** Virtual memory path (/memories/<scope>/...). */
  path: z.string().min(1),
  /** Destination virtual path (rename only). */
  newPath: z.string().optional(),
});
export type MemoryRefinementAction = z.infer<typeof MemoryRefinementActionSchema>;

/** Action payload for `data.kind === "skill"` rows (agent_skill_write/delete). */
export const SkillRefinementActionSchema = z.object({
  op: z.enum(["write", "delete-file", "delete-skill"]),
  skillName: z.string().min(1),
  /** Skill-relative file path (absent for delete-skill). */
  filePath: z.string().optional(),
});
export type SkillRefinementAction = z.infer<typeof SkillRefinementActionSchema>;

/**
 * Action payload for rollback rows (r6). A rollback applies the target row's
 * inverse, so the row carries the same `kind` as its target (memory | skill)
 * and is itself a legal rollback target (double inversion).
 */
export const RollbackRefinementActionSchema = z.object({
  op: z.literal("rollback"),
  /** Envelope `id` of the row this rollback applied the inverse of. */
  of: z.string().min(1),
  /** Caller-supplied justification (model tool calls record it here). */
  reason: z.string().optional(),
});
export type RollbackRefinementAction = z.infer<typeof RollbackRefinementActionSchema>;

/** Attribution for a refinement row: who/what performed the mutation. */
export const RefinementEvidenceSchema = z.object({
  workspaceId: z.string().min(1),
  toolName: z.string().min(1),
  /** Provider tool call id, when the mutation came from a model tool call. */
  toolCallId: z.string().optional(),
  /** Memory mutations record the acting party ("agent" | "user"). */
  actor: z.string().optional(),
});
export type RefinementEvidence = z.infer<typeof RefinementEvidenceSchema>;
