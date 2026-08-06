import { z } from "zod";
import { MAX_WORKSPACE_TURN_ATTACH_FILE_ARTIFACTS } from "@/common/constants/taskArtifacts";

export const TaskAttachFileArtifactSchema = z
  .object({
    path: z.string().min(1).max(4096),
    filename: z.string().min(1).max(255).optional(),
    mediaType: z.string().min(1).max(255),
    displayOnly: z.literal(true).optional(),
    sourceToolCallId: z.string().min(1).max(512).optional(),
  })
  .strict();

export type TaskAttachFileArtifact = z.infer<typeof TaskAttachFileArtifactSchema>;

export const TaskAttachFileArtifactsSchema = z
  .array(TaskAttachFileArtifactSchema)
  .max(MAX_WORKSPACE_TURN_ATTACH_FILE_ARTIFACTS);
