import { z } from "zod";

/** Why a foreground task wait returned before the task itself reached a terminal state. */
export const ForegroundWaitInterruptionSchema = z.discriminatedUnion("reason", [
  z
    .object({
      reason: z.literal("progress_report_received"),
      sourceTaskId: z.string().min(1),
    })
    .strict(),
  z
    .object({
      reason: z.literal("message_queued"),
    })
    .strict(),
]);

export type ForegroundWaitInterruption = z.infer<typeof ForegroundWaitInterruptionSchema>;

export const GENERIC_FOREGROUND_WAIT_INTERRUPTION = {
  reason: "message_queued",
} as const satisfies ForegroundWaitInterruption;
