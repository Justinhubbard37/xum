/**
 * RLM family messaging bounds (task_message_parent / task_message_sibling).
 *
 * A kernel guest can synthesize a multi-megabyte string in code_execution
 * without spending equivalent output tokens; without a cap the whole value
 * would be queued into a parent/sibling transcript, persisted, and sent to
 * that workspace's provider. 16K chars is generous for a status/handoff
 * message while keeping the receiving transcript bounded.
 */
export const TASK_FAMILY_MESSAGE_MAX_CHARS = 16 * 1024;
