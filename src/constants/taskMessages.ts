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

/**
 * Aggregate family-message budgets per sender→target pair, for the sender's
 * process-session lifetime. The per-message cap alone is not enough: a short
 * code_execution loop can invoke task_message_parent repeatedly with valid
 * 16K messages, and a busy target's message queue appends every one to a
 * single unbounded entry before joining it into history/provider input — a
 * prompt-influenced child could push tens of MB into another workspace.
 * These totals absolutely bound what one sender can deliver to one target:
 * 32 messages / 256K chars (= 16 max-size messages) is far beyond legitimate
 * status-update traffic, and the final result travels via agent_report,
 * which is not part of this budget.
 */
export const TASK_FAMILY_MESSAGE_MAX_TOTAL_MESSAGES = 32;
export const TASK_FAMILY_MESSAGE_MAX_TOTAL_CHARS = 256 * 1024;
