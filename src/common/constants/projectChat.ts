export const PROJECT_CHAT_VERSION = 1 as const;
export const PROJECT_CHAT_AGENT_ID = "orchestrator" as const;
export const PROJECT_CHAT_SESSION_ID_PREFIX = "project-session_" as const;

export function isProjectSessionId(sessionId: string): boolean {
  return sessionId.startsWith(PROJECT_CHAT_SESSION_ID_PREFIX);
}
