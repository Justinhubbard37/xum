import { AlertTriangle, RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";

import { ProjectChatHeader } from "@/browser/components/ProjectChatHeader/ProjectChatHeader";
import { AIView } from "@/browser/components/AIView/AIView";
import { Button } from "@/browser/components/Button/Button";
import { useAPI } from "@/browser/contexts/API";
import {
  getAgentIdKey,
  getModelKey,
  getReasoningModeKey,
  getThinkingLevelKey,
  getWorkspaceAISettingsByAgentKey,
} from "@/common/constants/storage";
import { updatePersistedState } from "@/browser/hooks/usePersistedState";
import { useWorkspaceStoreRaw } from "@/browser/stores/WorkspaceStore";
import { setWorkspaceModelWithOrigin } from "@/browser/utils/modelChange";
import { getErrorMessage } from "@/common/utils/errors";
import type { ProjectChatInfo } from "@/common/types/project";
import type { OpenAIReasoningMode, ThinkingLevel } from "@/common/types/thinking";

interface ProjectChatPageProps {
  projectPath: string;
  projectName: string;
  leftSidebarCollapsed: boolean;
  onToggleLeftSidebarCollapsed: () => void;
}

type ProjectChatLoadState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; info: ProjectChatInfo };

function seedProjectChatAiSettings(info: ProjectChatInfo): void {
  const workspaceId = info.sessionId;
  const agentId = info.agentId;
  const settings = info.aiSettingsByAgent?.[agentId] ?? info.metadata.aiSettingsByAgent?.[agentId];

  updatePersistedState(getAgentIdKey(workspaceId), agentId);
  if (!settings) {
    return;
  }

  updatePersistedState(getWorkspaceAISettingsByAgentKey(workspaceId), {
    [agentId]: settings,
  });
  setWorkspaceModelWithOrigin(workspaceId, settings.model, "sync");
  updatePersistedState<ThinkingLevel>(getThinkingLevelKey(workspaceId), settings.thinkingLevel);
  updatePersistedState<OpenAIReasoningMode>(
    getReasoningModeKey(workspaceId),
    settings.reasoningMode ?? "standard"
  );
}

/** Persistent, project-owned control-plane chat. Its session never appears as a workspace row. */
export function ProjectChatPage(props: ProjectChatPageProps) {
  const { api } = useAPI();
  const workspaceStore = useWorkspaceStoreRaw();
  const [reloadKey, setReloadKey] = useState(0);
  const [loadState, setLoadState] = useState<ProjectChatLoadState>({ status: "loading" });

  useEffect(() => {
    let ignore = false;
    let registeredSessionId: string | null = null;
    setLoadState({ status: "loading" });

    const load = async () => {
      if (!api) {
        return;
      }

      try {
        const result = await api.projects.chat.getOrCreate({ projectPath: props.projectPath });
        if (ignore) {
          return;
        }
        if (!result.success) {
          setLoadState({ status: "error", message: result.error });
          return;
        }

        const info = result.data;
        registeredSessionId = info.sessionId;
        seedProjectChatAiSettings(info);
        workspaceStore.addAuxiliaryChat(info.metadata);
        workspaceStore.setActiveWorkspaceId(info.sessionId);
        setLoadState({ status: "ready", info });
      } catch (error) {
        if (!ignore) {
          setLoadState({ status: "error", message: getErrorMessage(error) });
        }
      }
    };

    void load();
    return () => {
      ignore = true;
      if (registeredSessionId) {
        workspaceStore.removeAuxiliaryChat(registeredSessionId);
      }
    };
  }, [api, props.projectPath, reloadKey, workspaceStore]);

  if (loadState.status === "loading") {
    return (
      <div className="bg-surface-primary flex flex-1 flex-col overflow-hidden">
        <ProjectChatHeader
          projectName={props.projectName}
          projectPath={props.projectPath}
          leftSidebarCollapsed={props.leftSidebarCollapsed}
          onToggleLeftSidebarCollapsed={props.onToggleLeftSidebarCollapsed}
        />
        <div className="flex flex-1 items-center justify-center" role="status">
          <div className="text-content-secondary text-sm">Opening Project Chat…</div>
        </div>
      </div>
    );
  }

  if (loadState.status === "error") {
    return (
      <div className="bg-surface-primary flex flex-1 flex-col overflow-hidden">
        <ProjectChatHeader
          projectName={props.projectName}
          projectPath={props.projectPath}
          leftSidebarCollapsed={props.leftSidebarCollapsed}
          onToggleLeftSidebarCollapsed={props.onToggleLeftSidebarCollapsed}
        />
        <div className="flex flex-1 items-center justify-center p-6">
          <div className="border-border-light bg-background-secondary flex max-w-md flex-col items-center gap-3 rounded-lg border p-5 text-center">
            <AlertTriangle className="text-warning h-6 w-6" aria-hidden="true" />
            <div>
              <div className="font-medium text-content-primary">Could not open Project Chat</div>
              <div className="mt-1 text-sm text-content-secondary">{loadState.message}</div>
            </div>
            <Button variant="outline" onClick={() => setReloadKey((value) => value + 1)}>
              <RefreshCw className="h-4 w-4" aria-hidden="true" />
              Try again
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <AIView
      workspaceId={loadState.info.sessionId}
      projectPath={props.projectPath}
      projectName={props.projectName}
      workspaceName="Project Chat"
      namedWorkspacePath={props.projectPath}
      runtimeConfig={{ type: "local" }}
      leftSidebarCollapsed={props.leftSidebarCollapsed}
      onToggleLeftSidebarCollapsed={props.onToggleLeftSidebarCollapsed}
      surface="project"
    />
  );
}
