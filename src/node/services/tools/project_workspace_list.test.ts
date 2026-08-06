import { describe, expect, it, mock } from "bun:test";

import type { TaskService } from "@/node/services/taskService";
import { createTestToolConfig, mockToolCallOptions, TestTempDir } from "./testHelpers";
import { createProjectWorkspaceListTool } from "./project_workspace_list";
import { Ok } from "@/common/types/result";

describe("project_workspace_list tool", () => {
  it("returns canonical same-project workspace summaries in one service call", async () => {
    using tempDir = new TestTempDir("project-workspace-list-tool");
    const listProjectWorkspaces = mock(() =>
      Promise.resolve(
        Ok({
          projectPath: "/project",
          workspaces: [
            {
              workspaceId: "canonical-workspace-id",
              name: "feature",
              archived: false,
              workspaceTurn: {
                taskId: "wst_turn",
                status: "running" as const,
                updatedAt: "2026-08-06T00:00:00.000Z",
              },
            },
          ],
        })
      )
    );
    const taskService = { listProjectWorkspaces } as unknown as TaskService;
    const workspaceId = "project-session_aaaaaaaaaa";
    const listTool = createProjectWorkspaceListTool({
      ...createTestToolConfig(tempDir.path, { workspaceId }),
      projectChat: true,
      taskService,
    });

    const result: unknown = await Promise.resolve(
      listTool.execute!({ include_archived: true }, mockToolCallOptions)
    );

    expect(listProjectWorkspaces).toHaveBeenCalledWith(workspaceId, { includeArchived: true });
    expect(result).toEqual({
      projectPath: "/project",
      workspaces: [
        {
          workspaceId: "canonical-workspace-id",
          name: "feature",
          archived: false,
          workspaceTurn: {
            taskId: "wst_turn",
            status: "running",
            updatedAt: "2026-08-06T00:00:00.000Z",
          },
        },
      ],
    });
  });

  it("rejects non-Project-Chat callers", async () => {
    using tempDir = new TestTempDir("project-workspace-list-scope");
    const listTool = createProjectWorkspaceListTool(createTestToolConfig(tempDir.path));

    try {
      await listTool.execute!({}, mockToolCallOptions);
      throw new Error("Expected project_workspace_list to reject a non-Project-Chat caller");
    } catch (error) {
      expect(error).toBeInstanceOf(Error);
      expect((error as Error).message).toBe(
        "project_workspace_list is only available in Project Chat"
      );
    }
  });
});
