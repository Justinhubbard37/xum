import * as fsPromises from "fs/promises";
import * as path from "path";
import { tool } from "ai";

import { SkillNameSchema } from "@/common/orpc/schemas";
import {
  REFINEMENT_CAPTURE_MAX_FILE_BYTES,
  REFINEMENT_CAPTURE_MAX_FILES,
  REFINEMENT_CAPTURE_MAX_TOTAL_BYTES,
} from "@/common/types/refinement";
import type { AgentSkillDeleteToolResult } from "@/common/types/tools";
import { getErrorMessage } from "@/common/utils/errors";
import { TOOL_DEFINITIONS } from "@/common/utils/tools/toolDefinitions";
import type { ToolConfiguration, ToolFactory } from "@/common/utils/tools/tools";
import type { FileStat, Runtime } from "@/node/runtime/Runtime";
import { resolveSkillStorageContext } from "@/node/services/agentSkills/skillStorageContext";
import {
  appendRefinementEventFromTool,
  type RefinementFileCapture,
} from "@/node/services/refinement/refinementJournal";
import { log } from "@/node/services/log";
import { execBuffered, readFileString } from "@/node/utils/runtime/helpers";
import { quoteRuntimeProbePath } from "./runtimePathShellQuote";
import {
  ensureRuntimePathWithinWorkspace,
  inspectContainmentOnRuntime,
  resolveContainedSkillFilePathOnRuntime,
} from "./runtimeSkillPathUtils";
import {
  hasErrorCode,
  resolveContainedSkillFilePath,
  validateLocalSkillDirectory,
} from "./skillFileUtils";

interface AgentSkillDeleteToolArgs {
  name: string;
  target?: string | null;
  filePath?: string | null;
  confirm: boolean;
}

/** Capture budget violation: skip journaling entirely (never a partial inverse). */
class CaptureBudgetExceededError extends Error {}

/**
 * Enforce the inverse-capture budgets. `sizeBytes` is the file's on-disk size
 * (checked BEFORE reading so an attacker-sized file is never buffered).
 * Returns the new running total; throws when any budget is exceeded.
 */
function assertCaptureBudget(fileCount: number, sizeBytes: number, totalBytes: number): number {
  if (fileCount >= REFINEMENT_CAPTURE_MAX_FILES) {
    throw new CaptureBudgetExceededError(
      `skill has more than ${REFINEMENT_CAPTURE_MAX_FILES} files`
    );
  }
  if (sizeBytes > REFINEMENT_CAPTURE_MAX_FILE_BYTES) {
    throw new CaptureBudgetExceededError(
      `file exceeds ${REFINEMENT_CAPTURE_MAX_FILE_BYTES} bytes (${sizeBytes})`
    );
  }
  const newTotal = totalBytes + sizeBytes;
  if (newTotal > REFINEMENT_CAPTURE_MAX_TOTAL_BYTES) {
    throw new CaptureBudgetExceededError(
      `skill exceeds ${REFINEMENT_CAPTURE_MAX_TOTAL_BYTES} total bytes`
    );
  }
  return newTotal;
}

/**
 * Capture every regular file under a local skill dir (refinement inverse for a
 * whole-skill delete). Contents are captured as UTF-8 text; symlinks are
 * skipped (the tool refuses symlink targets anyway). Returns null when capture
 * fails or exceeds the capture budgets: the delete then proceeds unjournaled
 * (log-only) rather than failing.
 */
async function captureLocalSkillFiles(skillDir: string): Promise<RefinementFileCapture[] | null> {
  try {
    const captures: RefinementFileCapture[] = [];
    let totalBytes = 0;
    const walk = async (dir: string): Promise<void> => {
      const entries = await fsPromises.readdir(dir, { withFileTypes: true });
      entries.sort((a, b) => (a.name < b.name ? -1 : 1));
      for (const entry of entries) {
        const entryPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          await walk(entryPath);
        } else if (entry.isFile()) {
          const { size } = await fsPromises.stat(entryPath);
          totalBytes = assertCaptureBudget(captures.length, size, totalBytes);
          captures.push({
            path: entryPath,
            content: await fsPromises.readFile(entryPath, "utf-8"),
          });
        }
      }
    };
    await walk(skillDir);
    return captures;
  } catch (error) {
    if (error instanceof CaptureBudgetExceededError) {
      log.debug("[agent_skill_delete] skipping refinement inverse: capture budget exceeded", {
        skillDir,
        reason: error.message,
      });
      return null;
    }
    log.debug("[agent_skill_delete] failed to capture skill files for refinement inverse", {
      skillDir,
      error,
    });
    return null;
  }
}

/**
 * Runtime-path variant of captureLocalSkillFiles. `find` runs relative to the
 * skill dir so its output stays namespace-agnostic (remote runtimes translate
 * paths embedded in commands); results are resolved back to runtime paths.
 */
async function captureRuntimeSkillFiles(
  runtime: Runtime,
  skillDir: string
): Promise<RefinementFileCapture[] | null> {
  try {
    const findResult = await execBuffered(runtime, "find . -type f", {
      cwd: skillDir,
      timeout: 10,
    });
    if (findResult.exitCode !== 0) {
      log.debug("[agent_skill_delete] find failed while capturing refinement inverse", {
        skillDir,
        stderr: findResult.stderr,
      });
      return null;
    }
    const relPaths = findResult.stdout
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.length > 0)
      .map((line) => line.replace(/^\.\//, ""))
      .sort();
    const captures: RefinementFileCapture[] = [];
    let totalBytes = 0;
    for (const relPath of relPaths) {
      const runtimePath = runtime.normalizePath(relPath, skillDir);
      const { size } = await runtime.stat(runtimePath);
      totalBytes = assertCaptureBudget(captures.length, size, totalBytes);
      captures.push({ path: runtimePath, content: await readFileString(runtime, runtimePath) });
    }
    return captures;
  } catch (error) {
    if (error instanceof CaptureBudgetExceededError) {
      log.debug("[agent_skill_delete] skipping refinement inverse: capture budget exceeded", {
        skillDir,
        reason: error.message,
      });
      return null;
    }
    log.debug("[agent_skill_delete] failed to capture skill files for refinement inverse", {
      skillDir,
      error,
    });
    return null;
  }
}

/**
 * Tool that deletes skills/files under the contextual skills directory.
 */
export const createAgentSkillDeleteTool: ToolFactory = (config: ToolConfiguration) => {
  return tool({
    description: TOOL_DEFINITIONS.agent_skill_delete.description,
    inputSchema: TOOL_DEFINITIONS.agent_skill_delete.schema,
    execute: async (
      { name, target, filePath, confirm }: AgentSkillDeleteToolArgs,
      { toolCallId }
    ): Promise<AgentSkillDeleteToolResult> => {
      if (!confirm) {
        return {
          success: false,
          error: "Refusing to delete skill content without confirm: true",
        };
      }

      const parsedName = SkillNameSchema.safeParse(name);
      if (!parsedName.success) {
        return {
          success: false,
          error: parsedName.error.message,
        };
      }

      try {
        const skillCtx = resolveSkillStorageContext({
          runtime: config.runtime,
          workspacePath: config.cwd,
          muxScope: config.muxScope ?? null,
        });

        if (skillCtx.kind === "project-runtime") {
          const skillsRoot = config.runtime.normalizePath(".mux/skills", skillCtx.workspacePath);
          const skillDir = config.runtime.normalizePath(parsedName.data, skillsRoot);
          await ensureRuntimePathWithinWorkspace(
            config.runtime,
            skillCtx.workspacePath,
            skillDir,
            "Skill directory"
          );
          const targetMode = target ?? "file";

          if (targetMode === "skill") {
            let skillDirStat: FileStat;
            try {
              skillDirStat = await config.runtime.stat(skillDir);
            } catch (error) {
              const message = getErrorMessage(error);
              if (/enoent|no such file|does not exist/i.test(message)) {
                return {
                  success: false,
                  error: `Skill not found: ${parsedName.data}`,
                };
              }

              return {
                success: false,
                error: message,
              };
            }

            if (!skillDirStat.isDirectory) {
              return {
                success: false,
                error: `Skill not found: ${parsedName.data}`,
              };
            }

            // Prior contents must be captured before removal (refinement inverse).
            const skillCaptures = await captureRuntimeSkillFiles(config.runtime, skillDir);

            const rmSkillResult = await execBuffered(
              config.runtime,
              `rm -rf ${quoteRuntimeProbePath(skillDir)}`,
              {
                cwd: skillCtx.workspacePath,
                timeout: 10,
              }
            );

            if (rmSkillResult.exitCode !== 0) {
              const details = (rmSkillResult.stderr || rmSkillResult.stdout).trim();
              return {
                success: false,
                error: details || `Failed to delete skill directory '${parsedName.data}'`,
              };
            }

            if (skillCaptures !== null) {
              await appendRefinementEventFromTool(config, {
                kind: "skill",
                action: { op: "delete-skill", skillName: parsedName.data },
                inverse: { op: "restore-files", files: skillCaptures },
                evidence: { toolName: "agent_skill_delete", toolCallId },
              });
            }

            return {
              success: true,
              deleted: "skill",
            };
          }

          if (filePath == null) {
            return {
              success: false,
              error: "filePath is required when target is 'file'",
            };
          }

          let resolvedPath: string;
          try {
            ({ resolvedPath } = await resolveContainedSkillFilePathOnRuntime(
              config.runtime,
              skillDir,
              filePath
            ));
            const targetContainment = await inspectContainmentOnRuntime(
              config.runtime,
              skillDir,
              resolvedPath
            );
            if (targetContainment.leafSymlink) {
              return {
                success: false,
                error: `Target file is a symbolic link and cannot be accessed: ${filePath}`,
              };
            }
            await ensureRuntimePathWithinWorkspace(
              config.runtime,
              skillCtx.workspacePath,
              resolvedPath,
              "Skill file"
            );
          } catch (error) {
            return {
              success: false,
              error: getErrorMessage(error),
            };
          }

          // Prior content must be captured before removal (refinement inverse).
          // Null capture (e.g. unreadable or over-budget file) skips
          // journaling, never the delete.
          let fileCapture: RefinementFileCapture | null = null;
          try {
            const { size } = await config.runtime.stat(resolvedPath);
            if (size > REFINEMENT_CAPTURE_MAX_FILE_BYTES) {
              log.debug(
                "[agent_skill_delete] skipping refinement inverse: capture budget exceeded",
                { resolvedPath, size }
              );
            } else {
              fileCapture = {
                path: resolvedPath,
                content: await readFileString(config.runtime, resolvedPath),
              };
            }
          } catch (error) {
            log.debug("[agent_skill_delete] failed to capture file for refinement inverse", {
              resolvedPath,
              error,
            });
          }

          const rmFileResult = await execBuffered(
            config.runtime,
            `rm ${quoteRuntimeProbePath(resolvedPath)}`,
            {
              cwd: skillCtx.workspacePath,
              timeout: 10,
            }
          );

          if (rmFileResult.exitCode !== 0) {
            const details = (rmFileResult.stderr || rmFileResult.stdout).trim();
            if (/No such file/i.test(details)) {
              return {
                success: false,
                error: `File not found in skill '${parsedName.data}': ${filePath}`,
              };
            }

            return {
              success: false,
              error: details || `Failed to delete file in skill '${parsedName.data}'`,
            };
          }

          if (fileCapture !== null) {
            await appendRefinementEventFromTool(config, {
              kind: "skill",
              action: { op: "delete-file", skillName: parsedName.data, filePath },
              inverse: { op: "restore-files", files: [fileCapture] },
              evidence: { toolName: "agent_skill_delete", toolCallId },
            });
          }

          return {
            success: true,
            deleted: "file",
          };
        }

        const { muxScope } = config;
        if (!muxScope) {
          throw new Error("agent_skill_delete requires muxScope");
        }

        const skillsRoot =
          muxScope.type === "project"
            ? path.join(muxScope.projectRoot, ".mux", "skills")
            : path.join(muxScope.muxHome, "skills");
        // Containment is anchored at workspace root (project) or mux home (global).
        const containmentRoot =
          muxScope.type === "project" ? muxScope.projectRoot : muxScope.muxHome;

        const skillDir = path.join(skillsRoot, parsedName.data);

        let skillDirStat;
        try {
          ({ skillDirStat } = await validateLocalSkillDirectory(containmentRoot, skillDir));
        } catch (error) {
          if (hasErrorCode(error, "ENOENT")) {
            // A missing mux home/workspace root means there cannot be a contained skill to delete.
            return {
              success: false,
              error: `Skill not found: ${parsedName.data}`,
            };
          }

          return {
            success: false,
            error: getErrorMessage(error),
          };
        }

        if (!skillDirStat) {
          return {
            success: false,
            error: `Skill not found: ${parsedName.data}`,
          };
        }

        if (!skillDirStat.isDirectory()) {
          return {
            success: false,
            error: `Skill path is not a directory: ${parsedName.data}`,
          };
        }

        const targetMode = target ?? "file";
        if (targetMode === "skill") {
          // Prior contents must be captured before removal (refinement inverse).
          const skillCaptures = await captureLocalSkillFiles(skillDir);
          await fsPromises.rm(skillDir, { recursive: true });
          if (skillCaptures !== null) {
            await appendRefinementEventFromTool(config, {
              kind: "skill",
              action: { op: "delete-skill", skillName: parsedName.data },
              inverse: { op: "restore-files", files: skillCaptures },
              evidence: { toolName: "agent_skill_delete", toolCallId },
            });
          }
          return {
            success: true,
            deleted: "skill",
          };
        }

        if (filePath == null) {
          return {
            success: false,
            error: "filePath is required when target is 'file'",
          };
        }

        let targetPath: string;
        try {
          ({ resolvedPath: targetPath } = await resolveContainedSkillFilePath(skillDir, filePath, {
            allowMissingLeaf: true,
          }));
        } catch (error) {
          return {
            success: false,
            error: getErrorMessage(error),
          };
        }

        let targetStat;
        try {
          targetStat = await fsPromises.lstat(targetPath);
        } catch (error) {
          if (hasErrorCode(error, "ENOENT")) {
            return {
              success: false,
              error: `File not found in skill '${parsedName.data}': ${filePath}`,
            };
          }
          throw error;
        }

        if (targetStat.isSymbolicLink()) {
          return {
            success: false,
            error: "Refusing to delete a symlinked skill file target",
          };
        }

        if (targetStat.isDirectory()) {
          return {
            success: false,
            error: `Path is a directory, not a file: ${filePath}`,
          };
        }

        // Prior content must be captured before removal (refinement inverse).
        // Null capture (e.g. unreadable or over-budget file) skips journaling,
        // never the delete. lstat size is checked before reading so an
        // attacker-sized file is never buffered.
        let localFileCapture: RefinementFileCapture | null = null;
        if (targetStat.size > REFINEMENT_CAPTURE_MAX_FILE_BYTES) {
          log.debug("[agent_skill_delete] skipping refinement inverse: capture budget exceeded", {
            targetPath,
            size: targetStat.size,
          });
        } else {
          try {
            localFileCapture = {
              path: targetPath,
              content: await fsPromises.readFile(targetPath, "utf-8"),
            };
          } catch (error) {
            log.debug("[agent_skill_delete] failed to capture file for refinement inverse", {
              targetPath,
              error,
            });
          }
        }

        await fsPromises.unlink(targetPath);

        if (localFileCapture !== null) {
          await appendRefinementEventFromTool(config, {
            kind: "skill",
            action: { op: "delete-file", skillName: parsedName.data, filePath },
            inverse: { op: "restore-files", files: [localFileCapture] },
            evidence: { toolName: "agent_skill_delete", toolCallId },
          });
        }

        return {
          success: true,
          deleted: "file",
        };
      } catch (error) {
        return {
          success: false,
          error: `Failed to delete skill: ${getErrorMessage(error)}`,
        };
      }
    },
  });
};
