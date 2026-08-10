import { describe, expect, test } from "bun:test";

import { createMCPToolInputSchema } from "./mcpClient";

describe("createMCPToolInputSchema", () => {
  test("exposes a nullable model contract and restores the server contract", async () => {
    const inputSchema = createMCPToolInputSchema({
      type: "object",
      required: ["issueId"],
      properties: {
        issueId: { type: "string" },
        cursor: { type: "string" },
        statusUpdateType: { type: "string", enum: ["project", "initiative"] },
      },
      additionalProperties: false,
    });

    expect(inputSchema.jsonSchema).toMatchObject({
      required: ["issueId"],
      additionalProperties: false,
      properties: {
        issueId: { type: "string" },
        cursor: { anyOf: [{ type: "string" }, { type: "null" }] },
        statusUpdateType: {
          anyOf: [{ type: "string", enum: ["project", "initiative"] }, { type: "null" }],
        },
      },
    });

    expect(
      await inputSchema.validate?.({
        issueId: "CODAGT-709",
        cursor: "",
        statusUpdateType: null,
      })
    ).toEqual({
      success: true,
      value: { issueId: "CODAGT-709", cursor: "" },
    });
  });

  test("preserves dictionary schemas", () => {
    const inputSchema = createMCPToolInputSchema({
      type: "object",
      additionalProperties: { type: "string" },
    });

    expect(inputSchema.jsonSchema).toEqual({
      type: "object",
      properties: {},
      additionalProperties: { type: "string" },
    });
  });
});
