import { describe, expect, it } from "bun:test";
import * as fs from "node:fs/promises";
import * as nodePath from "node:path";

import { LocalRuntime } from "@/node/runtime/LocalRuntime";
import { DisposableTempDir } from "@/node/services/tempDir";

import { createKernelFileLoader } from "./kernelFileLoad";

describe("createKernelFileLoader line counting", () => {
  it("does not count a trailing newline as an extra line", async () => {
    // The {lines} summary is model-visible and used directly for exact-count
    // tasks; a conventional newline-terminated file must not report one more
    // line than it contains.
    using tmp = new DisposableTempDir("kernel-load-lines");
    await fs.writeFile(nodePath.join(tmp.path, "terminated.txt"), "line1\nline2\n", "utf8");
    await fs.writeFile(nodePath.join(tmp.path, "unterminated.txt"), "line1\nline2", "utf8");
    await fs.writeFile(nodePath.join(tmp.path, "empty.txt"), "", "utf8");
    await fs.writeFile(nodePath.join(tmp.path, "blank-line.txt"), "line1\n\nline3\n", "utf8");

    const load = createKernelFileLoader({ cwd: tmp.path, runtime: new LocalRuntime(tmp.path) });

    expect((await load({ path: "terminated.txt" })).lines).toBe(2);
    expect((await load({ path: "unterminated.txt" })).lines).toBe(2);
    expect((await load({ path: "empty.txt" })).lines).toBe(0);
    // Interior blank lines still count as records.
    expect((await load({ path: "blank-line.txt" })).lines).toBe(3);
  });
});
