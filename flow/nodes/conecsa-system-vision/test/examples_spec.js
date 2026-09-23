// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: Apache-2.0

"use strict";
const fs = require("fs");
const path = require("path");

// The example flows are shipped in the npm package and imported by users
// through Import -> Examples, so a malformed export or a node type this
// package does not register would only fail in the editor. Parse them here.
const EXAMPLES_DIR = path.join(__dirname, "..", "examples");
const PACKAGE_TYPES = Object.keys(require("../package.json")["node-red"].nodes);
// Node-RED core node types the examples are allowed to use.
const CORE_TYPES = ["tab", "inject", "function", "trigger", "debug"];

describe("example flows", () => {
  const files = fs.readdirSync(EXAMPLES_DIR).filter((f) => f.endsWith(".json"));

  it("ships at least the direct, hub and face access examples", () => {
    expect(files).toEqual(expect.arrayContaining(["direct-mode.json", "hub-mode.json", "face-access.json"]));
  });

  for (const file of files) {
    describe(file, () => {
      const flow = JSON.parse(fs.readFileSync(path.join(EXAMPLES_DIR, file), "utf8"));

      it("is an array of nodes with unique ids", () => {
        expect(Array.isArray(flow)).toBe(true);
        const ids = flow.map((n) => n.id);
        expect(ids.every((id) => typeof id === "string" && id.length > 0)).toBe(true);
        expect(new Set(ids).size).toBe(ids.length);
      });

      it("uses only this package's node types and core ones", () => {
        for (const node of flow) {
          expect(CORE_TYPES.concat(PACKAGE_TYPES)).toContain(node.type);
        }
      });

      it("wires only to nodes of the same flow", () => {
        const ids = new Set(flow.map((n) => n.id));
        for (const node of flow) {
          for (const wire of node.wires || []) {
            for (const target of wire) expect(ids.has(target)).toBe(true);
          }
          if (node.z) expect(ids.has(node.z)).toBe(true);
        }
      });
    });
  }

  it("the face access example gates the pin on a badge and an authorized name", () => {
    const flow = JSON.parse(fs.readFileSync(path.join(EXAMPLES_DIR, "face-access.json"), "utf8"));
    const detection = flow.find((n) => n.type === "conecsa-detection");
    // Polled, not on change: a person who stays in frame must keep the face
    // factor fresh for a badge presented later.
    expect(detection.mode).toBe("interval");
    expect(Number(detection.interval)).toBeLessThanOrEqual(2);
    const fn = flow.find((n) => n.type === "function");
    // The gate is the point of the example: a face alone must not reach the
    // pin (no liveness check), an unknown face never does, and the pulse
    // must be a trigger, not a permanent HIGH.
    expect(fn.func).toContain("AUTHORIZED");
    expect(fn.func).toContain("MIN_CONFIDENCE");
    expect(fn.func).toContain("WINDOW_S");
    expect(fn.func).toContain('msg.topic === "badge"');
    // A detection device with a class named like a person is not a face.
    expect(fn.func).toContain('msg.payload.task !== "face"');
    const badge = flow.find((n) => n.type === "inject");
    expect(badge.topic).toBe("badge");
    expect(badge.wires[0]).toEqual([fn.id]);
    expect(detection.wires[0]).toEqual([fn.id]);
    expect(fn.wires[0]).toEqual([expect.stringContaining("pulse")]);
    const trigger = flow.find((n) => n.type === "trigger");
    expect(trigger.units).toBe("s");
    const gpio = flow.find((n) => n.type === "conecsa-gpio");
    expect(gpio.action).toBe("payload");
    expect(trigger.wires[0]).toContain(gpio.id);
  });
});
