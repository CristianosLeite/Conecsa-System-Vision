// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: Apache-2.0

"use strict";
const helper = require("node-red-node-test-helper");
const detectionNode = require("../nodes/detection/detection.js");
const { startMockGateway } = require("./_mock-gateway");

helper.init(require.resolve("node-red"));

describe("detection node", () => {
  let gw;

  beforeEach((done) => { helper.startServer(done); });
  afterEach(async () => {
    await helper.unload();
    await new Promise((r) => helper.stopServer(r));
    if (gw) await gw.close();
    gw = null;
  });

  it("emits the snapshot payload when there are detections", async () => {
    gw = await startMockGateway({
      "GET /api/v1/detections/snapshot": (req, res) =>
        res.end(
          JSON.stringify({
            total: 1,
            model: "yolo",
            detections: [{ class_name: "cap", area: { label: "zone-1" } }],
          })
        ),
    });
    const flow = [
      { id: "n1", type: "conecsa-detection", inferenceUrl: gw.url, mode: "on-change", includeFrame: false, wires: [["n2"]] },
      { id: "n2", type: "helper" },
    ];
    await helper.load(detectionNode, flow);
    const n2 = helper.getNode("n2");

    const msg = await new Promise((resolve) => n2.on("input", resolve));
    expect(msg.payload.total).toBe(1);
    expect(msg.payload.detections[0].class_name).toBe("cap");
  }, 8000);

  it("tags the payload with the configured device id", async () => {
    gw = await startMockGateway({
      "GET /api/v1/detections/snapshot": (req, res) =>
        res.end(JSON.stringify({ total: 1, model: "m", detections: [{ class_name: "cap" }] })),
    });
    const flow = [
      { id: "n1", type: "conecsa-detection", inferenceUrl: gw.url, mode: "on-change", deviceId: "cam-7", wires: [["n2"]] },
      { id: "n2", type: "helper" },
    ];
    await helper.load(detectionNode, flow);
    const n2 = helper.getNode("n2");

    const msg = await new Promise((resolve) => n2.on("input", resolve));
    expect(msg.payload.device_id).toBe("cam-7");
  }, 8000);

  it("does not emit when there are no detections", async () => {
    gw = await startMockGateway({
      "GET /api/v1/detections/snapshot": (req, res) =>
        res.end(JSON.stringify({ total: 0, model: "m", detections: [] })),
    });
    const flow = [
      { id: "n1", type: "conecsa-detection", inferenceUrl: gw.url, mode: "on-change", wires: [["n2"]] },
      { id: "n2", type: "helper" },
    ];
    await helper.load(detectionNode, flow);
    const n2 = helper.getNode("n2");

    let emitted = false;
    n2.on("input", () => (emitted = true));
    // Wait past the initial fetch (setTimeout 1s + 300ms poll) to be sure.
    await new Promise((r) => setTimeout(r, 1500));
    expect(emitted).toBe(false);
  }, 8000);

  it("passes a classification result through: one item without bbox, the task and the candidates", async () => {
    gw = await startMockGateway({
      "GET /api/v1/detections/snapshot": (req, res) =>
        res.end(
          JSON.stringify({
            task: "classify",
            total: 1,
            model: "pets.engine",
            detections: [{ class_name: "cat", confidence: 0.91, area: null, color: "#00ff00" }],
            candidates: [
              { class_id: 0, class_name: "cat", confidence: 0.91 },
              { class_id: 1, class_name: "dog", confidence: 0.09 },
            ],
          })
        ),
    });
    const flow = [
      { id: "n1", type: "conecsa-detection", inferenceUrl: gw.url, mode: "on-change", includeFrame: false, wires: [["n2"]] },
      { id: "n2", type: "helper" },
    ];
    await helper.load(detectionNode, flow);
    const n2 = helper.getNode("n2");

    const msg = await new Promise((resolve) => n2.on("input", resolve));
    expect(msg.payload.task).toBe("classify");
    expect(msg.payload.total).toBe(1);
    expect(msg.payload.detections).toHaveLength(1);
    expect(msg.payload.detections[0].bbox).toBeUndefined();
    expect(msg.payload.candidates.map((c) => c.class_name)).toEqual(["cat", "dog"]);
  }, 8000);

  it("passes a segmentation result through: each item keeps its outline", async () => {
    const ring = [
      [0.1, 0.2],
      [0.3, 0.2],
      [0.3, 0.4],
    ];
    gw = await startMockGateway({
      "GET /api/v1/detections/snapshot": (req, res) =>
        res.end(
          JSON.stringify({
            task: "segment",
            total: 1,
            model: "parts.engine",
            detections: [
              { class_name: "bolt", confidence: 0.88, area: null, color: "#ff0000", bbox: [0.1, 0.2, 0.3, 0.4], polygons: [ring] },
            ],
          })
        ),
    });
    const flow = [
      { id: "n1", type: "conecsa-detection", inferenceUrl: gw.url, mode: "on-change", includeFrame: false, wires: [["n2"]] },
      { id: "n2", type: "helper" },
    ];
    await helper.load(detectionNode, flow);
    const n2 = helper.getNode("n2");

    const msg = await new Promise((resolve) => n2.on("input", resolve));
    expect(msg.payload.task).toBe("segment");
    expect(msg.payload.detections[0].polygons).toEqual([ring]);
  }, 8000);
});
