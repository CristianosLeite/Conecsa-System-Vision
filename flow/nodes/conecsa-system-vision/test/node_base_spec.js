// The shared node scaffold must register the node, resolve the gateway base
// URL, and seed the status ring exactly like the copies it replaced.
const { initNode } = require("../lib/node-base");

function fakeRed() {
  return { nodes: { createNode: jest.fn() } };
}

function fakeNode() {
  return { status: jest.fn() };
}

describe("initNode", () => {
  it("registers, resolves the base URL, and seeds an idle ring", () => {
    const RED = fakeRed();
    const node = fakeNode();
    const config = { inferenceUrl: "http://gateway:5000" };
    initNode(RED, node, config);
    expect(RED.nodes.createNode).toHaveBeenCalledWith(node, config);
    expect(node.inferenceUrl).toBe("http://gateway:5000");
    expect(node.status).toHaveBeenCalledWith({
      fill: "grey", shape: "ring", text: "idle",
    });
  });

  it("accepts a custom seed status", () => {
    const node = fakeNode();
    initNode(fakeRed(), node, {}, "connecting");
    expect(node.status).toHaveBeenCalledWith({
      fill: "grey", shape: "ring", text: "connecting",
    });
  });
});
