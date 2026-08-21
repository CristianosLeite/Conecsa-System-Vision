/**
 * @file Shared node scaffolding. Every Conecsa node starts the same way —
 *   createNode, resolve the api-gateway base URL from the node config, paint
 *   an initial ring status — and that boilerplate lived copied in each node.
 *   The HTTP/SSE transport is in ./http-client; this module owns only the
 *   node lifecycle scaffold.
 */
"use strict";

const { inferenceBaseUrl } = require("./http-client");

/**
 * Standard node initialization: registers the node with the runtime, resolves
 * `node.inferenceUrl`, and seeds the status ring.
 *
 * @param {object} RED Node-RED runtime.
 * @param {object} node The node instance (`this` inside the constructor).
 * @param {object} config The node's editor configuration.
 * @param {string} [statusText] Initial grey-ring status ("idle" by default).
 */
function initNode(RED, node, config, statusText = "idle") {
  RED.nodes.createNode(node, config);
  node.inferenceUrl = inferenceBaseUrl(config);
  node.status({ fill: "grey", shape: "ring", text: statusText });
}

module.exports = { initNode };
