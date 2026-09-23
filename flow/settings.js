// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

// Credential-encryption secret: REQUIRED. Third-party credentials stored in
// flows are encrypted with it; a source-known fallback would make them
// recoverable from any deployment, so startup fails loudly instead. Note:
// changing the secret invalidates credentials already stored in flows.
if (!process.env.NODE_RED_CREDENTIAL_SECRET) {
  throw new Error(
    "NODE_RED_CREDENTIAL_SECRET is not set; refusing to start with a " +
    "publicly-known credential secret. Provision it in the compose file."
  );
}

// Editor authentication: bearer tokens minted by the api-gateway for the
// operator the hub vouched for (see admin-token.js). Without it, any local
// process that could reach the device's /flow could deploy flows.
const adminAuth = require("./admin-token.js").adminAuth(process.env);
if (!adminAuth) {
  console.warn(
    "[conecsa] Node-RED editor authentication is OFF (FLOW_ADMIN_AUTH=0 or " +
    "no secret); anyone reaching /flow can edit and deploy flows."
  );
}

module.exports = {
  // Listen on all interfaces so the container is reachable
  uiHost: "0.0.0.0",
  uiPort: 1880,

  // Served under /flow on the device's single origin (nginx proxies /flow/ →
  // flow:1880/flow/), so the WASM UI reaches it same-origin through the hub's
  // reverse proxy. Both the editor (admin) and HTTP-in nodes live under /flow.
  httpAdminRoot: "/flow",
  httpNodeRoot: "/flow",

  // Editor/admin API authentication (null = open, development only). The
  // device UI opens the editor with `?access_token=<gateway token>`; the
  // editor keeps it in localStorage and sends it as a bearer on every admin
  // call and on the comms websocket. HTTP-in nodes (httpNodeRoot) are user
  // flows and stay public.
  ...(adminAuth ? { adminAuth } : {}),

  // HTTP-in nodes are consumed same-origin (through nginx /flow or the hub's
  // reverse proxy); no cross-origin caller exists, so no CORS is granted.

  // Allow iframe embedding by the WASM UI on the same origin (served directly,
  // behind :443 mTLS, or through the hub's 127.0.0.1 reverse proxy).
  headers: {
    "Content-Security-Policy":
      "frame-ancestors 'self' http://localhost:* http://127.0.0.1:*",
  },

  // Persist flows and credentials in the mounted volume
  flowFile: "flows.json",
  // Secret used to encrypt stored credentials — required, validated above.
  credentialSecret: process.env.NODE_RED_CREDENTIAL_SECRET,

  // Logging
  logging: {
    console: {
      level: "info",
      metrics: false,
      audit: false,
    },
  },

  // Editor theme
  // theme-auto.css contains the Conecsa industrial dark theme used by the
  // embedded editor and the standalone Node-RED UI.
  editorTheme: {
    page: {
      title: "Conecsa Flow Editor",
      css: "/data/theme-auto.css",
    },
    header: {
      title: "Conecsa Flow",
      image: "/data/conecsa_white_logo.png",
    },
    palette: {
      // Pre-install packages are handled at container build time (package.json)
    },
  },
};
