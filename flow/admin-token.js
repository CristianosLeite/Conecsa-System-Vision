// Verifier for the editor tokens the api-gateway mints (gateway/flow_token.py).
//
// Format: "v1.<base64url(JSON payload)>.<base64url(HMAC-SHA256 over the first
// two parts)>", payload {"sub": username, "role": role, "exp": unix seconds}.
// Both sides sign with FLOW_ADMIN_TOKEN_SECRET, falling back to the required
// NODE_RED_CREDENTIAL_SECRET, so no extra deployment secret is needed.
//
// Node-RED calls `tokens(token)` for the bearer on every admin request and on
// the comms websocket; it expects a user object ({username, permissions}) or
// null. Owner/admin may edit and deploy ("*"); a plain user may only read.
"use strict";

const crypto = require("crypto");

const VERSION = "v1";

function b64urlDecode(text) {
  return Buffer.from(text, "base64url");
}

function signature(secret, signedPart) {
  return crypto.createHmac("sha256", secret).update(signedPart).digest();
}

/** Permissions Node-RED grants for a role stamped by the hub. */
function permissionsFor(role) {
  if (role === "owner" || role === "admin") {
    return "*";
  }
  if (role === "user") {
    return "read";
  }
  return null;
}

/**
 * Verify `token` with `secret`; returns the Node-RED user or null.
 * `now` (seconds) is injectable for tests.
 */
function verify(token, secret, now) {
  if (typeof token !== "string" || !secret) {
    return null;
  }
  const parts = token.split(".");
  if (parts.length !== 3 || parts[0] !== VERSION) {
    return null;
  }
  const signedPart = `${parts[0]}.${parts[1]}`;
  const expected = signature(secret, signedPart);
  let given;
  try {
    given = b64urlDecode(parts[2]);
  } catch (e) {
    return null;
  }
  if (given.length !== expected.length || !crypto.timingSafeEqual(given, expected)) {
    return null;
  }
  let payload;
  try {
    payload = JSON.parse(b64urlDecode(parts[1]).toString("utf8"));
  } catch (e) {
    return null;
  }
  if (!payload || typeof payload !== "object" || !Number.isInteger(payload.exp)) {
    return null;
  }
  const current = now === undefined ? Date.now() / 1000 : now;
  if (payload.exp <= current) {
    return null;
  }
  const permissions = permissionsFor(payload.role);
  if (permissions === null) {
    return null;
  }
  return { username: String(payload.sub || ""), permissions };
}

/** Build the `adminAuth` block for settings.js, or null when no secret is set. */
function adminAuth(env) {
  const secret = env.FLOW_ADMIN_TOKEN_SECRET || env.NODE_RED_CREDENTIAL_SECRET || "";
  if (!secret || env.FLOW_ADMIN_AUTH === "0") {
    return null;
  }
  return {
    type: "credentials",
    // No password users: the only way in is a gateway-minted bearer token.
    users: [],
    tokens: (token) => Promise.resolve(verify(token, secret)),
  };
}

module.exports = { verify, adminAuth, permissionsFor };
