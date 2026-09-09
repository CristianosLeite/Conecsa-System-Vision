// The editor-token verifier must accept exactly what gateway/flow_token.py
// mints (same secret, same format) and nothing else.
const crypto = require("crypto");
const path = require("path");

const { verify, adminAuth, permissionsFor } = require(
  path.join(__dirname, "..", "..", "..", "admin-token.js")
);

const SECRET = "test-secret";

function mint(payload, secret = SECRET) {
  // Mirrors flow_token.mint(): compact, key-sorted JSON, base64url, HMAC over
  // "v1.<payload>".
  const body = Buffer.from(JSON.stringify(payload)).toString("base64url");
  const signedPart = `v1.${body}`;
  const sig = crypto.createHmac("sha256", secret).update(signedPart).digest("base64url");
  return `${signedPart}.${sig}`;
}

const future = Math.floor(Date.now() / 1000) + 600;

describe("admin-token verify", () => {
  it("accepts a valid admin token with full permissions", () => {
    const user = verify(mint({ exp: future, role: "admin", sub: "ana" }), SECRET);
    expect(user).toEqual({ username: "ana", permissions: "*" });
  });

  it("gives a plain user read-only access", () => {
    const user = verify(mint({ exp: future, role: "user", sub: "bob" }), SECRET);
    expect(user).toEqual({ username: "bob", permissions: "read" });
  });

  it("rejects an expired token", () => {
    const token = mint({ exp: 1000, role: "admin", sub: "ana" });
    expect(verify(token, SECRET, 1000)).toBeNull();
    expect(verify(token, SECRET, 999)).not.toBeNull();
  });

  it("rejects a bad signature or another secret", () => {
    expect(verify(mint({ exp: future, role: "admin", sub: "ana" }, "other"), SECRET)).toBeNull();
    const token = mint({ exp: future, role: "admin", sub: "ana" });
    expect(verify(token.slice(0, -2) + "AA", SECRET)).toBeNull();
  });

  it("rejects a tampered payload", () => {
    const token = mint({ exp: future, role: "user", sub: "bob" });
    const [v, , sig] = token.split(".");
    const escalated = Buffer.from(JSON.stringify({ exp: future, role: "admin", sub: "bob" }))
      .toString("base64url");
    expect(verify(`${v}.${escalated}.${sig}`, SECRET)).toBeNull();
  });

  it("rejects malformed input and unknown roles", () => {
    expect(verify("", SECRET)).toBeNull();
    expect(verify("v1.only", SECRET)).toBeNull();
    expect(verify("v2.a.b", SECRET)).toBeNull();
    expect(verify(mint({ exp: future, role: "root", sub: "x" }), SECRET)).toBeNull();
    expect(verify(mint({ exp: "soon", role: "admin", sub: "x" }), SECRET)).toBeNull();
    expect(verify(mint({ exp: future, role: "admin", sub: "x" }), "")).toBeNull();
  });

  it("maps roles to Node-RED permissions", () => {
    expect(permissionsFor("owner")).toBe("*");
    expect(permissionsFor("admin")).toBe("*");
    expect(permissionsFor("user")).toBe("read");
    expect(permissionsFor("")).toBeNull();
  });
});

describe("adminAuth block", () => {
  it("is built from the dedicated secret, else the credential secret", async () => {
    const auth = adminAuth({ NODE_RED_CREDENTIAL_SECRET: SECRET });
    expect(auth.type).toBe("credentials");
    expect(auth.users).toEqual([]);
    const user = await auth.tokens(mint({ exp: future, role: "admin", sub: "ana" }));
    expect(user.permissions).toBe("*");

    const dedicated = adminAuth({ NODE_RED_CREDENTIAL_SECRET: "other", FLOW_ADMIN_TOKEN_SECRET: SECRET });
    expect(await dedicated.tokens(mint({ exp: future, role: "admin", sub: "ana" }))).not.toBeNull();
  });

  it("is off without a secret or with FLOW_ADMIN_AUTH=0", () => {
    expect(adminAuth({})).toBeNull();
    expect(adminAuth({ NODE_RED_CREDENTIAL_SECRET: SECRET, FLOW_ADMIN_AUTH: "0" })).toBeNull();
  });
});
