// The container's settings.js must refuse to start without a real
// credential-encryption secret: the old fallback was a source-known constant,
// which made every deployment's stored third-party credentials recoverable.
const path = require("path");

const SETTINGS_PATH = path.join(__dirname, "..", "..", "..", "settings.js");

function loadSettingsFresh() {
  delete require.cache[require.resolve(SETTINGS_PATH)];
  return require(SETTINGS_PATH);
}

describe("settings.js credential secret", () => {
  const saved = process.env.NODE_RED_CREDENTIAL_SECRET;

  afterEach(() => {
    if (saved === undefined) {
      delete process.env.NODE_RED_CREDENTIAL_SECRET;
    } else {
      process.env.NODE_RED_CREDENTIAL_SECRET = saved;
    }
  });

  it("refuses to start without NODE_RED_CREDENTIAL_SECRET", () => {
    delete process.env.NODE_RED_CREDENTIAL_SECRET;
    expect(loadSettingsFresh).toThrow(/NODE_RED_CREDENTIAL_SECRET/);
  });

  it("uses the provisioned secret verbatim, with no fallback", () => {
    process.env.NODE_RED_CREDENTIAL_SECRET = "per-deployment-secret";
    const settings = loadSettingsFresh();
    expect(settings.credentialSecret).toBe("per-deployment-secret");
  });

  it("grants no cross-origin access to HTTP-in nodes", () => {
    process.env.NODE_RED_CREDENTIAL_SECRET = "x";
    const settings = loadSettingsFresh();
    expect(settings.httpNodeCors).toBeUndefined();
  });
});
