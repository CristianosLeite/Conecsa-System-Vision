# flow

The device's Node-RED image: automation flows on `:1880`, reached in production only through the
nginx `/flow` proxy. It ships the Conecsa custom nodes and a seeded default flow. In production the
editor accepts only tokens minted by the api-gateway; the dev compose stack turns editor
authentication off (`FLOW_ADMIN_AUTH=0`, set it to `1` to exercise the tokens).

Full description: [docs/services/flow.md](../docs/services/flow.md).

## Layout

| Path | What it is |
|---|---|
| `Dockerfile` | The Node-RED image, pinned by digest, with the node package and theme installed |
| `settings.js` | Node-RED settings (`/flow` root, credential secret, editor authentication) |
| `admin-token.js` | Verifies the editor bearer tokens the api-gateway mints |
| `conecsa-entrypoint.sh` | Refreshes the image's config into the `/data` volume on each start and seeds `flows.json` without overwriting user edits |
| `flows.json` | The default flow |
| `theme-auto.css`, `conecsa_white_logo.png` | Editor theme and logo |
| `nodes/conecsa-system-vision/` | The custom node package, also published to npm; see its [README](nodes/conecsa-system-vision/README.md) |

## Run

```bash
docker compose -f docker-compose.dev.yml up -d --build flow   # editor on http://localhost:1880
```

`scripts/dev.sh` builds and runs this image with host networking. Production requires
`NODE_RED_CREDENTIAL_SECRET` in `.env`, and compose refuses to start without it.

## Test

```bash
cd flow/nodes/conecsa-system-vision && npm install && npm test   # jest
```

The suite covers the nodes, `lib/`, `settings.js` and `admin-token.js`.

## Constraints

- The node package is Apache-2.0 and must not import AGPL code.
- The token format in `admin-token.js` must match `api-gateway/gateway/flow_token.py`.
- Publishing the package to npm is a manual step (see the package README).

## Reference

- [Configuration: `flow`](../docs/configuration.md#flow)
- [HTTP API reference](../docs/api-reference.md)
