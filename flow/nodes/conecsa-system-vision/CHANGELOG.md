# Changelog

## Unreleased

- **examples**: new `face-access.json` — two-factor access control on a face
  recognition device: a badge reader (an **inject** node standing in for it)
  and the **detection** node (polled every second) both feed a function node that
  releases only when the badge holder's face is recognized above a minimum
  similarity within a few seconds of the badge → a core **trigger** node
  pulsing → the **gpio** node on pin 29. Import it from Import → Examples.
  Face recognition has no liveness check, so a face alone opens nothing.
- **detection**: the help and README document face recognition devices — the
  payload's `task` value `face`, one item per face whose `class_name` is the
  person's name (or `unknown`) and whose `confidence` is the similarity to
  that person, and `total` counting the faces in the frame. Passed through
  as-is; no code changes.

- **detection**: the help and README document classification devices — the
  payload's `task`, the single frame-class item without `bbox`, and the top-k
  under `candidates`. The node already passed them through; no code changes.
- **detection**: the help and README document segmentation devices — each
  item's `polygons` (its outline as rings of normalized vertices) and the
  payload's `polygons_truncated`. Passed through as-is; no code changes.

## 1.1.3 — 2026-09-22

- Require a badge beside the face in the access example.

## 1.1.2 — 2026-09-03

- The bundled `NOTICE` now names the scoped packageeq
  `@conecsa/node-red-contrib-conecsa-system-vision`. No functional changes.

## 1.1.1 — 2026-08-22

- Published under the `@conecsa` npm scope:
  `@conecsa/node-red-contrib-conecsa-system-vision`. README and install
  instructions updated; no code changes.

## 1.1.0 — 2026-08-22

First public release on npm.

- **Hub mode.** New `conecsa-hub` configuration node (host, port, API key,
  CA certificate upload, verify) holding the connection to a Conecsa hub's
  Developer API. Every node gained a **Hub** and a **Device** field; with both
  set, requests go to `https://<hub>:<port>/devices/<device>/api/...` carrying
  `X-Api-Key` and trusting the hub CA. Without a hub the nodes keep calling an
  api-gateway directly.
- **"Inference URL" is now "API endpoint"** (same stored property; existing
  flows keep their value). Read-only when a hub is selected.
- **Node type ids are prefixed** `conecsa-` (`conecsa-stats`,
  `conecsa-start-stop`, …) so they cannot collide with other packages. Palette
  labels are unchanged. Flows built with the unprefixed types (1.0.0, only ever
  shipped inside the device image) must be re-imported: export, replace the
  `"type"` values, import.
- HTTP error statuses (401, 403, 503, …) are now reported as node errors
  instead of being parsed as successful replies.
- `detection` tags its payload with the hub device id when no explicit
  device id is configured.
- License: Apache-2.0.

## 1.0.0

Internal release bundled in the Conecsa System Vision device image.
