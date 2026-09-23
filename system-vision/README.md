# system-vision

The device UI: a Leptos single-page app compiled to WASM with Trunk and served by nginx. The same
container runs the nginx mTLS terminator on `:443`, which fronts the UI, the api-gateway and the
Node-RED editor. In production that port is the device's only published one, and the hub reaches it
with a client certificate.

Architecture context: [docs/architecture.md](../docs/architecture.md#frontend-rust).

## Layout

| Path | What it is |
|---|---|
| `src/lib.rs` | Crate root and the WASM entry point (`#[wasm_bindgen(start)]`); `src/main.rs` is an empty bin stub |
| `src/app.rs` | The root `App` component and shared helpers (API base URL, fetch wrapper) |
| `src/components/` | Shared UI: live view, control panel, models, camera, settings, statistics, training, Flow |
| `src/apps/` | Per-application UI (object detection, classification, segmentation) |
| `src/api/` | Backend access over HTTP and SSE |
| `src/models.rs` | Serde types for the API payloads |
| `build.rs` | Compiles `proto/detection.proto` and the `i18n/system-vision/` catalogs |
| `index.html`, `Trunk.toml`, `public/` | Trunk entry page, config and static assets |
| `styles.css` | Generated from `styles/input.css`; never edit it |
| `config/` | nginx: plaintext `:80` (`nginx.conf`), the pairing block (`nginx-enroll.conf`), the mTLS block (`nginx-enforcing.conf`) and the `entrypoint.sh` that switches between them |
| `Dockerfile.system-vision[.dev]` | Builds the WASM bundle and the nginx image (`.dev` uses the x86 Tailwind CLI) |

This crate is its own cargo workspace, and the app modules compile only for `wasm32`.

## Run

- In the stack: `docker compose -f docker-compose.dev.yml up -d --build system-vision` (the dev stack
  also publishes `:80`).
- On the host, against a gateway on `:5000`:

  ```bash
  bin/tailwindcss -i styles/input.css -o system-vision/styles.css --watch   # scripts/fetch-tailwind.sh first
  cd system-vision && trunk serve --proxy-rewrite=/api --proxy-backend=http://localhost:5000 --port=18080
  ```

  `scripts/dev.sh` runs both. See [Local development](../docs/getting-started.md#local-development).

## Test and lint

```bash
cargo check --manifest-path system-vision/Cargo.toml --target wasm32-unknown-unknown
wasm-pack test --headless --firefox system-vision   # CI uses --chrome
```

## Constraints

- Layout is conventional responsive CSS with a pinned header, never a global `transform: scale()`.
- Never compute `Date::now() - <server timestamp>`: the device clock is stepped. Use the backend's
  `elapsed_secs` and add only local deltas.
- Every backend SSE event needs a branch in `src/components/main_view/main_view.rs`.
- Add new strings to `i18n/system-vision/en` first and keep `pt-BR` and `es` in parity. The UI has no
  language selector; the language arrives as `?lang=`.
- Styles live in the shared `styles/` design system.

## Reference

- [HTTP API reference](../docs/api-reference.md)
- [Configuration: `system-vision`](../docs/configuration.md#system-vision)
- [`i18n/README.md`](../i18n/README.md), [`styles/README.md`](../styles/README.md)
