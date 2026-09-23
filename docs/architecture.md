# Architecture

## Naming

**Conecsa System Vision** is the product as a whole — the devices plus the hub that
fronts them. Two of its parts have similar names, so this documentation keeps them
apart:

- **Hub Vision** (`hub-vision`) — the native desktop fleet hub.
- the **device screen** (`system-vision`) — one device's own web UI container, listed
  in the table below.

`system-vision` in code font always means that container (and the crate, compose
service, Docker context, `i18n/system-vision/` catalog and device hostname that share
its name), never the product.

## Services

| Service | Language/Framework | Port | IPC |
|---|---|---|---|
| **os-base** | CUDA 12.6 + Python 3.10 (base image) **+ hardware gRPC agent** | gRPC `50051` | Named volumes + gRPC + GPIO SHM |
| **system-vision** | Rust + Leptos (WASM); nginx serves `index.html` uncacheable (`Cache-Control: no-cache`) and the content-hashed bundles cacheable | `443` (host, mTLS — the only production port) / 80 (Nginx, dev `${SYSTEM_VISION_PORT:-80}`) | HTTP |
| **api-gateway** | Python + Flask + Waitress (HTTP↔gRPC/SHM interface) | `5000` (internal; host-published in dev only) | HTTP out; gRPC + Shared Memory in |
| **inference-service** | Python (headless: gRPC + TensorRT pipeline) | gRPC `50061` | Shared Memory (camera in, processed out) |
| **training-service** | Python (headless: gRPC + child-process torch) | gRPC `50071` | Shared Memory (camera in) + gRPC |
| **webcam-server** | Rust (synchronous capture via nokhwa/v4l, or an MJPEG pull from a remote camera) | — | Shared Memory (producer) |
| **flow** | Node-RED + Conecsa custom nodes | `1880` (internal; host-published in dev only) | HTTP (→ api-gateway) |

In production only system-vision's `:443` mTLS endpoint is published; the
other host ports in the table apply to the dev stack
(`docker-compose.dev.yml`).

The backend follows an **`app → api → service`** split. The **api-gateway** is
the only HTTP surface: the web app and Flow talk to
it on `:5000`, and it translates each call to the headless inference-service
(gRPC `:50061`), the training-service (gRPC `:50071`) or the `os-base` hardware agent
(gRPC `:50051`), or fans the camera / processed MJPEG feeds out of shared
memory. Per-frame media never crosses gRPC.

Each device runs one **application type** — object detection,
classification, segmentation or face recognition — owned and persisted by the
inference-service (see [Application type](services/inference-service.md#application-type));
every model and dataset carries the task it belongs to, and the hub shows the
application of every device in the fleet. Face recognition is the one type
that runs a second engine: the face detector takes the normal pipeline lanes
and the face embedder sits on a private TensorRT worker beside them, as
model-assisted labeling does (see [Face recognition](face-recognition.md)).

The **`os-base`** service builds the `conecsa-os-base:base` image (CUDA + Python + ML
stack: torch, torchvision, tensorrt, pycuda, opencv, ultralytics, numpy,
etc.) which is inherited via `FROM` by the other Python services
(inference-service, api-gateway and training-service). It also owns the shared
volumes (`/data/models`, `/data/runs`).

Beyond the base image, the `os-base` container runs the **privileged hardware
agent** — see [`os-base` hardware agent](services/os-hardware-agent.md) for the
full breakdown (network/Wi-Fi/GPIO, GPIO SHM channel, performance-clock pinning).

### Fleet hub (`hub-vision`)

The table above lists the per-device **compose** services. A separate native
(Tauri 2 + Leptos) app, **`hub-vision`**, is the authenticated entry point to a
fleet of these devices. It is **not** in the compose stack and
runs on its own host on the LAN: it discovers devices over mDNS (`_conecsa._tcp`),
pairs with each (acting as a private CA), and reaches them **only over mutual
TLS** — pulling their detection records and their audit trails, not receiving
pushed ones. It is also the only place operators are authenticated, so it is
the only place a user action can be attributed to a person. See
[Fleet hub](services/hub-vision.md).

## Communication between services

![Communication diagram](communication.png)

Two transport rules: **per-frame media** (raw + processed JPEG) crosses **POSIX
shared memory**, never gRPC; **control / config / status / events / stats** go
over **gRPC**.

- The `webcam-server` captures frames — from a local V4L2 camera or, as a
  [remote camera](remote-camera.md), pulled over TCP from a phone — and
  publishes them to the **camera SHM ring** (`/dev/shm/conecsa_frame_shm`). Its header is defined in
  `proto/shm.proto` (frame metadata + raw/JPEG bytes). The inference-service,
  the api-gateway and the training-service all read it.
- The `inference-service` consumes the camera ring, runs the
  decode∥infer∥encode pipeline, and publishes the overlaid JPEGs to a second
  **processed SHM ring** (`/dev/shm/conecsa_processed_shm`). The api-gateway fans
  both rings out to MJPEG clients. These containers share the `ipc:`
  namespace (`ipc: shareable` on webcam-server; `ipc: "service:webcam-server"`
  on inference-service, api-gateway and training-service).
- Control/telemetry travels over gRPC: the api-gateway drives the
  inference-service's `DetectionControl` / `ModelControl` / `ManagementControl`
  (`proto/inference.proto`, `:50061`), the training-service's `TrainingControl`
  (`proto/training.proto`, `:50071`) and the hardware agent's `HardwareService`
  (`proto/hardware.proto`, `:50051`, the `os-base` container),
  re-publishing inference's event/stats streams onto a single unified SSE.
- Off-device, the [Fleet hub](services/hub-vision.md) (`hub-vision`) discovers
  devices over mDNS and **pulls** their detection records over mutual TLS (it
  polls each paired device's `/api/v1/detections/snapshot` every second); the
  device exposes only its `:443` mTLS endpoint. This is a LAN mTLS/mDNS path, separate from the
  intra-device gRPC + SHM transports above.
- The same path carries the **audit trail**: the hub drains each device's record
  of user actions (`/api/v1/audit/backlog`, persist-then-ack) and, in the other
  direction, stamps the signed-in operator onto every request it proxies into a
  device (`X-Conecsa-User`, `X-Conecsa-Role`, `X-Conecsa-Origin-Ip`) — the
  device has no authentication of its own, so that is the only identity it ever
  sees. nginx adds `X-Forwarded-For`, and both are trusted only when it relayed
  the request. See [Audit trail](services/hub-vision.md#audit-trail).
- The same LAN path carries the **time**: the hub relays its clock at pairing
  and on every status poll, and the api-gateway hands it to the hardware agent's
  `SetSystemTime` — see
  [Clock synchronization](services/hub-vision.md#clock-synchronization).

See the [Protocol Buffers reference](reference/proto.md) for the full message
and service catalogue.

## Frontend (Rust)

The device UI, **`system-vision`**, is a Leptos app compiled to **WASM** via
Trunk and served by Nginx (port `:80`). It talks to the api-gateway over
HTTP/SSE/MJPEG on the same origin it was loaded from.

The native **Tauri 2** desktop app in this repo is the separate
[`hub-vision`](services/hub-vision.md) fleet hub — a different application, not a
desktop build of `system-vision`.

### Localization

Both frontends are localized (**en** default, **pt-BR**, **es**) with
[`leptos_i18n`](https://crates.io/crates/leptos_i18n). Translations are
compiled in: each crate's `build.rs` generates the i18n module from the shared
catalogs under the `i18n/` directory at the monorepo root
(`i18n/system-vision/<locale>/<namespace>.json`,
`i18n/hub-vision/<locale>.json` — layout, parity rules and the cross-app
terminology glossary are documented in `i18n/README.md`).

The device UI has **no language selector**. The only selector lives in the
hub's **Settings** page; the hub persists the choice (`hub-settings.json`) and
appends `?lang=<locale>` to the embedded device page's iframe URL. The device
UI resolves its locale as `?lang=` → localStorage (`conecsa.lang`, written on
every change so direct browser access keeps the last language) →
`navigator.languages` → `en`.

### Interface

A pinned header above a dashboard that fills the viewport in three columns on
a desktop and stacks on narrower screens. State follows the unified event
stream, with a 5-second status poll as the fallback.

- **Main pane**: the live stream with detections, or the view selected in the
  navigation (camera settings, Flow editor, device settings). On an
  object-detection device the video carries the detection-area controls:
    - **`▦`** (top-right): creates a new detection area centered in the frame
      and enters editing mode.
    - **Area chips** (top-left): one chip per area. The `□`/`○` glyph shows
      the shape; clicking the number enters editing; clicking the `✗` deletes
      the area. The chip of the area being edited is highlighted amber.
    - **Editing toolbar** (bottom, only while an area is being edited):
      movement (`↑ ↓ ← →`), axis resize (`W+`/`W−` width, `H+`/`H−` height),
      uniform resize (`⤢`/`⤡`), shape toggle (`□` ↔ `○`); `✓` saves and `✗`
      discards the edit.
- **Configuration panel**: model upload/selection, the confidence and overlay
  thresholds (segmentation adds the instance limit, face recognition the
  match threshold, minimum face size and faces per frame), class labels
- **Control panel**: view navigation (including training), Start/Stop with the
  detection status, performance statistics
- **Status row** (full width): system metrics

## Protocol Buffers

All `.proto` files live under the `proto/` directory at the monorepo root. The
generated [Protocol Buffers reference](reference/proto.md) documents every
message and service.

| File | Use |
|---|---|
| `proto/detection.proto` | Schema of the REST message bodies (consumed by the Rust frontend (system-vision) via `prost`; used by the api-gateway for HTTP protobuf content-negotiation) |
| `proto/shm.proto` | Schema of the camera shared-memory header (webcam-server → inference-service / api-gateway / training-service) |
| `proto/inference.proto` | gRPC control/telemetry contract between the api-gateway and the headless inference-service (`:50061`) |
| `proto/hardware.proto` | gRPC contract for the `os-base` hardware agent (`HardwareService`, `:50051`) |
| `proto/training.proto` | gRPC contract for the training-service (`TrainingControl`, `:50071`) |

**Compilation**:

- **Rust** (system-vision, webcam-server): compiled automatically via `build.rs` with
  `prost-build` during `cargo build`
- **Python** (inference-service, api-gateway, training-service): compiled with
  `grpc_tools.protoc` in the Dockerfile (or locally with
  `scripts/compile-proto.sh`)
- **Script**: `scripts/compile-proto.sh` compiles every `.proto` for Python
  and triggers `cargo build` for Rust

## Tech stack

| Layer | Technology |
|---|---|
| Frontend UI | Leptos 0.8 (Rust WASM, served by Nginx) |
| Fleet hub | `hub-vision` — Leptos + Tauri 2 (native desktop, off-device) |
| Frontend build | Trunk, TailwindCSS |
| Localization | leptos_i18n — compile-time catalogs (en / pt-BR / es) under `i18n/` |
| Protobuf client | prost |
| Webcam server | Rust — nokhwa (+ v4l / rscam on aarch64), synchronous capture |
| Frame transport | POSIX Shared Memory (zero-copy IPC) — camera + processed rings |
| API gateway | Python 3.10 + Flask + Waitress (HTTP↔gRPC/SHM) |
| Inference | Headless Python — TensorRT pipeline + gRPC control |
| Training | Headless Python — ultralytics + SAM3 in child processes + gRPC control |
| Control plane | gRPC (grpcio): inference `:50061`, training `:50071`, `os-base` hardware agent `:50051` |
| Detection | TensorRT |
| Serialization | Protocol Buffers (protobuf / prost) |
| Automation | Node-RED + `conecsa-system-vision` package (9 nodes plus the `conecsa-hub` configuration node) |
| Containerization | Docker Compose + NVIDIA Container Runtime |
