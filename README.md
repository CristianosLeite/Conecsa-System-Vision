# Conecsa System Vision

Real-time computer vision — object detection, image classification and instance
segmentation — built with a Rust web frontend (Leptos/WASM), a Rust camera
server and Python backend services. Designed for the **NVIDIA
Jetson Orin Nano** (ARM64, JetPack 6.2.2 / L4T R36.5.0, CUDA 12.6), with
**TensorRT** inference. A separate native **`hub-vision`** desktop app is the
authenticated entry point to a fleet of devices (see
[Architecture](docs/architecture.md#fleet-hub-hub-vision)).

A thin **api-gateway** owns the entire external HTTP/SSE/MJPEG contract; the
headless inference-service, training-service and `os-base` hardware agent sit
behind it on **gRPC** (control) and **POSIX shared memory** (frames) — see
[docs/architecture.md](docs/architecture.md).

![Communication diagram](docs/communication.png)

## Features

- **Real-time detection**: YOLO26 models via TensorRT (`.pt`, `.engine`,
  `.plan`, `.onnx`)
- **Detection areas**: rectangular/circular regions restricting inference
  (normalized coordinates, persistent across restarts)
- **Streaming via shared memory**: zero-copy camera → inference frame transfer
- **MJPEG streaming**: processed stream with detection overlays over HTTP
- **Model management**: upload, selection, deletion and async `.pt → .engine`
  conversion
- **On-device training**: capture datasets, label with SAM3 assistance, train
  YOLO models without leaving the device
- **Camera configuration**: live index/resolution/framerate/exposure changes,
  no restart
- **System monitoring**: real-time CPU, RAM, disk, temperature and GPU usage
- **Web frontend**: Leptos compiled to WASM, served by Nginx
- **Multilingual UI**: English (default), Brazilian Portuguese and Spanish in
  both frontends (`leptos_i18n`, compile-time catalogs under `i18n/`). The
  language is chosen in the hub's **Settings** and propagates to embedded
  device pages via `?lang=` (persisted in the device UI's localStorage)
- **Fleet hub**: a separate native (Tauri) `hub-vision` app logs operators in,
  discovers devices over mDNS and pulls their detections over **mutual TLS**
  (off-device; built with `scripts/build-hub.sh`)
- **Detections buffered while the hub is offline**: each device buffers
  detection records on disk whenever the hub stops polling, in a SQLite ring
  that survives reboots and is capped at 5 000 records / 1 GB (the oldest are
  evicted first). The hub drains the backlog on reconnect, deleting the device
  copy only after its own store confirms the write, with timestamps
  reconstructed to the real detection time
- **Audit trail**: user actions on the hub and on its devices are recorded
  with their actor, origin IP and outcome. Devices buffer their own events in
  a bounded on-disk ring that the hub drains over mTLS, and the hub's own
  events go through a bounded on-disk outbox (overflow is counted and shown in
  Settings) before they reach its database. History is kept for a configurable
  window and exports to CSV
  (see [Audit trail](docs/services/hub-vision.md#audit-trail))
- **Secure by default**: the device exposes only a `:443` **mTLS** endpoint; the
  hub acts as a private CA, enrolls devices by a one-click pairing, and is the
  sole client holding a valid certificate — no root certificate is ever installed

## Quick start

**On the Jetson (production):**

```bash
# Build and bring everything up in one go.
docker compose up -d --build
docker compose logs -f
```

The production stack publishes **only `:443` (mTLS)** — the device's UI, REST/
MJPEG/SSE API and Flow editor are reachable solely through the `hub-vision` app
([Fleet hub](docs/services/hub-vision.md)). On first run, pair the device from
the hub (one click on the trusted LAN); mTLS then locks it to that hub.

**On an x86_64 workstation with an NVIDIA GPU (local dev):** the root
`docker-compose.yml` is Jetson-specific (aarch64 wheels, Tegra host-library
bind-mounts, GPIO). Use the dev stack, which matches production except for the
Jetson-specific parts (x86 `tensorrt-cu12`/`pycuda`, x86 Tailwind).
Requires the [NVIDIA Container Toolkit](docs/getting-started.md#prerequisite-nvidia-container-toolkit):

```bash
docker compose -f docker-compose.dev.yml up -d --build
```

After startup the dev stack also publishes the plaintext ports: web
app on `http://localhost:80`, api-gateway on `http://localhost:5000`, Flow on
`http://localhost:1880` — plus the `:443` mTLS terminator for testing enrollment
and the hub.

A non-container path is also available — `./scripts/init.sh` bootstraps a local
`.venv` and `./scripts/dev.sh` runs the services on the host (see
[docs/getting-started.md](docs/getting-started.md)).

## Documentation

Full documentation lives under [`docs/`](docs/index.md) and is published as a
[MkDocs](https://www.mkdocs.org/) site (`scripts/build-docs.sh`).

| Topic | Page |
|---|---|
| Architecture & transports | [docs/architecture.md](docs/architecture.md) |
| Prerequisites & local dev | [docs/getting-started.md](docs/getting-started.md) |
| Environment variables | [docs/configuration.md](docs/configuration.md) |
| HTTP API reference | [docs/api-reference.md](docs/api-reference.md) |
| Troubleshooting | [docs/troubleshooting.md](docs/troubleshooting.md) |
| inference-service | [docs/services/inference-service.md](docs/services/inference-service.md) |
| api-gateway | [docs/services/api-gateway.md](docs/services/api-gateway.md) |
| webcam-server | [docs/services/webcam-server.md](docs/services/webcam-server.md) |
| `os-base` hardware agent | [docs/services/os-hardware-agent.md](docs/services/os-hardware-agent.md) |
| training-service | [docs/services/training-service.md](docs/services/training-service.md) |
| Flow nodes | [docs/services/flow.md](docs/services/flow.md) |
| Fleet hub (hub-vision) | [docs/services/hub-vision.md](docs/services/hub-vision.md) |
| Protocol Buffers | [`proto/`](proto/) (rendered reference generated into the docs site at build time) |
| Yocto host image | [docs/yocto-build.md](docs/yocto-build.md) |

### Building the docs locally

```bash
pip install -r docs/requirements-docs.txt
mkdocs serve -f docs/mkdocs.yml   # live preview at http://127.0.0.1:8000
# or the full site (MkDocs + cargo doc) into ./site:
scripts/build-docs.sh
```

## Project structure

```
system-vision/
├── proto/                  # Protocol Buffers (single source): detection, shm,
│                           #   inference, hardware, training
├── i18n/                   # Shared translation catalogs (en/pt-BR/es) for
│                           #   system-vision + hub-vision (see i18n/README.md)
├── styles/                 # Shared Tailwind v4 design system (entry styles/input.css)
├── os-base/                # Base CUDA/ML image + privileged hardware agent + SHM helpers
├── system-vision/          # Rust web frontend (Leptos WASM, served by Nginx)
├── webcam-server/          # Camera server (Rust) — camera SHM producer
├── api-gateway/            # Public HTTP↔gRPC/SHM interface (Python, no ML stack)
├── inference-service/      # Headless TensorRT inference backend (Python)
├── training-service/       # Headless dataset/labeling/training backend (Python)
├── flow/                   # Flow automation (Node-RED) + Conecsa custom nodes
├── hub-vision/             # Native Tauri fleet hub (auth, CA, mDNS, mTLS pull);
│                           #   built via scripts/build-hub.sh, not in Docker
├── manual/                 # Interactive user manual (EN/PT-BR/ES): Markdown
│                           #   content + Leptos shell + simulators running the
│                           #   real hub/device UI on fixtures (build-manual.sh);
│                           #   private like hub-vision — only the published
│                           #   site is public, NOT in the open-source mirror
├── scripts/                # init.sh, dev.sh, test.sh, compile-proto.sh, build.sh,
│                           #   build-hub.sh, build-hub-jetson.sh, build-docs.sh,
│                           #   build-manual.sh, publish-manual.sh, gen-proto-docs.py,
│                           #   export-mirror.sh, pin/fetch helpers (check-pins.sh,
│                           #   fetch-tailwind.sh, fetch-trunk.sh)
├── docs/                   # Documentation site (MkDocs config + pages)
├── yocto/                  # Lean Yocto host image for the Jetson (see yocto/README.md)
├── requirements-dev.txt    # Single dev venv (all services + docs toolchain)
├── pyrightconfig.json      # Pyright/Pylance: type checking (editor + CI)
├── docker-compose.yml      # Production stack (Jetson)
└── docker-compose.dev.yml  # Local dev stack (x86_64 + NVIDIA GPU)
```

> Each service directory has a `README.md` with its layout and its run and test
> commands.

> Each service builds from its own `Dockerfile.<service>`; the dev stack reuses
> them and only swaps `os-base/Dockerfile.os-base.dev` (x86 GPU wheels) and
> `system-vision/Dockerfile.system-vision.dev` (x86 Tailwind).

> **Dev environment & type-checking:** run `./scripts/init.sh` (creates the root
> `.venv`, installs `requirements-dev.txt`, and compiles the proto stubs so the
> editor resolves the generated modules). See
> [docs/getting-started.md](docs/getting-started.md) for details.

## License

Conecsa System Vision is licensed in layers, and every file declares its own
license with an SPDX header or an annotation in [`REUSE.toml`](REUSE.toml)
([REUSE](https://reuse.software/) compliant; `scripts/check-licenses.sh` checks it):

- **Apache-2.0**: the gRPC contracts (`proto/`), the design system (`styles/`),
  the translation catalogs (`i18n/`), the shared Python libraries
  (`os-base/conecsa_shm`, `os-base/conecsa_common`), the Node-RED node package
  (`flow/nodes/conecsa-system-vision`), `scripts/`, `yocto/` and `docs/`. These
  never import AGPL code.
- **AGPL-3.0-only**: the application stack, which is everything else, including
  the services, the device web UI and the webcam server ([LICENSE](LICENSE)).

License texts live in [`LICENSES/`](LICENSES); Apache-2.0 directories also carry
their own `LICENSE` file. Third-party dependencies are listed in
`os-base/THIRD-PARTY-NOTICES.md` and `api-gateway/THIRD-PARTY-NOTICES.md`.
The Conecsa names and logo are covered by [TRADEMARKS.md](TRADEMARKS.md), not by
these licenses.
