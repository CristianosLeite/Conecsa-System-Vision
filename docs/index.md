# Conecsa System Vision

Real-time computer vision — object detection, image classification, instance
segmentation and face recognition — built with a Rust web frontend (Leptos/WASM), a Rust camera
server and Python backend services. Designed for the **NVIDIA
Jetson Orin Nano** (ARM64, JetPack 6.2.2 / L4T R36.5.0, CUDA 12.6), with
**TensorRT** inference.

A thin **api-gateway** owns the external HTTP/SSE/MJPEG contract and reaches the
headless services over gRPC and shared memory; a separate native **`hub-vision`**
desktop app is the authenticated entry point to a fleet of devices. See
[Architecture](architecture.md) for both.

Operators (rather than integrators) should start with the **user manual** — an
interactive, trilingual guide that embeds a simulation of the hub and device
screens: <https://conecsa.com.br/system-vision/2026-6/manual/>.

## Documentation map

| Page | Contents |
|---|---|
| [Architecture](architecture.md) | Services, transports, communication rules, the frontend |
| [Getting started](getting-started.md) | Prerequisites, Docker quick start, local development |
| [Configuration](configuration.md) | Every `docker-compose` environment variable |
| [Face recognition](face-recognition.md) | The `face` application: enrolment, gallery build, settings, biometric-data duties |
| [HTTP API reference](api-reference.md) | All REST/SSE/MJPEG endpoints on the api-gateway |
| [Troubleshooting](troubleshooting.md) | Common failures + Yocto runtime notes |
| **Services** | [inference-service](services/inference-service.md) · [api-gateway](services/api-gateway.md) · [webcam-server](services/webcam-server.md) · [`os-base` hardware agent](services/os-hardware-agent.md) · [training-service](services/training-service.md) · [Flow](services/flow.md) |
| [Fleet hub](services/hub-vision.md) | The native `hub-vision` app: auth, mDNS discovery, mTLS detection pull and the audit trail across many devices |
| **Reference** | [Protocol Buffers](reference/proto.md) · [Python API](reference/python-api/index.md) · Rust API (`cargo doc`) |
| [Yocto build](yocto-build.md) | Building the lean Yocto host image for the Jetson |

## Features

- **Real-time detection**: YOLO26 models via TensorRT (`.pt`, `.engine`,
  `.plan`, `.onnx`); object detection, classification or segmentation per
  device
- **Face recognition**: an enrolled gallery built and kept on the device names
  the people it sees (bundled YuNet + SFace models) — see
  [Face recognition](face-recognition.md)
- **Detection areas**: one or more rectangular or circular regions defined in
  the UI (normalized coordinates in `[0,1]`) where inference is restricted.
  Persistent across restarts.
- **Streaming via shared memory**: camera frames travel from the webcam-server
  to the inference-service through POSIX shared memory (zero-copy IPC)
- **MJPEG streaming**: the processed stream with detection overlays is
  exposed over HTTP for the frontend
- **Model management**: upload, selection, deletion and asynchronous
  `.pt` → `.engine` conversion (TensorRT)
- **On-device training**: capture datasets, label with SAM3 assistance and
  train YOLO models without leaving the device (see
  [training-service](services/training-service.md))
- **Camera configuration**: real-time adjustment of index, resolution,
  framerate and exposure via shared memory, no restart required
- **System monitoring**: real-time CPU, RAM, disk, temperature and GPU usage
- **Web frontend**: Leptos compiled to WASM, served by Nginx
- **Multilingual UI**: English (default), Brazilian Portuguese and Spanish in
  both frontends (see [Architecture § Localization](architecture.md#localization))
- **Fleet hub**: `hub-vision` discovers devices over mDNS and pulls their
  detections over mutual TLS; it runs on a hub machine or as a **boot-time
  kiosk on the device's DisplayPort** (see [Fleet hub](services/hub-vision.md))
- **Audit trail**: operator actions on the hub and on its devices are
  recorded with their actor, origin and outcome in bounded on-disk buffers,
  kept for a configurable window and exportable as CSV (see
  [Audit trail](services/hub-vision.md#audit-trail))
- **Efficient API**: REST with Protocol Buffers serialization (JSON fallback)

## Quick start

```bash
# Build and bring everything up in one go.
docker compose up -d --build
docker compose logs -f
```

The production stack publishes only `:443` (mTLS) — the device's UI, API and Flow
editor are reached through the `hub-vision` app. The dev stack
(`docker-compose.dev.yml`) also exposes the plaintext ports (`:80`, `:5000`,
`:1880`) for local work. See [Getting started](getting-started.md) for the full
setup.
