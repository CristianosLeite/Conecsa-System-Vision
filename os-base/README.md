# os-base

Two things share this directory: the `conecsa-os-base:base` image every Python service builds
`FROM`, and the privileged **`os-base` hardware agent** that runs in its container. The agent is a
gRPC `HardwareService` on `:50051` (`proto/hardware.proto`) that owns network, Wi-Fi, GPIO, system
metrics, power and the system clock; its only client is the api-gateway.

Full description: [docs/services/os-hardware-agent.md](../docs/services/os-hardware-agent.md).

## Layout

| Path | What it is |
|---|---|
| `Dockerfile.os-base` | The Jetson base image: CUDA, Python 3.10 and the ML stack, plus the agent |
| `Dockerfile.os-base.dev` | The x86_64 dev base image (x86 TensorRT/PyCUDA wheels); it does not run the agent |
| `requirements-common.txt` | The shared CUDA/ML pins baked into the base image |
| `agent/` | The hardware agent (`python3 -m agent`): `server.py` wires the per-area agents (`network_agent`, `ap_agent`, `gpio_agent`, `system_agent`, `time_agent`, `clocks_agent`) |
| `conecsa_shm/` | SHM ring helpers for the Python services on the rings (inference-service, api-gateway, training-service): camera ring reader, processed-frame ring, stereo combine; the Rust webcam-server writes the camera ring itself |
| `conecsa_common/` | Plain-Python helpers shared by the services (atomic writes, bounded SQLite queue, events, polygons, tiling, tasks) |
| `tests/` | Host-side pytest suite |

`conecsa_shm/` and `conecsa_common/` are Apache-2.0 and are installed into the base image's
`dist-packages`, so services import them without vendoring.

## Run

- Device stack: `docker compose up -d --build os-base` (compose builds this image first for the
  derived services through `additional_contexts: base=service:os-base`).
- The agent needs the Jetson host (networkd, wpa_supplicant, GPIO character devices); off-device
  the dev stack runs this container only as the base image and volume owner.

## Test and lint

```bash
cd os-base && pytest -q          # or scripts/test.sh for every suite
ruff check . && .venv/bin/pyright   # from the repo root
```

## Constraints

- The SHM layout is versioned: a change to `conecsa_shm/camera_ring.py` bumps `SHM_VERSION` here
  and in `webcam-server/src/webcam_server/shm.rs`, and the four SHM services deploy together.
- A Wi-Fi or static IP change must never strand the device: save only after success, roll back on
  failure, and test with a wired fallback.
- `conecsa_common` must stay free of third-party imports at the package root; the api-gateway
  imports it without the ML stack.
- Timeouts and durations use `time.monotonic()`; the agent itself steps the wall clock.

## Reference

- [Configuration: `os-base` hardware agent](../docs/configuration.md#os-base-hardware-agent)
- [`proto/hardware.proto`](../proto/hardware.proto)
- [Architecture](../docs/architecture.md)
