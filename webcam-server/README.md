# webcam-server

The Rust camera process. It captures frames from a V4L2 device and publishes them to the camera SHM
ring (`SHM_NAME`) that the inference-service, api-gateway and training-service read. It has no HTTP
and no gRPC: camera configuration comes back through the same shared-memory segment. With no camera
it publishes only the `no_camera` health status and keeps retrying the device.

Full description: [docs/services/webcam-server.md](../docs/services/webcam-server.md).

## Layout

| Path | What it is |
|---|---|
| `src/main.rs` | Entry point: logging, then the capture loop |
| `src/webcam_server.rs` | Server state and `run_server` |
| `src/webcam_server/capture*` | The acquire → process → publish loop, camera re-attach, direct V4L2 paths, formats and hardware controls |
| `src/webcam_server/processing*` | Software image processing: gain, RGB levels, Bayer debayering |
| `src/webcam_server/config*` | Camera configuration from the environment, updated live through the SHM config region |
| `src/webcam_server/shm.rs` | The SHM ring producer and its versioned header |
| `build.rs` | Compiles `proto/shm.proto` (camera config and health regions) |
| `Dockerfile.webcam-server` | Multi-stage release build |

This crate is its own cargo workspace (it is not a member of the root workspace). Unit tests live in
the sibling `tests.rs` modules.

## Run

```bash
docker compose -f docker-compose.dev.yml up -d --build webcam-server   # in the stack
cargo run --release --manifest-path webcam-server/Cargo.toml           # on the host
```

`scripts/dev.sh` starts it first, since the other services consume its ring.

## Test and lint

```bash
cargo test --manifest-path webcam-server/Cargo.toml
cargo clippy --manifest-path webcam-server/Cargo.toml --all-targets -- -D warnings
```

## Constraints

- The SHM layout is versioned: any layout change bumps `SHM_VERSION` in `src/webcam_server/shm.rs`
  and in `os-base/conecsa_shm/camera_ring.py`, and webcam-server, inference-service, api-gateway and
  training-service deploy together.
- The container is `ipc: shareable` and the three consumers join its IPC namespace. Recreate them as
  a set; a consumer left running after webcam-server was recreated needs `--force-recreate`.

## Reference

- [Configuration: `webcam-server`](../docs/configuration.md#webcam-server)
- [`proto/shm.proto`](../proto/shm.proto)
- Rust API: `scripts/build-docs.sh` (cargo doc)
