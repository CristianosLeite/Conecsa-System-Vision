# Webcam Server (Rust)

Captures MJPEG directly via V4L2 (`/dev/video0`). Supports native MJPEG
cameras (zero-CPU passthrough), bilinear debayering of RGGB8 Bayer frames
and a YUYV fallback. Configuration (index, resolution, framerate,
automatic/manual exposure) can be changed in real time via shared memory
with no restart. When no camera is available it publishes **no frames at
all** — only the `no_camera` health status that gates detection downstream —
and it **re-attempts the real camera every few seconds** so a re-plugged /
re-powered camera self-heals without restarting the container. A camera that
stops delivering frames mid-stream is given up after ~30 consecutive failures
(a few hundred milliseconds), the health flips to `no_camera`, and the same
re-open cycle runs; it never spins on a dead device.
It is `privileged` (no static `/dev/video0` mapping — that would block startup
when the camera is unplugged), so it sees `/dev/video*` including hot-plug.

## Network source (remote camera)

Instead of a local camera, the server can pull a Motion-JPEG stream from a remote
camera over a direct Wi-Fi link and publish it into the same shared-memory ring. It
stays the only producer of that ring, so frames still reach every consumer over
shared memory and nothing downstream changes. The wire format, the limits and
the health states are specified in the
[remote camera protocol](remote-camera-protocol.md).

One source is active at a time, selected by `source` in the shared-memory
`CameraConfig`:

- A **local** source restarts capture on a device, resolution or framerate
  change and applies exposure, gamma, gain and RGB levels live.
- A **network** source restarts only when the source, address, port or token
  changes. Resolution, framerate and the V4L2 controls mean nothing to a remote camera:
  they are stored but never run `v4l2-ctl`, never reopen a device and never
  report `capturing` — only an arriving frame does that.
- A `CameraConfig` **without** `source` changes neither the source nor the
  network fields. A writer that only tunes the local camera therefore cannot
  switch sources, which also keeps a mixed-version deployment from doing so.
- An unknown `source` value, or an incomplete or invalid address/port/token, is
  ignored and logged; it is never treated as "local".

A network source that fails keeps retrying and reports why through
`HealthStatus.detail`. It never falls back to a local camera on its own.

`CAMERA_SOURCE` and `CAMERA_NETWORK_*` choose the source at start-up for
development and bootstrap only; a source published over shared memory replaces
them. In production the source is set from the device screen and persisted by
the inference-service (see the [remote camera guide](../remote-camera.md)). The token never appears in a log line, in `Debug` output or in serialized
configuration.

## Shared memory

Captured frames are published to a POSIX shared memory segment (`SHM_NAME`):
a 256-byte header followed by two frame slots. The frame metadata (magic,
version, width, height, format, frame size, active slot, write sequence) is
a raw fixed-offset header; Protocol Buffers are used only for the camera
config and health regions inside that header (`proto/shm.proto`).

The segment carries **SHM protocol version 2**, a seqlock publication: the
writer makes the frame sequence odd, writes the payload into the inactive slot
and then all its metadata, and makes the sequence even again; readers copy the
frame only on an even sequence and accept it only when the sequence is
unchanged afterwards, retrying a torn copy. The version is `SHM_VERSION` in
`webcam-server/src/webcam_server/shm.rs` and
`os-base/conecsa_shm/camera_ring.py`, and readers reject any other version,
so a mixed deployment fails loudly instead of tearing frames. Any layout
change bumps both and must deploy webcam-server, inference-service,
api-gateway and training-service together.

The SHM slot is sized for
the largest frame the camera can deliver (`SHM_SLOT_MIN_BYTES`, code default
8 MB; compose raises it to 16 MB so the stereo camera's native 3840×1080 RAW
fallback fits).
The container runs with `ipc: shareable` so the inference-service, api-gateway
and training-service share the same IPC namespace.

## Reference

- Rust API: `cargo doc` → `webcam_server` crate (see the Rust API link in the
  nav, or run `scripts/build-docs.sh`)
- SHM config/health schema: [`proto/shm.proto`](../reference/proto.md)
- Network source wire format: [remote camera protocol](remote-camera-protocol.md)
- Configuration: [webcam-server env vars](../configuration.md#webcam-server)
