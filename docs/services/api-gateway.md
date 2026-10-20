# API Gateway (Python)

The thin HTTP↔gRPC/SHM interface and the **only HTTP surface** of a device
(port 5000, Flask + Waitress), serving the web app, Flow and the hub:

- **Control / config / models / classes / areas / system / GPIO / network /
  training**: translated to gRPC calls (inference-service `:50061`,
  training-service `:50071`, `os-base` hardware agent `:50051`).
- **Both MJPEG feeds** (`/api/v1/video_feed`, `/api/v1/video_feed_processed`):
  fanned out directly from the camera and processed SHM rings.
- **Unified SSE** (`/api/v1/events/stream`): invalidation events plus a
  multiplexed stats channel, fed by background relays of the inference-service's
  `StreamEvents` / `StreamStats` and the training-service's `StreamEvents`
  (see [Event stream](../api-reference.md#event-stream)).
- **Protocol Buffers content negotiation** on status/start/stop/threshold/
  overlay_threshold/classes and their `/api/*` aliases; every other route is
  JSON only.
- **Mutating routes are role-checked** for hub-relayed operator traffic
  (`ROUTE_POLICIES` in `gateway/authz.py`, see
  [Roles](../api-reference.md#roles)).
- **Audit trail**: mutating requests (with the exclusions listed in the
  [HTTP API reference](../api-reference.md)) are appended to a SQLite ring
  buffer under `AUDIT_DIR` (`audit.db`, volume `conecsa-audit-data` at
  `/data/audit`), which the hub drains over mTLS via `/api/v1/audit/backlog`
  and clears with `/api/v1/audit/backlog/ack`. See
  [Audit trail](hub-vision.md#audit-trail).

It ships no ML stack (no torch/tensorrt) — only the web layer and the compiled
proto stubs on top of `conecsa-os-base:base`.

!!! note "Auditing is per request, not per handler"
    The trail is written from an `after_request` hook. Several control
    endpoints are served by more than one view — the `/api/*` aliases are
    separate view functions — so a decorator on the handler would silently miss
    half of them. A mutating route with no event key of its own is still
    recorded, under `device.request` with its method and path, so coverage does
    not depend on remembering to register a new endpoint. `detail` is built by
    explicit per-route extraction of one safe field: request bodies, uploads
    and Wi-Fi pre-shared keys never reach the buffer.

!!! note "SSE thread budget"
    Waitress is thread-per-connection; long-lived MJPEG/SSE streams pin one
    task thread each. Size `WAITRESS_THREADS` above the worst-case number of
    concurrent streams.

!!! note "Slow viewers get the newest frame, not a backlog"
    Each MJPEG client has its own generator (`gateway/media.py`) that copies
    the newest frame from the SHM ring whenever it is allowed to run, so a
    viewer that drains slower than the detector publishes skips frames instead
    of queueing them. That only works once waitress pauses the generator, which
    it does when `WAITRESS_OUTBUF_HIGH_WATERMARK` bytes are pending on the
    connection — hence the small default (about two processed frames) instead
    of waitress's 16 MB. The compose files also cap the kernel TCP buffers of
    this container and of the nginx terminator (`sysctls`), since every MB
    queued between the generator and the viewer is a second of stale video on
    a 1 MB/s link. Inference is unaffected either way: the processed ring is a
    seqlock the writer never waits on, and a stalled viewer only stalls its
    own task thread.

!!! note "Every backend call has a deadline"
    The gRPC channels are wrapped by an interceptor that gives every unary and client-streaming call a deadline when the call
    site passes none — `GATEWAY_GRPC_TIMEOUT` for control calls,
    `GATEWAY_GRPC_LONG_TIMEOUT` for the slow ones (runtime swaps, training
    start/stop, dataset deletion) and `GATEWAY_GRPC_UPLOAD_TIMEOUT` for
    uploads — so a backend that stays connected but stops answering cannot
    park a request thread forever. A deadline that expires answers `504`;
    an unreachable backend answers `503`. Server streams (the event relays,
    downloads) are unbounded by design.

## Reference

- Full endpoint catalogue: [HTTP API reference](../api-reference.md)
- Python API: [`gateway` package](../reference/python-api/index.md)
- Configuration: [api-gateway env vars](../configuration.md#api-gateway)
