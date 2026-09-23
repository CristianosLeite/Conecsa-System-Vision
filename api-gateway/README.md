# api-gateway

The only HTTP surface of a device: REST, SSE and MJPEG on `:5000` (Flask on waitress). It is public
only through the nginx mTLS terminator on `:443`. Every handler translates to gRPC (inference-service
`:50061`, training-service `:50071`, `os-base` hardware agent `:50051`) or reads a SHM ring for the
two MJPEG feeds. It ships no ML stack.

Full description: [docs/services/api-gateway.md](../docs/services/api-gateway.md).

## Layout

| Path | What it is |
|---|---|
| `main.py` | Entry point for `waitress main:app`; starts the event relays, mDNS advertising and the training orphan watchdog at import |
| `gateway/app.py` | Assembles the Flask app |
| `gateway/controllers/` | Route handlers per resource (detection, models, camera, areas, GPIO, network, system, streams, …) |
| `gateway/training/` | The training routes (datasets, images, jobs, SAM, weights) |
| `gateway/authz.py` | `ROUTE_POLICIES`: the role each mutating route needs |
| `gateway/events.py` | The unified SSE bus and the gRPC event relays |
| `gateway/audit.py`, `gateway/audit_events.py` | The audit trail buffer and event keys |
| `gateway/enroll.py` | Device pairing (`/enroll/*`) |
| `gateway/grpc_clients.py`, `gateway/rpc_deadlines.py` | gRPC channels and per-call deadlines |
| `gateway/media.py` | MJPEG framing from the SHM rings |
| `deploy/` | Example avahi service for advertising the device from the host when the container cannot emit multicast |
| `tests/` | Host-side pytest suite |

## Run

- In the stack: `docker compose -f docker-compose.dev.yml up -d --build api-gateway`.
- On the host: `python3 api-gateway/main.py` from the repo `.venv`, after `scripts/compile-proto.sh`.
  `scripts/dev.sh --gateway-only` runs it with the device UI and no GPU services. See
  [Local development](../docs/getting-started.md#local-development).

## Test and lint

```bash
cd api-gateway && pytest -q         # or scripts/test.sh for every suite
ruff check . && .venv/bin/pyright   # from the repo root
```

## Constraints

- Every new or changed mutating route needs a `ROUTE_POLICIES` entry; `tests/test_authz.py` fails
  otherwise.
- Waitress is thread-per-connection: long-lived SSE and MJPEG streams pin threads, so count them
  against `WAITRESS_THREADS` and put new event kinds on `/api/v1/events/stream`.
- A new SSE event needs a branch in `system-vision/src/components/main_view/main_view.rs`, or the
  device UI drops it.
- The gateway steps `CLOCK_REALTIME` when the hub polls: durations use `time.monotonic()`.
- Keep the proto `sys.path` shim in `gateway/__init__.py` above any `*_pb2` import.

## Reference

- [HTTP API reference](../docs/api-reference.md)
- [Configuration: `api-gateway`](../docs/configuration.md#api-gateway)
