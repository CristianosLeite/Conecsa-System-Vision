# inference-service

The headless TensorRT inference backend. Its only surface is a gRPC control server on `:50061`
(`proto/inference.proto`); it has no HTTP. A decode∥infer∥encode pipeline reads the camera SHM ring,
runs the active model in a TensorRT worker and publishes the overlaid JPEGs to the processed SHM
ring, which the api-gateway serves.

Full description: [docs/services/inference-service.md](../docs/services/inference-service.md).

## Layout

| Path | What it is |
|---|---|
| `main.py` | Entry point: builds the application, restores the last model, serves gRPC and blocks |
| `api/composition.py` | Composition root that wires the services and the pipeline |
| `api/inference_grpc.py` | The gRPC servicers |
| `api/services/` | Detection, models, conversion, configuration, areas, events, stats, GPIO gate and the processing pipeline |
| `api/runtime_management/` | TensorRT runtime, the worker subprocess and its client, `.pt`/`.onnx` → engine conversion |
| `api/postprocess/` | Output decoding per application type (detect, classify, segment) |
| `api/views/` | Overlay drawing (boxes, masks, detection areas) |
| `api/model_manager.py`, `api/yolo_detector.py` | Model loading and the detector |
| `api/models/`, `api/repositories/` | Model metadata and class label storage |
| `tests/` | Host-side pytest suite (no GPU needed) |
| `uploaded_models/` | Local model storage when run outside Docker; excluded from the image build context |

## Run

- In the stack: `docker compose -f docker-compose.dev.yml up -d --build inference-service`.
- On the host: `cd inference-service && python3 -m main`, from the repo `.venv` installed with
  `scripts/init.sh --gpu`. TensorRT and PyCUDA are required, so in practice this runs on the Jetson
  or in the dev stack. See [Local development](../docs/getting-started.md#local-development).

## Test and lint

```bash
cd inference-service && pytest -q   # or scripts/test.sh for every suite
ruff check . && .venv/bin/pyright   # from the repo root
```

## Constraints

- Keep `tensorrt`, `pycuda`, `torch` and `ultralytics` imports lazy: the host tests and CI run
  without the GPU stack.
- `ultralytics` (AGPL) is confined to `api/_pt_onnx_converter.py`.
- No HTTP here: new external behavior is a gRPC call that the api-gateway exposes.
- The camera and processed rings follow the versioned SHM protocol (see `os-base/conecsa_shm`).

## Reference

- [Configuration: `inference-service`](../docs/configuration.md#inference-service)
- [`proto/inference.proto`](../proto/inference.proto)
- [HTTP API reference](../docs/api-reference.md) (the gateway routes that reach this service)
