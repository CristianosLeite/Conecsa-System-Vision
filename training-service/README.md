# training-service

On-device datasets, SAM3-assisted labeling and YOLO training. It is headless: the only surface is a
gRPC control server on `:50071` (`proto/training.proto`), and the api-gateway owns all HTTP. The
service is the control plane only; training and SAM3 run in child processes.

Full description: [docs/services/training-service.md](../docs/services/training-service.md).

## Layout

| Path | What it is |
|---|---|
| `main.py` | Entry point: builds the application, serves gRPC and blocks |
| `service/composition.py` | Composition root |
| `service/training_grpc.py` | The gRPC servicer |
| `service/dataset_registry.py`, `dataset_service.py`, `dataset_import.py` | Datasets: storage, labels, ZIP import and export |
| `service/capture_service.py` | Captures the current camera frame into a dataset |
| `service/training_service.py`, `train_overrides.py` | Training jobs and the allowlisted hyperparameter overrides |
| `service/tile_split.py` | Builds the training split from the same tile crops the inference-service runs on |
| `service/_yolo_trainer.py` | The training child process (ultralytics) |
| `service/sam_service.py`, `_sam_worker.py` | SAM3 labeling and its child process |
| `service/weights_store.py`, `_weights_averager.py` | Round checkpoints for federated training and the averaging child process |
| `service/model_fetch.py` | Fetches an existing device model's weights through the gateway as a fine-tuning base |
| `assets/` | Base weights per application type (`yolo26s*.pt`, committed) and the SAM3 checkpoint (`sam3.pt`, gitignored, provided by the operator) |
| `tests/` | Host-side pytest suite |

## Run

- In the stack: `docker compose -f docker-compose.dev.yml up -d --build training-service`. The image
  bakes in `assets/`, so place `sam3.pt` there first or build with `--build-arg SAM3_INSTALL=0`.
- On the host: `cd training-service && python3 -m main` from the repo `.venv` with the GPU stack.
  See [Local development](../docs/getting-started.md#local-development).

## Test and lint

```bash
cd training-service && pytest -q    # or scripts/test.sh for every suite
ruff check . && .venv/bin/pyright   # from the repo root
```

## Constraints

- Heavy work (torch, ultralytics, SAM3) runs in child processes; keep those imports lazy so the
  host tests and CI run without the GPU stack.
- `ultralytics` (AGPL) is confined to `service/_yolo_trainer.py`.
- Training and inference share one GPU: see
  [GPU handover](../docs/services/training-service.md#gpu-handover).
- Capture reads the camera SHM ring, so this service joins webcam-server's IPC namespace and is
  recreated with it.

## Reference

- [Configuration: `training-service`](../docs/configuration.md#training-service)
- [`proto/training.proto`](../proto/training.proto)
- [HTTP API reference: Training](../docs/api-reference.md#training-relayed-to-the-training-service)
