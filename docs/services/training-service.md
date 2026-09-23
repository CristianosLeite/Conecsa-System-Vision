# Training Service (Python, headless)

On-device dataset capture, SAM3-assisted labeling and YOLO training. Like the
inference-service it is **headless**: the only surface is a **gRPC control
server** on `:50071` (`proto/training.proto`, `TrainingControl`), after which
the process blocks. The api-gateway owns all HTTP/SSE. Heavy work (torch /
ultralytics / SAM3) always runs in **child processes**; this service is only the
control plane.

## Responsibilities

- **Datasets** — multiple datasets live under
  `{DATA_DIR}/datasets/{dataset_id}/` (`{DATA_DIR}` = `TRAINING_DATA_DIR`,
  default `/data/training`; camera captures + YOLO labels + classes).
  Every dataset-scoped RPC carries the `dataset_id` explicitly; the server keeps
  no "active dataset" state. Each dataset is labeled for one **task**
  (`meta.json["task"]`, absent = `detect`), fixed at creation from the
  device's application, and accepts only that task's label kind: boxes for
  detection, polygon rings for segmentation (one `class x1 y1 … xn yn` row
  per ring), one class per image for classification **and for face
  recognition**, where the class is the enrolled person (stored as a
  one-line label file holding the class id). Datasets can also be imported
  from / exported to a ZIP in the task's standard layout: YOLO
  (`images/` + `labels/` + `data.yaml`) for detection and, with polygon rows,
  for segmentation; one folder of images per class (optionally under
  `train/`, `val/`, `test/`, with a `classes.txt` fixing the class order) for
  classification. An archive of another task (box rows for segmentation,
  polygon rows for detection, class folders for either) is refused rather
  than converted behind the operator's back; the device's task travels with
  the upload.
- **Capture** — captures the current camera frame (read from the camera SHM
  ring) into a dataset. The compose file sets `STEREO_COMBINE` /
  `STEREO_BLEND_ALPHA` to the same values on this service and the
  inference-service so captured images match the geometry the live detector
  sees (there is no runtime sync; both code defaults are `none`).
- **Dataset geometry** — `TRAIN_DATASET_IMG_SIZE` decides how every image
  (capture, ZIP import, hub ingest) is stored. The default `0` stores the
  stereo-combined frame at its native resolution (e.g. 1280×720) and leaves
  letterboxing to ultralytics at train time, so the same dataset trains any
  `imgsz` on real pixels (training at 1280 on a 640×640 dataset would only
  upscale already-downscaled pixels). A value > 0 stores the frame
  letterboxed to that square with YOLO gray padding, the format of datasets
  created by older firmware (`640`), which stay usable with their recorded
  geometry. Labels are always normalized to the stored image, and `GetImage`
  reports the real stored `width`/`height`. Each dataset records its geometry
  in `meta.json` (`{"letterbox": 640}` or `{"native": true}`; a dataset
  without the key is letterbox 640) when it is created, and the service refuses to add
  images to a dataset whose recorded geometry differs from the current
  setting (`FAILED_PRECONDITION`: *dataset X stores 640×640 letterboxed
  images; set TRAIN_DATASET_IMG_SIZE=640 or create a new dataset*), so a
  dataset never mixes formats.
- **Training geometry** — `TRAIN_TILE` decides what the trainer actually sees.
  The inference-service slices every frame into square tiles by default
  (`TILING_MODE=grid`, tile side = the frame's short side) and a model only
  performs at the scale it was trained at, so with the default `auto` the
  split builder writes every image of both the train and the valid split as
  its tile crops — the same `conecsa_common.tiling` grid, side = the image's
  short side, so any 16:9 camera gives two crops per frame at any
  resolution — with the labels clipped and re-normalised per tile: a box
  survives in a tile when at least `TRAIN_TILE_MIN_VISIBLE` of its area lies
  inside it, a tile whose only content was a smaller fragment is skipped
  rather than taught as background, and a tile no box touches is a genuine
  negative. Validation runs on tiles too, so the reported metrics describe
  the geometry the model is deployed in. Images the grid cannot slice (a
  640×640 letterboxed dataset, a square image) are symlinked whole,
  as is everything with `TRAIN_TILE=off` — the pairing for a device running
  `TILING_MODE=off`. The crops are job scratch under `runs/<job>/dataset/`
  and are removed when the job ends (`best.pt` stays under
  `runs/<job>/weights/`). The effective geometry (`frames`, `tiles:auto` or
  `tiles:<px>`) is declared on the `best.pt` upload and recorded in the
  model's settings sidecar, where the inference-service checks it against
  `TILING_MODE` on activation.
- **Classification datasets** — train a classifier on whole images: the
  split builder writes `train/<class>/` and `val/<class>/` symlink folders
  (ultralytics' `data=<root>` layout) instead of tiles, splitting each class
  on its own and giving every dataset class a folder in both splits, even an
  empty one, so the class indices are identical across splits and federated
  shards; unlabeled images are left out. Training needs at least 2 classes
  with 2 or more labeled images each, starts from `TRAIN_BASE_WEIGHTS_CLS`
  (YOLO26s-cls) at `TRAIN_IMG_SIZE_CLS` (224), and the resulting `best.pt` is
  uploaded with `task: classify` at that size.
- **Segmentation datasets** — label objects with polygon rings. Every ring
  any producer writes — the label editor, SAM3's masks, the hub's ingest, a
  ZIP import — is normalized at the stored image size
  (`conecsa_common.polygons`): rasterised and re-extracted, so crossing edges
  and holes cannot survive, pieces smaller than 0.05 % of the image are
  dropped, and a ring saved again unchanged stays exactly as it was. The
  split crops the same tiles as detection: each ring is rasterised, cut to
  the tile and re-extracted, so a concave outline split by a tile edge
  becomes one ring per visible piece, and `TRAIN_TILE_MIN_VISIBLE` applies to
  the visible area. Training starts from `TRAIN_BASE_WEIGHTS_SEG`
  (YOLO26s-seg) at `TRAIN_IMG_SIZE`, and `best.pt` is uploaded with
  `task: segment` and its tile geometry.
- **Replicate** — duplicates a labeled image (the JPEG plus its YOLO labels)
  1–50 times to quickly reach the training minimum (`TRAIN_MIN_IMAGES`, 20).
  Replicas are flagged in `meta.json` (`replica_image_ids`) so the gallery can
  mark them; an unlabeled source or an already-replicated image is rejected.
- **Pre-labeled ingest** (`AddDatasetImage`) — accepts an externally captured
  image plus pre-labels carried by **class name** with normalized corner
  coordinates on the uploaded image. The JPEG is stored in the dataset's
  geometry like a camera capture: letterboxed to the square with the
  coordinates mapped into letterbox space (with the same rounding as the
  letterbox itself), or kept at its own resolution with the corners converted
  to center/size form. Class names are resolved
  against `classes.json` — missing names are
  appended — so the image arrives already labeled. A segmentation dataset
  takes rings by class name (`polygons`, mapped into the stored geometry
  point by point) and a classification dataset the image's class by name
  (`image_class`) instead of boxes. This is
  how the hub feeds a record's clean frame back into a dataset for
  retraining; the operator only adjusts the class in the label editor when
  the model got it wrong.
- **SAM3-assisted labeling** — an on-demand SAM3 worker
  (`SAM3_CHECKPOINT`, HF-gated and baked into the image at build time) turns a
  user prompt into boxes and, for each, its mask as normalized rings (a
  segmentation dataset keeps the rings, a detection dataset the box), loaded
  and unloaded explicitly to free GPU memory.
- **Model-assisted labeling** — any engine already on the device can be the
  labeling assistant instead of SAM3. It does not run here: the gateway hands
  the dataset image to the [inference-service](inference-service.md), which
  runs the engine on a private TensorRT worker with the same preprocessing
  (`TILING_MODE`) and decode path as live detection, so the suggestions are
  exactly what the device would detect. Boxes come back tagged with the
  engine's class names; the editor resolves them against the dataset's
  classes (creating missing ones) on accept. A classification engine suggests
  the image's class instead, a segmentation engine each object's rings, and
  only engines of the dataset's task are
  offered (the gateway refuses the others). The engine is unloaded when the
  training page exits, when detection starts and by every GPU handover.
- **Fine-tuning from an existing model** — `StartTraining.base_model` (a
  model-list name such as `X.engine`, one with `has_weights`) makes the run
  start from that model's checkpoint sidecar — the inference-service keeps the
  `.pt` a conversion started from (an earlier on-device run's `best.pt`, a
  manual or federated `.pt` upload) under `weights/` next to the engine —
  fetched through the gateway's model weights route into `{DATA_DIR}/base/`
  instead of the baked-in `TRAIN_BASE_WEIGHTS`. Retraining under the same model
  name replaces the engine and its sidecar, so a model's checkpoint on the
  device is always its latest `best.pt`. A model converted by older firmware
  keeps no checkpoint until retrained or re-uploaded. The base model must be a
  model of the dataset's task. Exclusive with the federated
  `initial_weights_id`.
- **Training job** — one ultralytics run at a time, executed in a child process
  that streams one JSON line per epoch; the service folds those into the job
  state and publishes `training_progress` events. On success the resulting
  `best.pt` is uploaded through the api-gateway's model-upload route, which renames it to the user-chosen model name and starts
  the `pt → onnx → engine` conversion on the inference-service — the same path a
  manual upload takes.

## Face galleries

A dataset of task `face` holds one class per person and that person's photos
(camera captures and uploads alike). "Train" on it is **not** a YOLO run: the
service packages the labeled images as `<name>.faces` — a ZIP with a
`manifest.json` (format 1) and the images — and uploads it through the normal
model upload with `task=face`, where the inference-service builds the gallery.
Epochs, batch size, an initial-weights checkpoint, a base model and federated
training do not apply and are refused; a face dataset validates for training
with a single person holding a single photo. See
[Face recognition](../face-recognition.md).

## Federated training (hub-orchestrated FedAvg)

The [hub](hub-vision.md) can train one model across every paired device
without any device-to-device traffic: it ferries opaque `.pt` blobs over its
per-device mTLS channel. The service-side building blocks:

- **Shard export** — `ExportDatasetShard` exports one deterministic IID shard
  of a dataset (shuffle by a shared seed + round-robin assignment): the N
  shards of one seed are disjoint, cover the full dataset and differ in size
  by at most one image. `data.yaml` (a classification dataset: `classes.txt`)
  always carries the full class list so per-shard checkpoints stay
  averageable.
- **Weights stash** — opaque checkpoints under `{DATA_DIR}/weights/{id}.pt`
  (`UploadWeights` / `DownloadWeights` / `DeleteWeights`), capped by
  `TRAINING_MAX_WEIGHTS_MB` and pruned after `TRAINING_WEIGHTS_TTL_SEC`. A
  checkpoint may carry the task it was trained for (`WeightsUploadMeta.task`,
  or the job's dataset task for a stashed result) in a `<id>.task` sidecar: a
  job refuses weights of another task and `AverageWeights` refuses to mix
  tasks.
- **Federated train mode** — `TrainRequest.federated` starts the regular job
  from a stashed checkpoint (`initial_weights_id`, falling back to the baked-in
  base weights of the dataset's task — `TRAIN_BASE_WEIGHTS`,
  `TRAIN_BASE_WEIGHTS_CLS` or `TRAIN_BASE_WEIGHTS_SEG`) and, on success, stashes the resulting **`last.pt`** (same
  number of local epochs on every device) as `result_weights_id` instead of
  uploading a model. The round trains on the same split as a local job (tile
  crops under the default `TRAIN_TILE=auto`) and reports its effective
  `geometry` in the job status; the hub requires every participant to report
  the same value and declares it on the final model upload.
- **Averaging** — `AverageWeights` FedAvg-merges ≥2 stashed checkpoints in a
  CPU child process (the same isolation rule as the trainer):
  float tensors of the `model` **and** `ema` state dicts averaged in fp32 and
  cast back, non-float buffers kept from the first checkpoint, optimizer state
  dropped. CPU-only, so it never competes with a GPU training job.

## GPU handover

Training and inference share a single Jetson GPU. Entering training mode hands
the GPU over from the inference runtime (`ReleaseRuntime` / `ResumeRuntime` on the
inference-service), and exiting resumes it. The gateway exposes this as
`/api/v1/training/enter` and `/api/v1/training/exit`. While a training job is
active the GPU stays with the trainer: `POST /api/v1/start` answers `409`
(the device UI shows the Start button disabled as "Training model…"), and a
`/training/exit` neither resumes detection nor ends the handover — leaving the
training page mid-run keeps inference stopped until the job ends and its model
conversion finishes. Exiting training mode ends the handover even when
detection is left stopped for the model conversion, and so does an explicit
Start; until then the application type cannot change.

!!! note "Sizing for the Orin Nano 8 GB"
    `TRAIN_BATCH` defaults to 4 and `TRAIN_WORKERS` to 0 (single-process
    DataLoader — the small shared `/dev/shm` from webcam-server's IPC namespace
    cannot back worker tensors). `TRAIN_STALL_TIMEOUT_SEC` is a liveness
    watchdog (kills a *hung* trainer with no output), not a cap on total
    duration; set `TRAIN_TIMEOUT_SEC` for an overall wall-clock cap.

## Reference

- Workflow endpoints: [HTTP API reference — Training](../api-reference.md#training-relayed-to-the-training-service)
- Python API: [`service` package](../reference/python-api/index.md)
- gRPC contract: [`proto/training.proto`](../reference/proto.md)
- Configuration: [training-service env vars](../configuration.md#training-service)
