# Inference Service (Python, headless)

No HTTP server — the only surface is a **gRPC control server** on `:50061`
(`proto/inference.proto`: `DetectionControl`, `ModelControl`,
`ManagementControl`, plus the standard gRPC health service), started
alongside the decode∥infer∥encode pipeline; the process then blocks on it. A
port that cannot be bound ends the process with a non-zero exit so the
container restart policy retries. `ReloadModel`, `SetDetectionAreas` and
`SetSettings` are reserved and answer `UNIMPLEMENTED`; model activation, areas
and settings go through the other RPCs.

The pipeline reads the camera SHM ring, runs inference on **TensorRT** and
publishes the overlaid JPEGs to the processed SHM ring. Accepted model
formats: `.engine`, `.plan`, `.pt`, `.onnx`. A `.pt` or `.onnx` upload
starts an asynchronous conversion to `.engine`; clients follow it on
`/api/v1/model/conversion/<job_id>` and the event stream.

Each loaded engine runs in a **TensorRT worker** subprocess (one per
`TENSORRT_CONTEXTS` lane). The service starts the first worker in the
background at boot, so the cold start (see
[Troubleshooting](../troubleshooting.md)) is paid before the first request.

**Frames via SHM**: a consumer thread reads the camera ring; camera
configuration (resolution, framerate, exposure) is written back through the
same segment, with no HTTP to the webcam-server. The encode stage publishes
the processed JPEGs to the **processed SHM ring** for the api-gateway to fan
out.

**Detection areas**: `DetectionAreaService` keeps a persistent list of areas
**per model**: each model's areas live next to its weights
(`weights.engine` → `weights.areas.json`), so switching models switches area
sets. Each area is a rectangle or circle with normalized coordinates in
`[0,1]`, so it survives camera resolution changes. When at least one area is
saved, detections whose center falls outside the union of the areas are
dropped. Areas in *editing* mode filter nothing; they only draw the overlay
(dashed border, dimmed outside) so the user can position them. Saved areas
affect inference but are invisible in the stream.

**Pipeline**: decode, inference and encode run on separate threads connected
by bounded queues; `TENSORRT_CONTEXTS` adds parallel TensorRT contexts
(pipeline lanes).

**Stage timings**: the postprocess and encode stages are single-threaded, so
their service time bounds the frame rate whatever `TENSORRT_CONTEXTS` is. The
service keeps the last 512 samples of each (mean, p95 and p99) plus the p95
age of a published frame, measured from its pickup off the camera ring — the
ring carries no capture timestamp. They reach clients on `/api/v1/stats` and
the stats streams (`/api/v1/stats/stream`, `/api/v1/events/stream?stats=1`);
`/api/v1/status` carries only the four basic counters.

**Tiled inference (SAHI-style)**: with `TILING_MODE=grid` (the default) the
prepare stage slices the decoded frame into overlapping square tiles
(`TILING_TILE`/`TILING_OVERLAP`, geometry from `conecsa_common.tiling` — the
same module the training-service crops datasets with, so the deployed layout
is exactly the one trained on). Each lane runs one inference per tile, and
the postprocess stage shifts every tile's boxes back into frame space and
removes cross-tile duplicates with a class-aware NMS (`TILING_MERGE_IOU`)
before the overlay-threshold NMS, area filter and drawing. A small object
then reaches the model at close to native scale. The default tile side is
`auto` — the frame's short side — so the layout depends only on the camera's
aspect ratio: any 16:9 camera gives two tiles. The reported `inference_time`
is the summed per-tile GPU time. `TILING_MODE=off` keeps a single full-frame
inference for models trained on whole frames. A model only performs at the
geometry it was trained at, so grid mode must be paired with a model trained
on the same crops — the training-service does that by default
(`TRAIN_TILE=auto`). Grid mode needs `PROCESSING_DECODE_SCALE=1` (a warning is
logged otherwise).

**Overlay threshold**: the IoU above which overlapping boxes are suppressed
(NMS). It is not a score gate; the confidence threshold decides which boxes
count.

**Capture source**: whether the device inspects with its local camera or a
[remote camera](../remote-camera.md) is device-level state, kept in
`camera_source.json` in the models directory (owner-readable only: it holds the
stream token). It is loaded before any model settings, published to the
webcam-server at start-up even with no model selected, and published again
whenever the webcam-server recreates its shared-memory segment. A source update
is one transaction — validate, persist, publish — and a failed publish restores
the previous file.

**Model settings sidecar**: each model's `weights.settings.json` carries its
thresholds, camera and stereo snapshot (local camera tuning only — never the
capture source), plus two informational fields the
`.pt` conversion writes: `imgsz` (the export size) and `training.geometry`
(`frames`, `tiles:auto` or `tiles:<px>`, as declared by the uploader — the
training-service always declares it, a federated upload declares the geometry
the participating devices reported, and a browser upload leaves it unknown). Activating a model (`select`, snapshot restore, startup default)
compares the geometry with the live `TILING_MODE`/`TILING_TILE` and logs a
warning when they disagree — a whole-frame model under grid tiling loses
confidence inside the tiles, and a tile-trained model is blind on the whole
letterboxed frame. The sidecar also records the model's `task` (see
[Application type](#application-type)), which no threshold edit erases. A
segmentation model can carry `segment.max_instances`, the instance limit the
operator set: it replaces both `SEGMENT_MAX_MASKS` caps while that model is
active.

**Detections snapshot**: the `Snapshot` RPC backs
`/api/v1/detections/snapshot`. Each detection carries its normalized `bbox`
corners (`[x1, y1, x2, y2]`, 0..1). Besides the annotated `frame` (which
Node-RED flows consume) the snapshot can include the **clean** frame
(`include_raw_frame` → `raw_frame`): the image the boxes were detected on,
which the hub stores so detection records can be re-labeled and fed back into
a training dataset. On a classification device the snapshot's `detections`
hold at most the frame's class above the confidence threshold (no `bbox`,
`area: null`, so `total` is 1 or 0) and `candidates` the top-k classes; see
[Classification](#classification). On a segmentation device each detection
also carries `polygons`, its outline as normalized exterior rings, capped at
64 KiB per snapshot (the smallest rings are left out first and
`polygons_truncated` says so); see [Segmentation](#segmentation).

**Offline detection buffer (store-and-forward)**: the hub's snapshot poll
(every second) doubles as a hub-is-online heartbeat. Only pulls that arrive
through the mTLS terminator count: the gateway marks them `hub_pull` from the
`X-Conecsa-Client-Verify` header nginx stamps on `:443` traffic, and honors
that header only when the TCP peer is the terminator container itself. Local
snapshot consumers such as the Flow detection node poll the same endpoint and
must not make the device believe the hub is online. When no hub poll arrives
for `HUB_OFFLINE_THRESHOLD_SEC` (default 5 s), `DetectionBufferService`
persists on-change detection records — the same class@area change signature
the hub collector uses — to a SQLite ring buffer at
`/data/detections/buffer.db` (the `conecsa-detections-data` volume, so records
survive container restarts and reboots). Each record stores the detection
list, the task that produced it, `captured_at`, and the clean JPEG frame; caps
are 5 000 records / 1 GB (oldest evicted first, discards logged). The snapshot
advertises the row count as `pending_backlog`; the hub drains it via the
`ListBacklog`/`AckBacklog` RPCs (`GET /api/v1/detections/backlog` +
`POST /api/v1/detections/backlog/ack` through the gateway) and rows are
deleted **only after the hub acks having persisted them**. A persisted
`hub_seen` flag keeps a device that was never paired from writing, and
buffering is offline-only by design — zero eMMC writes while the hub is
polling. Database errors never take the pipeline down: the buffer recreates a
corrupt file and disables itself as a last resort.

## Application type

The device runs one application — `detect`, `classify`, `segment` or `face`
(the first three ids are ultralytics' `model.task`). `ApplicationService`
persists it in `/data/models/application.json`;
`ManagementControl.GetApplication` / `SetApplication` expose it and
`StatusResponse.task` carries it on every status poll. The supported tasks are
the postprocess strategies the build registers: the first three always, and
`face` only when the image carries the bundled face models, so
`supported_tasks` never offers an application this build cannot serve.

- **First start without `application.json`**: a models directory holding any
  model artifact (`*.engine`, `*.plan`, `*.onnx`, `*.pt`, a `.current_model`
  marker — even one naming a missing engine — or a `weights/` directory) is an
  existing installation and becomes `{"task": "detect", "migrated": true}`; an
  empty one is a blank device and records `{"task": null}`. A blank device
  starts nothing until an administrator chooses the application.
- **Models carry their task** in the settings sidecar (`"task"`, absent =
  `detect`), declared at upload (`ModelUploadMeta.task`, the device's task by
  default) and verified: a `.pt` export or an `.onnx` graph of another task
  fails its conversion before the engine build, and every engine is checked
  against its task's output layout when it is activated, so a mismatch rolls
  back to the previous model. A model of another task than the device's is
  refused outright (`FAILED_PRECONDITION`, `409` at the gateway).
- **Switching** is serialized with model activation and with the GPU
  handover: it is refused while the GPU is handed over (from `ReleaseRuntime`
  until `ResumeRuntime` or an explicit Start) and while a conversion runs. It
  stops detection, stores the choice, deselects an active model of another
  task (its files are kept), drops the runtime and the last result, and
  publishes `application_changed`. Detection is not restarted. The handover
  itself is announced with `runtime_changed` (keys `status`, data
  `runtime_released`): `true` from a successful `ReleaseRuntime`, `false` from
  `ResumeRuntime` and from a `Start` that started detection.
- **Restart**: a boot-default model (`.current_model`) that is missing or of
  another task is cleared with a warning before the TensorRT warm-up reads it.

### Classification

A classification engine (YOLO26s-cls, output `[1, classes]`) labels the
**whole frame** with one class:

- **Input** — the frame is never tiled and detection areas do not apply. It
  is prepared exactly like ultralytics' `classify_transforms`: converted to
  RGB, the short side resized to the engine's input size with Pillow's
  bilinear filter (antialiased when downscaling), the centre square cropped,
  scaled to 0..1.
- **Activation** — besides the `[1, classes]` shape (at least 2 classes), one
  inference on a blank frame must sum to 1: ultralytics exports classifiers
  with softmax in the graph, and an engine that outputs logits is refused
  (the previous model stays).
- **Result** — the top-1 class counts only when its probability is strictly
  above the confidence threshold (the same strict gate as detection); the
  overlay threshold does not apply. The snapshot reports that class (or
  nothing) plus the `CLASSIFY_TOPK` best candidates, highest first, ties to
  the lower class index. Nothing is drawn on the processed stream: the device
  UI's classification panel names the class under the video.
- **Counter** — the detection counter counts class **transitions**: it grows
  by one when the frame's class changes to a class (`none → A`, `A → B`, and
  `A` again after `none`), never while the class stays the same or goes away.
  The transition state restarts on start, stop, stats reset, model change and
  application change; a frame frozen by the GPIO trigger is not processed and
  leaves it alone. `frames_with_detections` counts frames that have a class.
- **Offline buffer and hub** — the buffer's `class@area` change signature
  yields one record per transition to a class and none for a static scene or
  a class going away; the hub records the same changes and stores the clean
  frame.

### Segmentation

A segmentation engine (YOLO26s-seg) outlines every object. Its two outputs
are picked by rank: the rows `[1, N ≤ 300, 6 + nm]` (box, score, class and
`nm` mask coefficients) and the mask prototypes `[1, nm, S/4, S/4]`.

- **Activation** — only YOLO26's end-to-end head is accepted: exactly those
  two outputs, with the same `nm` in both and prototypes at a quarter of the
  input size. An engine with a one-to-many head
  (`[1, 4 + classes + nm, 8400]`) is refused (the previous model stays).
- **Masks** — computed only for rows strictly above the confidence threshold
  that the overlay threshold's NMS keeps within their tile (a duplicate of a
  better row gets no mask), at most `SEGMENT_MAX_MASKS_PER_TILE` per tile:
  `coefficients · prototypes` on the prototype cells the box covers, resized
  to the box in frame pixels (the letterbox undone) and thresholded at 0, as
  ultralytics does. After the tiles merge, the overlay NMS applies again as for
  detection, at most `SEGMENT_MAX_MASKS` instances remain, and the detection
  areas filter on each instance's centre.
- **Tiles** — segmentation runs on the same `TILING_MODE=grid` tiles as
  detection. Fragments of one object seen by two tiles are grouped (same
  class, different tiles, both touching the tiles' overlap band, and
  `TILING_MERGE_IOU` or `TILING_MERGE_IOS` measured inside that band) and
  their masks joined into one instance, whose box is the union's and whose
  score is the best fragment's. `SEGMENT_TILE_STITCH=0` keeps detection's
  IoU-only merge instead, which leaves such an object cut.
- **Drawing and results** — the kept masks are filled into the processed
  frame in one pass, with boxes and class labels drawn as for detection. Each
  instance's outline becomes exterior rings normalized to the frame
  (`conecsa_common.polygons`: rasterised and re-extracted, holes dropped,
  pieces under `SEGMENT_MIN_COMPONENT_AREA` left out, at most 200 vertices per
  ring and 8 rings per instance) and rides the snapshot as `polygons`; the
  offline buffer's change signature ignores them.

### Face recognition

A face device identifies people against a gallery enrolled on the device. The
bundled **YuNet** detector is the main engine (whole frame, no tiling, 5
landmarks per face) and the bundled **SFace** embedder runs on a private
TensorRT worker (`TENSORRT_FACE_WORKER_PORT`, past the labeling worker),
closed when the device leaves the application.

- **Gate and cost** — `CONFIDENCE_THRESHOLD` is the face score gate and
  `OVERLAY_THRESHOLD` the NMS IoU, as for detection; faces below
  `FACE_MIN_SIZE_PX` are dropped and at most `FACE_MAX_FACES` faces per frame
  (largest first) are aligned to 112×112 and embedded, so the recognition cost
  is bounded per frame.
- **Matching** — each 128-d signature is compared with the model's gallery by
  cosine similarity; above `FACE_MATCH_THRESHOLD` the face takes that person's
  name, otherwise `unknown`. Detection areas filter on the face's centre.
- **Result** — the snapshot item is the detection one, with `confidence`
  holding the similarity; `total` is the number of faces and the counter grows
  on each arrival of a known person (`unknown` never counts). Boxes and names
  are drawn on a copy of the frame as detection does.
- **Model files** — `<name>.engine` (detector), `<name>.txt` (the names),
  `<name>.settings.json` and the `<name>.gallery.npz` sidecar holding the
  L2-normalized embeddings and the embedder hash. The gallery is never
  downloadable and never federated; a gallery built with another embedder is
  refused at activation and the previous model stays. Shared detector/embedder
  engines are cached in `<models dir>/face/` and reused by later builds.
- **Gallery build** — a `.faces` package uploaded with `task=face` is run as a
  `ConversionJob` (`services/face_gallery_builder.py`) with its own detector
  and embedder workers: every photo is detected, aligned and embedded, photos
  without a usable face are counted as skipped, and the job fails when no
  person ends up with a signature. The model's files are published as one set
  under the model lifecycle lock, and a gallery rebuilt under the name of the
  **active** model is reloaded there, before the job reports done — so live
  recognition, and a rename made right after, use the new gallery without a
  new Select. A photo whose second face reaches 65 % of the largest one's area
  is skipped as ambiguous. Uploading a file that is not a `.faces`
  package while declaring `face` is refused, and a face model cannot be loaded
  for model-assisted labeling.

See [Face recognition](../face-recognition.md) for the operator-facing view,
the bundled models' licenses and the biometric-data duties.

## Model-assisted labeling

The training page can pre-label a dataset image with any engine already on
the device. `ModelControl.LoadLabelModel` pins that engine to a private
TensorRT worker (`TENSORRT_LABEL_WORKER_PORT`, past the live model's context
lanes) and `LabelDetect` runs one encoded image through the same
preprocessing (`TILING_MODE` grid, letterbox) and decode/merge as live
detection, over the engine's own classes sidecar, so the suggestions are
exactly what the device would detect. A classification engine suggests the
image's class instead (`LabelDetectResult.image_class` above the threshold,
plus the top-k `candidates`), a segmentation engine adds each object's mask
as rings (`LabelDetection.rings`), and `GetLabelModelStatus` reports the
engine's `task` so the gateway can refuse suggestions for a dataset of another
task. `ReleaseRuntime` (the training GPU handover) and `UnloadLabelModel`
(training page exit, detection start) terminate the worker. The `.pt` a
conversion started from is kept as a `weights/<stem>.pt` sidecar
(`ModelInfo.has_weights`, `DownloadModelWeights`) so the training-service can
fine-tune from the model.

## Reference

- Python API: [`api` package](../reference/python-api/index.md)
- gRPC contract: [`proto/inference.proto`](../reference/proto.md)
- Configuration: [inference-service env vars](../configuration.md#inference-service)
