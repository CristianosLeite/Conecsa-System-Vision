# Configuration

Environment variables read by each service. **Default** is the value built into
the code (what applies when the variable is unset); where the production
`docker-compose.yml` sets a different value, the **Compose** column shows it.

## `webcam-server`

| Variable | Default | Compose | Description |
|---|---|---|---|
| `CAMERA_INDEX` | `0` | — | Camera device index |
| `CAPTURE_WIDTH` | `2560` (code) / `640` (image `ENV`) | `2560` | Capture width. The binary defaults to 2560, but `Dockerfile.webcam-server` bakes `640`, so a container started without compose captures 640 |
| `CAPTURE_HEIGHT` | `720` (code) / `640` (image `ENV`) | `720` | Capture height (same split as `CAPTURE_WIDTH`) |
| `CAPTURE_FRAMERATE` | `60` (code) / `30` (image `ENV`) | `60` | Capture FPS (same split as `CAPTURE_WIDTH`); the initial exposure is `10000 / framerate` |
| `SHM_NAME` | `conecsa_frame_shm` | — | Shared memory segment name |
| `SHM_SLOT_MIN_BYTES` | `8388608` (8 MB) | `16777216` (16 MB) | Minimum SHM slot size — must fit the largest possible frame (the slot is the larger of this and `width × height × 3`); values below `1500000` or unparseable are ignored and the 8 MB default applies. Compose raises it so the stereo camera's native 3840×1080 RAW fallback (12.44 MB/frame) fits |
| `RUST_LOG` | `info` (image `ENV`) | `info` | `env_logger` log level (`error`, `warn`, `info`, `debug`, `trace`) |
| `CAMERA_SOURCE` | `local` | — | Start-up capture source: `local` (V4L2) or `network` (a remote camera, see the [remote camera protocol](services/remote-camera-protocol.md)). Any other value stops the server at start-up rather than falling back to `local`. **Development/bootstrap only**: a source published over shared memory replaces it |
| `CAMERA_NETWORK_HOST` | — | — | Remote camera address for `CAMERA_SOURCE=network`: an IPv4 literal. Development/bootstrap only |
| `CAMERA_NETWORK_PORT` | — | — | Remote camera TCP port, `1`–`65535`. Development/bootstrap only |
| `CAMERA_NETWORK_TOKEN` | — | — | Stream token: 8–32 Crockford base32 characters, hyphens and case ignored. A secret — never logged. Development/bootstrap only; production compose does not set it |

## `inference-service`

| Variable | Default | Compose | Description |
|---|---|---|---|
| `SHM_NAME` | `conecsa_frame_shm` | — | Camera SHM segment name (must match webcam-server) |
| `PROCESSED_SHM_NAME` | `conecsa_processed_shm` | — | Processed SHM ring the encode stage publishes to (must match api-gateway) |
| `PROCESSED_SHM_SLOT_BYTES` | `1048576` (1 MiB) | — | Per-slot size of the processed SHM ring (`os-base/conecsa_shm/processed_ring.py`); must fit one overlaid JPEG |
| `INFERENCE_GRPC_LISTEN` | `0.0.0.0:50061` | — | gRPC control server bind address |
| `CONFIDENCE_THRESHOLD` | `0.75` | — | Initial confidence threshold (a model's settings sidecar overrides it on activation) |
| `OVERLAY_THRESHOLD` | `0.45` | — | Initial overlay threshold: the IoU above which overlapping boxes are suppressed (NMS); a model's settings sidecar overrides it on activation |
| `MODELS_DIR` | `/data/models` | — | Model directory; when unset and `/data/models` is not writable, a repo-local directory is used (host runs) |
| `MODEL_PATH` | `{MODELS_DIR}/weights.engine` | — | Model path used when no model is selected |
| `CLASSES_FILE_PATH` | `MODEL_PATH` with `.txt` | — | Class labels file used when no model is selected (selecting a model switches to its sibling `.txt`) |
| `MAX_MODEL_UPLOAD_BYTES` | `629145600` (600 MB) | — | Cap on a streamed model upload, counted while the chunks arrive |
| `CAMERA_BOOT_TIMEOUT` | `15` | — | Seconds the boot auto-start waits for the webcam-server to report a streaming camera before leaving detection stopped |
| `CAMERA_INDEX` | `0` | `0` | Base camera config the service merges partial camera patches into (written back to webcam-server over SHM) |
| `CAPTURE_WIDTH` / `CAPTURE_HEIGHT` | `640` / `640` | `2560` / `720` | Base capture resolution for the same merge; these code defaults differ from webcam-server's (2560×720) |
| `CAPTURE_FRAMERATE` | `30` | `60` | Base capture FPS for the same merge; also seeds the `CAPTURE_EXPOSURE_TIME` default |
| `CAPTURE_DEVICE` | `/dev/media0` (`0` on Windows) | — | Capture device reported in `GET /api/v1/config`; it does not select the camera (webcam-server captures, from `CAMERA_INDEX`) |
| `CAPTURE_RESOLUTION_X` / `_Y` | `640` / `640` | — | Model input size assumed when scaling boxes back to the frame if the engine's input size is unknown; also reported in `GET /api/v1/config`. The camera resolution comes from `CAPTURE_WIDTH` / `CAPTURE_HEIGHT` |
| `PROCESSING_DECODE_SCALE` | `1` | — | Reduced-scale JPEG decode for inference/overlay (1 = full, 2 = half, 4 = quarter). Must stay `1` for the default tiled inference (`TILING_MODE=grid`) and for models trained above 640; `2` is the cheaper choice only with `TILING_MODE=off` and a 640 model trained on whole frames |
| `PROCESSED_OUTPUT_SCALE` | `1` | — | Downscale factor (`1`, `2`, `4`) applied to the drawn frame before JPEG encoding/publishing; keeps the single-thread encode stage and the MJPEG stream small when the decode scale is `1` |
| `INFER_LETTERBOX_PAD` | `0` | — | Letterbox pad value (0–255) for the model input; training letterboxes with `114` |
| `INFER_RESIZE_INTERP` | `nearest` | — | Resize interpolation for the model input (`nearest`, `linear`, `area`); training uses linear |
| `TILING_MODE` | `grid` | — | SAHI-style tiled inference. `grid` (default) slices the decoded frame into overlapping square tiles, runs one inference per tile and merges the shifted boxes, so small objects reach the model near native scale; `off` keeps the single full-frame letterboxed inference for models trained on whole frames. A model only performs at the geometry it was trained at: pair `grid` with a model trained by the training-service's default `TRAIN_TILE=auto` (the model's settings sidecar records its geometry and activation logs a warning on a mismatch). Requires `PROCESSING_DECODE_SCALE=1` so tiles are cut from native pixels |
| `TILING_TILE` | `auto` | — | Square tile side for `TILING_MODE=grid`: `auto` = the frame's short side, resolved per frame, so any 16:9 camera gives two tiles whatever its resolution; a pixel value pins the side and must then match the `TRAIN_TILE` the model was trained with |
| `TILING_OVERLAP` | `0.2` | — | Fraction of `TILING_TILE` that adjacent tiles overlap (`0 ≤ f < 1`); the band must exceed the largest object the tiles are meant to catch |
| `TILING_MERGE_IOU` | `0.5` | — | IoU above which boxes from different tiles are merged as one object (class-aware greedy NMS, `conecsa_common.tiling.merge_tiles`) |
| `CLASSIFY_TOPK` | `5` | — | Classification only: how many candidate classes each frame reports in the snapshot's `candidates` (clamped to the model's class count; invalid or non-positive values fall back to `5`). Classification always runs on the whole frame, never on tiles |
| `TILING_MERGE_IOS` | `0.5` | — | Segmentation on tiles: intersection over the smaller box, measured inside the two tiles' overlap band, at or above which fragments of one object seen by different tiles are stitched into one instance (`conecsa_common.tiling.merge_tiles_grouped`; `TILING_MERGE_IOU` also groups them). Accepts `0`–`1`; anything else falls back to the default with a warning |
| `SEGMENT_TILE_STITCH` | `1` | — | Segmentation on tiles (boolean: `1`/`true`/`yes`/`on` or `0`/`false`/`no`/`off`; anything else keeps the default): `1` stitches the masks of an object straddling the overlap band into one instance with one outline; `0` keeps detection's IoU-only merge, so such an object stays cut |
| `SEGMENT_MAX_MASKS_PER_TILE` | `32` | — | Segmentation: at most this many rows above the confidence threshold, highest first and after the overlay NMS drops duplicates within the tile, get a mask per tile (`1`–`300`) |
| `SEGMENT_MAX_MASKS` | `32` | — | Segmentation: at most this many instances per frame after merging and NMS (`1`–`255`); the lowest scores are dropped. Both segmentation caps are only defaults: a model whose **Instance limit** was set in the device's Configuration panel (`POST /api/v1/segment/max_instances`) uses that value for both |
| `SEGMENT_MIN_COMPONENT_AREA` | `0.0005` | — | Segmentation: pieces of an outline smaller than this fraction of the frame are left out of the instance's polygons (`0`–`1`; anything else falls back to the default with a warning) |
| `FACE_MATCH_THRESHOLD` | `0.363` | — | Face recognition: cosine similarity a face must reach against the model's gallery to take that person's name; below it the face is `unknown` (`0`–`1`). The per-model value set through `POST /api/v1/face/settings` wins |
| `FACE_MIN_SIZE_PX` | `40` | — | Face recognition: faces whose box is smaller than this many pixels are ignored (`0`–`1024`) |
| `FACE_MAX_FACES` | `5` | — | Face recognition: at most this many faces per frame are aligned and identified, largest first (`1`–`20`); each one costs an embedder inference |
| `FACE_MODELS_DIR` | `assets/face` (in the image) | — | Directory holding the bundled YuNet detector and SFace embedder ONNX files (`scripts/fetch-face-models.sh`, checksums in `scripts/face-models.pin`). Without them the build does not register the `face` strategy and the device does not offer the application |
| `YOLO_MAX_CANDIDATES` | `300` | — | Maximum decoded candidates (highest confidence first) kept per inference |
| `YOLO_NMS_TOPK` | `120` | — | Maximum boxes (highest confidence first) passed to the overlay-threshold NMS |
| `AREA_MOVE_DELTA` | `0.02` | — | Normalized step for moving a detection area with the move commands |
| `AREA_RESIZE_DELTA` | `0.02` | — | Normalized step for resizing a detection area with the resize commands |
| `GPIO_SHM_PATH` | `/run/conecsa-gpio/state` | — | GPIO SHM channel the trigger gate reads every frame (must match the hardware agent) |
| `STEREO_COMBINE` | `none` | `none` | Stereo combine mode — split the side-by-side frame and blend both eyes into one image. Starts off: it would tear an ordinary camera's picture in half, so the Camera Settings toggle (shown only for a 3D camera) enables it and the per-model settings snapshot restores it |
| `STEREO_BLEND_ALPHA` | `0.5` | — | Blend factor for `STEREO_COMBINE=blend` |
| `STEREO_OFFSET` / `STEREO_OFFSET_Y` | `0.0` | — | Initial horizontal / vertical shift of the right eye over the left before blending, as a fraction of the eye's width / height (clamped to `-0.5`–`0.5`) |
| `CAPTURE_AUTO_EXPOSURE` | `false` | — | Camera auto-exposure |
| `CAPTURE_EXPOSURE_TIME` | `10000 / CAPTURE_FRAMERATE` (`333` at the code default of 30 fps) | `166` | Manual exposure time |
| `CAPTURE_RGB_RED` / `_GREEN` / `_BLUE` | `128` | — | Per-channel white-balance gains |
| `CAPTURE_GAMMA` | `100` | — | Camera gamma |
| `CAPTURE_GAIN` | `0` | — | Camera gain |
| `TENSORRT_WORKSPACE_MB` | `256` | `512` | TensorRT builder workspace (MB) for `.pt → .engine` conversion; `192` is enough for 640 engines (the dev compose uses it), 1280 engines usually need `512` (the builder retries 256/192/128) |
| `TENSORRT_AUTO_REBUILD_ENGINE` | `1` | — | Rebuilds the engine when the model changes |
| `TENSORRT_CONTEXTS` | `1` | `2` | Parallel TensorRT contexts / pipeline lanes |
| `TENSORRT_WORKER_PORT` | `5501` | — | Loopback port of the first TensorRT worker subprocess; lane *n* uses `TENSORRT_WORKER_PORT + n` |
| `WORKER_REQUEST_TIMEOUT_SEC` | `8.0` | — | Deadline (s) for one request to a TensorRT worker |
| `TENSORRT_BUILD_TIMEOUT_SEC` | `1800` | — | Deadline (s) for an engine build in a TensorRT worker |
| `PT_ONNX_TIMEOUT_SEC` | `600` | — | Deadline (s) for the `.pt → .onnx` export subprocess |
| `YOLO_AUTOINSTALL` | _(ultralytics default)_ | `false` | Stops ultralytics from pip-installing packages at runtime during export (a runtime upgrade would break pycuda in the workers); also set on training-service |
| `LD_PRELOAD` | _(unset)_ | jemalloc | Preloads jemalloc (installed in the os-base image) into the service and every subprocess; also set on training-service |
| `MALLOC_CONF` | _(unset)_ | `background_thread:true,dirty_decay_ms:5000,muzzy_decay_ms:5000` | jemalloc tuning: background decay thread, pages returned to the OS after 5 s; also set on training-service |
| `PYTORCH_CUDA_ALLOC_CONF` | _(unset)_ | `max_split_size_mb:128,garbage_collection_threshold:0.8` | PyTorch CUDA allocator tuning; only affects torch subprocesses (the `.pt → .onnx` conversion here, training on training-service) |
| `TENSORRT_FACE_WORKER_PORT` | `TENSORRT_WORKER_PORT + 17` | — | Loopback port of the private TensorRT worker running the face embedder beside the live detector's context lanes (a gallery build uses the next two ports for its own detector and embedder and closes both); the worker is closed when the device leaves the `face` application. Every private worker sits past 16 lanes, or past the last lane when `TENSORRT_CONTEXTS` exceeds 16, so it never collides with a live one |
| `TENSORRT_LABEL_WORKER_PORT` | `TENSORRT_WORKER_PORT + 16` | — | Loopback port of the private TensorRT worker that runs an existing engine for model-assisted labeling on the training page (beside, never instead of, the live model's context lanes; past the last lane when `TENSORRT_CONTEXTS` exceeds 16); released with the runtime |
| `OPENBLAS_NUM_THREADS` | `1` (image) | — | Threads of the OpenBLAS pools that numpy and OpenCV bundle. The pipeline already runs one thread per stage and lane; a pool sized to every core busy-waits between the small matrix products of segmentation masks and starves those threads. Keep it at `1` |
| `OMP_NUM_THREADS` | `1` (image) | — | OpenMP thread count for any library built with OpenMP; `1` for the same reason as `OPENBLAS_NUM_THREADS` |
| `CUDA_VISIBLE_DEVICES` | `0` | — | GPU visible to CUDA |
| `HUB_OFFLINE_THRESHOLD_SEC` | `5.0` | — | Seconds without a hub snapshot poll before the device considers the hub offline and starts buffering detections |
| `DETECTION_BUFFER_MAX_RECORDS` | `5000` | — | Offline detection buffer cap (records); oldest evicted first |
| `DETECTION_BUFFER_MAX_BYTES` | `1073741824` (1 GB) | — | Offline detection buffer cap (bytes); whichever cap hits first evicts |
| `DETECTION_BUFFER_SAMPLE_SEC` | `1.0` | — | While the hub is offline, frames are compared for a change at this interval (the hub's poll cadence), so a flickering detection set writes at most one record per interval |
| `DETECTIONS_DIR` | `/data/detections` | — | Offline buffer directory (`buffer.db`); falls back to a repo-local dir on host runs |

## `api-gateway`

| Variable | Default | Compose | Description |
|---|---|---|---|
| `INFERENCE_GRPC_ADDR` | `inference-service:50061` | — | Headless inference gRPC control surface |
| `TRAINING_GRPC_ADDR` | `training-service:50071` | — | Training-service gRPC control surface |
| `HARDWARE_AGENT_ADDR` | `os-base:50051` | `os-base:50051` | `os-base` hardware agent (network/Wi-Fi/GPIO) |
| `SHM_NAME` | `conecsa_frame_shm` | — | Camera SHM ring (raw feed) |
| `PROCESSED_SHM_NAME` | `conecsa_processed_shm` | — | Processed SHM ring (overlaid feed) |
| `PROCESSED_SHM_SLOT_BYTES` | `1048576` (1 MiB) | — | Per-slot size of the processed SHM ring (must match inference-service) |
| `GATEWAY_PORT` | `5000` | — | Internal HTTP port |
| `GATEWAY_DEBUG_ERRORS` | _(off)_ | — | `1`/`true` includes the exception class and text in `500` bodies (development only) |
| `MAX_IMAGE_UPLOAD_BYTES` | `20971520` (20 MB) | — | Cap on a single labeled-image upload relayed to the training-service |
| `TRUSTED_PROXY_HOST` | `system-vision` | — | Host name of the nginx mTLS terminator — the only peer whose `X-Conecsa-Client-Verify` header is honored |
| `TRAINING_ORPHAN_TIMEOUT_SEC` | `120` | `1800` | Client silence (s) after which training mode is exited automatically, so a dead hub or closed page cannot leave inference stopped; `0` disables |
| `WAITRESS_THREADS` | `32` | — | Waitress task threads (MJPEG/SSE pin one each) |
| `GATEWAY_GRPC_TIMEOUT` | `12` | — | Deadline (s) for every unary gRPC call to a backend that passes no explicit timeout; a call that exceeds it answers `504` |
| `GATEWAY_GRPC_LONG_TIMEOUT` | `120` | — | Deadline (s) for the slow unary calls (model select/reload, detection start, runtime release/resume, training start/cancel/finish, dataset delete, capture, SAM unload) |
| `GATEWAY_GRPC_UPLOAD_TIMEOUT` | `600` | — | Deadline (s) for client-streaming uploads (model, dataset, weights) without an explicit timeout |
| `STEREO_COMBINE` | `none` | `none` | Stereo combine for the training preview (matches inference-service); fallback only — the live inference config wins when reachable |
| `STEREO_BLEND_ALPHA` | `0.5` | — | Blend factor for the training preview |
| `STEREO_OFFSET` / `STEREO_OFFSET_Y` | `0.0` | — | Eye shift for the training preview (matches inference-service) |
| `DEVICE_VERSION` | `unknown` | `2026.6-LTS` | Device software version, surfaced on `/api/v1/status` + `/api/v1/health` for the hub (the in-container mDNS record advertises it empty when unset) |
| `DEVICE_ID` | _(see description)_ | — | Device identity used by enrollment, the cert SAN and mDNS. Resolved in order: this variable, the hostname read from `CONECSA_HOST_HOSTNAME`, the container's hostname |
| `CONECSA_HOST_HOSTNAME` | `/etc/conecsa/host_hostname` | — | File holding the host's hostname (compose bind-mounts the host's `/etc/hostname` there); the `DEVICE_ID` fallback, equal to the host avahi instance name |
| `CONECSA_CERT_DIR` | `/etc/conecsa/certs` | — | Device key/CSR + hub-signed cert/CA (volume shared with the nginx TLS terminator) |
| `DEVICE_PAIR_TOKEN` | _(unset)_ | `${DEVICE_PAIR_TOKEN:-}` | Optional shared pairing secret; unset = first hub on the trusted LAN to pair wins |
| `HUB_MDNS_ENABLED` | `1` | `0` | In-container mDNS advertiser; disabled in production (the host avahi-daemon advertises instead) |
| `DEVICE_ADVERTISE_IP` | _(auto-detected)_ | — | In-container mDNS only: IPv4 address to advertise instead of the detected primary address |
| `DEVICE_NAME` | _(container hostname)_ | — | In-container mDNS only: instance name |
| `DEVICE_HTTP_PORT` | `80` | — | In-container mDNS only: HTTP port advertised in the record |
| `CLOCK_SYNC_THRESHOLD_SEC` | `30` | — | Drift from the hub's clock that triggers a step (the board has no RTC battery; see [Clock synchronization](services/hub-vision.md#clock-synchronization)) |
| `CLOCK_SYNC_MIN_INTERVAL_SEC` | `60` | — | Minimum spacing between clock-step attempts, so a failing step is not retried on every hub status poll (every 2 s) |
| `AUDIT_DIR` | `/data/audit` | `/data/audit` | Audit trail directory (`audit.db`); needs a writable volume (`conecsa-audit-data`) |
| `FLOW_ADMIN_TOKEN_SECRET` | _(falls back to `NODE_RED_CREDENTIAL_SECRET`)_ | `${FLOW_ADMIN_TOKEN_SECRET:-}` | HMAC secret for the Node-RED editor tokens (`POST /api/v1/flow/token`); must equal the value the `flow` service verifies with |
| `FLOW_ADMIN_TOKEN_TTL_SEC` | `43200` (12 h) | — | Lifetime of an editor token |
| `AUDIT_MAX_RECORDS` | `50000` | — | Audit ring cap (records); oldest evicted first |
| `AUDIT_MAX_BYTES` | `67108864` (64 MB) | — | Audit ring cap (bytes); whichever cap hits first evicts |

## `training-service`

| Variable | Default | Compose | Description |
|---|---|---|---|
| `TRAINING_GRPC_LISTEN` | `0.0.0.0:50071` | — | gRPC control server bind address |
| `TRAINING_DATA_DIR` | `/data/training` | — | Dataset, run, weights-stash and base-checkpoint storage — the `{DATA_DIR}` in the [training-service page](services/training-service.md) |
| `SHM_NAME` | `conecsa_frame_shm` | — | Camera SHM ring (capture source; must match webcam-server) |
| `STEREO_COMBINE` | `none` | `none` | Stereo combine mode — compose sets inference-service to the same value so captured images match the live detector geometry (there is no runtime sync; the live inference config wins when reachable) |
| `STEREO_BLEND_ALPHA` | `0.5` | — | Blend factor for `STEREO_COMBINE=blend` |
| `STEREO_OFFSET` / `STEREO_OFFSET_Y` | `0.0` | — | Eye shift for captures, as in inference-service (clamped to `-0.5`–`0.5`) |
| `GATEWAY_ADDR` | `http://api-gateway:5000` | — | Gateway URL used to hand `best.pt` back through the model-upload route |
| `TRAIN_IMG_SIZE` | `640` | — | Model input size (`imgsz`) for training and for the exported ONNX/engine. 640 is the production default (a plain 1280 engine runs well below the live-pipeline fps floor on the Orin Nano); larger values train on real pixels only because datasets are stored at native resolution |
| `TRAIN_IMG_SIZE_CLS` | `224` | — | Model input size for training a **classification** dataset and for its exported engine (ultralytics' default for `-cls` models) |
| `TRAIN_BASE_WEIGHTS` | `/app/training-service/assets/yolo26s.pt` | — | Starting checkpoint for a detection dataset (committed and baked into the image; the device is often offline) |
| `TRAIN_BASE_WEIGHTS_CLS` | `/app/training-service/assets/yolo26s-cls.pt` | — | Starting checkpoint for a classification dataset (ImageNet-pretrained YOLO26s-cls, committed and baked the same way); ultralytics infers the task from the checkpoint |
| `TRAIN_BASE_WEIGHTS_SEG` | `/app/training-service/assets/yolo26s-seg.pt` | — | Starting checkpoint for a segmentation dataset (COCO-pretrained YOLO26s-seg, committed and baked the same way); it trains at `TRAIN_IMG_SIZE` on the same tile crops as detection |
| `TRAIN_DATASET_IMG_SIZE` | `0` | `0` | Dataset storage geometry. `0` (default) stores the stereo-combined camera frame at native resolution and leaves letterboxing to ultralytics at train time; a value > 0 stores images letterboxed to that square (the `640` format of datasets created by older firmware). Recorded per dataset in `meta.json`; adding images to a dataset created with a different geometry is refused |
| `TRAIN_TILE` | `auto` | — | Training geometry (`off`, `0`, `none`, `false` or `no` = whole frames; an unparseable or non-positive value falls back to `auto`). `auto` (default) slices every image of the train/valid split into the square tile crops the inference-service runs on — side = the image's short side, the same `conecsa_common.tiling` grid as `TILING_TILE=auto` — and rewrites the labels per tile, so the model is trained at the scale it is deployed at; a pixel value mirrors an explicit `TILING_TILE`; `off` trains on whole frames for a device running `TILING_MODE=off`. Images the grid cannot slice (a 640×640 letterboxed dataset) stay whole. The effective geometry (`frames`, `tiles:auto`, `tiles:<px>`) is declared on the model upload and recorded in the model's settings sidecar |
| `TRAIN_TILE_OVERLAP` | `0.2` | — | Overlap fraction between adjacent training tiles (`0 ≤ f < 1`); mirrors `TILING_OVERLAP` |
| `TRAIN_TILE_MIN_VISIBLE` | `0.25` | — | A box is kept in a tile when at least this fraction of its area lies inside it (`0 < f ≤ 1`); a tile whose only content was a smaller fragment is skipped rather than taught as background |
| `TRAIN_OVERRIDES` | _(empty)_ | — | Space-separated allowlisted ultralytics `model.train` hyperparameters, e.g. `freeze=10 lr0=0.002 close_mosaic=5` (allowlist in `training-service/service/train_overrides.py`) |
| `TRAIN_MIN_IMAGES` | `20` | — | Training gate: minimum images a dataset needs before a job can start |
| `TRAIN_DEFAULT_EPOCHS` | `50` | — | Epochs when the request does not set them |
| `TRAIN_DEFAULT_PATIENCE` | `50` | — | Early-stopping patience when the request does not set it |
| `TRAIN_AMP` | `1` | — | Mixed-precision training; `0`, `false` or `no` disables it |
| `TRAIN_BATCH` | `4` | — | YOLO training batch size (sized for the Orin Nano 8 GB) |
| `TRAIN_WORKERS` | `0` | — | DataLoader workers (0 = single-process; the small shared `/dev/shm` cannot back worker tensors) |
| `TRAIN_STALL_TIMEOUT_SEC` | `3600` | — | Liveness watchdog — kills the trainer after this long with **no** output (a hang), not a cap on total duration |
| `TRAIN_TIMEOUT_SEC` | `0` (disabled) | — | Optional overall wall-clock cap (s) on a training run |
| `TRAINING_MAX_UPLOAD_MB` | `512` | — | Cap on an uploaded dataset ZIP (the spooled file and its uncompressed total) |
| `TRAINING_MAX_ZIP_ENTRIES` | `20000` | — | Maximum entries in an imported dataset ZIP |
| `TRAINING_MAX_ZIP_ENTRY_MB` | `256` | — | Maximum uncompressed size of one ZIP entry |
| `TRAINING_MAX_ZIP_RATIO` | `100` | — | Maximum compression ratio of a ZIP entry, checked once the entry has expanded past 1 MB (zip-bomb guard) |
| `TRAINING_MAX_IMAGE_PIXELS` | `64000000` | — | Maximum pixels of an imported image, checked from the file header before decoding |
| `TRAINING_MAX_WEIGHTS_MB` | `200` | — | Cap per uploaded federated checkpoint (`last.pt` carries optimizer state, ~2-3× the model size) |
| `TRAINING_WEIGHTS_TTL_SEC` | `86400` | — | TTL of stashed federated checkpoints under `{DATA_DIR}/weights/` (hub deletes best-effort; the prune is the backstop) |
| `SAM3_CHECKPOINT` | `/app/training-service/assets/sam3.pt` | — | SAM3 checkpoint (HF-gated; downloaded locally and baked into the image at build time) |
| `SAM_WORKER_PORT` | `5601` | — | Loopback port of the SAM3 worker subprocess |
| `SAM_IDLE_UNLOAD_SEC` | `300` | — | Idle time (s) after which the SAM3 worker is unloaded to free GPU memory |
| `SAM3_WARM_CACHE` | `1` | — | Reads the checkpoint into the page cache in the background while the worker starts; any other value disables |
| `SAM3_SKIP_INIT` | `1` | — | Skips random parameter initialization while building the model (the checkpoint overwrites it); any other value disables |
| `SAM3_DTYPE` | `fp32` | — | SAM3 weight dtype; `bf16` is experimental (inference already uses bf16 autocast over fp32 weights) |
| `SAM3_INSTALL` | `1` | — | Build argument of `Dockerfile.training-service`: installs the SAM3 package into the image; `0` skips it and AI labeling reports unavailable |
| `SAM3_COMMIT` | _(pinned commit)_ | — | Build argument of `Dockerfile.training-service`: SAM3 source commit to install |

## `os-base` hardware agent

| Variable | Default | Compose | Description |
|---|---|---|---|
| `HARDWARE_AGENT_LISTEN` | `0.0.0.0:50051` | `0.0.0.0:50051` | gRPC `HardwareService` bind address |
| `AP_ADDRESS_CIDR` | `10.98.76.1/24` | — | Address and subnet the device takes as a Wi-Fi access point (see [os-base hardware agent](services/os-hardware-agent.md#wi-fi-access-point)); a start is refused while the subnet overlaps an active network, so change it at sites that use that range. The access point also needs the `/run/systemd/network` bind mount `docker-compose.yml` gives `os-base`; without it a start is refused with "This deployment cannot manage the access point network file" |
| `CONECSA_CLOCK_FLOOR` | `/var/lib/conecsa/fake-hwclock` | — | Clock floor file shared with the host: `SetSystemTime` refuses times older than it and rewrites it after each accepted step (see [os-base hardware agent](services/os-hardware-agent.md#system-clock)) |
| `PIN_PERFORMANCE_CLOCKS` | `1` | — | Pins the Jetson GPU/CPU performance clocks at startup; any other value disables |
| `GPIO_POLL_MS` | `5` | — | GPIO poll loop interval (ms) |
| `GPIO_SHM_PATH` | `/run/conecsa-gpio/state` | — | GPIO SHM channel the agent writes (must match inference-service) |

## `system-vision`

| Variable | Default | Compose | Description |
|---|---|---|---|
| `SYSTEM_VISION_TLS_PORT` | `443` | — | Published mTLS port — the **only** port the production stack exposes |
| `SYSTEM_VISION_PORT` | `80` | _(dev only)_ | Plaintext web port; used only by `docker-compose.dev.yml`, never published in production |

## `flow`

| Variable | Default | Compose | Description |
|---|---|---|---|
| `INFERENCE_URL` | `http://api-gateway:5000` | — | Base URL the Conecsa nodes use to reach the api-gateway |
| `NODE_RED_CREDENTIAL_SECRET` | _(required)_ | `${NODE_RED_CREDENTIAL_SECRET:?}` | Encrypts credentials stored in flows; also signs the editor tokens unless `FLOW_ADMIN_TOKEN_SECRET` is set |
| `FLOW_ADMIN_TOKEN_SECRET` | _(falls back to the credential secret)_ | `${FLOW_ADMIN_TOKEN_SECRET:-}` | Verifies the editor tokens the api-gateway mints (`flow/admin-token.js`) |
| `FLOW_ADMIN_AUTH` | `1` | _(dev: `0`)_ | `0` leaves the editor open (no `adminAuth`); the dev stack publishes the editor on the host and sets it |
| `DEVICE_ID` | _(empty)_ | — | Device id stamped on detection messages (node config takes precedence) |
| `TZ` | — | `America/Sao_Paulo` | Timezone |

> The Flow editor's port (`1880`) is published only by the dev stack
> (`docker-compose.dev.yml`); in production it is reachable through the
> hub's mTLS proxy.

## `hub-vision`

The fleet hub is a desktop app, not a compose service; see
[Fleet hub](services/hub-vision.md) for details.

| Variable | Default | Description |
|---|---|---|
| `HUB_INGEST_QUEUE_CAP` | `10000` | Capacity of the detection ingest queue; beyond it the pull collector waits rather than dropping records (unparseable or non-positive values use the default) |
| `HUB_KEK_FILE` | _(unset)_ | Path of a file-based key-encryption key for `secrets.bin`. Unset: the OS keychain, falling back to `kek.bin` in the data directory when no keychain is available; set but empty is an error |
| `CONECSA` | _(unset)_ | `true` or `1` enables the hidden built-in `conecsa` owner sign-in; otherwise it cannot sign in |
