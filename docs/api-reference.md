# HTTP API reference

All HTTP endpoints are served by the **api-gateway** on port 5000. The gateway
relays each to the headless inference-service, the training-service or the
`os-base` hardware agent over gRPC, or fans the MJPEG feeds out of shared
memory.

Every mutating request is recorded in the device's
[audit trail](services/hub-vision.md#audit-trail) before the response leaves —
including the [short `/api/*` aliases](#short-api-aliases) — except five
routes that carry no user intent: the two backlog acknowledgements
(`/api/v1/detections/backlog/ack`, `/api/v1/audit/backlog/ack`), the training
heartbeat (`/api/v1/training/heartbeat`) and the per-click labeling calls
(`/api/v1/training/sam/segment`, `/api/v1/training/label-model/detect`). Reads
are not recorded. The routes marked **hub-only** below answer to the paired hub
alone (mTLS, verified by the nginx terminator) and return `403` to anyone else.

Third-party systems reach these routes through the hub's
[Developer API](services/hub-vision.md#developer-api):
`https://<hub>:8443/devices/<device_id>/api/...` with an `X-Api-Key` header,
forwarded over the hub's mTLS channel.

## Roles

The device authenticates nobody itself; the role is the `X-Conecsa-Role` header
the hub stamps on operator traffic. `ROUTE_POLICIES` in
`api-gateway/gateway/authz.py` gives every mutating route (`POST`, `PUT`,
`PATCH`, `DELETE`) a minimum role, ranked `user` < `admin` < `owner`. It is
enforced only on requests the mTLS terminator verified as coming from the hub:

- **Not hub-verified** (the internal compose network, a dev stack): no check;
  the network boundary applies.
- **Hub-verified, no role header**: the hub acting on its own behalf (applying a
  recipe, draining backlogs); allowed.
- **Hub-verified with a role**: an unknown role, a role below the route's
  minimum, or a mutating route without a policy answers
  `403 {"error": "forbidden"}` before the handler runs (the rejected attempt is
  still audited).

Reads are not role-checked, and the `/enroll/*` routes follow the pairing
policy instead. `user` suffices for operating the detector:

| Method | Endpoints (with their `/api/*` aliases) |
|---|---|
| `POST` | `/api/v1/start`, `/api/v1/stop`, `/api/v1/threshold`, `/api/v1/overlay_threshold`, `/api/v1/segment/max_instances`, `/api/v1/face/settings`, `/api/v1/stats/reset`, `/api/v1/counter/reset`, `/api/v1/trigger/enable`, `/api/v1/trigger/disable`, `/api/v1/flow/token` |

Every other mutating route requires `admin`. That includes the two backlog
acknowledgements (`/api/v1/detections/backlog/ack`, `/api/v1/audit/backlog/ack`),
which the hub normally issues without a role header and so passes as itself.

## Detection

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/status` | System state, active model, thresholds, runtime, FPS, the device's application `task` (see [Application type](#application-type)) and the settings of the task the device runs (`segment_max_instances` for segmentation; `face_match_threshold`, `face_min_size_px`, `face_max_faces` for face recognition), absent for every other task and on older firmware — in JSON keys and, for `Accept: application/x-protobuf`, as the optional fields of the same names |
| `POST` | `/api/v1/start` | Start detection. Answers `409` while no application type is chosen, while no camera is connected, while a training job is active or while a model conversion (TensorRT build) runs — the single GPU is busy |
| `POST` | `/api/v1/stop` | Stop detection |
| `GET` | `/api/v1/stats` | FPS, latency, detection count, frames with detections, and the pipeline's stage service times in ms: `finish_mean_ms`/`finish_p95_ms`/`finish_p99_ms` (postprocess), `encode_mean_ms`/`encode_p95_ms` (JPEG encode + publish) and `frame_age_p95_ms` (from the frame's pickup off the camera ring to its publication) |
| `POST` | `/api/v1/stats/reset` | Reset the stats counters |
| `GET` | `/api/v1/stats/stream` | Server-Sent Events stream of the stats. Heartbeat (`: keepalive`) every 15s when idle. Used by the Node-RED `stats` node |
| `GET` | `/api/v1/events/stream` | Unified Server-Sent Events stream: invalidation events (each carries a `keys` list to re-fetch) from any client; opt-in `?stats=1` multiplexes the live stats channel onto the same connection. Used by the web app (one connection instead of two) and the Node-RED nodes |
| `GET` | `/api/v1/detections/snapshot` | Latest detections snapshot (JSON); `task` names the device's application, and each detection carries its normalized `bbox` corners. On a classification device `detections` holds at most the frame's class above the confidence threshold — no `bbox`, `area: null` — so `total` is 1 or 0, and `candidates` lists the top-k classes (`class_id`, `class_name`, `confidence`, highest first; the key is absent for other tasks). On a face recognition device each detection is one face, in the same shape as a detection: `class_name` is the recognized person's name or `unknown`, `confidence` is the cosine similarity to that person (not a detection score), and `total` counts the faces in the frame. On a segmentation device each detection also carries `polygons` — its outline as exterior rings of normalized `[x, y]` vertices; when the rings of one snapshot would exceed 64 KiB the smallest are left out and `polygons_truncated: true` is set (the key is absent otherwise). `?include_frame=false` omits the annotated JPEG frame; `?include_raw_frame=true` adds the clean frame (`raw_frame`, no overlay — used for dataset ingest). `?passive=true` marks a reader that never counts as the hub's heartbeat for the offline buffer (the device UI's classification panel). `pending_backlog` reports how many records the offline buffer holds. Polled by the [hub](services/hub-vision.md) over mTLS |
| `GET` | `/api/v1/detections/backlog` | One page of offline-buffered detection records, oldest first (`?limit=N`, default 25, max 100). Each record carries `id`, `captured_at`, `task` (the application that produced it — records outlive an application switch; absent on records buffered by older firmware) and the snapshot-format detections/frame; the envelope adds `device_now` (the device clock, so the hub can offset-correct timestamps) and `pending` |
| `POST` | `/api/v1/detections/backlog/ack` | Delete buffered records the hub confirmed persisting — body `{"ids": [..]}`, idempotent. The hub calls this only after its own store insert committed |
| `POST` | `/api/v1/flow/token` | Mint a short-lived bearer token for the embedded Node-RED editor (`{"token", "expires_in"}`), carrying the operator's identity and role the hub vouched for; a request that is not hub-verified (or a hub request without an identity) gets the user `local` with role `admin`. The device UI opens `/flow/?access_token=<token>` with it. `503` when no secret is configured |
| `GET` | `/api/v1/health` | Liveness: the gateway process answers (constant body) |
| `GET` | `/api/v1/ready` | Readiness: probes the gRPC health service of the inference-service, training-service and hardware agent; `503` `degraded` when the inference-service is not serving. The device UI's header status pill polls it every 5 s |

## Event stream

`GET /api/v1/events/stream` is a Server-Sent Events stream; each `data:` line is
one JSON envelope:

```json
{"version": 42, "type": "thresholds_changed", "timestamp": 1767225600.0,
 "source": "api", "keys": ["status", "thresholds"], "data": {"confidence_threshold": 0.5}}
```

`version` is the gateway bus's own counter, `timestamp` is the device's wall
clock (do not diff it against the client's), `keys` names the state to re-fetch
and `data` is an optional hint. A client that connects first receives a
`state_snapshot` with `keys` `status`, `models`, `classes`, `thresholds`,
`camera`, `network`, `gpio`, `trigger`, `areas` and `application` (empty
`data`), and receives one again whenever it has fallen behind the replay buffer.
When idle the stream sends `: keepalive` every 15 s. With `?stats=1` (or `true`,
`yes`) the live stats are multiplexed as `{"version", "type": "stats",
"source": "api", "keys": ["stats"], "data": {...}}` messages (the fields of
`GET /api/v1/stats`, no `timestamp`).

The gateway publishes the events of its own routes and relays the
`StreamEvents` of both the inference-service and the training-service, keeping
their `type`, `keys` and `source`. Each relay forwards the backend's own
`state_snapshot` when it (re)connects: the inference-service's carries the
keys above, the training-service's `training`, `dataset` and `sam`.

| Source | `type` | `keys` | Meaning |
|---|---|---|---|
| gateway | `detection_state_changed` | `status` | Detection started or stopped, including by training enter/exit |
| gateway | `thresholds_changed` | `status`, `thresholds` | Confidence or overlay threshold, or segmentation instance limit set |
| gateway | `stats_changed` | `stats` | Stats counters reset |
| gateway | `config_changed` | `status`, `thresholds`, `camera` | `PUT /api/v1/config` applied |
| gateway | `camera_config_changed` | `camera` | Camera configuration changed |
| inference | `camera_health_changed` | `camera_health` | The camera's health status or detail changed; `data` is `{status, detail, source}` for the applied capture source |
| gateway | `trigger_changed` | `trigger` | Trigger enabled or disabled |
| gateway | `counter_changed` | `trigger` | Detection counter reset |
| gateway | `detection_areas_changed` | `areas` | A detection area was created, edited or removed |
| gateway | `classes_changed` | `classes` | Classes uploaded or cleared |
| gateway | `conversion_started` | `models`, `conversion` | A model upload started an asynchronous conversion |
| gateway | `model_changed` | `models`, `status`, `classes`, `areas`, `thresholds`, `camera` | A model was uploaded (loaded directly) or selected |
| gateway | `models_changed` | `models` | A model was deleted |
| gateway | `network_config_changed` | `network` | IPv4 settings applied, a Wi-Fi network connected or forgotten, or the access point started or stopped |
| gateway | `gpio_changed` | `gpio` | GPIO trigger mode or an output pin changed |
| inference | `runtime_changed` | `status` | GPU handover began (`runtime_released: true`) or ended (`false`) |
| inference | `application_changed` | `application`, `models`, `status` | The application type changed (`data.task`) |
| inference | `conversion_changed` | `conversion` | A conversion job was created or changed status/progress (`source` `conversion`) |
| inference | `classes_changed` | `classes` | A conversion saved the model's class labels (`source` `conversion`) |
| inference | `models_changed` | `models` | A conversion finished its engine (`source` `conversion`) |
| inference | `label_model_changed` | `label_model` | The labeling engine was loaded or unloaded (`source` `labeling`) |
| training | `datasets_changed` | `datasets` | The dataset list changed, or a dataset's cover was set |
| training | `dataset_changed` | `dataset` | An image was captured, added, deleted or replicated, or the classes changed |
| training | `training_progress` | `training` | A training job's status or progress changed |
| training | `sam_changed` | `sam` | The SAM3 worker was loaded or unloaded |

Gateway events take their `source` from the client's `X-Conecsa-Source` header
(default `api`); inference events without a noted source use `api`, training
events `training`.

## Application type

A device runs one application — object detection, image classification,
instance segmentation or face recognition — and every model, dataset and
detection snapshot carries the task it belongs to: `detect`, `classify`,
`segment` or `face`. This build runs `detect`, `classify` (whole-frame
classification), `segment` (instance segmentation with YOLO26's end-to-end
head) and, when the image carries the bundled face models, `face`
(identification against an enrolled gallery — see
[Face recognition](face-recognition.md)); `supported_tasks` is authoritative. A device
that already held models before application types existed is an
object-detection device; a blank device (no model at all) has no application
until an administrator chooses one, and refuses `POST /api/v1/start` (`409`)
until then.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/application` | `{"task", "supported_tasks", "migrated"}`: `task` is `null` while none is chosen, `supported_tasks` lists what this build can run, `migrated` is `true` when the task was set automatically for a device that already held models |
| `PUT` | `/api/v1/application` | **admin** — `{"task": "detect"}`. Stops detection and deselects an active model of another task (its files are kept); detection is not restarted; `application_changed` is published on `/api/v1/events/stream`. Choosing the current task changes nothing. `400` unknown task; `409` a task this build cannot run, or while a training job or a model conversion holds the GPU; `503` when that check cannot reach the training or inference service; `500` when the device could not stop detection or store the choice (the previous application stays) |

How clients read the task:

| Where | None chosen | Chosen | An id the client does not know | Older firmware |
|---|---|---|---|---|
| JSON (`/api/v1/status`, `/api/v1/application`, SSE) | `"task": null` | `"task": "detect"` | a newer device's task: show it as unsupported, never as unset | `/api/v1/status` has no `task` key: no guard, object detection |
| Protobuf (`StatusResponse.task`) | empty string | `"detect"` | as above | the field is unset (no presence) |
| Hub device list | no badge | the task's name | *Unsupported application* | the last known task, else no badge |

A face model is built on the device from a `face` dataset and its gallery
never leaves it: it is not downloadable and not federated.

A model without a recorded task, and a dataset created before application
types existed, are `detect`. `POST /api/v1/training/datasets` creates the
dataset for the device's task (`409` while none is chosen, or when the body's
`task` names another one).

## Models

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/models` | List available models (name, size, date, active, `has_weights` = the model keeps the `.pt` it was converted from as a training-checkpoint sidecar, `task` = the model's task). `?task=<task>` lists only that task's models |
| `POST` | `/api/v1/model` | Upload a model. The optional form field `task` declares the model's task (default: the device's application; required on a device without one); a `.pt`/`.onnx` of another task fails its conversion and an engine of another task is refused (`409`). The optional form field `imgsz` is the conversion input size; without it a `.pt` is exported at the size it was trained at (ultralytics records it in the checkpoint), else at the task's default (`224` for classification, `640` otherwise). The optional form field `train_geometry` (`frames`, `tiles:auto` or `tiles:<px>`) records the geometry the model was trained at; a malformed value is ignored. A classification engine must output probabilities (softmax in the exported graph, as ultralytics exports it): one that outputs logits is refused at activation. A segmentation engine must have YOLO26's end-to-end head (detection rows plus mask prototypes); a segmentation engine with a one-to-many head is refused at activation. `.pt` and `.onnx` → async conversion to an engine (202 + `job_id`); `.engine`/`.plan` load immediately |
| `POST` | `/api/v1/model/select` | Select active model by name |
| `DELETE` | `/api/v1/model/<name>` | Remove a model |
| `GET` | `/api/v1/model/<name>/download` | Download the model file |
| `GET` | `/api/v1/model/<name>/weights` | Download the model's training checkpoint (the `.pt` it was converted from, see `has_weights`); 404 when it keeps none. Used by the training-service to fine-tune from the model |
| `GET` | `/api/v1/model/conversion` | List active conversion jobs |
| `GET` | `/api/v1/model/conversion/<job_id>` | Conversion status: `pending`, `converting_to_onnx`, `converting_to_engine`, `done`, `failed` |

## Configuration

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/config` | Current configuration (device, resolution, framerate, thresholds) |
| `PUT` | `/api/v1/config` | Update configuration |
| `POST` | `/api/v1/threshold` | Set confidence threshold (0.0–1.0) |
| `POST` | `/api/v1/overlay_threshold` | Set the overlay threshold (0.0–1.0): the IoU above which overlapping boxes are suppressed (NMS) |
| `POST` | `/api/v1/segment/max_instances` | Set the segmentation instance limit, JSON `{"max_instances": n}` with `n` in 1–255: the most instances per frame (and per tile) that get a mask, saved with the active model's settings; `GET /api/v1/status` reports the value in effect as `segment_max_instances` while the device runs segmentation |

## Face recognition

On a face recognition device (`task` = `face`) `GET /api/v1/status` reports
three more fields — `face_match_threshold` (`0`–`1`), `face_min_size_px`
(`0`–`1024`) and `face_max_faces` (`1`–`20`) — and one route changes them.
The confidence threshold is the face **score** gate and the overlay threshold
the NMS IoU, as for detection. See [Face recognition](face-recognition.md) for
the whole application.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/face/settings` | **user** — JSON with any of `{"match_threshold": 0.0–1.0, "min_size_px": 0–1024, "max_faces": 1–20}`; only the given fields change, saved with the active model's settings. `match_threshold` is the cosine similarity a face must reach to take an enrolled person's name (below it: `unknown`), `min_size_px` ignores smaller faces and `max_faces` caps how many faces of a frame are identified (largest first). `400` on a value out of range; audited as `detection.face_settings_changed` with the changed fields |

Uploading a face model is the gallery build described on that page: the
`.faces` package goes through `POST /api/v1/model` with `task=face` (a file
that is not a `.faces` package declared `face` is refused with `400`), and the
resulting `<name>.gallery.npz` is never downloadable.

## Camera

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/camera/devices` | List V4L2 devices + current configuration, the capture source and the camera health |
| `POST` | `/api/v1/camera/config` | Update the camera tuning (index, width, height, framerate, exposure, ...) and/or the capture source via shared memory. Admin only |

`GET /api/v1/camera/devices` also reports the capture source: `current_source`
(`local` or `network`), `current_network_host`, `current_network_port`,
`network_token_set` (a boolean — the token itself is never returned),
`camera_connected` (`true` only while frames arrive),
`camera_status` (`starting`, `capturing`, `no_camera`) and `camera_detail`, which
says why a [remote camera](remote-camera.md) is not capturing: `unspecified`,
`connecting`, `unauthorized`, `rate_limited`, `unreachable`, `stalled` or
`bad_stream`.

`POST /api/v1/camera/config` accepts the capture-source fields next to the tuning
ones:

| Field | Value |
|---|---|
| `source` | `local` or `network` |
| `network_host` | The remote camera's IPv4 address. Unspecified, loopback, multicast and broadcast addresses are rejected |
| `network_port` | `1`–`65535` |
| `network_token` | 8–32 Crockford base32 characters; hyphens and case are ignored. **Write-only** |

The merged state is validated before anything is written: switching to `network`
needs a complete address, port and token. An omitted `network_token` keeps the
stored one; an empty one is an error, not a way to clear it. Switching to `local`
keeps the stored remote camera address and token. The source is device-level — it is not
part of any model's settings. The response, the `camera_config_changed` event and
the audit entry (`camera.configured`) never contain the token.

## Video (MJPEG fanned out of shared memory)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/video_feed` | Raw MJPEG stream (camera SHM ring) |
| `GET` | `/api/v1/video_feed_processed` | MJPEG stream with detection overlays (processed SHM ring) |

## Trigger and counter

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/trigger/status` | Trigger state (`trigger_enabled`) and `detection_count` |
| `POST` | `/api/v1/trigger/enable` | Enables frame processing |
| `POST` | `/api/v1/trigger/disable` | Freezes the last processed frame |
| `GET` | `/api/v1/counter` | Accumulated detection counter |
| `POST` | `/api/v1/counter/reset` | Resets the counter |

## Detection areas

Normalized coordinates in `[0, 1]`; the per-click delta size is controlled by
`AREA_MOVE_DELTA` and `AREA_RESIZE_DELTA` (default `0.02` each). Every
endpoint returns the full state (`{"areas": [...]}`).

| Method | Endpoint | Description |
|---|---|---|
| `GET`    | `/api/v1/detection-areas`               | List every area |
| `POST`   | `/api/v1/detection-areas`               | Create a new area (40%×40%, centered, `is_editing=true`); turns off the previous `editing` flag |
| `DELETE` | `/api/v1/detection-areas/<id>`          | Remove the area |
| `POST`   | `/api/v1/detection-areas/<id>/save`     | Leave editing mode (commit — overlay disappears, filter stays) |
| `POST`   | `/api/v1/detection-areas/<id>/discard`  | Discard pending edits: restore the pre-edit geometry/shape, or remove the area entirely if it was newly created |
| `POST`   | `/api/v1/detection-areas/<id>/edit`     | Promote a saved area back to editing mode |
| `POST`   | `/api/v1/detection-areas/<id>/shape`    | Body `{"shape": "rectangle"\|"circle"}` |
| `POST`   | `/api/v1/detection-areas/<id>/command`  | Body `{"action": "<command>"}`. Commands: `move_up`, `move_down`, `move_left`, `move_right`, `grow`, `shrink`, `grow_horizontal`, `shrink_horizontal`, `grow_vertical`, `shrink_vertical` |

## Classes and system

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/classes` | List labels/classes |
| `POST` | `/api/v1/classes` | Upload `classes.txt` or JSON with a list of names |
| `DELETE` | `/api/v1/classes` | Clear custom classes (revert to the model default) |
| `GET` | `/api/system/status` | CPU%, RAM%, disk%, temperature, GPU% (use/temp/freq — Jetson sysfs) |
| `POST` | `/api/v1/system/power` | Shut down or restart the host. Body `{"action": "shutdown"\|"restart"}` (relayed to the `os-base` hardware agent's `SystemPower` RPC) |

## GPIO and network (relayed to the `os-base` hardware agent)

Each of these routes relays one RPC to the agent. An agent the gateway cannot
reach answers `503`; one that does not answer within the call's deadline
(60 s for an access point start or stop, 30 s for a Wi-Fi connect, 12 s
otherwise) answers `504`. Neither body carries the agent's address. A
Wi-Fi change the agent refuses because the radio is serving the
[access point](remote-camera.md#using-the-device-as-the-access-point)
answers `409` with the reason.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/gpio/status` | GPIO availability, trigger mode, output pins and their current levels |
| `POST` | `/api/v1/gpio/trigger` | Enable/disable GPIO trigger mode (`{"enabled": bool}`) |
| `POST` | `/api/v1/gpio/pin` | Drive an output pin HIGH/LOW (`{"pin": 29\|31\|33, "level": bool}`). Used by the Node-RED `gpio` node |
| `GET` | `/api/v1/network/config` | Current wired + Wi-Fi configuration |
| `POST` | `/api/v1/network/config` | Apply IPv4 settings: `{"interface": "wired"\|"wifi", "method": "auto"\|"static", "address", "prefix", "gateway", "dns": [...]}`. With `static`, `address` and `prefix` (1–32) are required and `address`, `gateway` and each `dns` entry must be IPv4 addresses (`400` otherwise); a change networkd cannot apply is rolled back to the previous configuration; `409` for the `wifi` interface while the access point is up |
| `GET` | `/api/v1/network/wifi/scan` | List available Wi-Fi networks |
| `POST` | `/api/v1/network/wifi/connect` | Connect to a network (`{ssid, password}`); `409` while the access point is up |
| `POST` | `/api/v1/network/wifi/forget` | Remove a saved network (`{ssid}`); `409` while the access point is up |
| `GET` | `/api/v1/network/ap` | The device's Wi-Fi access point: `active`, `ssid` (the device id, also when inactive), `frequency_mhz`, `address`, `prefix`, `stations` (`address`, `hostname`, `signal` per joined remote camera), `join_deadline_remaining_secs`, `wired_ready`, `message`, `channels` (the ones the radio may start on right now). Never the passphrase |
| `POST` | `/api/v1/network/ap/start` | Start the access point (`{passphrase, channel?}`; the SSID is the device id, `channel` 36, 40, 44 or 48, or 0/absent for automatic). A passphrase outside 8–63 characters or a channel outside that set is `400`. Every other refusal is `200` with `success: false` and the reason in `message`: already active, no wireless interface, no wired link (carrier plus an IPv4 address), the subnet overlapping an active network (or a link whose addresses cannot be read), another transition running, a passphrase that is not printable ASCII, the deployment lacking the `/run/systemd/network` mount, no regulatory country in wpa_supplicant, or the chosen channel (or, with automatic, every channel) blocked right now, naming the usable ones. A failed start rolls the radio back to station mode. Emits `network_config_changed`; audited as `network.ap_started` with the SSID only, never the passphrase |
| `POST` | `/api/v1/network/ap/stop` | Return the radio to station mode (idempotent). Emits `network_config_changed`; audited as `network.ap_stopped` |

## Training (relayed to the training-service)

See [training-service](services/training-service.md) for the workflow. Every
dataset-scoped route carries the `dataset_id` explicitly.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/training/enter` | Enter training mode (acquire GPU handover from inference) |
| `POST` | `/api/v1/training/exit` | Leave training mode (resume inference runtime). Body `{"resume_detection": false}` keeps detection stopped and the engine unloaded for the model conversion, but still ends the GPU handover, so the application type can change again; the resume is also skipped while a training job is active (the handover then lasts until the job ends and detection is started) |
| `POST` | `/api/v1/training/heartbeat` | Keep the training orphan watchdog fed while the training page is open (the device UI beats about every 10 s); returns `{"status": "ok"}` and changes nothing else |
| `GET` | `/api/v1/training/preview` | Capture-preview frame (current camera image) |
| `GET` | `/api/v1/training/datasets` | List datasets (`dataset_id`, `name`, `task`, `created_at`, `image_count`, `labeled_count`, `class_count`, `cover_image_id`) |
| `POST` | `/api/v1/training/datasets` | Create a dataset |
| `POST` | `/api/v1/training/datasets/upload` | Multipart `file` (the ZIP) and `name` (both required, `400` otherwise). Import a ZIP of a pre-existing dataset laid out for the device's task: YOLO format (`images/` + `labels/` + `data.yaml`) for detection, the same layout with polygon rows (`class x1 y1 … xn yn`) for segmentation; one folder of images per class for classification (optionally under `train/`, `val/`, `test/`, with a `classes.txt` fixing the class order). An archive of the other task is refused; `409` while no application type is chosen |
| `GET` | `/api/v1/training/datasets/<dataset_id>` | Dataset metadata |
| `PUT` | `/api/v1/training/datasets/<dataset_id>` | Rename a dataset |
| `DELETE` | `/api/v1/training/datasets/<dataset_id>` | Delete a dataset |
| `GET` | `/api/v1/training/datasets/<dataset_id>/export` | Export the dataset as a ZIP in the layout the import accepts (a classification dataset as `train/<class>/` folders plus `classes.txt`, unlabeled images left out); `?shards=N&index=I[&seed=S]` exports one deterministic IID shard instead (federated training): `shards` 2–16, `index` (default `0`) below `shards`; a non-integer `shards` or `index` answers `400`. A face dataset is never exported, with or without shards (`409`): its enrolment photos are biometric data that stay on the device, so the hub fetches none of its images and offers neither gallery, download nor transfer for it |
| `PUT` | `/api/v1/training/datasets/<dataset_id>/cover` | Set the dataset cover image |
| `POST` | `/api/v1/training/datasets/<dataset_id>/capture` | Capture the current camera frame into the dataset |
| `POST` | `/api/v1/training/datasets/<dataset_id>/images` | Add an external image (multipart `file`) with optional pre-labels: `boxes` (JSON `[{class_name, x1, y1, x2, y2}]`, normalized corners) for a detection dataset, `polygons` (JSON `[{class_name, points: [[x, y], …], instance}]`, normalized on the uploaded image; `instance`, optional, groups the rings of one object; a ring without one is an object of its own) for a segmentation dataset, or `image_class` (a class name) for a classification dataset — the dataset's task decides which kind it accepts (`400` for the other). Stored like a capture; class names are resolved/created. Used by the hub's record-to-dataset flow |
| `GET` | `/api/v1/training/datasets/<dataset_id>/images` | List images (+ label state, replica flag, and `image_class` for a classification dataset) |
| `GET` | `/api/v1/training/datasets/<dataset_id>/images/<image_id>` | Fetch an image (JPEG) |
| `DELETE` | `/api/v1/training/datasets/<dataset_id>/images/<image_id>` | Delete an image (and its labels) |
| `GET` | `/api/v1/training/datasets/<dataset_id>/images/<image_id>/labels` | Get the image's labels: `boxes` (normalized center/size), `polygons` (`[{class_id, instance, points}]`, the rings of a segmentation image; read back from its label file, every ring is its own instance) and `image_class` (the class index of a classification image, or `null`) |
| `PUT` | `/api/v1/training/datasets/<dataset_id>/images/<image_id>/labels` | Replace the image's labels: `{"boxes": [...]}` for a detection dataset, `{"polygons": [{class_id, instance, points}]}` for a segmentation dataset (`instance` optional as on upload; each object's rings are normalized on save: rasterised and re-extracted, holes dropped), `{"image_class": <index>}` (`null` clears it) for a classification dataset; the other kind is refused |
| `POST` | `/api/v1/training/datasets/<dataset_id>/images/<image_id>/replicate` | Replicate a labeled image (image + labels) N times to grow the dataset — body `{"count": N}`, 1–50, default 1 |
| `GET` | `/api/v1/training/datasets/<dataset_id>/classes` | List dataset classes |
| `POST` | `/api/v1/training/datasets/<dataset_id>/classes` | Add one class (`{"name": ...}`); returns the full class list |
| `PUT` | `/api/v1/training/datasets/<dataset_id>/classes/<index>` | Rename the class at `index` |
| `DELETE` | `/api/v1/training/datasets/<dataset_id>/classes/<index>` | Remove the class at `index` |
| `GET` | `/api/v1/training/sam` | SAM3 worker status: `{"available", "loaded", "message"}` (`available` is `false`, with the reason in `message`, when the package or checkpoint is missing) |
| `POST` | `/api/v1/training/sam/load` | Load the SAM3 segmentation worker |
| `POST` | `/api/v1/training/sam/unload` | Unload the SAM3 worker (free GPU memory) |
| `POST` | `/api/v1/training/sam/segment` | SAM3-assisted segmentation for a text or point prompt — body `{"dataset_id", "image_id", "text_prompt", "points": [{"x", "y", "positive"}], "threshold"}` (`dataset_id` and `image_id` required; `positive` defaults to `true`). Returns `boxes` (normalized center/size), `scores` and, parallel to them, `polygons` — each object's mask as normalized rings (`[]` for an object without one) |
| `GET` | `/api/v1/training/label-model` | Model-assisted labeling status (`loaded`, `model_name`, `class_names`, `task`): an existing engine on the inference-service's private TensorRT worker |
| `POST` | `/api/v1/training/label-model/load` | Load an existing engine (`{"model_name": "X.engine"}`, any `.engine`/`.plan` of `GET /api/v1/models`) as the labeling assistant, on the same TensorRT runtime and `TILING_MODE` preprocessing as live detection |
| `POST` | `/api/v1/training/label-model/unload` | Unload the labeling engine (terminates its private worker) |
| `POST` | `/api/v1/training/label-model/detect` | Run the loaded engine on a dataset image (`{"dataset_id", "image_id", "threshold"}`); returns `boxes` (normalized center/size, `class_id` = the engine's index), `scores` the parallel `class_names` the dataset resolves them by and, for a segmentation engine, the parallel `polygons` (each object's mask as normalized rings). Every response also carries `image_class` and `candidates`: a classification engine fills `image_class` (`{class_id, class_name, score}` above the threshold) and the top-k `candidates`, with `boxes`, `scores`, `class_names` and `polygons` empty; other engines answer `image_class: null` and `candidates: []`. `409` when the engine's task is not the dataset's |
| `POST` | `/api/v1/training/train` | Start a training job. The dataset's task picks the stock weights (YOLO26s; YOLO26s-seg for segmentation, on the same tile crops; YOLO26s-cls at `TRAIN_IMG_SIZE_CLS` for classification). `base_model` (`"X.engine"`, a `GET /api/v1/models` entry with `has_weights`) fine-tunes from that model's training checkpoint (its last `best.pt`) instead; it must be a model of the dataset's task. Federated round: `{"federated": true, "initial_weights_id": ...}` trains from a stashed checkpoint and retains the resulting `last.pt` (`model_name` optional); exclusive with `base_model`, refused when the checkpoint was stashed for another task |
| `GET` | `/api/v1/training/train/status` | Current job status / progress. `geometry` is the effective training geometry once the split is built (`frames`, `tiles:auto` or `tiles:<px>`, see `TRAIN_TILE`); federated jobs expose `result_weights_id` when done |
| `POST` | `/api/v1/training/train/cancel` | Cancel the running job |
| `POST` | `/api/v1/training/train/finish` | Finish: hand `best.pt` to the model-upload route (pt→onnx→engine) |
| `POST` | `/api/v1/training/weights` | Stash a checkpoint (multipart `file`, optional `task` = the task it was trained for, so a job started from it refuses a dataset of another task) for federated training; answers `201` with `{"weights_id", "size"}` |
| `GET` | `/api/v1/training/weights/<weights_id>` | Download a stashed checkpoint (`.pt` blob) |
| `DELETE` | `/api/v1/training/weights/<weights_id>` | Delete a stashed checkpoint (TTL prune is the backstop) |
| `POST` | `/api/v1/training/weights/average` | FedAvg: average ≥2 stashed checkpoints (`{"weights_ids": [...]}`) into a new one (CPU child process) |

## Enrollment (device pairing)

Served under `/enroll/*` and used by the [hub](services/hub-vision.md) to pair
a device. Before pairing, nginx serves these routes over a self-signed cert;
once enrolled it flips to mTLS-enforcing mode automatically. Pairing needs no
token by default (first hub on the trusted LAN wins); set `DEVICE_PAIR_TOKEN`
to require a shared secret.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/enroll/info` | Public pairing info: `device_id`, `logical_name`, `enrolled`, `token_required`, `key_fingerprint` |
| `POST` | `/enroll/csr` | Return a CSR for the hub to sign (authorized per the pairing policy) |
| `POST` | `/enroll/complete` | Install the hub-signed `device_cert` + `ca_cert`; nginx reloads into mTLS-enforcing mode |
| `POST` | `/enroll/reset` | Unpair: clear the cert + CA and return to enrollment mode. Requires the owning hub (mTLS) or the pairing token |

## Audit trail (hub-only)

The device keeps its own record of what users did to it, in a SQLite ring
buffer the hub drains and then clears. Delivery is at-least-once: the hub
persists a page before acknowledging it, so an unacknowledged page is simply
re-delivered — duplicates are tolerated, losing a record of what someone did is
not. See [Audit trail](services/hub-vision.md#audit-trail).

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/audit/backlog` | One page of recorded actions, oldest first (`?limit=N`, default 25, max 200). Each record carries `id`, `captured_at`, `username`, `role`, `event`, `detail`, `source_ip` and `outcome`; the envelope adds `device_now` (the device clock, so the hub can offset-correct timestamps) and `pending` |
| `POST` | `/api/v1/audit/backlog/ack` | Delete recorded actions the hub confirmed persisting — body `{"ids": [..]}`, idempotent, returns `{"success": true, "deleted": N}` |

`event` is a stable machine-readable key (`detection.start`, `dataset.deleted`),
never a phrase — the hub composes the sentence in the operator's language. A
mutating route with no key of its own is still recorded, under `device.request`
with its method and path, so a new endpoint appears in the trail as soon as it
exists. `detail` names the target and nothing else: request bodies, uploaded
files and Wi-Fi pre-shared keys never reach the buffer.

The device authenticates nobody, so `username`/`role` are whatever the hub
stamped on the request (`X-Conecsa-User`, `X-Conecsa-Role`) and are kept only
when the mTLS terminator verified the caller. `source_ip` comes from
`X-Conecsa-Origin-Ip` or `X-Forwarded-For`, both read only when nginx relayed
the request; a caller reaching the gateway directly is recorded by its actual
peer address.

## Short `/api/*` aliases

Short paths for the most common routes; each relays to its `/api/v1/*`
counterpart above.

| Method | Endpoint | Alias of |
|---|---|---|
| `GET` | `/api/status` | `/api/v1/status` |
| `POST` | `/api/start` | `/api/v1/start` |
| `POST` | `/api/stop` | `/api/v1/stop` |
| `POST` | `/api/threshold` | `/api/v1/threshold` |
| `POST` | `/api/overlay_threshold` | `/api/v1/overlay_threshold` |
| `GET` | `/api/models` | `/api/v1/models` (returns the bare model list) |
| `GET` | `/api/health` | `/api/v1/health` |
| `GET` | `/api/ready` | `/api/v1/ready` |
| `GET` | `/api/classes` | `/api/v1/classes` |
| `POST` | `/api/classes` | `/api/v1/classes` |
