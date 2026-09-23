# Face recognition

Face recognition is the device application `face`: the device finds every face
in the frame and gives it the name of the enrolled person it matches, or
`unknown`. It follows the same path as the other application types —
application selector → dataset → labels → build on the device → model list →
activate → live results, records, SSE and Flow — and its results have the
shape of a detection, so the offline buffer, the hub records, the overlay and
the Node-RED nodes work unchanged.

!!! danger "No liveness detection"
    Nothing checks that the camera is looking at a living person: a printed
    photo, a phone screen or a mask can be recognized as the person it shows.
    Never let a recognized name be the only thing that opens a door, releases a
    machine or authorizes a transaction — pair it with a badge, a PIN, a
    second factor or an operator.

!!! warning "Face signatures are biometric data (LGPD / GDPR)"
    Enrolment photos and the numeric signatures built from them are personal
    biometric data. Processing them is the **operator's** responsibility:
    collect informed consent, tell people what is stored and for how long, and
    keep a way to remove a person. The product's part is to keep the data
    local: the gallery is built on the device, is **never downloadable, never
    federated and never sent to the hub**, and a deleted model takes its
    signatures with it. The enrolment dataset stays too: a face dataset
    cannot be exported, downloaded or transferred to another device (`409`),
    and Hub Vision fetches none of its photos — the dataset is managed on the
    device screen.
    Records and snapshots carry the recognized *name*, not the signature.

## How it runs

| Stage | What happens |
|---|---|
| Detection | The bundled **YuNet** detector runs on the whole frame (no tiling) as the device's main engine and returns each face's box, score and 5 landmarks |
| Gate | Faces below `CONFIDENCE_THRESHOLD` (the face score), smaller than `FACE_MIN_SIZE_PX` or suppressed by the `OVERLAY_THRESHOLD` NMS are dropped; at most `FACE_MAX_FACES` faces survive, largest first |
| Alignment | The landmarks align each face to 112×112 |
| Recognition | The bundled **SFace** embedder — a second engine on a private TensorRT worker (`TENSORRT_FACE_WORKER_PORT`) — turns each aligned face into a 128-dimension signature, compared with the model's gallery by cosine similarity |
| Result | Above `FACE_MATCH_THRESHOLD` the face takes the person's name, otherwise `unknown`; detection areas filter on the face's centre as they do for detection |

The snapshot item is the detection one:
`{"class_name": "<person>" | "unknown", "color", "confidence", "area", "bbox"}`,
where `confidence` is the **cosine similarity** to the matched person rather
than a detection score. `total` is the number of faces in the frame; the
detection counter grows by one on each **arrival of a known person** — an
`unknown` face never counts.

## Enrolling people and building the gallery

A dataset created on a face device is a dataset of task `face`: **one class
per person**, one image per photo, from camera capture or upload, labeled with
one image class per photo (as a classification dataset is). The photos are
kept in the dataset, so people can be added or removed and the model rebuilt;
rebuilding the model the device is running under the same name puts the new
gallery live as soon as the build finishes.
The name `unknown` (in any letter case) is reserved for a face nobody
matches and is refused as a person, on enrolment as on import.

"Train" on a face dataset **builds a gallery**; there is no YOLO training run,
and epochs, batch size, initial weights, a base model and federated training
do not apply (they are refused). The training-service packages the labeled
images as `<name>.faces` (a ZIP with a `manifest.json`, format 1) and uploads
it to `POST /api/v1/model` with `task=face`; the inference-service runs it as
a normal conversion job (the `CONVERTING_TO_ENGINE` status and the usual
events), detecting, aligning and embedding every photo. Photos with no usable
face are counted as skipped, and the job fails when no person ends up with a
signature. The `.faces` file is listed while it builds — never selectable,
the device screen shows it as *building* — and deleted when the build
finishes; one left behind by an interrupted build can simply be deleted.

| File | Content |
|---|---|
| `<name>.engine` | The face detector engine for this model |
| `<name>.txt` | The enrolled people's names, one per gallery index: the Class Names panel renames a person in place, and a list that adds, drops or duplicates a person, or names one `unknown`, is refused (`400`) |
| `<name>.settings.json` | `{"task": "face", …}` like any model's sidecar |
| `<name>.gallery.npz` | The signatures, their labels and the embedder's hash — never downloadable, never federated, never sent to the hub |

The shared detector/embedder engines are built once from the bundled ONNX
files and cached in `<models dir>/face/` under a name that carries the graph's
hash, so a second build reuses them and an upgraded graph is never served by
a stale engine.
Uploading a file that is not a `.faces` package while declaring `task=face`
is refused (`400`), and a gallery built with a different embedder is refused
at activation (the previous model stays).

## Settings

`GET /api/v1/status` reports `face_match_threshold`, `face_min_size_px` and
`face_max_faces` only while the device runs face recognition (as JSON keys
and as optional fields of the protobuf response alike), and
`POST /api/v1/face/settings` (role **user**) changes any of them as one
request: a field out of range leaves every other field untouched — see the
[HTTP API reference](api-reference.md#face-recognition). Their defaults come
from `FACE_MATCH_THRESHOLD`, `FACE_MIN_SIZE_PX` and `FACE_MAX_FACES` (see
[Configuration](configuration.md)). Raise the match threshold when one person
is mistaken for another; lower it when a known person often reads as
`unknown`.

## Access control with Node-RED

Detections do not drive GPIO by themselves — the device's GPIO routes are
explicit writes, not a rule engine. Access control is therefore a Flow, built
from the nodes the package already ships and available as
**Import → Examples → @conecsa/node-red-contrib-conecsa-system-vision →
face-access**. Because there is no liveness check, the example needs **two
factors**: a badge (or PIN) from a reader, and the badge holder's face.

```text
badge reader (inject, replace with yours) ─┐
                                           ├→ function (badge + authorized
conecsa-detection (every second) ──────────┘   face within WINDOW_S)
                                → trigger (pulse, e.g. 3 s) → conecsa-gpio (pin 29)
```

The function node maps each authorized person — the classes of the face
model — to their badge id, holds the similarity a face must reach, and
releases only when the badge and a matching face arrive within a few seconds
of each other; a face alone, a badge alone, an `unknown` face or a face that
is not the badge holder's opens nothing. The trigger node turns the release
into a pulse, so the pin returns LOW on its own. See [Flow](services/flow.md).

## Bundled third-party models

Both models come from the [OpenCV Model Zoo](https://github.com/opencv/opencv_zoo)
and are fetched into `inference-service/assets/face/` by
`scripts/fetch-face-models.sh` against the checksums in
`scripts/face-models.pin` (the image build verifies them; `FACE_MODELS_DIR`
points at the directory). A build without these assets does not register the
`face` strategy, and the device then does not list `face` in
`supported_tasks`.

| Model | File | License | Source |
|---|---|---|---|
| YuNet (face detection) | `face_detection_yunet_2023mar.onnx` | MIT © Shiqi Yu et al. | <https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet> |
| SFace (face recognition) | `face_recognition_sface_2021dec.onnx` | Apache-2.0 | <https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface> |

The full notices ship with the image in `inference-service/assets/face/NOTICE.md`.
Conecsa reviewed both models' licenses and training-data provenance and cleared
them for commercial use as bundled (2026-09-22); the biometric-data duties and
the missing liveness check above are unaffected by that review.
