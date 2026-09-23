# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
gRPC server for the headless inference-service.

Exposes the `DetectionControl`, `ModelControl` and `ManagementControl` services
(proto/inference.proto) backed by the in-process services. It is the service's
only control surface: the api-gateway talks to it over gRPC while the processed
frames cross via shared memory.

Started by :func:`serve_grpc` from `main.py`.
"""
import contextlib
import json
import logging
import os
import sys
import time
import uuid
from concurrent import futures
from typing import List, Optional

import grpc
from conecsa_common.atomic import fsync_dir
from conecsa_common.tasks import (
    FACE,
    SEGMENT,
    is_reserved_face_name,
    is_safe_class_name,
    person_key,
)

from .model_paths import validate_model_filename
from .services.errors import InvalidTask, PreconditionFailed

# Generated *_pb2 / *_pb2_grpc modules do a flat `import inference_pb2`, so their
# directory must be importable. In the image the stubs are generated next to
# this file (api/proto, see Dockerfile.inference-service). For local dev they
# come from scripts/compile-proto.sh, which writes them to api-gateway/gateway/
# proto — fall back to that when the co-located dir is absent.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROTO_DIR = os.path.join(_HERE, "proto")
if not os.path.isdir(_PROTO_DIR):
    _repo_root = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
    _PROTO_DIR = os.path.join(_repo_root, "api-gateway", "gateway", "proto")
if _PROTO_DIR not in sys.path:
    sys.path.insert(0, _PROTO_DIR)

import inference_pb2 as pb  # noqa: E402
import inference_pb2_grpc as pb_grpc  # noqa: E402

logger = logging.getLogger(__name__)

LISTEN_ADDR = os.environ.get("INFERENCE_GRPC_LISTEN", "0.0.0.0:50061")
_EVENT_KEEPALIVE_S = 15.0
_STATS_INTERVAL_S = 0.25


def _unimplemented(context, what: str):
    """Mark the current RPC ``UNIMPLEMENTED`` with a placeholder message."""
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details(f"{what} not implemented yet (pending gateway cutover)")


def _refuse(context, code, message: str) -> None:
    """Fail the current RPC with ``code`` (the caller returns an empty reply).

    ``set_code``/``set_details`` rather than ``abort``, like the
    training-service: the client still raises ``RpcError`` with this code, and
    the gateway maps FAILED_PRECONDITION to 409 and INVALID_ARGUMENT to 400.
    Client-caused refusals are logged here with their reason; INTERNAL
    callers log the underlying failure themselves.
    """
    if code != grpc.StatusCode.INTERNAL:
        logger.warning("refused (%s): %s", code.name, message)
    if context is not None:
        context.set_code(code)
        context.set_details(message)


def _application_pb(info: dict) -> "pb.ApplicationInfo":
    """ApplicationService.info() as the wire message (empty task = unset)."""
    return pb.ApplicationInfo(
        task=str(info.get("task") or ""),
        supported_tasks=[str(t) for t in info.get("supported_tasks") or []],
        migrated=bool(info.get("migrated", False)),
    )


class DetectionControlServicer(pb_grpc.DetectionControlServicer):
    """`DetectionControl` RPCs — detection lifecycle, tuning, trigger/counter
    and the event/stats telemetry streams. Thin adapters over the in-process
    detection/stats/event services."""

    def __init__(self, application):
        self._app = application

    @property
    def _det(self):
        """The shared DetectionService."""
        return self._app.detection_service

    @property
    def _stats(self):
        """The shared StatsService."""
        return self._app.stats_service

    @property
    def _events(self):
        """The shared EventService."""
        return self._app.event_service

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def Start(self, request, context):
        """RPC: start the detection loop; this also ends a GPU handover."""
        try:
            if self._app.model_service.start_detection():
                self._events.publish(
                    "runtime_changed", keys=["status"], data={"runtime_released": False}
                )
            return pb.Result(success=True, message="Detection started")
        except Exception as exc:  # noqa: BLE001
            return pb.Result(success=False, message=str(exc))

    def Stop(self, request, context):
        """RPC: stop the detection loop."""
        try:
            self._det.stop()
            return pb.Result(success=True, message="Detection stopped")
        except Exception as exc:  # noqa: BLE001
            return pb.Result(success=False, message=str(exc))

    def GetStatus(self, request, context):
        """RPC: return run state, active model, thresholds, runtime and stats."""
        det = self._det
        cfg = det.config
        s = self._stats.get_stats()
        # An application change deselects a model of another task: report no
        # model then, not the default path the config falls back to.
        models = getattr(self._app, "model_service", None)
        selected = models is None or bool(models.current_model)
        # Presence matters (optional field): set means "this build knows
        # application types", and an empty value means none is chosen.
        app = getattr(self._app, "application_service", None)
        extra: dict = {"task": app.task or ""} if app is not None else {}
        # The per-task settings are present only for the task the device
        # runs: a client reads presence as "applies here".
        task = app.task if app is not None else None
        if task == SEGMENT:
            from .postprocess.segment import segment_settings_from_env
            extra["segment_max_instances"] = int(
                cfg.SEGMENT_MAX_INSTANCES or segment_settings_from_env().max_masks)
        elif task == FACE:
            extra["face_match_threshold"] = float(cfg.FACE_MATCH_THRESHOLD)
            extra["face_min_size_px"] = int(cfg.FACE_MIN_SIZE_PX)
            extra["face_max_faces"] = int(cfg.FACE_MAX_FACES)
        return pb.StatusResponse(
            **extra,
            is_running=bool(det.is_running),
            model=os.path.basename(cfg.MODEL_PATH or "") if selected else "",
            confidence_threshold=float(cfg.CONFIDENCE_THRESHOLD),
            overlay_threshold=float(cfg.OVERLAY_THRESHOLD),
            acceleration_type=str(det.acceleration_type()),
            runtime_type=str(det.runtime_api()),
            trigger_enabled=bool(det.get_trigger_status()),
            detection_count=int(det.get_detection_count()),
            stats=_stats_pb(s),
            camera_connected=bool(self._app.video_service.camera_connected()),
        )

    # ── tuning ────────────────────────────────────────────────────────────────

    def SetThreshold(self, request, context):
        """RPC: set the confidence threshold (0–1)."""
        ok = self._det.set_confidence_threshold(request.threshold)
        return pb.Result(success=bool(ok),
                         message="Threshold updated" if ok else "Threshold must be between 0 and 1")

    def _update_model_setting(self, context, name, set_value, what, invalid):
        """Change and save a per-model setting under the model lifecycle lock.

        A refused value is an unsuccessful Result (the gateway's 400); a value
        that could not be saved aborts with INTERNAL, so no success is
        reported or audited for a change that would be lost on restart.
        """
        models = self._app.model_service
        outcome = models.update_setting(name, set_value)
        if outcome == models.SETTING_UNSAVED:
            _refuse(context, grpc.StatusCode.INTERNAL, f"{what} could not be saved")
            return pb.Result(success=False, message=f"{what} could not be saved")
        ok = outcome == models.SETTING_SAVED
        return pb.Result(success=ok, message=f"{what} updated" if ok else invalid)

    def SetOverlayThreshold(self, request, context):
        """RPC: set the IoU/NMS overlay threshold (0–1) and persist it."""
        cfg = self._det.config
        return self._update_model_setting(
            context, "OVERLAY_THRESHOLD",
            lambda: cfg.set_overlay_threshold(request.threshold),
            "Overlay threshold", "Threshold must be between 0 and 1")

    def SetSegmentMaxInstances(self, request, context):
        """RPC: set the segmentation instance limit (1–255) and persist it."""
        cfg = self._det.config
        return self._update_model_setting(
            context, "SEGMENT_MAX_INSTANCES",
            lambda: cfg.set_segment_max_instances(int(request.max_instances)),
            "Instance limit", "The instance limit must be between 1 and 255")

    def SetFaceSettings(self, request, context):
        """RPC: set the face recognition settings present in the request and persist them.

        One transaction under the model lifecycle lock: a refused field leaves
        every other field as it was, and the values are saved in one write.
        """
        cfg = self._det.config
        changes = []
        invalid = {}
        if request.HasField("match_threshold"):
            changes.append(("FACE_MATCH_THRESHOLD",
                            lambda: cfg.set_face_match_threshold(float(request.match_threshold))))
            invalid["FACE_MATCH_THRESHOLD"] = "The match threshold must be between 0 and 1"
        if request.HasField("min_size_px"):
            changes.append(("FACE_MIN_SIZE_PX",
                            lambda: cfg.set_face_min_size(int(request.min_size_px))))
            invalid["FACE_MIN_SIZE_PX"] = "The minimum face size must be between 0 and 1024 pixels"
        if request.HasField("max_faces"):
            changes.append(("FACE_MAX_FACES",
                            lambda: cfg.set_face_max_faces(int(request.max_faces))))
            invalid["FACE_MAX_FACES"] = "The faces per frame must be between 1 and 20"
        if not changes:
            return pb.Result(success=False, message="No face setting given")
        models = self._app.model_service
        outcome, refused = models.update_settings(changes)
        if outcome == models.SETTING_UNSAVED:
            _refuse(context, grpc.StatusCode.INTERNAL, "Face settings could not be saved")
            return pb.Result(success=False, message="Face settings could not be saved")
        if outcome == models.SETTING_INVALID:
            return pb.Result(success=False, message=invalid[refused or ""])
        return pb.Result(success=True, message="Face settings updated")

    # ── trigger / counter / stats ───────────────────────────────────────────────

    def EnableTrigger(self, request, context):
        """RPC: enable frame processing (trigger gate on)."""
        self._det.enable_trigger()
        return pb.Result(success=True, message="Trigger enabled")

    def DisableTrigger(self, request, context):
        """RPC: freeze the last processed frame (trigger gate off)."""
        self._det.disable_trigger()
        return pb.Result(success=True, message="Trigger disabled")

    def GetCounter(self, request, context):
        """RPC: return the accumulated detection count and trigger state."""
        return pb.CounterResponse(
            count=int(self._det.get_detection_count()),
            trigger_enabled=bool(self._det.get_trigger_status()),
        )

    def ResetCounter(self, request, context):
        """RPC: reset the detection counter."""
        self._det.reset_detection_count()
        return pb.Result(success=True, message="Counter reset")

    def ResetStats(self, request, context):
        """RPC: reset the performance statistics."""
        self._stats.reset()
        # A stats reset restarts the classification transition state.
        self._det.reset_task_state()
        return pb.Result(success=True, message="Stats reset")

    def Snapshot(self, request, context):
        """RPC: return the current detections (optionally with the frame) as JSON."""
        # The hub polls this ~1x/s: it doubles as the hub-is-online heartbeat
        # for the offline detection buffer. Only hub pulls count — the Flow
        # detection node (and any local script) polls the same snapshot, and
        # treating those as hub contact would keep the buffer disarmed forever.
        if request.hub_pull:
            self._app.detection_buffer.note_snapshot_pull()
        # Thin: DetectionService owns the snapshot logic; carry it as a JSON
        # blob the gateway relays verbatim (mirrors config/camera/system).
        snap = self._det.detections_snapshot(
            bool(request.include_frame), bool(request.include_raw_frame)
        )
        return pb.SnapshotResponse(json=json.dumps(snap))

    def ListBacklog(self, request, context):
        """RPC: one page of offline-buffered detection records, oldest first.

        Deliberately does NOT count as a hub pull: changes that happen during
        a long drain are buffered and picked up on the next cycle.
        """
        page = self._app.detection_buffer.list_backlog(int(request.limit))
        return pb.BacklogResponse(json=json.dumps(page))

    def AckBacklog(self, request, context):
        """RPC: delete buffered records the hub confirmed persisting (idempotent)."""
        n = self._app.detection_buffer.ack(list(request.ids))
        return pb.Result(success=True, message=f"{n} records acknowledged")

    # ── model activation ─────────────────────────────────────────────────────────

    def ReloadModel(self, request, context):
        """RPC (not implemented): reload the live model."""
        _unimplemented(context, "ReloadModel")
        return pb.Result(success=False, message="not implemented")

    def SetDetectionAreas(self, request, context):
        """RPC (not implemented): replace the detection areas."""
        _unimplemented(context, "SetDetectionAreas")
        return pb.Result(success=False, message="not implemented")

    def SetSettings(self, request, context):
        """RPC (not implemented): push model settings."""
        _unimplemented(context, "SetSettings")
        return pb.Result(success=False, message="not implemented")

    # ── telemetry streams ────────────────────────────────────────────────────────

    def StreamEvents(self, request, context):
        """RPC (server stream): emit a snapshot then each invalidation event.

        Keepalive-bounded: ``wait_for_changes`` returns periodically so the
        stream stays responsive to client cancellation.
        """
        version, snap = self._events.snapshot()
        yield _event_pb(snap)
        last = version
        while context.is_active():
            new_version, events, _sv, _st, changed = self._events.wait_for_changes(
                last, None, _EVENT_KEEPALIVE_S
            )
            if changed:
                for ev in events:
                    yield _event_pb(ev)
            last = new_version

    def StreamStats(self, request, context):
        """RPC (server stream): emit the live stats every ``_STATS_INTERVAL_S``."""
        while context.is_active():
            s = self._stats.get_stats()
            yield pb.StatsUpdate(version=0, stats=_stats_pb(s))
            time.sleep(_STATS_INTERVAL_S)


class ModelControlServicer(pb_grpc.ModelControlServicer):
    """Model lifecycle. Stays inference-side because SelectModel reloads the
    live detector (activate_model)."""

    def __init__(self, application):
        self._app = application

    @property
    def _models(self):
        """The shared ModelService."""
        return self._app.model_service

    @property
    def _conv(self):
        """The shared ConversionService."""
        return self._app.conversion_service

    def ListModels(self, request, context):
        """RPC: list available models (name, path, size, modified, active)."""
        out = []
        for m in self._models.list_models():
            out.append(pb.ModelInfo(
                name=m.name, path=getattr(m, "path", ""), size=int(getattr(m, "size", 0) or 0),
                modified=float(getattr(m, "modified", 0.0) or 0.0),
                is_active=bool(getattr(m, "is_active", False)),
                has_weights=bool(getattr(m, "has_weights", False)),
                task=str(getattr(m, "task", "detect") or "detect"),
            ))
        return pb.ModelList(models=out)

    def SelectModel(self, request, context):
        """RPC: activate a model by name (reloads the live detector).

        A model of another task than the device's application fails with
        FAILED_PRECONDITION naming both tasks; the running model is untouched.
        """
        try:
            success, result, _was_running = self._models.activate_model(request.name)
        except PreconditionFailed as exc:
            _refuse(context, grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            return pb.Result(success=False, message=str(exc))
        return pb.Result(success=bool(success), message=str(result))

    def DeleteModel(self, request, context):
        """RPC: delete a model by name."""
        success, message = self._models.delete_model(request.name)
        return pb.Result(success=bool(success), message=str(message))

    def ListConversions(self, request, context):
        """RPC: list active `.pt`→`.engine` conversion jobs."""
        return pb.ConversionList(jobs=[_conversion_pb(j) for j in self._conv.get_active_jobs()])

    def GetConversion(self, request, context):
        """RPC: return one conversion job by id (NOT_FOUND if unknown)."""
        job = self._conv.get_job(request.job_id)
        if job is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"No conversion job '{request.job_id}'")
            return pb.ConversionJob()
        return _conversion_pb(job)

    def UploadModel(self, request_iterator, context):
        """RPC (client stream): stream an uploaded model to disk and save/convert it.

        The first message carries the metadata, the rest are file chunks. The
        chunks are written straight to an exclusive staging file inside the
        model directory with a running byte cap — never accumulated in memory,
        which the 8 GB device cannot afford.
        ModelService owns the save/convert/activate decision; its result and the
        intended HTTP status are carried back as a JSON blob for the gateway.
        """
        def error(status: int, message: str) -> "pb.UploadResult":
            if status < 500:
                logger.warning("UploadModel refused (%d): %s", status, message)
            return pb.UploadResult(ok=False, http_status=int(status),
                                   json=json.dumps({"error": message}))

        model_dir = self._models.model_directory
        limit = int(self._app.config.MAX_MODEL_UPLOAD_BYTES)
        meta = None
        task: Optional[str] = None
        fd = None
        staged_path = None
        received = 0
        try:
            for msg in request_iterator:
                which = msg.WhichOneof("data")
                if which == "meta":
                    if fd is not None:
                        return error(400, "Duplicate upload metadata")
                    meta = msg.meta
                    if not meta.filename:
                        return error(400, "Missing upload metadata")
                    # Reject a bad name before any byte lands on disk.
                    _, invalid = validate_model_filename(meta.filename, model_dir)
                    if invalid:
                        return error(400, invalid)
                    # …and a task this device cannot record.
                    app = getattr(self._app, "application_service", None)
                    if app is not None:
                        try:
                            task = app.resolve_upload_task(meta.task)
                        except InvalidTask as exc:
                            return error(400, str(exc))
                        except PreconditionFailed as exc:
                            return error(409, str(exc))
                    else:
                        task = meta.task or None
                    staged_path = os.path.join(
                        model_dir, f".upload-{uuid.uuid4().hex}.part")
                    fd = os.open(staged_path,
                                 os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                elif which == "chunk":
                    if fd is None:
                        return error(400, "Upload chunks arrived before metadata")
                    received += len(msg.chunk)
                    if received > limit:
                        return error(
                            413,
                            f"Model upload exceeds the {limit} byte limit")
                    os.write(fd, msg.chunk)
            if meta is None or staged_path is None:
                return error(400, "Missing upload metadata")
            if fd is not None:
                os.fsync(fd)
                os.close(fd)
                fd = None
            # No size given: a .pt is exported at the size it was trained at,
            # else at the task's own default (224 for classification).
            from .postprocess import default_imgsz
            imgsz = int(meta.imgsz) or default_imgsz(task)
            body, status = self._models.process_upload(
                meta.filename, _StagedFile(staged_path), imgsz,
                train_geometry=meta.train_geometry or None, task=task,
                imgsz_from_checkpoint=not meta.imgsz)
            return pb.UploadResult(ok=(200 <= status < 300),
                                   http_status=int(status),
                                   json=json.dumps(body))
        finally:
            # Cancellation, over-cap, or a failed save: never leave a
            # half-written .part behind (a successful save renamed it away).
            if fd is not None:
                os.close(fd)
            if staged_path is not None and os.path.exists(staged_path):
                try:
                    os.unlink(staged_path)
                except OSError:
                    logger.warning("could not remove staged upload %s", staged_path)

    def DownloadModel(self, request, context):
        """RPC (server stream): stream a model file's bytes in ~1 MiB chunks."""
        path = self._models.model_file_path(request.name)
        if not path:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Model '{request.name}' not found")
            return
        yield from self._stream_file(path, request.name, context)

    def DownloadModelWeights(self, request, context):
        """RPC (server stream): the model's training-checkpoint sidecar."""
        path = self._models.weights_file_path(request.name)
        if not path:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Model '{request.name}' has no training checkpoint")
            return
        yield from self._stream_file(path, request.name, context)

    # ── model-assisted labeling ───────────────────────────────────────────────

    def GetLabelModelStatus(self, request, context):
        """RPC: state of the labeling engine on the private worker."""
        s = self._app.labeling_service.status()
        return pb.LabelModelStatus(
            loaded=bool(s["loaded"]), model_name=s["model_name"],
            class_names=list(s["class_names"]), message=s["message"],
            task=s.get("task") or "",
        )

    def LoadLabelModel(self, request, context):
        """RPC: load a listed engine as the labeling assistant."""
        try:
            self._app.labeling_service.load(request.name)
            return pb.Result(success=True, message=f"Model '{request.name}' loaded for labeling")
        except Exception as exc:  # noqa: BLE001
            return pb.Result(success=False, message=str(exc))

    def UnloadLabelModel(self, request, context):
        """RPC: drop the labeling engine and its worker."""
        try:
            self._app.labeling_service.unload()
            return pb.Result(success=True, message="Labeling model unloaded")
        except Exception as exc:  # noqa: BLE001
            return pb.Result(success=False, message=str(exc))

    def LabelDetect(self, request, context):
        """RPC: suggestions of the labeling engine on one encoded image.

        A detection engine yields boxes (with their rings for a segmentation
        engine); a classification engine the image's class (when above the
        threshold) and its top-k candidates.
        """
        try:
            found = self._app.labeling_service.detect(
                bytes(request.jpeg), float(request.threshold))
            result = pb.LabelDetectResult(
                success=True,
                detections=[
                    pb.LabelDetection(
                        class_id=int(d["class_id"]), class_name=d["class_name"],
                        score=d["score"], x1=d["x1"], y1=d["y1"], x2=d["x2"], y2=d["y2"],
                        rings=[pb.Ring(points=[v for point in ring for v in point])
                               for ring in d.get("rings", ())],
                    )
                    for d in found["detections"]
                ],
                candidates=[_label_class_pb(c) for c in found["candidates"]],
            )
            if found["image_class"] is not None:
                result.image_class.CopyFrom(_label_class_pb(found["image_class"]))
            return result
        except Exception as exc:  # noqa: BLE001
            return pb.LabelDetectResult(success=False, message=str(exc))

    @staticmethod
    def _stream_file(path: str, name: str, context):
        """Yield a file as ~1 MiB ``ModelFileChunk`` messages."""
        try:
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(1 << 20)
                    if not chunk:
                        break
                    yield pb.ModelFileChunk(chunk=chunk)
        except OSError as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to read model '{name}': {exc}")
            return

def _parse_json_or_abort(raw: str, context) -> dict:
    """Decode a ``ConfigJson`` body; a malformed one is INVALID_ARGUMENT."""
    data: object = None
    try:
        data = json.loads(raw or "{}")
    except ValueError as exc:
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"Invalid JSON: {exc}")
    if not isinstance(data, dict):
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Body must be a JSON object")
        raise AssertionError("abort() returns")  # pragma: no cover - abort raises
    return data


def _abort_for_status(context, status: int, message: str) -> None:
    """Map a service ``(ok, message, status)`` failure to a gRPC status.

    400 → INVALID_ARGUMENT (the gateway answers 400), 503 → UNAVAILABLE (503),
    anything else → INTERNAL (502).
    """
    code = {
        400: grpc.StatusCode.INVALID_ARGUMENT,
        503: grpc.StatusCode.UNAVAILABLE,
    }.get(status, grpc.StatusCode.INTERNAL)
    context.abort(code, message)


class ManagementControlServicer(pb_grpc.ManagementControlServicer):
    """Classes + detection-areas: per-model state consumed by the live detector,
    so it stays inference-side. Area ops return the full state JSON (mirroring
    the REST controller's state response) for the gateway to relay verbatim."""

    def __init__(self, application):
        self._app = application

    # ── classes (current model's sibling .txt) ────────────────────────────────
    def _repo(self):
        """Build a ClassLabelsRepository for the current model's labels file."""
        from api.repositories import ClassLabelsRepository
        return ClassLabelsRepository(self._app.config.CLASSES_FILE_PATH)

    def GetClasses(self, request, context):
        """RPC: list the current model's class labels."""
        return pb.ClassList(classes=list(self._repo().load_labels()))

    def _apply_labels_live(self) -> None:
        """Hand the saved labels to the running strategy (best-effort).

        A rename must show in the overlay, the snapshot and the records at
        once; failing to apply it is not worth failing the save, which the
        next model load would pick up anyway.
        """
        try:
            self._app.detection_service.refresh_class_labels()
        except Exception as exc:  # noqa: BLE001 - the labels are already saved
            logger.warning("Saved classes could not be applied to the live model: %s", exc)

    def _face_gallery(self):
        """``(gallery, path)`` of the active face model, or ``None`` for any other."""
        model_path = getattr(self._app.config, "MODEL_PATH", "") or ""
        if not model_path:
            return None
        from api.services.model_service import ModelService
        from api.services.model_settings_service import ModelSettingsService
        if ModelSettingsService.task_of(ModelService.settings_file_for_model(model_path)) != FACE:
            return None
        from api.postprocess import _face_gallery
        from api.postprocess.face import gallery_file_for_model
        path = gallery_file_for_model(model_path)
        try:
            return _face_gallery.load(path), path
        except _face_gallery.GalleryError:
            return None  # nothing to keep in step with; activation refuses such a model

    def _stamp_face_gallery(self, labels: List[str]) -> bool:
        """Record ``labels`` as the names the active face gallery goes with.

        The strategy refuses a gallery whose stamp does not match the sidecar
        (a torn build); a rename or a clear must therefore move the stamp
        along, or the next activation would refuse the model. False when the
        gallery could not be rewritten (nothing else may change then).
        """
        found = self._face_gallery()
        if found is None:
            return True
        from api.postprocess import _face_gallery
        gallery, path = found
        gallery.labels_sha256 = _face_gallery.labels_stamp(labels)
        try:
            _face_gallery.save(path, gallery)
        except OSError as exc:
            logger.error("Could not restamp the face gallery %s: %s", path, exc)
            return False
        return True

    def _save_labels(self, labels: List[str], context) -> "pb.Result":
        """Persist ``labels`` for the current model and apply them live.

        Under the model lifecycle lock, so a concurrent select cannot swap the
        model between the validation, the two writes and the live apply. For
        a face model the gallery is restamped first and the sidecar second: a
        failed restamp changes nothing, and a failed sidecar write moves the
        stamp back, so the two never disagree.
        """
        models = getattr(self._app, "model_service", None)
        lock = models.op_lock if models is not None else contextlib.nullcontext()
        with lock:
            # Clearing reverts to the enrolled names (the gallery pads them).
            error = self._face_class_list_error(labels) if labels else None
            if error:
                _refuse(context, grpc.StatusCode.INVALID_ARGUMENT, error)
                return pb.Result(success=False, message=error)
            repo = self._repo()
            previous = repo.load_labels()
            if not self._stamp_face_gallery(labels):
                _refuse(context, grpc.StatusCode.INTERNAL, "Failed to save classes")
                return pb.Result(success=False, message="Failed to save classes")
            if not repo.save_labels(labels):
                self._stamp_face_gallery(previous)
                return pb.Result(success=False, message="Failed to save classes")
            self._apply_labels_live()
        return pb.Result(success=True,
                         message="Classes updated" if labels else "Classes cleared")

    def _face_class_list_error(self, labels: List[str]) -> Optional[str]:
        """Why ``labels`` cannot be a face model's people, or ``None``.

        The gallery keeps the indices and the sidecar the names, so the list
        must name exactly the enrolled people in the gallery's order: a
        shorter, longer or reordered list would show one person's face under
        another's name — and an access-control flow would open for it.
        ``unknown`` is what a face nobody matches is called.
        """
        found = self._face_gallery()
        if found is None:
            return None
        from api.utils import parse_label_and_color
        people = found[0].names
        names = [(parse_label_and_color(label)[0] or "").strip() for label in labels]
        if len(names) != len(people):
            return (f"A face model lists exactly its {len(people)} enrolled people, in the "
                    f"gallery's order; {len(names)} names were given")
        if any(not person_key(label) for label in labels):
            return "A person's name must not be empty or a colour alone"
        if any(not is_safe_class_name(name) for name in names):
            return ("A person's name is too long or has characters that are not letters, "
                    "digits, space, '_', '-' or '.'")
        if any(is_reserved_face_name(name) for name in names):
            return "'unknown' is reserved for a face nobody matches, not a person"
        keys = [person_key(name) for name in names]
        if len(set(keys)) != len(keys):
            return "Two people cannot share a name"
        # A name may only stay in its slot or be new: moving one to another
        # index (a swap, a rotation) would give that person another's face.
        current = self._repo().load_labels()
        current = current + people[len(current):]
        slots = {person_key(entry): i for i, entry in enumerate(current)}
        for i, key in enumerate(keys):
            if key in slots and slots[key] != i:
                return (f"'{names[i]}' is person {slots[key] + 1} of this model and cannot "
                        f"move to slot {i + 1}: rename people in place")
        return None

    def SetClasses(self, request, context):
        """RPC: replace the current model's class labels, applied live.

        A face model's list is checked against its gallery first
        (INVALID_ARGUMENT, the gateway's 400).
        """
        return self._save_labels(list(request.classes), context)

    def ClearClasses(self, request, context):
        """RPC: clear custom labels (revert to the model default), applied live."""
        return self._save_labels([], context)

    # ── config (thin: ConfigService owns the logic) ──────────────────────────
    def GetConfig(self, request, context):
        """RPC: return the current configuration as JSON."""
        return pb.ConfigJson(json=json.dumps(self._app.config_service.get_config()))

    def UpdateConfig(self, request, context):
        """RPC: apply a configuration update from a JSON body.

        Failures are gRPC statuses (INVALID_ARGUMENT / UNAVAILABLE / INTERNAL)
        so the gateway's shared error mapper answers 400 / 503 / 502.
        """
        data = _parse_json_or_abort(request.json, context)
        ok, message, status = self._app.config_service.update_config(data)
        if not ok:
            _abort_for_status(context, status, message)
        return pb.Result(success=True, message=message)

    # ── camera (thin: VideoService owns the logic) ───────────────────────────
    def GetCamera(self, request, context):
        """RPC: list V4L2 devices + current camera config as JSON."""
        return pb.ConfigJson(json=json.dumps(self._app.video_service.list_camera_devices()))

    def UpdateCamera(self, request, context):
        """RPC: apply a camera update (via SHM) and persist it on success."""
        data = _parse_json_or_abort(request.json, context)
        ok, message, status = self._app.video_service.apply_camera_update(data)
        if not ok:
            _abort_for_status(context, status, message)
        ms = getattr(self._app, "model_settings_service", None)
        # A capture-source change is device-level and already persisted by the
        # video service; only model-scoped fields touch the model's settings.
        if ms is not None and not self._app.video_service.is_source_only_update(data):
            try:
                ms.save()
            except Exception as exc:  # noqa: BLE001
                logger.warning("persist camera config failed: %s", exc)
        return pb.Result(success=True, message=message)

    # System metrics are served by the `os-base` hardware agent
    # (HardwareService.GetSystemStatus), not here.

    # ── detection areas ────────────────────────────────────────────────────────
    @property
    def _areas(self):
        """The shared DetectionAreaService."""
        return self._app.detection_area_service

    def _state(self, ok: bool = True) -> "pb.AreaResult":
        """Wrap the full areas state as an ``AreaResult`` for the gateway."""
        state = {"areas": [a.to_dict() for a in self._areas.list()]}
        return pb.AreaResult(ok=ok, state_json=json.dumps(state))

    def ListAreas(self, request, context):
        """RPC: return all detection areas."""
        return self._state(True)

    def CreateArea(self, request, context):
        """RPC: create a new (centered, editing) detection area."""
        self._areas.add()
        return self._state(True)

    def DeleteArea(self, request, context):
        """RPC: delete a detection area by id."""
        return self._state(self._areas.delete(request.area_id))

    def SaveArea(self, request, context):
        """RPC: commit an area (leave editing mode; filter stays active)."""
        return self._state(self._areas.save(request.area_id) is not None)

    def EditArea(self, request, context):
        """RPC: promote a saved area back to editing mode."""
        return self._state(self._areas.edit(request.area_id) is not None)

    def DiscardArea(self, request, context):
        """RPC: discard the editing area."""
        return self._state(self._areas.discard(request.area_id))

    def SetAreaShape(self, request, context):
        """RPC: set an area's shape (rectangle/circle); no-op on invalid shape."""
        from api.services.detection_area_service import VALID_SHAPES
        if request.shape not in VALID_SHAPES:
            return pb.AreaResult(ok=False, state_json=self._state(True).state_json)
        return self._state(self._areas.set_shape(request.area_id, request.shape) is not None)

    def AreaCommand(self, request, context):
        """RPC: apply a move/resize command to an area; no-op on invalid action."""
        from api.services.detection_area_service import VALID_ACTIONS
        if request.action not in VALID_ACTIONS:
            return pb.AreaResult(ok=False, state_json=self._state(True).state_json)
        return self._state(self._areas.apply_command(request.area_id, request.action) is not None)

    # ── GPU handover (training-service) ───────────────────────────────────────

    def ReleaseRuntime(self, request, context):
        """RPC: release the GPU runtime so the training-service can use it.

        Refuses while a model conversion is in progress; otherwise stops
        detection, releases the TensorRT workers, and publishes a status event.
        Until the handover ends (ResumeRuntime or an explicit Start) the
        application type cannot change.
        """
        try:
            ok, message = self._app.model_service.release_runtime()
            if ok:
                self._app.event_service.publish(
                    "runtime_changed", keys=["status"], data={"runtime_released": True}
                )
            return pb.Result(success=ok, message=message)
        except Exception as exc:  # noqa: BLE001
            return pb.Result(success=False, message=str(exc))

    def ResumeRuntime(self, request, context):
        """RPC: end the GPU handover; re-initialize and restart detection unless
        ``keep_detection_stopped`` (the post-training conversion handover)."""
        try:
            ok, message = self._app.model_service.resume_runtime(
                restart_detection=not request.keep_detection_stopped)
            self._app.event_service.publish(
                "runtime_changed", keys=["status"], data={"runtime_released": False}
            )
            return pb.Result(success=ok, message=message)
        except Exception as exc:  # noqa: BLE001
            return pb.Result(success=False, message=str(exc))

    # ── application type ──────────────────────────────────────────────────────

    def GetApplication(self, request, context):
        """RPC: the device's application task, the supported tasks, migration flag."""
        return _application_pb(self._app.application_service.info())

    def SetApplication(self, request, context):
        """RPC: switch the application type (ModelService.set_task)."""
        try:
            self._app.model_service.set_task(request.task)
        except InvalidTask as exc:
            _refuse(context, grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            return pb.ApplicationInfo()
        except PreconditionFailed as exc:
            _refuse(context, grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            return pb.ApplicationInfo()
        except Exception as exc:  # noqa: BLE001 - stop or write failure
            logger.exception("Application change to %r failed", request.task)
            _refuse(context, grpc.StatusCode.INTERNAL,
                    f"Could not change the application type: {exc}")
            return pb.ApplicationInfo()
        return _application_pb(self._app.application_service.info())


class _StagedFile:
    """Adapts an upload already staged on disk to the ``.save(path)``
    interface ModelService.save_model expects (Flask FileStorage over HTTP).
    ``save`` is an atomic rename — the staging file lives in the model
    directory, so the final name appears only complete, never half-written."""

    def __init__(self, staged_path: str):
        self._staged_path = staged_path

    def save(self, path: str) -> None:
        """Atomically move the staged upload to *path* (durable: dir fsync)."""
        os.replace(self._staged_path, path)
        fsync_dir(os.path.dirname(path))


def _conversion_pb(job) -> "pb.ConversionJob":
    """Convert a conversion-job object to a ``ConversionJob`` message."""
    status = getattr(job, "status", "")
    return pb.ConversionJob(
        job_id=str(getattr(job, "job_id", "")),
        original_filename=str(getattr(job, "original_filename", "")),
        status=str(getattr(status, "value", status) or ""),
        progress=int(getattr(job, "progress", 0) or 0),
        message=str(getattr(job, "message", "") or ""),
        error=str(getattr(job, "error", "") or ""),
        engine_filename=str(getattr(job, "engine_filename", "") or ""),
        started_at=float(getattr(job, "started_at", 0.0) or 0.0),
        elapsed_secs=float(getattr(job, "elapsed_secs", 0.0) or 0.0),
    )


def _label_class_pb(c: dict) -> "pb.LabelClass":
    """One ``LabelingService`` class suggestion as a ``LabelClass`` message."""
    return pb.LabelClass(class_id=int(c["class_id"]), class_name=str(c["class_name"]),
                         score=float(c["score"]))


def _stats_pb(s) -> "pb.Stats":
    """Convert a stats object to a ``Stats`` message."""
    return pb.Stats(
        fps=float(getattr(s, "fps", 0.0) or 0.0),
        inference_time=float(getattr(s, "inference_time", 0.0) or 0.0),
        detections=int(getattr(s, "detections", 0) or 0),
        frames_with_detections=int(getattr(s, "frames_with_detections", 0) or 0),
        finish_mean_ms=float(getattr(s, "finish_mean_ms", 0.0) or 0.0),
        finish_p95_ms=float(getattr(s, "finish_p95_ms", 0.0) or 0.0),
        finish_p99_ms=float(getattr(s, "finish_p99_ms", 0.0) or 0.0),
        encode_mean_ms=float(getattr(s, "encode_mean_ms", 0.0) or 0.0),
        encode_p95_ms=float(getattr(s, "encode_p95_ms", 0.0) or 0.0),
        frame_age_p95_ms=float(getattr(s, "frame_age_p95_ms", 0.0) or 0.0),
    )


def _event_pb(ev: dict) -> "pb.Event":
    """Convert an event dict to an ``Event`` message (data carried as JSON)."""
    return pb.Event(
        version=int(ev.get("version", 0)),
        type=str(ev.get("type", "")),
        timestamp=float(ev.get("timestamp", 0.0)),
        source=str(ev.get("source", "")),
        keys=list(ev.get("keys", []) or []),
        data=json.dumps(ev.get("data", {}) or {}),
    )


def _bind_or_raise(server: grpc.Server, addr: str, what: str) -> None:
    """``add_insecure_port`` with one failure mode: a RuntimeError naming the port.

    grpcio returns 0 on a bind failure in some releases and raises in others;
    both become the same RuntimeError here.
    """
    try:
        bound = server.add_insecure_port(addr)
    except RuntimeError as exc:
        raise RuntimeError(f"could not bind the {what} to {addr}: {exc}") from exc
    if bound == 0:
        raise RuntimeError(f"could not bind the {what} to {addr}")


def serve_grpc(application, listen_addr: Optional[str] = None) -> grpc.Server:
    """Bind and start the DetectionControl gRPC server; return it.

    Runs on the calling thread so a bind failure is a plain exception the
    entry point can turn into a non-zero exit, never a live process with no
    listener. The standard
    gRPC health service is registered and reports SERVING once the port is
    open (the gateway's readiness probe checks it). ``grpc.so_reuseport`` is
    off so a second instance cannot silently share the port.
    """
    addr = listen_addr or LISTEN_ADDR
    from grpc_health.v1 import health, health_pb2, health_pb2_grpc

    # Explicit receive cap: upload chunks are ~1 MiB, so 8 MiB is ample
    # headroom while still bounding what one message can buffer.
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=16),
        options=[
            ("grpc.max_receive_message_length", 8 * 1024 * 1024),
            ("grpc.so_reuseport", 0),
        ],
    )
    pb_grpc.add_DetectionControlServicer_to_server(
        DetectionControlServicer(application), server
    )
    pb_grpc.add_ModelControlServicer_to_server(
        ModelControlServicer(application), server
    )
    pb_grpc.add_ManagementControlServicer_to_server(
        ManagementControlServicer(application), server
    )
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    _bind_or_raise(server, addr, "inference gRPC server")
    server.start()
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    logger.info("Inference DetectionControl gRPC server listening on %s", addr)
    return server
