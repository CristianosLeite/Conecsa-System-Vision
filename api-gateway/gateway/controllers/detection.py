# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Detection controller: run state (status/start/stop), confidence thresholds,
stats and the detections snapshot."""
import logging

# Compiled detection schemas (protobuf content-negotiation for the Tauri/native
# and Leptos protobuf endpoints). Sits next to the other stubs in gateway/proto,
# which `..grpc_clients` (imported above) puts on sys.path.
import detection_pb2 as det_pb  # noqa: E402
import grpc
from flask import Response, request

from ..grpc_clients import clients, inf, trn
from ..helpers import (
    DEVICE_VERSION,
    _accepts_protobuf,
    _grpc_error,
    _hub_verified,
    _json,
    _json_error,
    _json_success,
    _protobuf,
    _publish_if_success,
)
from ..training.helpers import conversion_active, training_job_active
from . import api_bp

logger = logging.getLogger(__name__)


def _camera_connected(status) -> bool:
    """Read StatusResponse.camera_connected, treating "unset" as connected.

    The field has explicit presence, so an inference-service that predates the
    camera gate leaves it absent rather than sending false. Defaulting that to
    false would fail closed — the gateway would refuse every Start with a 409
    until inference is upgraded too. Unset means "no gate on the producer",
    which is exactly the ungated behavior: let detection start.
    """
    return status.camera_connected if status.HasField("camera_connected") else True


def _status_task(status) -> dict:
    """StatusResponse.task for JSON clients, as the keys to merge into the body.

    ``{"task": None}`` while none is chosen. An inference-service that predates
    application types leaves the field absent, and the key is omitted then:
    the hub reads a missing key as older firmware (no guard) and ``null`` as
    "unset", so collapsing the two would guard a device that merely lags.
    """
    if not status.HasField("task"):
        return {}
    return {"task": status.task or None}


def _application_unset(status) -> bool:
    """True only when the producer knows application types and none is chosen.

    Presence decides, like ``camera_connected``: an absent field comes from an
    inference-service that predates the feature, so it must not block Start
    during a rolling upgrade.
    """
    return status.HasField("task") and not status.task


def _status_segment(status) -> dict:
    """The segmentation instance limit in effect, when the inference-service reports one."""
    if status.HasField("segment_max_instances"):
        return {"segment_max_instances": status.segment_max_instances}
    return {}


def _status_face(status) -> dict:
    """The face recognition settings in effect, when the inference-service reports them."""
    return {key: getattr(status, key)
            for key in ("face_match_threshold", "face_min_size_px", "face_max_faces")
            if status.HasField(key)}


@api_bp.route('/api/v1/status', methods=['GET'])
def get_status():
    """GET /api/v1/status — gateway relay."""
    try:
        s = clients.detection.GetStatus(inf.Empty())
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    if _accepts_protobuf():
        r = det_pb.StatusResponse()
        r.is_running = s.is_running
        r.current_model = s.model
        r.confidence_threshold = s.confidence_threshold
        r.stats.fps = float(s.stats.fps)
        r.stats.inference_time = float(s.stats.inference_time)
        r.stats.detections = int(s.stats.detections)
        r.stats.frames_with_detections = int(s.stats.frames_with_detections)
        r.protocols.http_port = 5000
        r.camera_connected = _camera_connected(s)
        # Left unset (not empty) for an older inference-service, like the JSON key.
        if s.HasField("task"):
            r.task = s.task
        for key, value in {**_status_segment(s), **_status_face(s)}.items():
            setattr(r, key, value)
        return _protobuf(r)
    return _json({
        "is_running": s.is_running,
        "version": DEVICE_VERSION,
        "model": s.model,
        "confidence_threshold": s.confidence_threshold,
        "overlay_threshold": s.overlay_threshold,
        "acceleration_type": s.acceleration_type,
        "runtime_type": s.runtime_type,
        "camera_connected": _camera_connected(s),
        **_status_task(s),
        **_status_segment(s),
        **_status_face(s),
        "stats": {
            "fps": s.stats.fps,
            "inference_time": s.stats.inference_time,
            "detections": s.stats.detections,
            "frames_with_detections": s.stats.frames_with_detections,
        },
        "protocols": {
            "http_port": 5000,
        },
    })


@api_bp.route('/api/v1/start', methods=['POST'])
def start_detection():
    """POST /api/v1/start — gateway relay."""
    try:
        status = clients.detection.GetStatus(inf.Empty())
        if status.is_running:
            if _accepts_protobuf():
                return _protobuf(det_pb.StartDetectionResponse(
                    success=False, message="Detection already running"), 400)
            return _json_error("Detection already running", 400)
        # A blank device has no application type: nothing may run until an
        # administrator chooses one (the inference-service refuses as well).
        if _application_unset(status):
            msg = ("No application type is selected. An administrator must choose "
                   "one before starting detection.")
            if _accepts_protobuf():
                return _protobuf(det_pb.StartDetectionResponse(success=False, message=msg), 409)
            return _json_error(msg, 409)
        # Without a camera the webcam-server publishes no frames at all, so
        # detection would run blind — refuse before touching inference.
        if not _camera_connected(status):
            msg = "No camera connected. Connect a camera before starting detection."
            if _accepts_protobuf():
                return _protobuf(det_pb.StartDetectionResponse(success=False, message=msg), 409)
            return _json_error(msg, 409)
        # The single Jetson GPU is owned by a training job or a TensorRT engine
        # build while one runs; starting detection would re-spawn the TensorRT
        # workers on top of it. The device UI disables Start for the same
        # states, but the hub, Node-RED and a second tab reach this route
        # directly, so the gateway is the authority.
        if training_job_active():
            msg = ("A model training is in progress; wait for it to finish "
                   "before starting detection.")
            if _accepts_protobuf():
                return _protobuf(det_pb.StartDetectionResponse(success=False, message=msg), 409)
            return _json_error(msg, 409)
        if conversion_active():
            msg = ("A model conversion is in progress; wait for it to finish "
                   "before starting detection.")
            if _accepts_protobuf():
                return _protobuf(det_pb.StartDetectionResponse(success=False, message=msg), 409)
            return _json_error(msg, 409)
        # Training's labeling assistants (SAM3 in the training-service, the
        # labeling engine on the inference-service's private worker) must
        # never stay GPU-pinned once detection runs. Best-effort with their
        # own except: an unreachable training-service must not block the
        # start (nor be mistaken for an inference failure), and the status
        # probes avoid a spurious *_changed event from unloading nothing.
        try:
            if clients.training.GetSamStatus(trn.Empty()).loaded:
                clients.training.UnloadSam(trn.Empty())
        except grpc.RpcError as sam_exc:
            logger.warning("Best-effort SAM unload before start failed: %s",
                           sam_exc)
        try:
            if clients.model.GetLabelModelStatus(inf.Empty()).loaded:
                clients.model.UnloadLabelModel(inf.Empty())
        except grpc.RpcError as lm_exc:
            logger.warning("Best-effort label-model unload before start failed: %s",
                           lm_exc)
        r = clients.detection.Start(inf.Empty())
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    if not r.success:
        if _accepts_protobuf():
            return _protobuf(det_pb.StartDetectionResponse(success=False, message=r.message), 500)
        return _json_error(r.message, 500)
    video_feed_url = "/api/v1/video_feed_processed"
    if _accepts_protobuf():
        resp = _protobuf(det_pb.StartDetectionResponse(
            success=True, message="Detection started", video_feed_url=video_feed_url))
    else:
        resp = _json_success(message="Detection started", video_feed_url=video_feed_url)
    return _publish_if_success(resp, "detection_state_changed", ["status"],
                               data={"is_running": True})


@api_bp.route('/api/v1/stop', methods=['POST'])
def stop_detection():
    """POST /api/v1/stop — gateway relay."""
    try:
        if not clients.detection.GetStatus(inf.Empty()).is_running:
            if _accepts_protobuf():
                return _protobuf(det_pb.StopDetectionResponse(
                    success=False, message="Detection not running"), 400)
            return _json_error("Detection not running", 400)
        clients.detection.Stop(inf.Empty())
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    if _accepts_protobuf():
        resp = _protobuf(det_pb.StopDetectionResponse(success=True, message="Detection stopped"))
    else:
        resp = _json_success(message="Detection stopped")
    return _publish_if_success(resp, "detection_state_changed", ["status"],
                               data={"is_running": False})


def _parse_threshold():
    """Return (threshold, error_response). Mirrors DetectionController."""
    if "application/json" in request.headers.get("Content-Type", ""):
        try:
            threshold = (request.get_json() or {}).get("threshold")
        except Exception:  # noqa: BLE001
            return None, _json({"success": False, "message": "Invalid JSON request"}, 400)
        if threshold is None:
            return None, _json({"success": False, "message": "Missing threshold parameter"}, 400)
        return threshold, None
    req = det_pb.SetThresholdRequest()
    try:
        req.ParseFromString(request.data)
    except Exception:  # noqa: BLE001
        return None, _protobuf(det_pb.SetThresholdResponse(
            success=False, message="Invalid protobuf request"), 400)
    return req.threshold, None


def _threshold_success(threshold, message):
    """Build a threshold-set success Response (JSON or protobuf)."""
    if "application/json" in request.headers.get("Content-Type", ""):
        return _json({"success": True, "message": message, "threshold": threshold})
    return _protobuf(det_pb.SetThresholdResponse(success=True, message=message, threshold=threshold))


def _threshold_invalid():
    """Build a 400 'threshold out of range' Response (JSON or protobuf)."""
    if "application/json" in request.headers.get("Content-Type", ""):
        return _json({"success": False, "message": "Threshold must be between 0 and 1"}, 400)
    return _protobuf(det_pb.SetThresholdResponse(
        success=False, message="Threshold must be between 0 and 1"), 400)


@api_bp.route('/api/v1/threshold', methods=['POST'])
def set_threshold():
    """POST /api/v1/threshold — gateway relay."""
    threshold, err = _parse_threshold()
    if err is not None:
        return err
    if threshold is None:
        return _threshold_invalid()
    try:
        r = clients.detection.SetThreshold(inf.ThresholdRequest(threshold=float(threshold)))
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    if not r.success:
        return _threshold_invalid()
    resp = _threshold_success(threshold, "Threshold updated")
    return _publish_if_success(resp, "thresholds_changed", ["status", "thresholds"],
                               data={"confidence_threshold": threshold})


@api_bp.route('/api/v1/overlay_threshold', methods=['POST'])
def set_overlay_threshold():
    """POST /api/v1/overlay_threshold — gateway relay."""
    threshold, err = _parse_threshold()
    if err is not None:
        return err
    if threshold is None:
        return _threshold_invalid()
    try:
        r = clients.detection.SetOverlayThreshold(inf.ThresholdRequest(threshold=float(threshold)))
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    if not r.success:
        return _threshold_invalid()
    resp = _threshold_success(threshold, "Overlay threshold updated")
    return _publish_if_success(resp, "thresholds_changed", ["status", "thresholds"],
                               data={"overlay_threshold": threshold})


@api_bp.route('/api/v1/segment/max_instances', methods=['POST'])
def set_segment_max_instances():
    """POST /api/v1/segment/max_instances — segmentation instance limit.

    JSON ``{"max_instances": n}``, an integer 1..255 that caps the instances
    per frame and per tile; saved with the active model's settings, like the
    thresholds, and announced on the same ``thresholds`` event key.
    """
    body = request.get_json(silent=True)
    value = body.get("max_instances") if isinstance(body, dict) else None
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 255:
        return _json({"success": False,
                      "message": "max_instances must be an integer between 1 and 255"}, 400)
    try:
        r = clients.detection.SetSegmentMaxInstances(
            inf.SegmentMaxInstancesRequest(max_instances=value))
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    if not r.success:
        return _json({"success": False, "message": r.message}, 400)
    resp = _json({"success": True, "message": "Instance limit updated", "max_instances": value})
    return _publish_if_success(resp, "thresholds_changed", ["status", "thresholds"],
                               data={"segment_max_instances": value})


def _face_setting(body: dict, key: str, valid):
    """``(present, value, error)`` for one optional field of the face settings body."""
    if key not in body:
        return False, None, None
    value = body[key]
    if isinstance(value, bool) or not valid(value):
        return True, None, key
    return True, value, None


@api_bp.route('/api/v1/face/settings', methods=['POST'])
def set_face_settings():
    """POST /api/v1/face/settings — face recognition settings.

    JSON with any of ``match_threshold`` (number 0..1, the cosine similarity a
    face must exceed to be named), ``min_size_px`` (integer 0..1024) and
    ``max_faces`` (integer 1..20); saved with the active model's settings and
    announced on the ``thresholds`` event key.
    """
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return _json({"success": False, "message": "A JSON object is required"}, 400)
    fields = {
        "match_threshold": lambda v: isinstance(v, (int, float)) and 0.0 <= v <= 1.0,
        "min_size_px": lambda v: isinstance(v, int) and 0 <= v <= 1024,
        "max_faces": lambda v: isinstance(v, int) and 1 <= v <= 20,
    }
    changes = {}
    for key, valid in fields.items():
        present, value, error = _face_setting(body, key, valid)
        if error:
            return _json({"success": False,
                          "message": f"{error} is out of range or not a number"}, 400)
        if present:
            changes[key] = value
    if not changes:
        return _json({"success": False,
                      "message": "Give match_threshold, min_size_px or max_faces"}, 400)
    try:
        r = clients.detection.SetFaceSettings(inf.FaceSettingsRequest(**changes))
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    if not r.success:
        return _json({"success": False, "message": r.message}, 400)
    resp = _json({"success": True, "message": "Face settings updated", **changes})
    return _publish_if_success(resp, "thresholds_changed", ["status", "thresholds"],
                               data={f"face_{k}": v for k, v in changes.items()})


@api_bp.route('/api/v1/stats', methods=['GET'])
def get_stats():
    """GET /api/v1/stats — gateway relay."""
    try:
        s = clients.detection.GetStatus(inf.Empty()).stats
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    return _json({
        "fps": s.fps,
        "inference_time": s.inference_time,
        "detections": s.detections,
        "frames_with_detections": s.frames_with_detections,
        # Pipeline service times in ms (benchmark protocol): stage C
        # (postprocess), stage D (encode + publish) and the age of a frame
        # from its pickup off the camera ring to its publication.
        "finish_mean_ms": s.finish_mean_ms,
        "finish_p95_ms": s.finish_p95_ms,
        "finish_p99_ms": s.finish_p99_ms,
        "encode_mean_ms": s.encode_mean_ms,
        "encode_p95_ms": s.encode_p95_ms,
        "frame_age_p95_ms": s.frame_age_p95_ms,
    })


@api_bp.route('/api/v1/stats/reset', methods=['POST'])
def reset_stats():
    """POST /api/v1/stats/reset — gateway relay."""
    try:
        clients.detection.ResetStats(inf.Empty())
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    resp = _json({"success": True, "reset": "stats"})
    return _publish_if_success(resp, "stats_changed", ["stats"], data={"reset": True})


@api_bp.route('/api/v1/detections/snapshot', methods=['GET'])
def get_detections_snapshot():
    """GET /api/v1/detections/snapshot — gateway relay."""
    include_frame = request.args.get("include_frame", "true").lower() != "false"
    include_raw = request.args.get("include_raw_frame", "false").lower() == "true"
    # Hub pulls arrive through the mTLS terminator, which stamps the verified
    # client-cert result (system-vision/config/nginx-enforcing.conf). Local
    # consumers (Flow nodes hitting the gateway directly) must not feed the
    # offline-buffer's hub-is-online heartbeat, even if they spoof the header —
    # _hub_verified also checks that the peer is the terminator itself.
    # ``passive=true`` marks a reader that is never the heartbeat even through
    # the terminator: the device UI's classification panel polls this route
    # while the device is open in the hub. The flag can only
    # switch counting off, so a spoofed value is harmless.
    passive = request.args.get("passive", "false").lower() == "true"
    hub_pull = not passive and _hub_verified()
    try:
        # proto3 bool default is false; set it explicitly to match HTTP default-true.
        r = clients.detection.Snapshot(inf.SnapshotRequest(
            include_frame=include_frame, include_raw_frame=include_raw,
            hub_pull=hub_pull))
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    return Response(r.json, status=200, mimetype="application/json")


@api_bp.route('/api/v1/detections/backlog', methods=['GET'])
def get_detections_backlog():
    """GET /api/v1/detections/backlog — one page of offline-buffered records."""
    limit = request.args.get("limit", type=int)
    if limit is None:
        limit = 0
    elif limit < 0:
        return _json_error('"limit" must be >= 0', 400)
    elif limit > 100:
        limit = 100
    try:
        r = clients.detection.ListBacklog(inf.BacklogRequest(limit=limit))
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    return Response(r.json, status=200, mimetype="application/json")


@api_bp.route('/api/v1/detections/backlog/ack', methods=['POST'])
def ack_detections_backlog():
    """POST /api/v1/detections/backlog/ack — delete records the hub persisted."""
    ids = (request.get_json(silent=True) or {}).get("ids")
    if not isinstance(ids, list) or not all(type(i) is int for i in ids):
        return _json_error('Body must be {"ids": [int, ...]}', 400)
    try:
        r = clients.detection.AckBacklog(inf.BacklogAckRequest(ids=ids))
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    return _json({"success": r.success, "message": r.message})
