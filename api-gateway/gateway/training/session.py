# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Training session lifecycle: GPU handover (enter/exit) and the CPU-only
combined camera preview."""
import logging

import grpc
from flask import Response, request

from .. import media
from ..events import event_service
from ..grpc_clients import clients, inf, trn
from . import training_bp
from .helpers import (
    _grpc_error,
    _json,
    _json_error,
    _release_runtime,
    training_job_active,
)
from .orphan import tracker

logger = logging.getLogger(__name__)


def _do_exit(resume_detection: bool) -> tuple[bool, str]:
    """Exit training mode: best-effort assistant unloads, then resume (or skip).

    Shared by the /training/exit route and the orphan watchdog. Returns
    ``(success, message)``; raises ``grpc.RpcError`` when the resume RPC
    itself fails (callers map/log it). While a training job is active the
    handover is left alone, whatever ``resume_detection`` asks: the trainer
    owns the GPU, so leaving the training page mid-run keeps detection stopped
    (the dashboard's Start stays disabled until the job ends) and the
    application type locked, exactly like the post-training conversion handover.
    """
    # Best-effort unload of both labeling assistants first (SAM3 in the
    # training-service, the labeling engine in the inference-service);
    # freeing inference is what matters.
    try:
        clients.training.UnloadSam(trn.Empty())
    except grpc.RpcError as exc:
        logger.warning("UnloadSam on exit failed: %s", exc)
    try:
        clients.model.UnloadLabelModel(inf.Empty())
    except grpc.RpcError as exc:
        logger.warning("UnloadLabelModel on exit failed: %s", exc)

    if training_job_active():
        event_service.publish("detection_state_changed", keys=["status"],
                              data={"is_running": False})
        return True, "Inference left stopped: a model training is in progress"
    if not resume_detection:
        # End the GPU handover without restarting detection: the runtime stays
        # unloaded for the model conversion, but the application type can
        # change again. Best-effort, like the unloads above:
        # an explicit Start ends the handover too.
        try:
            clients.management.ResumeRuntime(
                inf.ResumeRuntimeRequest(keep_detection_stopped=True))
        except grpc.RpcError as exc:
            logger.warning("Ending the GPU handover on exit failed: %s", exc)
        event_service.publish("detection_state_changed", keys=["status"],
                              data={"is_running": False})
        return True, "Inference left stopped for model conversion"

    r = clients.management.ResumeRuntime(inf.ResumeRuntimeRequest())
    if not r.success:
        return False, r.message
    event_service.publish("detection_state_changed", keys=["status"],
                          data={"is_running": True})
    return True, r.message


@training_bp.route("/api/v1/training/enter", methods=["POST"])
def training_enter():
    """POST /api/v1/training/enter — gateway relay."""
    try:
        r = _release_runtime()
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    if not r.success:
        return _json_error(r.message, 409)
    event_service.publish("detection_state_changed", keys=["status"],
                          data={"is_running": False})
    tracker.arm()
    return _json({"status": "success", "message": r.message})


@training_bp.route("/api/v1/training/exit", methods=["POST"])
def training_exit():
    # `resume_detection` (default true): a manual exit restores inference. When
    # training just finished, the client passes false — the model is being
    # converted/optimized, so the runtime stays released (GPU free for the
    # TensorRT build, old model not running). The conversion's auto-select then
    # loads the new engine with detection still stopped, for the user to start.
    """POST /api/v1/training/exit — gateway relay."""
    body = request.get_json(silent=True) or {}
    resume_detection = bool(body.get("resume_detection", True))

    try:
        ok, message = _do_exit(resume_detection)
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    if not ok:
        # Still in training mode — keep the orphan watchdog armed.
        return _json_error(message, 500)
    tracker.disarm()
    return _json({"status": "success", "message": message})


@training_bp.route("/api/v1/training/heartbeat", methods=["POST"])
def training_heartbeat():
    """POST /api/v1/training/heartbeat — keep the orphan watchdog fed.

    The device UI beats every ~10s while the training page is mounted; the
    blueprint's before_request hook (tracker.touch) is the whole effect. It
    never arms the tracker, so a beat racing a normal exit cannot re-arm it.
    Answers JSON, not 204, because the frontend parses every response as JSON."""
    return _json({"status": "ok"})


@training_bp.route("/api/v1/training/preview", methods=["GET"])
def training_preview():
    """GET /api/v1/training/preview — gateway relay."""
    return Response(media.generate_training_preview(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")
