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
from ..gpu import (
    GpuBusy,
    Purpose,
    busy_response,
    gpu_lock,
    gpu_session,
    training_job_active,
    unload_assistants,
)
from ..grpc_clients import clients, inf
from . import training_bp
from .helpers import _grpc_error, _json, _json_error
from .orphan import tracker

logger = logging.getLogger(__name__)


def _do_exit(resume_detection: bool) -> tuple[bool, str]:
    """Exit training mode: best-effort assistant unloads, then resume (or skip).

    Shared by the /training/exit route and the orphan watchdog. Returns
    ``(success, message)``; raises ``grpc.RpcError`` when the resume RPC
    itself fails (callers map/log it). While a training job is active the
    handover is left alone, whatever ``resume_detection`` asks: the trainer
    owns the GPU, so leaving the training page mid-run keeps detection stopped
    (the dashboard's Start stays disabled until the job ends). The handover
    left behind locks nothing: the next consumer's shared GPU release
    (``gateway.gpu``) ends it once the job is over.
    """
    # Both labeling assistants go first (SAM3 in the training-service, the
    # labeling engine in the inference-service), verified: a resume must not
    # load the detection runtime beside one that is still resident, so an
    # unconfirmed unload refuses it (the orphan watchdog retries). Without a
    # resume nothing is loaded here, so the handover still ends.
    try:
        unload_assistants()
    except GpuBusy as exc:
        if resume_detection:
            return False, exc.message
        logger.warning("Assistant unload on exit failed: %s", exc.message)

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
    """POST /api/v1/training/enter — hand the GPU over to training.

    The shared GPU release for ``Purpose.TRAINING``: detection stops (the
    device UI confirms it first), every other holder goes, and the handover is
    marked so that leaving resumes detection. Refused only while a model
    conversion runs.
    """
    try:
        with gpu_session(Purpose.TRAINING):
            # Armed inside the lock: a /training/train that follows sees
            # training mode, and an exit cannot slip between handover and arm.
            tracker.arm()
    except GpuBusy as exc:
        return busy_response(exc)
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    event_service.publish("detection_state_changed", keys=["status"],
                          data={"is_running": False})
    return _json({"status": "success", "message": "Runtime released"})


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
        # A resume loads the engine: it takes the GPU lock like any consumer.
        with gpu_lock():
            ok, message = _do_exit(resume_detection)
            # Disarmed inside the lock, like the enter arms it; a failed exit
            # is still in training mode and keeps the watchdog armed.
            if ok:
                tracker.disarm()
    except GpuBusy as exc:
        return busy_response(exc)
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    if not ok:
        return _json_error(message, 500)
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
