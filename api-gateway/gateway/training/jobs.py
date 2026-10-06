# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Training-job routes: start (with GPU-handover re-assert), status polling,
cancel and finish."""
import logging

import grpc
from flask import request

from ..gpu import GpuBusy, Purpose, acquire_gpu, busy_response, gpu_lock
from ..grpc_clients import clients, inf, trn
from . import training_bp
from .helpers import _grpc_error, _job_dict, _json, _json_error, _result
from .orphan import tracker

logger = logging.getLogger(__name__)

def _detection_running() -> bool:
    """Whether detection runs now (False when it cannot be asked)."""
    try:
        return bool(clients.detection.GetStatus(inf.Empty()).is_running)
    except grpc.RpcError:
        return False


def _undo_handover(resume_detection: bool) -> None:
    """End the handover a failed start began, restoring detection if it ran."""
    from .session import _do_exit

    try:
        ok, message = _do_exit(resume_detection)
    except grpc.RpcError as exc:
        ok, message = False, str(exc)
    if not ok:
        logger.warning("Could not undo the GPU handover of a failed start: %s", message)


@training_bp.route("/api/v1/training/train", methods=["POST"])
def training_start():
    """POST /api/v1/training/train — gateway relay."""
    body = request.get_json(silent=True) or {}
    federated = bool(body.get("federated"))
    model_name = (body.get("model_name") or "").strip()
    if not model_name and not federated:
        # Federated rounds keep the result on-device, so no name is needed.
        return _json_error("'model_name' is required")
    dataset_id = (body.get("dataset_id") or "").strip()
    if not dataset_id:
        return _json_error("'dataset_id' is required")
    # Every local check first: the GPU is taken only for a request that can
    # actually start.
    try:
        train_request = trn.TrainRequest(
            model_name=model_name,
            dataset_id=dataset_id,
            epochs=int(body.get("epochs") or 0),
            batch=int(body.get("batch") or 0),
            patience=int(body.get("patience") or 0),
            initial_weights_id=(body.get("initial_weights_id") or "").strip(),
            federated=federated,
            base_model=str(body.get("base_model") or "").strip(),
        )
    except (TypeError, ValueError):
        return _json_error("'epochs', 'batch' and 'patience' must be non-negative integers")
    try:
        with gpu_lock():
            # Read under the lock (an enter or exit may have run while this
            # request waited): a client inside training mode keeps it when the
            # start fails; one that skipped /training/enter must not be left
            # with a handover and no job.
            entered = tracker.armed
            was_running = not entered and _detection_running()
            # Re-assert the GPU handover (the shared release): idempotent,
            # and protects against a client that skipped /training/enter.
            try:
                acquire_gpu(Purpose.TRAINING)
            except grpc.RpcError as exc:
                detail = exc.details() if hasattr(exc, "details") else str(exc)
                return _json_error(f"Could not release inference runtime: {detail}", 503)
            try:
                job = clients.training.StartTraining(train_request)
            except grpc.RpcError as exc:
                if not entered:
                    _undo_handover(was_running)
                return _grpc_error(exc)
            # Covers clients that skipped /training/enter (the handover was
            # re-asserted above): a run is active, so the orphan watchdog must
            # be armed — inside the lock, so an exit cannot land in between.
            tracker.arm()
    except GpuBusy as exc:
        return busy_response(exc)
    return _json(_job_dict(job), 202)


@training_bp.route("/api/v1/training/train/status", methods=["GET"])
def training_status():
    """GET /api/v1/training/train/status — gateway relay."""
    try:
        job = clients.training.GetTraining(trn.Empty())
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    return _json(_job_dict(job))


@training_bp.route("/api/v1/training/train/cancel", methods=["POST"])
def training_cancel():
    """POST /api/v1/training/train/cancel — gateway relay."""
    try:
        return _result(clients.training.CancelTraining(trn.Empty()))
    except grpc.RpcError as exc:
        return _grpc_error(exc)


@training_bp.route("/api/v1/training/train/finish", methods=["POST"])
def training_finish():
    """POST /api/v1/training/train/finish — gateway relay."""
    try:
        return _result(clients.training.FinishTraining(trn.Empty()))
    except grpc.RpcError as exc:
        return _grpc_error(exc)
