# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Model-assisted labeling routes: an existing engine on the device, loaded
into the inference-service's private TensorRT labeling worker, and per-image
detection for label suggestions. The image itself comes from the
training-service (the dataset's stored JPEG), so the suggestions land in the
same normalized space the labels use."""
import logging

import grpc
from flask import request

from ..grpc_clients import clients, inf, trn
from . import training_bp
from .helpers import _grpc_error, _json, _json_error, _pairs, _result

logger = logging.getLogger(__name__)


def _status_dict(s) -> dict:
    """Serialize a LabelModelStatus message."""
    return {
        "loaded": s.loaded,
        "model_name": s.model_name,
        "class_names": list(s.class_names),
        "message": s.message,
        # The loaded engine's task ("" = none loaded).
        "task": s.task,
    }


def _class_dict(c) -> dict:
    """A classification engine's whole-image class suggestion (class_id = the engine's)."""
    return {"class_id": c.class_id, "class_name": c.class_name, "score": c.score}


def _suggestion_dict(d) -> dict:
    """Normalized corners → the editor's center/size box (class_id = the engine's)."""
    x1, y1 = min(d.x1, d.x2), min(d.y1, d.y2)
    x2, y2 = max(d.x1, d.x2), max(d.y1, d.y2)
    return {
        "class_id": d.class_id,
        "cx": (x1 + x2) / 2.0,
        "cy": (y1 + y2) / 2.0,
        "w": x2 - x1,
        "h": y2 - y1,
    }


@training_bp.route("/api/v1/training/label-model", methods=["GET"])
def training_label_model_status():
    """GET /api/v1/training/label-model — gateway relay."""
    try:
        s = clients.model.GetLabelModelStatus(inf.Empty())
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    return _json(_status_dict(s))


@training_bp.route("/api/v1/training/label-model/load", methods=["POST"])
def training_label_model_load():
    """POST /api/v1/training/label-model/load — gateway relay."""
    body = request.get_json(silent=True) or {}
    model_name = str(body.get("model_name") or "").strip()
    if not model_name:
        return _json_error("'model_name' is required")
    # One labeling assistant at a time (8 GB GPU budget): drop SAM3 in the
    # training-service first, best-effort and status-probed.
    try:
        if clients.training.GetSamStatus(trn.Empty()).loaded:
            clients.training.UnloadSam(trn.Empty())
    except grpc.RpcError as exc:
        logger.warning("Best-effort SAM unload before label-model load failed: %s", exc)
    try:
        # Spawning a TensorRT worker + engine deserialization on the Orin:
        # allow well beyond the default control-call deadline.
        return _result(clients.model.LoadLabelModel(inf.ModelName(name=model_name), timeout=300))
    except grpc.RpcError as exc:
        return _grpc_error(exc)


@training_bp.route("/api/v1/training/label-model/unload", methods=["POST"])
def training_label_model_unload():
    """POST /api/v1/training/label-model/unload — gateway relay."""
    try:
        return _result(clients.model.UnloadLabelModel(inf.Empty()))
    except grpc.RpcError as exc:
        return _grpc_error(exc)


@training_bp.route("/api/v1/training/label-model/detect", methods=["POST"])
def training_label_model_detect():
    """POST /api/v1/training/label-model/detect — dataset image → inference-service."""
    body = request.get_json(silent=True) or {}
    image_id = body.get("image_id", "")
    dataset_id = body.get("dataset_id", "")
    if not image_id:
        return _json_error("'image_id' is required")
    if not dataset_id:
        return _json_error("'dataset_id' is required")
    try:
        threshold = float(body.get("threshold", 0.0) or 0.0)
    except (TypeError, ValueError):
        return _json_error("Malformed threshold")
    # Suggestions are only valid for a dataset of the engine's task: the check
    # is against the dataset, not the device.
    try:
        dataset = clients.training.GetDataset(trn.DatasetId(dataset_id=dataset_id))
        engine_task = clients.model.GetLabelModelStatus(inf.Empty()).task
    except grpc.RpcError as exc:
        if exc.code() == grpc.StatusCode.NOT_FOUND:
            return _json_error(f"Dataset '{dataset_id}' not found", 404)
        return _grpc_error(exc)
    dataset_task = dataset.task or "detect"
    if engine_task and engine_task != dataset_task:
        return _json_error(
            f"The labeling model is a '{engine_task}' model, but this dataset is labeled "
            f"for '{dataset_task}'; load a '{dataset_task}' model.", 409)
    try:
        blob = clients.training.GetImage(trn.ImageId(dataset_id=dataset_id, image_id=image_id))
    except grpc.RpcError as exc:
        if exc.code() == grpc.StatusCode.NOT_FOUND:
            return _json_error(f"Image '{image_id}' not found", 404)
        return _grpc_error(exc)
    try:
        r = clients.model.LabelDetect(
            inf.LabelDetectRequest(jpeg=blob.jpeg, threshold=threshold), timeout=120)
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    if not r.success:
        return _json_error(r.message, 400)
    return _json({
        "boxes": [_suggestion_dict(d) for d in r.detections],
        "scores": [d.score for d in r.detections],
        "class_names": [d.class_name for d in r.detections],
        # A segmentation engine: each box's mask as rings, parallel to boxes
        # (the same shape as the SAM route's `polygons`).
        "polygons": [[_pairs(ring.points) for ring in d.rings] for d in r.detections],
        # A classification engine: the class above the threshold (or null)
        # and its top-k candidates, highest first.
        "image_class": _class_dict(r.image_class) if r.HasField("image_class") else None,
        "candidates": [_class_dict(c) for c in r.candidates],
    })
