# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""SAM (Segment Anything) assisted-labeling routes: status, load/unload of the
checkpoint and point/text-prompted segmentation."""
import logging

import grpc
from flask import request

from ..grpc_clients import clients, inf, trn
from . import training_bp
from .helpers import _grpc_error, _json, _json_error, _pairs, _result

logger = logging.getLogger(__name__)


@training_bp.route("/api/v1/training/sam", methods=["GET"])
def training_sam_status():
    """GET /api/v1/training/sam — gateway relay."""
    try:
        s = clients.training.GetSamStatus(trn.Empty())
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    return _json({"available": s.available, "loaded": s.loaded, "message": s.message})


@training_bp.route("/api/v1/training/sam/load", methods=["POST"])
def training_sam_load():
    """POST /api/v1/training/sam/load — gateway relay.

    One labeling assistant at a time: the SAM3 cold load peaks near the
    Orin's whole memory, so the TensorRT labeling engine (inference-service)
    is dropped first, best-effort and status-probed.
    """
    try:
        if clients.model.GetLabelModelStatus(inf.Empty()).loaded:
            clients.model.UnloadLabelModel(inf.Empty())
    except grpc.RpcError as exc:
        logger.warning("Best-effort label-model unload before SAM load failed: %s", exc)
    try:
        # Cold load reads the multi-GB checkpoint; allow well beyond the
        # default control-call deadline.
        return _result(clients.training.LoadSam(trn.Empty(), timeout=300))
    except grpc.RpcError as exc:
        return _grpc_error(exc)


@training_bp.route("/api/v1/training/sam/unload", methods=["POST"])
def training_sam_unload():
    """POST /api/v1/training/sam/unload — gateway relay."""
    try:
        return _result(clients.training.UnloadSam(trn.Empty()))
    except grpc.RpcError as exc:
        return _grpc_error(exc)


@training_bp.route("/api/v1/training/sam/segment", methods=["POST"])
def training_sam_segment():
    """POST /api/v1/training/sam/segment — gateway relay."""
    body = request.get_json(silent=True) or {}
    image_id = body.get("image_id", "")
    dataset_id = body.get("dataset_id", "")
    if not image_id:
        return _json_error("'image_id' is required")
    if not dataset_id:
        return _json_error("'dataset_id' is required")
    points = body.get("points") or []
    if not isinstance(points, list) or not all(isinstance(p, dict) for p in points):
        return _json_error("'points' must be a list of objects")
    try:
        msg = trn.SamRequest(
            image_id=image_id,
            dataset_id=dataset_id,
            text_prompt=str(body.get("text_prompt", "") or ""),
            threshold=float(body.get("threshold", 0.0) or 0.0),
            points=[
                trn.Point(x=float(p.get("x", 0)), y=float(p.get("y", 0)),
                          positive=bool(p.get("positive", True)))
                for p in points
            ],
        )
    except (TypeError, ValueError):
        return _json_error("Malformed point entry or threshold")
    try:
        r = clients.training.SamSegment(msg, timeout=300)
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    if not r.success:
        return _json_error(r.message, 400)
    # Each box's mask as normalized rings, parallel to boxes ([] without one).
    rings: list = [[] for _ in r.boxes]
    for polygon in r.polygons:
        if polygon.instance < len(rings):
            rings[polygon.instance].append(_pairs(polygon.points))
    return _json({
        "boxes": [{"cx": b.cx, "cy": b.cy, "w": b.w, "h": b.h} for b in r.boxes],
        "scores": list(r.scores),
        "polygons": rings,
    })
