# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""SAM (Segment Anything) assisted-labeling routes: status, load/unload of the
checkpoint and point/text-prompted segmentation."""
import grpc
from flask import request

from ..gpu import GpuBusy, Purpose, acquire_gpu, busy_response, gpu_lock, gpu_session
from ..grpc_clients import clients, trn
from . import training_bp
from .helpers import _grpc_error, _json, _json_error, _pairs, _result


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

    The SAM3 cold load peaks near the Orin's whole memory, so the GPU is
    freed first through the shared release (``Purpose.SAM``): the labeling
    engine and every idle TensorRT worker go; refused while detection, a
    training or a conversion runs. The gateway GPU lock is held through the
    load, so no Start can begin between the release and the load.
    """
    try:
        with gpu_session(Purpose.SAM):
            # Cold load reads the multi-GB checkpoint; allow well beyond the
            # default control-call deadline.
            return _result(clients.training.LoadSam(trn.Empty(), timeout=300))
    except GpuBusy as exc:
        return busy_response(exc)
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
        # Under the GPU lock, so a Start cannot evict SAM3 (and a segment then
        # reload it lazily beside detection) between the check and the
        # segment. A segment that loads SAM3 acquires the GPU like /sam/load.
        with gpu_lock():
            if not clients.training.GetSamStatus(trn.Empty()).loaded:
                acquire_gpu(Purpose.SAM)
            r = clients.training.SamSegment(msg, timeout=300)
    except GpuBusy as exc:
        return busy_response(exc)
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
