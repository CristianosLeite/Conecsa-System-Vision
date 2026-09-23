# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Per-image routes: camera capture, labeled-image upload, listing, JPEG
retrieval, deletion, replication and the YOLO label editor."""
import grpc
from flask import Response, request

from ..config import settings
from ..grpc_clients import clients, trn
from . import training_bp
from .helpers import (
    _grpc_error,
    _image_info_dict,
    _json,
    _json_error,
    _labels_dict,
    _labels_message,
    _parse_named_boxes,
    _parse_named_polygons,
    _result,
)


@training_bp.route("/api/v1/training/datasets/<dataset_id>/capture", methods=["POST"])
def training_capture(dataset_id):
    """POST /api/v1/training/datasets/<dataset_id>/capture — gateway relay."""
    try:
        info = clients.training.CaptureImage(trn.DatasetId(dataset_id=dataset_id))
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    return _json(_image_info_dict(info))


@training_bp.route("/api/v1/training/datasets/<dataset_id>/images", methods=["POST"])
def training_image_add(dataset_id):
    """POST /api/v1/training/datasets/<dataset_id>/images — gateway relay."""
    if "file" not in request.files:
        return _json_error("No file provided")
    try:
        boxes = _parse_named_boxes(request.form.get("boxes", "[]"))
        # A segment dataset's pre-labels: rings by class name.
        polygons = _parse_named_polygons(request.form.get("polygons", "[]"))
    except ValueError as exc:
        return _json_error(str(exc))
    # A classify dataset's pre-label: the image's class by name ("" = none).
    image_class = (request.form.get("image_class") or "").strip()
    # Bounded read: the image travels as one gRPC message, so an oversized
    # body must be refused here rather than buffered whole and rejected later.
    cap = settings.MAX_IMAGE_UPLOAD_BYTES
    jpeg = request.files["file"].stream.read(cap + 1)
    if len(jpeg) > cap:
        return _json_error(
            f"Image exceeds the {cap} byte limit", 413)
    try:
        info = clients.training.AddDatasetImage(trn.LabeledImageUpload(
            dataset_id=dataset_id,
            jpeg=jpeg,
            boxes=boxes,
            polygons=polygons,
            image_class=image_class,
        ))
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    return _json(_image_info_dict(info), 201)


@training_bp.route("/api/v1/training/datasets/<dataset_id>/images", methods=["GET"])
def training_images(dataset_id):
    """GET /api/v1/training/datasets/<dataset_id>/images — gateway relay."""
    try:
        lst = clients.training.ListImages(trn.DatasetId(dataset_id=dataset_id))
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    return _json({"images": [_image_info_dict(i) for i in lst.images]})


@training_bp.route("/api/v1/training/datasets/<dataset_id>/images/<image_id>",
                   methods=["GET"])
def training_image(dataset_id, image_id):
    """GET /api/v1/training/datasets/<dataset_id>/images/<image_id> — gateway relay."""
    try:
        blob = clients.training.GetImage(
            trn.ImageId(dataset_id=dataset_id, image_id=image_id))
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    return Response(blob.jpeg, mimetype="image/jpeg",
                    headers={"Cache-Control": "max-age=3600"})


@training_bp.route("/api/v1/training/datasets/<dataset_id>/images/<image_id>",
                   methods=["DELETE"])
def training_image_delete(dataset_id, image_id):
    """DELETE /api/v1/training/datasets/<dataset_id>/images/<image_id> — gateway relay."""
    try:
        return _result(clients.training.DeleteImage(
            trn.ImageId(dataset_id=dataset_id, image_id=image_id)))
    except grpc.RpcError as exc:
        return _grpc_error(exc)


@training_bp.route("/api/v1/training/datasets/<dataset_id>/images/<image_id>/replicate",
                   methods=["POST"])
def training_image_replicate(dataset_id, image_id):
    """POST /api/v1/training/datasets/<dataset_id>/images/<image_id>/replicate — gateway relay."""
    body = request.get_json(silent=True) or {}
    try:
        count = int(body.get("count", 1))
    except (TypeError, ValueError):
        return _json_error("'count' must be an integer")
    if not 1 <= count <= 50:
        return _json_error("'count' must be between 1 and 50")
    try:
        return _result(clients.training.ReplicateImage(
            trn.ReplicateRequest(dataset_id=dataset_id, image_id=image_id, count=count)))
    except grpc.RpcError as exc:
        return _grpc_error(exc)


# ── labels ─────────────────────────────────────────────────────────────────────

@training_bp.route("/api/v1/training/datasets/<dataset_id>/images/<image_id>/labels",
                   methods=["GET"])
def training_labels_get(dataset_id, image_id):
    """GET /api/v1/training/datasets/<dataset_id>/images/<image_id>/labels — gateway relay."""
    try:
        labels = clients.training.GetLabels(
            trn.ImageId(dataset_id=dataset_id, image_id=image_id))
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    return _json(_labels_dict(labels))


@training_bp.route("/api/v1/training/datasets/<dataset_id>/images/<image_id>/labels",
                   methods=["PUT"])
def training_labels_put(dataset_id, image_id):
    """PUT /api/v1/training/datasets/<dataset_id>/images/<image_id>/labels — gateway relay."""
    body = request.get_json(silent=True) or {}
    try:
        msg = _labels_message(dataset_id, image_id, body)
    except ValueError as exc:
        return _json_error(str(exc))
    try:
        return _result(clients.training.SetLabels(msg))
    except grpc.RpcError as exc:
        return _grpc_error(exc)
