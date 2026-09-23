# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Dataset registry routes: list/create/upload, per-dataset get/rename/delete,
ZIP export (full or federated shard), cover image and the class list."""
import grpc
from flask import Response, request

from ..grpc_clients import clients, inf, trn
from ..helpers import _grpc_error as _inference_grpc_error
from . import training_bp
from .helpers import _grpc_error, _json, _json_error, _meta_dict, _result


def _device_task():
    """The device's application task for a new dataset, or an error Response.

    A dataset is labeled for one task, fixed at creation from the device's
    application. An inference-service that predates
    application types (UNIMPLEMENTED during a rolling upgrade) is object
    detection.
    """
    try:
        info = clients.management.GetApplication(inf.Empty())
    except grpc.RpcError as exc:
        if exc.code() == grpc.StatusCode.UNIMPLEMENTED:
            return "detect", None
        return None, _inference_grpc_error(exc)
    if not info.task:
        return None, _json_error("No application type is selected; an administrator must "
                                 "choose one before creating a dataset.", 409)
    return info.task, None


@training_bp.route("/api/v1/training/datasets", methods=["GET"])
def training_datasets():
    """GET /api/v1/training/datasets — gateway relay."""
    try:
        lst = clients.training.ListDatasets(trn.Empty())
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    return _json({"datasets": [_meta_dict(m) for m in lst.datasets]})


@training_bp.route("/api/v1/training/datasets", methods=["POST"])
def training_dataset_create():
    """POST /api/v1/training/datasets — gateway relay."""
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    if not name:
        return _json_error("'name' is required")
    task, error = _device_task()
    if error is not None:
        return error
    requested = body.get("task")
    if isinstance(requested, str) and requested.strip() and requested.strip() != task:
        return _json_error(f"This device runs the '{task}' application; a "
                           f"'{requested.strip()}' dataset cannot be created here.", 409)
    try:
        m = clients.training.CreateDataset(trn.DatasetName(name=name, task=task))
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    return _json(_meta_dict(m), 201)


@training_bp.route("/api/v1/training/datasets/upload", methods=["POST"])
def training_dataset_upload():
    """POST /api/v1/training/datasets/upload — gateway relay."""
    if "file" not in request.files:
        return _json_error("No file provided")
    name = (request.form.get("name") or "").strip()
    if not name:
        return _json_error("'name' is required")
    # The archive must be laid out for the device's task, which the new
    # dataset then records.
    task, error = _device_task()
    if error is not None:
        return error
    file = request.files["file"]

    def stream():
        """Yield DatasetUploadChunk messages (metadata first, then ZIP chunks)."""
        yield trn.DatasetUploadChunk(meta=trn.DatasetUploadMeta(name=name, task=task))
        while True:
            chunk = file.stream.read(1 << 20)
            if not chunk:
                break
            yield trn.DatasetUploadChunk(chunk=chunk)

    try:
        # The import re-encodes every image on the Jetson; allow well beyond
        # the default control-call deadline.
        r = clients.training.UploadDataset(stream(), timeout=600)
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    if not r.success:
        return _json_error(r.message, 400)
    return _json({"status": "success", "message": r.message,
                  "dataset": _meta_dict(r.dataset)}, 201)


@training_bp.route("/api/v1/training/datasets/<dataset_id>", methods=["GET"])
def training_dataset(dataset_id):
    """GET /api/v1/training/datasets/<dataset_id> — gateway relay."""
    try:
        d = clients.training.GetDataset(trn.DatasetId(dataset_id=dataset_id))
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    return _json({
        "image_count": d.image_count,
        "labeled_count": d.labeled_count,
        "classes": list(d.classes),
        "min_images": d.min_images,
        "dataset_id": d.dataset_id,
        "name": d.name,
        "cover_image_id": d.cover_image_id,
        "task": d.task or "detect",
    })


@training_bp.route("/api/v1/training/datasets/<dataset_id>", methods=["PUT"])
def training_dataset_rename(dataset_id):
    """PUT /api/v1/training/datasets/<dataset_id> — gateway relay."""
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    if not name:
        return _json_error("'name' is required")
    try:
        m = clients.training.RenameDataset(
            trn.DatasetRename(dataset_id=dataset_id, name=name))
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    return _json(_meta_dict(m))


@training_bp.route("/api/v1/training/datasets/<dataset_id>", methods=["DELETE"])
def training_dataset_delete(dataset_id):
    """DELETE /api/v1/training/datasets/<dataset_id> — gateway relay."""
    try:
        return _result(clients.training.DeleteDataset(
            trn.DatasetId(dataset_id=dataset_id)))
    except grpc.RpcError as exc:
        return _grpc_error(exc)


@training_bp.route("/api/v1/training/datasets/<dataset_id>/export", methods=["GET"])
def training_dataset_export(dataset_id):
    # Resolve the dataset name first (and 404 early on unknown ids) — the
    # streamed body can't change status once it starts.
    """GET /api/v1/training/datasets/<dataset_id>/export — gateway relay.

    With ``?shards=N&index=I[&seed=S]`` exports one deterministic IID shard
    (federated training) instead of the full dataset.
    """
    try:
        d = clients.training.GetDataset(trn.DatasetId(dataset_id=dataset_id))
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    filename = f"{d.name}.zip"
    if request.args.get("shards"):
        try:
            num_shards = int(request.args["shards"])
            shard_index = int(request.args.get("index") or 0)
        except ValueError:
            return _json_error("'shards' and 'index' must be integers")
        stream = clients.training.ExportDatasetShard(trn.ShardExportRequest(
            dataset_id=dataset_id, num_shards=num_shards,
            shard_index=shard_index, seed=request.args.get("seed") or "",
        ), timeout=600)
        filename = f"{d.name}-shard-{shard_index}.zip"
    else:
        stream = clients.training.ExportDataset(
            trn.DatasetId(dataset_id=dataset_id), timeout=600)
    # Pull the first chunk before answering so an INVALID_ARGUMENT (shard
    # geometry) or FAILED_PRECONDITION (a face dataset never leaves the
    # device) can still set the HTTP status (download_model precedent).
    try:
        first = next(stream, None)
    except grpc.RpcError as exc:
        return _grpc_error(exc)

    def generate():
        """Yield the ZIP bytes relayed from the training-service."""
        if first is not None:
            yield first.chunk
        for msg in stream:
            yield msg.chunk

    # Dataset names are validated server-side (ASCII letters/digits/space/
    # _-.) so they are safe inside a quoted filename.
    return Response(
        generate(),
        mimetype="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


@training_bp.route("/api/v1/training/datasets/<dataset_id>/cover", methods=["PUT"])
def training_dataset_cover(dataset_id):
    """PUT /api/v1/training/datasets/<dataset_id>/cover — gateway relay."""
    body = request.get_json(silent=True) or {}
    image_id = (body.get("image_id") or "").strip()
    if not image_id:
        return _json_error("'image_id' is required")
    try:
        return _result(clients.training.SetDatasetCover(
            trn.DatasetCover(dataset_id=dataset_id, image_id=image_id)))
    except grpc.RpcError as exc:
        return _grpc_error(exc)


# ── classes ────────────────────────────────────────────────────────────────────

@training_bp.route("/api/v1/training/datasets/<dataset_id>/classes", methods=["GET"])
def training_classes(dataset_id):
    """GET /api/v1/training/datasets/<dataset_id>/classes — gateway relay."""
    try:
        c = clients.training.GetClasses(trn.DatasetId(dataset_id=dataset_id))
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    return _json({"classes": list(c.classes)})


@training_bp.route("/api/v1/training/datasets/<dataset_id>/classes", methods=["POST"])
def training_class_add(dataset_id):
    """POST /api/v1/training/datasets/<dataset_id>/classes — gateway relay."""
    body = request.get_json(silent=True) or {}
    name = body.get("name", "")
    try:
        c = clients.training.AddClass(
            trn.ClassName(dataset_id=dataset_id, name=name))
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    return _json({"classes": list(c.classes)})


@training_bp.route("/api/v1/training/datasets/<dataset_id>/classes/<int:index>",
                   methods=["PUT"])
def training_class_rename(dataset_id, index):
    """PUT /api/v1/training/datasets/<dataset_id>/classes/<int:index> — gateway relay."""
    body = request.get_json(silent=True) or {}
    name = body.get("name", "")
    try:
        c = clients.training.RenameClass(
            trn.ClassEdit(dataset_id=dataset_id, index=index, name=name))
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    return _json({"classes": list(c.classes)})


@training_bp.route("/api/v1/training/datasets/<dataset_id>/classes/<int:index>",
                   methods=["DELETE"])
def training_class_remove(dataset_id, index):
    """DELETE /api/v1/training/datasets/<dataset_id>/classes/<int:index> — gateway relay."""
    try:
        c = clients.training.RemoveClass(
            trn.ClassIndex(dataset_id=dataset_id, index=index))
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    return _json({"classes": list(c.classes)})
