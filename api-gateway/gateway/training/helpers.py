# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Shared helpers for the training routes: JSON responses, gRPC error mapping
and message↔dict serializers.

The response/error primitives live in ``gateway.helpers``; this module keeps
only the training-specific serializers and a service-tagged error wrapper.
"""
import json
import logging

import grpc
from flask import Response

from ..grpc_clients import clients, inf, trn
from ..helpers import _grpc_error as _shared_grpc_error
from ..helpers import _json, _json_error  # noqa: F401  (re-exported for routes)

logger = logging.getLogger(__name__)


def _grpc_error(exc: grpc.RpcError) -> Response:
    """Map a training-service gRPC error to a JSON Response."""
    return _shared_grpc_error(exc, service="training")


def _result(r, ok_status=200) -> Response:
    """Map a training Result message to a JSON success/error Response."""
    if not r.success:
        return _json_error(r.message or "Operation failed (no detail reported)", 400)
    return _json({"status": "success", "message": r.message}, ok_status)


def _job_dict(job) -> dict:
    """Serialize a training job (status + metrics) to its JSON dict."""
    try:
        metrics = json.loads(job.metrics_json) if job.metrics_json else {}
    except ValueError:
        metrics = {}
    return {
        "job_id": job.job_id,
        "status": job.status or "idle",
        "progress": job.progress,
        "epoch": job.epoch,
        "total_epochs": job.total_epochs,
        "message": job.message,
        "error": job.error,
        "model_name": job.model_name,
        "conversion_job_id": job.conversion_job_id,
        "metrics": metrics,
        "started_at": job.started_at,
        "dataset_id": job.dataset_id,
        "federated": job.federated,
        "result_weights_id": job.result_weights_id,
        "base_model": getattr(job, "base_model", ""),
        "geometry": getattr(job, "geometry", ""),
    }


def _meta_dict(m) -> dict:
    """Serialize a dataset's metadata to its JSON dict."""
    return {
        "dataset_id": m.dataset_id,
        "name": m.name,
        "created_at": m.created_at,
        "cover_image_id": m.cover_image_id,
        "image_count": m.image_count,
        "labeled_count": m.labeled_count,
        "class_count": m.class_count,
        # A training-service that predates application types sends no task.
        "task": getattr(m, "task", "") or "detect",
    }


def _parse_named_boxes(raw: str) -> list:
    """Parse the multipart ``boxes`` field into NamedBox messages.

    Expects a JSON list of ``{"class_name", "x1", "y1", "x2", "y2"}``; raises
    ``ValueError`` with a client-facing message on any malformed entry.
    """
    try:
        data = json.loads(raw or "[]")
    except ValueError:
        raise ValueError("'boxes' must be valid JSON") from None
    if not isinstance(data, list):
        raise ValueError("'boxes' must be a JSON list")
    if not all(isinstance(b, dict) for b in data):
        raise ValueError("Malformed box entry")
    try:
        return [
            trn.NamedBox(class_name=str(b.get("class_name", "")),
                         x1=float(b.get("x1", 0)), y1=float(b.get("y1", 0)),
                         x2=float(b.get("x2", 0)), y2=float(b.get("y2", 0)))
            for b in data
        ]
    except (TypeError, ValueError):
        raise ValueError("Malformed box entry") from None


def _instance(value) -> int:
    """A polygon's ``instance`` (rings of one object share it): a non-negative int."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("'instance' must be a non-negative integer")
    return value


def _instances(entries: list) -> list:
    """Each polygon entry's ``instance``: its own when given, else a fresh id.

    An entry without one is an object of its own: the training-service
    normalizes the rings of one (instance, class) together, so defaulting
    every omitted id to 0 would union separate objects and cap them at
    eight rings. Fresh ids skip every explicit one.
    """
    explicit = [None if p.get("instance") is None else _instance(p["instance"])
                for p in entries]
    used = {value for value in explicit if value is not None}
    out, fresh = [], 0
    for value in explicit:
        if value is None:
            while fresh in used:
                fresh += 1
            value = fresh
            fresh += 1
        out.append(value)
    return out


def _flat_points(points) -> list:
    """``[[x, y], …]`` (at least 3 vertices) → the proto's flat ``x1, y1, x2, …`` list."""
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError("A polygon needs at least 3 [x, y] points")
    flat = []
    for point in points:
        if (not isinstance(point, (list, tuple)) or len(point) != 2
                or any(isinstance(v, bool) for v in point)):
            raise ValueError("Malformed polygon point")
        try:
            flat.extend(float(v) for v in point)
        except (TypeError, ValueError):
            raise ValueError("Malformed polygon point") from None
    return flat


def _pairs(flat) -> list:
    """The proto's flat point list → ``[[x, y], …]``, the JSON shape of a ring."""
    values = list(flat)
    return [[values[i], values[i + 1]] for i in range(0, len(values) - 1, 2)]


def _parse_named_polygons(raw: str) -> list:
    """Parse the multipart ``polygons`` field into NamedPolygon messages.

    Expects a JSON list of ``{"class_name", "points": [[x, y], …], "instance"}``
    (normalized on the uploaded image; ``instance``, optional, groups the
    rings of one object); raises ``ValueError`` with a client-facing message
    on any malformed entry.
    """
    try:
        data = json.loads(raw or "[]")
    except ValueError:
        raise ValueError("'polygons' must be valid JSON") from None
    if not isinstance(data, list) or not all(isinstance(p, dict) for p in data):
        raise ValueError("'polygons' must be a JSON list of objects")
    return [
        trn.NamedPolygon(class_name=str(p.get("class_name", "")),
                         instance=instance,
                         points=_flat_points(p.get("points")))
        for p, instance in zip(data, _instances(data), strict=True)
    ]


def _image_info_dict(info) -> dict:
    """Serialize an ImageInfo message: label state, box count, replica flag
    and the whole-image class of a classify dataset (``None`` otherwise)."""
    return {
        "image_id": info.image_id,
        "created_at": info.created_at,
        "labeled": info.labeled,
        "box_count": info.box_count,
        "replica": info.replica,
        # Presence, not the value: class 0 is a real class.
        "image_class": info.image_class if info.HasField("image_class") else None,
    }


def _labels_dict(labels) -> dict:
    """Serialize a Labels message: the boxes of a detect dataset, the polygon
    rings of a segment dataset and the whole-image class of a classify
    dataset (``None`` when it has none)."""
    return {
        "image_id": labels.image_id,
        "boxes": [
            {"class_id": b.class_id, "cx": b.cx, "cy": b.cy, "w": b.w, "h": b.h}
            for b in labels.boxes
        ],
        "polygons": [
            {"class_id": p.class_id, "instance": p.instance, "points": _pairs(p.points)}
            for p in labels.polygons
        ],
        # Presence, not the value: class 0 is a real class.
        "image_class": labels.image_class if labels.HasField("image_class") else None,
    }


def _labels_message(dataset_id: str, image_id: str, body) -> "trn.Labels":
    """Parse a label-editor PUT body into a Labels message.

    The body carries ``boxes`` (a list of center/size boxes, detect datasets),
    ``polygons`` (a list of ``{class_id, instance, points: [[x, y], …]}``
    rings, segment datasets) or ``image_class`` (a class index, or null for
    none, classify datasets). Which kind the dataset accepts is the
    training-service's check (against the dataset's task). Raises
    ``ValueError`` with a client-facing message.
    """
    if not isinstance(body, dict) or not any(
            key in body for key in ("boxes", "polygons", "image_class")):
        raise ValueError(
            "Body must contain a 'boxes' list, a 'polygons' list or an 'image_class'")
    boxes = body.get("boxes", [])
    if not isinstance(boxes, list):
        raise ValueError("Body must contain a 'boxes' list")
    if not all(isinstance(b, dict) for b in boxes):
        raise ValueError("Malformed box entry")
    polygons = body.get("polygons", [])
    if not isinstance(polygons, list) or not all(isinstance(p, dict) for p in polygons):
        raise ValueError("'polygons' must be a list of objects")
    image_class = body.get("image_class")
    if image_class is not None and (isinstance(image_class, bool)
                                    or not isinstance(image_class, int) or image_class < 0):
        raise ValueError("'image_class' must be a class index or null")
    try:
        msg = trn.Labels(dataset_id=dataset_id, image_id=image_id, boxes=[
            trn.Box(class_id=int(b.get("class_id", 0)),
                    cx=float(b.get("cx", 0)), cy=float(b.get("cy", 0)),
                    w=float(b.get("w", 0)), h=float(b.get("h", 0)))
            for b in boxes
        ])
    except (TypeError, ValueError):
        raise ValueError("Malformed box entry") from None
    for p, instance in zip(polygons, _instances(polygons), strict=True):
        points = _flat_points(p.get("points"))
        try:
            msg.polygons.append(trn.Polygon(class_id=int(p.get("class_id", 0)),
                                            instance=instance, points=points))
        except (TypeError, ValueError):
            raise ValueError("Malformed polygon entry") from None
    if image_class is not None:
        msg.image_class = image_class
    return msg


def _release_runtime() -> "inf.Result":
    """Release the inference GPU runtime (GPU handover to training)."""
    return clients.management.ReleaseRuntime(inf.Empty())


#: Training job statuses that own the GPU (the trainer subprocess is alive).
ACTIVE_JOB_STATUSES = ("preparing", "training", "uploading")
#: Conversion statuses meaning a TensorRT build may hold the GPU.
ACTIVE_CONVERSION_STATUSES = ("pending", "converting_to_onnx", "converting_to_engine")


class GpuProbeError(RuntimeError):
    """A strict GPU-busy probe could not reach the service it asks."""


def training_job_active(strict: bool = False) -> bool:
    """True while a training job holds the GPU (``ACTIVE_JOB_STATUSES``).

    An unreachable training-service cannot be training, so by default a gRPC
    failure answers False (logged): the callers gate detection on this and
    must not be blocked by a service that is merely down. ``strict`` callers
    (the application switch) refuse on an unanswered probe instead and get
    :class:`GpuProbeError`.
    """
    try:
        job = clients.training.GetTraining(trn.Empty())
    except grpc.RpcError as exc:
        if strict:
            raise GpuProbeError(f"training job probe failed: {exc.code()}") from exc
        logger.warning("Training job probe failed; assuming idle: %s", exc)
        return False
    return job.status in ACTIVE_JOB_STATUSES


def conversion_active(strict: bool = False) -> bool:
    """True while a model conversion (TensorRT build) may hold the GPU.

    Fail-open by default like :func:`training_job_active`; ``strict`` raises
    :class:`GpuProbeError` when the inference-service does not answer.
    """
    try:
        cl = clients.model.ListConversions(inf.Empty())
    except grpc.RpcError as exc:
        if strict:
            raise GpuProbeError(f"conversion probe failed: {exc.code()}") from exc
        logger.warning("Conversion probe failed; assuming idle: %s", exc)
        return False
    return any(j.status in ACTIVE_CONVERSION_STATUSES for j in cl.jobs)
