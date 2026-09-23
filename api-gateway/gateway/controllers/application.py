# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Application type controller: which application — object detection,
classification or segmentation — the device runs.

GET is open to every role; PUT is an admin action (``authz.ROUTE_POLICIES``),
audited as ``application.changed`` with the task as its detail. The switch is
the inference-service's transaction (``SetApplication``): it stops detection,
persists the task, deselects a model of another task and publishes
``application_changed``, which the event relay forwards verbatim — so this
route never publishes it a second time.
"""
import logging

import grpc
from flask import request

from ..grpc_clients import clients, inf
from ..helpers import _grpc_error, _json, _json_error
from ..training.helpers import GpuProbeError, conversion_active, training_job_active
from . import api_bp

logger = logging.getLogger(__name__)


def _application_dict(info) -> dict:
    """ApplicationInfo as JSON: an empty task (none chosen yet) is ``null``."""
    return {
        "task": info.task or None,
        "supported_tasks": list(info.supported_tasks),
        "migrated": bool(info.migrated),
    }


@api_bp.route('/api/v1/application', methods=['GET'])
def get_application():
    """GET /api/v1/application — the device's application type."""
    try:
        info = clients.management.GetApplication(inf.Empty())
    except grpc.RpcError as exc:
        return _grpc_error(exc)
    return _json(_application_dict(info))


@api_bp.route('/api/v1/application', methods=['PUT'])
def set_application():
    """PUT /api/v1/application — switch the application type (admin)."""
    body = request.get_json(silent=True)
    task = body.get("task") if isinstance(body, dict) else None
    if not isinstance(task, str) or not task.strip():
        return _json_error("'task' is required")
    # The switch stops detection and may deselect the active model, so the
    # GPU must be idle. Unlike /start, a probe that cannot answer refuses the
    # request (fail-closed): an unverifiable GPU is not an idle one.
    try:
        if training_job_active(strict=True):
            return _json_error("A model training is in progress; change the application "
                               "type after it finishes.", 409)
        if conversion_active(strict=True):
            return _json_error("A model conversion is in progress; change the application "
                               "type after it finishes.", 409)
    except GpuProbeError as exc:
        logger.warning("Application change refused: %s", exc)
        return _json_error("Cannot verify that the GPU is idle; try again when the training "
                           "and inference services are reachable.", 503)
    try:
        info = clients.management.SetApplication(
            inf.SetApplicationRequest(task=task.strip()))
    except grpc.RpcError as exc:
        # Stopping detection or writing the choice failed on the device: the
        # previous application is still in place.
        if exc.code() == grpc.StatusCode.INTERNAL:
            return _json_error(exc.details() or "Could not change the application type", 500)
        return _grpc_error(exc)
    return _json(_application_dict(info))
