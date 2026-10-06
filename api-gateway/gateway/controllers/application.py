# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Application type controller: which application — object detection,
classification or segmentation — the device runs.

GET is open to every role; PUT is an admin action (``authz.ROUTE_POLICIES``),
audited as ``application.changed`` with the task as its detail. The switch is
the inference-service's transaction (``SetApplication``), after the shared GPU
release (``gateway.gpu``, which refuses while detection runs): it persists the
task, deselects a model of another task and publishes
``application_changed``, which the event relay forwards verbatim — so this
route never publishes it a second time.
"""
import grpc
from flask import request

from ..gpu import GpuBusy, Purpose, acquire_gpu, busy_response, gpu_lock
from ..grpc_clients import clients, inf
from ..helpers import _grpc_error, _json, _json_error
from . import api_bp


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
    # The switch drops the runtime and may deselect the active model, so the
    # GPU must be free: refused while detection, a training or a conversion
    # runs (detection is never stopped for it), and fail-closed when a probe
    # cannot answer. A training handover left behind ends here.
    task = task.strip()
    try:
        # The whole decision runs under the GPU lock: the current task is read
        # there (a concurrent switch cannot make the no-op stale), and no
        # conversion can be queued between the release and the switch.
        with gpu_lock():
            current = clients.management.GetApplication(inf.Empty())
            if current.task == task:
                # Choosing the current application changes nothing: no GPU needed.
                return _json(_application_dict(current))
            # A task the device cannot run is refused by SetApplication before
            # it touches anything (400 unknown, 409 unsupported); freeing the
            # GPU first would make that refusal cost the device its runtime.
            if task in current.supported_tasks:
                acquire_gpu(Purpose.APPLICATION)
            info = clients.management.SetApplication(inf.SetApplicationRequest(task=task))
    except GpuBusy as exc:
        return busy_response(exc)
    except grpc.RpcError as exc:
        # Freeing the GPU or writing the choice failed on the device: the
        # previous application is still in place.
        if exc.code() == grpc.StatusCode.INTERNAL:
            return _json_error(exc.details() or "Could not change the application type", 500)
        return _grpc_error(exc)
    return _json(_application_dict(info))
