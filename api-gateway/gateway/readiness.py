# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Backend readiness: the gRPC health service of every peer.

``/api/v1/health`` is a constant document — it proves only that Flask can
answer. This module asks each backend's standard ``grpc.health.v1`` service
whether it is serving, with a short deadline, so ``/api/v1/ready`` reflects
what the gateway can actually relay to.
"""
import logging
from typing import Dict

import grpc
from grpc_health.v1 import health_pb2

from .grpc_clients import clients

logger = logging.getLogger(__name__)

SERVING = "SERVING"
_PROBE_TIMEOUT_S = 2.0


def _probe(stub) -> str:
    """One health check; the gRPC status name when the check itself fails."""
    try:
        response = stub.Check(health_pb2.HealthCheckRequest(service=""),
                              timeout=_PROBE_TIMEOUT_S)
    except grpc.RpcError as exc:
        code = exc.code() if hasattr(exc, "code") else None
        return code.name if code is not None else "UNAVAILABLE"
    return health_pb2.HealthCheckResponse.ServingStatus.Name(response.status)


def backends_status() -> Dict[str, str]:
    """``{"inference": ..., "training": ..., "hardware": ...}`` serving states."""
    return {
        "inference": _probe(clients.inference_health),
        "training": _probe(clients.training_health),
        "hardware": _probe(clients.hardware_health),
    }
