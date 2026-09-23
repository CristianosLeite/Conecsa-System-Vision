# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""GET /api/v1/ready reflects the backends' gRPC health."""
from types import SimpleNamespace

import grpc
import pytest
from flask import Flask
from gateway import grpc_clients
from gateway.controllers import api_bp
from grpc_health.v1 import health_pb2


class FakeHealth:
    def __init__(self, status=None, error=None):
        self._status = status
        self._error = error
        self.timeouts = []

    def Check(self, request, timeout=None):
        self.timeouts.append(timeout)
        if self._error is not None:
            raise self._error
        return health_pb2.HealthCheckResponse(status=self._status)


class _Down(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.UNAVAILABLE


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(api_bp)
    return app.test_client()


def _install(monkeypatch, inference, training, hardware):
    monkeypatch.setattr(grpc_clients.clients, "inference_health", inference)
    monkeypatch.setattr(grpc_clients.clients, "training_health", training)
    monkeypatch.setattr(grpc_clients.clients, "hardware_health", hardware)


SERVING = health_pb2.HealthCheckResponse.SERVING
NOT_SERVING = health_pb2.HealthCheckResponse.NOT_SERVING


def test_ready_when_every_backend_serves(client, monkeypatch):
    inference = FakeHealth(SERVING)
    _install(monkeypatch, inference, FakeHealth(SERVING), FakeHealth(SERVING))
    resp = client.get("/api/v1/ready")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ready"
    assert body["backends"] == {"inference": "SERVING", "training": "SERVING",
                                "hardware": "SERVING"}
    # The probe itself is bounded: a hung backend must not hang the probe.
    assert inference.timeouts == [2.0]


def test_degraded_when_inference_is_down(client, monkeypatch):
    _install(monkeypatch, FakeHealth(error=_Down()), FakeHealth(SERVING), FakeHealth(SERVING))
    resp = client.get("/api/v1/ready")
    assert resp.status_code == 503
    assert resp.get_json()["status"] == "degraded"
    assert resp.get_json()["backends"]["inference"] == "UNAVAILABLE"


def test_missing_hardware_agent_is_reported_but_does_not_gate(client, monkeypatch):
    # The x86 development stack has no hardware agent.
    _install(monkeypatch, FakeHealth(SERVING), FakeHealth(NOT_SERVING), FakeHealth(error=_Down()))
    resp = client.get("/api/ready")
    assert resp.status_code == 200
    assert resp.get_json()["backends"] == {"inference": "SERVING", "training": "NOT_SERVING",
                                           "hardware": "UNAVAILABLE"}


def test_health_stays_a_constant_liveness_document(client, monkeypatch):
    _install(monkeypatch, FakeHealth(error=_Down()), FakeHealth(error=_Down()),
             FakeHealth(error=_Down()))
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "healthy"


def test_the_stubs_exist_on_the_real_clients():
    for name in ("inference_health", "training_health", "hardware_health"):
        assert hasattr(grpc_clients.clients, name)
    assert isinstance(SimpleNamespace(), SimpleNamespace)
