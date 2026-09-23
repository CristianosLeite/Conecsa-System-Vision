# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for the app-level error handlers (sanitized 500s, preserved HTTP
semantics)."""
import logging
from types import SimpleNamespace

import grpc
import pytest
from gateway import grpc_clients
from gateway.config import settings


class _Refused(grpc.RpcError):
    """A FAILED_PRECONDITION the gateway relays as 409."""

    def code(self):
        return grpc.StatusCode.FAILED_PRECONDITION

    def details(self):
        return "A training handover is in progress"


class TestRefusalLog:
    def test_a_refused_mutation_logs_its_reason(self, real_app, monkeypatch, caplog):
        def refuse(*args, **kwargs):
            raise _Refused()

        monkeypatch.setattr(grpc_clients.clients, "detection", SimpleNamespace(ResetStats=refuse))
        with caplog.at_level(logging.WARNING, logger="gateway.app"):
            resp = real_app.test_client().post("/api/v1/stats/reset")
        assert resp.status_code == 409
        assert ("refused POST /api/v1/stats/reset -> 409: "
                "A training handover is in progress") in caplog.text

    def test_a_protobuf_refusal_logs_its_reason(self, real_app, monkeypatch, caplog):
        # Native clients get protobuf 4xx bodies; the log must still name why.
        monkeypatch.setattr(grpc_clients.clients, "detection",
                            SimpleNamespace(GetStatus=lambda _req: SimpleNamespace(is_running=True)))
        with caplog.at_level(logging.WARNING, logger="gateway.app"):
            resp = real_app.test_client().post(
                "/api/v1/start", headers={"Accept": "application/x-protobuf"})
        assert resp.status_code == 400
        assert resp.mimetype == "application/x-protobuf"
        assert "refused POST /api/v1/start -> 400: Detection already running" in caplog.text

    def test_a_missing_route_is_not_logged(self, real_app, caplog):
        with caplog.at_level(logging.WARNING, logger="gateway.app"):
            assert real_app.test_client().post("/api/v1/no-such-route").status_code == 404
        assert "refused" not in caplog.text


@pytest.fixture
def broken_backend(monkeypatch):
    """A backend whose calls raise a non-gRPC error (a genuine bug)."""

    def boom(*args, **kwargs):
        raise ValueError("secret internal state: /data/models")

    monkeypatch.setattr(grpc_clients.clients, "detection",
                        SimpleNamespace(GetStatus=boom, ResetStats=boom))


class TestCatchAll:
    def test_a_500_reveals_nothing(self, real_app, broken_backend):
        resp = real_app.test_client().post("/api/v1/stats/reset")
        assert resp.status_code == 500
        body = resp.get_json()
        assert body == {"error": "Internal server error"}

    def test_debug_mode_restores_the_detail(self, real_app, broken_backend,
                                            monkeypatch):
        monkeypatch.setattr(settings, "DEBUG_ERRORS", True)
        resp = real_app.test_client().post("/api/v1/stats/reset")
        body = resp.get_json()
        assert body["type"] == "ValueError"
        assert "secret internal state" in body["message"]


class TestHttpSemantics:
    def test_an_oversized_body_stays_413(self, real_app, monkeypatch):
        # Used to fall into the catch-all and become a 500. The limit fires
        # when the handler reads the body, so use a JSON-parsing route.
        monkeypatch.setitem(real_app.config, "MAX_CONTENT_LENGTH", 1024)
        resp = real_app.test_client().post("/api/v1/model/select",
                                           data=b"x" * 4096,
                                           content_type="application/json")
        assert resp.status_code == 413
        assert resp.get_json()["error"] == "Request Entity Too Large"

    def test_a_wrong_method_stays_405(self, real_app):
        resp = real_app.test_client().delete("/api/v1/stats/reset")
        assert resp.status_code == 405
        assert resp.get_json()["error"] == "Method Not Allowed"

    def test_unknown_routes_keep_the_dedicated_404_body(self, real_app):
        resp = real_app.test_client().get("/api/v1/nope")
        assert resp.status_code == 404
        assert resp.get_json() == {"error": "Route not found"}
