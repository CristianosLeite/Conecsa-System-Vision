# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""POST /api/v1/network/config: type errors are 400s, not 500s — and the
hardware agent's gRPC errors map like every other backend's."""
import grpc
import pytest
from flask import Flask
from gateway import hardware
from gateway.controllers import api_bp


class FakeRpcError(grpc.RpcError):
    def __init__(self, code, details):
        self._code, self._details = code, details

    def code(self):
        return self._code

    def details(self):
        return self._details


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(api_bp)
    return app.test_client()


@pytest.fixture
def agent(monkeypatch):
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return {"success": True, "message": "ok"}

    monkeypatch.setattr(hardware, "set_network_config", fake)
    return calls


@pytest.mark.parametrize("body,fragment", [
    ({}, "method"),
    ({"method": "manual"}, "auto"),
    ({"method": "static", "address": "10.0.0.2", "prefix": "twenty"}, "prefix"),
    ({"method": "static", "address": "10.0.0.2", "prefix": 24, "dns": "1.1.1.1"}, "dns"),
    ({"method": "static", "address": "10.0.0.2", "prefix": 24, "dns": [1]}, "dns"),
])
def test_malformed_bodies_are_400(client, agent, body, fragment):
    resp = client.post("/api/v1/network/config", json=body)
    assert resp.status_code == 400
    assert fragment in resp.get_json()["error"]
    assert agent == []


def test_a_valid_body_is_relayed_with_an_int_prefix(client, agent):
    resp = client.post("/api/v1/network/config", json={
        "interface": "wifi", "method": "static", "address": "10.0.0.2",
        "prefix": "24", "gateway": "10.0.0.1", "dns": ["1.1.1.1"]})
    assert resp.status_code == 200
    assert agent == [{"interface": "wifi", "method": "static", "address": "10.0.0.2",
                      "prefix": 24, "gateway": "10.0.0.1", "dns": ["1.1.1.1"]}]


class TestAgentErrors:
    """A Wi-Fi connect that fails inside the agent answers by the gRPC status,
    and an infrastructure failure never echoes the agent's address."""

    @pytest.fixture
    def failing(self, monkeypatch):
        holder = {}

        def connect_wifi(ssid, password):
            raise holder["exc"]

        monkeypatch.setattr(hardware, "connect_wifi", connect_wifi)
        return holder

    @pytest.mark.parametrize("code,status", [
        (grpc.StatusCode.UNAVAILABLE, 503),
        (grpc.StatusCode.DEADLINE_EXCEEDED, 504),
        (grpc.StatusCode.INTERNAL, 502),
    ])
    def test_infrastructure_failures_hide_the_agent_address(self, client, failing, code, status):
        failing["exc"] = FakeRpcError(code, "failed to connect to os-base:50051")
        resp = client.post("/api/v1/network/wifi/connect", json={"ssid": "site"})
        assert resp.status_code == status
        assert "50051" not in resp.get_data(as_text=True)

    def test_a_refusal_from_the_agent_is_a_409_with_its_reason(self, client, failing):
        failing["exc"] = FakeRpcError(grpc.StatusCode.FAILED_PRECONDITION,
                                      "The access point is active; stop it before changing Wi-Fi")
        resp = client.post("/api/v1/network/wifi/connect", json={"ssid": "site"})
        assert resp.status_code == 409
        assert "stop it before changing Wi-Fi" in resp.get_json()["error"]
