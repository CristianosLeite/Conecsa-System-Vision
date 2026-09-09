"""POST /api/v1/network/config: type errors are 400s, not 500s (review H2)."""
import pytest
from flask import Flask
from gateway import hardware
from gateway.controllers import api_bp


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
