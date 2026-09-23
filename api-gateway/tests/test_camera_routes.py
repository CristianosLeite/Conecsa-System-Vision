# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""GET /api/v1/camera/devices and POST /api/v1/camera/config.

The POST body can carry the remote camera's stream token, a secret the gateway
relays to the inference-service and must never keep: not in its response, not in
the SSE event, not in the audit trail and not in a log line. The route stays
admin-only and stays out of the audit detail fields on purpose.
"""
import json
import logging
from types import SimpleNamespace

import grpc
import pytest
from conftest import hub_request
from gateway import audit, audit_events, authz, helpers
from gateway.controllers import camera
from gateway.events import EventService

# The generated stubs are put on sys.path by gateway.grpc_clients itself, so
# they are taken from there rather than imported flat (which only works after
# some other test module happened to import the gateway first).
from gateway.grpc_clients import inf as inf_pb

ROUTE = "/api/v1/camera/config"
# Visibly fake stream token, as an operator types it and as it is normalized.
RAW_TOKEN = "test-t0ke-n123"
TOKEN = "TESTT0KEN123"
SECRETS = (RAW_TOKEN, TOKEN)
NETWORK = {"source": "network", "network_host": "192.0.2.1", "network_port": 8080,
           "network_token": RAW_TOKEN}

# What the inference-service reports: token-free by construction (its own suite
# proves that); the gateway relays it verbatim.
DEVICES = {"devices": [], "camera_status": "no_camera", "camera_detail": "unauthorized",
           "current_source": "network", "current_network_host": "192.0.2.1",
           "current_network_port": 8080, "network_token_set": True}


class FakeRpcError(grpc.RpcError):
    def __init__(self, code, details):
        self._code, self._details = code, details

    def code(self):
        return self._code

    def details(self):
        return self._details


@pytest.fixture
def wired(monkeypatch):
    """Stub inference-service plus a private event bus; returns what each saw."""
    seen = SimpleNamespace(updates=[], events=EventService(), fail=None)

    def update_camera(req):
        seen.updates.append(json.loads(req.json))
        if seen.fail is not None:
            raise seen.fail
        return inf_pb.Result(success=True, message="Camera configuration applied")

    fake = SimpleNamespace(management=SimpleNamespace(
        GetCamera=lambda _req: inf_pb.ConfigJson(json=json.dumps(DEVICES)),
        UpdateCamera=update_camera))
    monkeypatch.setattr(camera, "clients", fake)
    monkeypatch.setattr(helpers, "event_service", seen.events)
    return seen


def _events(seen):
    return seen.events.wait_for_changes(0, None, timeout=0)[1]


def _backlog():
    assert audit._buffer is not None, "real_app installs the audit buffer"
    return audit._buffer.list_backlog()


def _audit_text():
    return json.dumps(_backlog())


class TestPolicy:
    def test_the_route_is_admin_only(self):
        assert authz.ROUTE_POLICIES[("POST", ROUTE)] == authz.ROLE_ADMIN

    def test_a_non_admin_cannot_set_the_source(self, real_app, wired):
        resp = real_app.test_client().post(ROUTE, json=NETWORK, **hub_request(role="user"))
        assert resp.status_code == 403
        assert wired.updates == [], "the token never reached the backend"

    def test_the_body_is_never_an_audit_detail(self):
        # Record what was done, never what it was done with: adding this route
        # to the detail table would put the token in the trail.
        assert ROUTE not in audit_events._BODY_DETAIL_FIELDS
        assert audit_events.ROUTE_EVENTS[("POST", ROUTE)] == "camera.configured"


class TestRelay:
    def test_the_whole_body_reaches_the_inference_service(self, real_app, wired):
        resp = real_app.test_client().post(ROUTE, json=NETWORK, **hub_request())
        assert resp.status_code == 200
        assert wired.updates == [NETWORK]

    def test_devices_are_relayed_verbatim(self, real_app, wired):
        resp = real_app.test_client().get("/api/v1/camera/devices", **hub_request())
        assert resp.get_json() == DEVICES

    def test_the_change_is_announced_on_the_camera_key(self, real_app, wired):
        real_app.test_client().post(ROUTE, json=NETWORK, **hub_request())
        (event,) = _events(wired)
        assert event["type"] == "camera_config_changed"
        assert event["keys"] == ["camera"]

    def test_a_refused_update_announces_nothing(self, real_app, wired):
        wired.fail = FakeRpcError(grpc.StatusCode.INVALID_ARGUMENT,
                                  "network_token must be 8 to 32 Crockford base32 characters")
        resp = real_app.test_client().post(ROUTE, json=NETWORK, **hub_request())
        assert resp.status_code == 400
        assert _events(wired) == []


class TestTheTokenStaysOut:
    def _assert_clean(self, *texts):
        for text in texts:
            for secret in SECRETS:
                assert secret not in text

    def test_of_a_successful_update(self, real_app, wired, caplog):
        with caplog.at_level(logging.DEBUG):
            resp = real_app.test_client().post(ROUTE, json=NETWORK, **hub_request())
        assert resp.status_code == 200
        self._assert_clean(resp.get_data(as_text=True), json.dumps(_events(wired)),
                           _audit_text(), caplog.text)

        record = _backlog()["records"][0]
        assert record["event"] == "camera.configured"
        assert record["detail"] == "", "what was done, not what it was done with"

    @pytest.mark.parametrize("code,status", [
        (grpc.StatusCode.INVALID_ARGUMENT, 400),
        (grpc.StatusCode.UNAVAILABLE, 503),
        (grpc.StatusCode.INTERNAL, 502),
    ])
    def test_of_a_failed_update(self, real_app, wired, caplog, code, status):
        wired.fail = FakeRpcError(code, "Failed to persist the camera source")
        with caplog.at_level(logging.DEBUG):
            resp = real_app.test_client().post(ROUTE, json=NETWORK, **hub_request())
        assert resp.status_code == status
        self._assert_clean(resp.get_data(as_text=True), json.dumps(_events(wired)),
                           _audit_text(), caplog.text)

    def test_of_the_device_listing(self, real_app, wired, caplog):
        client = real_app.test_client()
        client.post(ROUTE, json=NETWORK, **hub_request())
        with caplog.at_level(logging.DEBUG):
            body = client.get("/api/v1/camera/devices", **hub_request()).get_data(as_text=True)
        self._assert_clean(body, caplog.text)
        assert "network_token" not in body.replace("network_token_set", "")
