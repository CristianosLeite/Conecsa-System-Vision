# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""GET /api/v1/network/ap and POST /api/v1/network/ap/{start,stop}.

The start body carries the access point passphrase, a secret the gateway relays
to the hardware agent and must never keep: not in a response, not in the SSE
event, not in the audit trail, not in a log line. Both mutating routes are
admin-only and neither has an audit detail field.
"""
import json
import logging
from types import SimpleNamespace

import grpc
import pytest
from conftest import hub_request
from gateway import audit, audit_events, authz, hardware, helpers
from gateway.controllers import network
from gateway.events import EventService

START, STOP, STATUS = "/api/v1/network/ap/start", "/api/v1/network/ap/stop", "/api/v1/network/ap"
PASSPHRASE = "test-passphrase-not-real"
STATUS_BODY = {"active": True, "ssid": "conecsa-000001", "frequency_mhz": 5180, "address": "10.98.76.1",
               "prefix": 24, "stations": [{"address": "10.98.76.11", "hostname": "pixel", "signal": -50}],
               "join_deadline_remaining_secs": 0, "wired_ready": True, "message": "Access point started",
               "channels": [44]}


class FakeRpcError(grpc.RpcError):
    def __init__(self, code, details):
        self._code, self._details = code, details

    def code(self):
        return self._code

    def details(self):
        return self._details


@pytest.fixture
def wired(monkeypatch):
    seen = SimpleNamespace(starts=[], stops=0, events=EventService(), fail=None, status=dict(STATUS_BODY))

    def start_ap(ssid, passphrase, channel):
        seen.starts.append((ssid, passphrase, channel))
        if seen.fail is not None:
            raise seen.fail
        return {"success": True, "message": "Access point started"}

    def stop_ap():
        seen.stops += 1
        return {"success": True, "message": "Access point stopped"}

    monkeypatch.setattr(hardware, "start_ap", start_ap)
    monkeypatch.setattr(hardware, "stop_ap", stop_ap)
    monkeypatch.setattr(hardware, "get_ap_status", lambda: dict(seen.status))
    monkeypatch.setattr(network, "device_id", lambda: "conecsa-000001")
    monkeypatch.setattr(helpers, "event_service", seen.events)
    return seen


def _events(seen):
    return seen.events.wait_for_changes(0, None, timeout=0)[1]


def _backlog():
    assert audit._buffer is not None, "real_app installs the audit buffer"
    return audit._buffer.list_backlog()


class TestPolicy:
    def test_both_mutating_routes_are_admin_only(self):
        assert authz.ROUTE_POLICIES[("POST", START)] == authz.ROLE_ADMIN
        assert authz.ROUTE_POLICIES[("POST", STOP)] == authz.ROLE_ADMIN

    def test_a_non_admin_cannot_start_the_access_point(self, real_app, wired):
        resp = real_app.test_client().post(START, json={"passphrase": PASSPHRASE}, **hub_request(role="user"))
        assert resp.status_code == 403 and wired.starts == []

    def test_the_start_body_is_never_an_audit_detail(self):
        assert START not in audit_events._BODY_DETAIL_FIELDS
        assert audit_events.ROUTE_EVENTS[("POST", START)] == "network.ap_started"
        assert audit_events.ROUTE_EVENTS[("POST", STOP)] == "network.ap_stopped"


class TestRelay:
    def test_start_uses_the_device_id_as_ssid_and_leaves_the_channel_to_the_agent(self, real_app, wired):
        resp = real_app.test_client().post(START, json={"passphrase": PASSPHRASE}, **hub_request())
        assert resp.status_code == 200
        assert wired.starts == [("conecsa-000001", PASSPHRASE, 0)], "0 = automatic"
        (event,) = _events(wired)
        assert event["type"] == "network_config_changed" and event["keys"] == ["network"]

    @pytest.mark.parametrize("channel", ["44", 48, 0])
    def test_an_explicit_channel_is_relayed(self, real_app, wired, channel):
        real_app.test_client().post(START, json={"passphrase": PASSPHRASE, "channel": channel}, **hub_request())
        assert wired.starts[0][2] == int(channel)

    @pytest.mark.parametrize("body,fragment", [
        ({}, "passphrase"),
        ({"passphrase": "short"}, "passphrase"),
        ({"passphrase": "p" * 64}, "passphrase"),
        ({"passphrase": 12345678}, "passphrase"),
        ({"passphrase": PASSPHRASE, "channel": "five"}, "channel"),
        ({"passphrase": PASSPHRASE, "channel": 52}, "channel"),
        ({"passphrase": PASSPHRASE, "channel": -1}, "channel"),
        ("a string, not an object", "passphrase"),
        ([{"passphrase": PASSPHRASE}], "passphrase"),
    ])
    def test_malformed_bodies_are_400_and_reach_no_agent(self, real_app, wired, body, fragment):
        resp = real_app.test_client().post(START, json=body, **hub_request())
        assert resp.status_code == 400 and fragment in resp.get_json()["error"]
        assert wired.starts == [] and _events(wired) == []

    def test_stop_is_relayed_and_announced(self, real_app, wired):
        resp = real_app.test_client().post(STOP, **hub_request())
        assert resp.status_code == 200 and wired.stops == 1
        assert _events(wired)[0]["type"] == "network_config_changed"

    def test_status_fills_the_ssid_a_start_would_use_and_relays_the_usable_channels(self, real_app, wired):
        wired.status.update({"active": False, "ssid": ""})
        body = real_app.test_client().get(STATUS, **hub_request()).get_json()
        assert body["ssid"] == "conecsa-000001" and body["channels"] == [44], "the agent's live list, not a constant"
        assert body["active"] is False and body["stations"] == STATUS_BODY["stations"]
        wired.status.pop("channels")
        body = real_app.test_client().get(STATUS, **hub_request()).get_json()
        assert body["channels"] == [], "an agent that cannot say reports none"

    def test_an_unavailable_agent_is_a_503_without_its_address(self, real_app, wired):
        wired.fail = FakeRpcError(grpc.StatusCode.UNAVAILABLE, "connect to os-base:50051 failed")
        resp = real_app.test_client().post(START, json={"passphrase": PASSPHRASE}, **hub_request())
        assert resp.status_code == 503
        assert "50051" not in resp.get_data(as_text=True)

    def test_an_agent_that_outruns_its_deadline_is_a_504_without_its_address(self, real_app, wired):
        # The peer is up but still bringing the radio up: a gateway timeout
        # the client may retry, not "unavailable".
        wired.fail = FakeRpcError(grpc.StatusCode.DEADLINE_EXCEEDED, "Deadline Exceeded os-base:50051")
        resp = real_app.test_client().post(START, json={"passphrase": PASSPHRASE}, **hub_request())
        assert resp.status_code == 504
        assert "50051" not in resp.get_data(as_text=True)
        assert _events(wired) == []


class TestThePassphraseStaysOut:
    def test_everywhere(self, real_app, wired, caplog):
        client = real_app.test_client()
        with caplog.at_level(logging.DEBUG):
            resp = client.post(START, json={"passphrase": PASSPHRASE, "channel": 40}, **hub_request())
            status = client.get(STATUS, **hub_request()).get_data(as_text=True)
        assert resp.status_code == 200
        for text in (resp.get_data(as_text=True), json.dumps(_events(wired)), json.dumps(_backlog()),
                     caplog.text, status):
            assert PASSPHRASE not in text
        record = _backlog()["records"][0]
        assert record["event"] == "network.ap_started" and record["detail"] == ""


class TestTheDeadlineOutlastsTheAgent:
    """The agent may spend 15 s on the radio, 10 s on the address and 10 s on
    the DHCP server before it answers, plus a rollback: a shorter gRPC deadline
    turns a slow but healthy start into a 503 while the radio is coming up."""

    @pytest.fixture
    def stub(self, monkeypatch):
        seen = SimpleNamespace(calls=[])

        class Stub:
            def StartAp(self, req, timeout=None):
                seen.calls.append(("StartAp", req.ssid, req.channel, timeout))
                return SimpleNamespace(success=True, message="started")

            def StopAp(self, req, timeout=None):
                seen.calls.append(("StopAp", None, None, timeout))
                return SimpleNamespace(success=True, message="stopped")

        monkeypatch.setattr(hardware, "clients", SimpleNamespace(hardware=Stub()))
        return seen

    def test_start_and_stop_get_the_long_deadline(self, stub):
        hardware.start_ap("conecsa-000001", PASSPHRASE, 40)
        hardware.stop_ap()
        assert stub.calls == [("StartAp", "conecsa-000001", 40, 60), ("StopAp", None, None, 60)]
        assert hardware._TIMEOUT_AP >= 15 + 10 + 10 + 10, "radio + address + DHCP server + rollback"
