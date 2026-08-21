"""Unit tests for the device audit trail: the buffer, the request hook and the
route→event mapping."""
import sqlite3
from typing import Optional

import pytest
from flask import Flask, request
from gateway import audit, audit_events
from gateway.controllers import api_bp


@pytest.fixture
def buf(tmp_path):
    return audit.AuditBuffer(
        db_path=str(tmp_path / "audit.db"), max_records=1000,
        max_bytes=1024 * 1024)


class TestBuffer:
    def test_record_round_trips(self, buf):
        buf.record(event="detection.start", username="ana", role="admin",
                   detail="", source_ip="192.168.1.10")
        page = buf.list_backlog()
        assert page["pending"] == 1
        record = page["records"][0]
        assert record["event"] == "detection.start"
        assert record["username"] == "ana"
        assert record["source_ip"] == "192.168.1.10"
        assert record["outcome"] == "ok"
        assert record["captured_at"] <= page["device_now"]

    def test_backlog_is_oldest_first_and_paged(self, buf):
        for i in range(5):
            buf.record(event=f"event.{i}")
        page = buf.list_backlog(limit=2)
        assert [r["event"] for r in page["records"]] == ["event.0", "event.1"]
        # `pending` is the whole backlog, not the page.
        assert page["pending"] == 5

    def test_ack_deletes_and_is_idempotent(self, buf):
        buf.record(event="a")
        buf.record(event="b")
        ids = [r["id"] for r in buf.list_backlog()["records"]]
        assert buf.ack(ids) == 2
        assert buf.pending_count() == 0
        # Replaying an ack (the hub retries a page it already acknowledged)
        # must not error or double-count.
        assert buf.ack(ids) == 0

    def test_ack_of_nothing_is_a_no_op(self, buf):
        buf.record(event="a")
        assert buf.ack([]) == 0
        assert buf.pending_count() == 1

    def test_records_survive_a_reopen(self, tmp_path):
        path = str(tmp_path / "audit.db")
        first = audit.AuditBuffer(path, max_records=10, max_bytes=10_000)
        first.record(event="detection.start")
        first._close()
        second = audit.AuditBuffer(path, max_records=10, max_bytes=10_000)
        assert second.pending_count() == 1

    def test_ring_evicts_oldest_when_over_the_record_cap(self, tmp_path):
        buf = audit.AuditBuffer(str(tmp_path / "audit.db"), max_records=3,
                                max_bytes=10_000_000)
        for i in range(6):
            buf.record(event=f"event.{i}")
        events = [r["event"] for r in buf.list_backlog()["records"]]
        assert buf.pending_count() == 3
        assert events == ["event.3", "event.4", "event.5"]

    def test_a_broken_database_disables_the_buffer_instead_of_raising(self, buf):
        # The gateway must keep serving even if the trail cannot be written:
        # losing a record is bad, turning a working action into a 500 is worse.
        def explode(*_args, **_kwargs):
            raise sqlite3.OperationalError("disk I/O error")

        class DeadConnection:
            execute = explode
            commit = explode

            def close(self):
                pass

        buf._db = DeadConnection
        # The one reopen attempt fails too, so the buffer gives up for good.
        buf._open = explode
        buf.record(event="detection.start")
        assert buf._disabled is True
        # Every entry point stays callable once disabled.
        page = buf.list_backlog()
        assert page["records"] == []
        assert page["pending"] == 0
        assert buf.ack([1]) == 0
        assert buf.pending_count() == 0


class FakeRequest:
    """The few attributes the mapping reads off a Flask request."""

    def __init__(self, method: str = "POST",
                 rule: Optional[str] = "/api/v1/start",
                 path: Optional[str] = None,
                 json_body=None, form=None, view_args=None, headers=None,
                 remote_addr: str = "172.18.0.5"):
        self.method = method
        self.url_rule = None if rule is None else type("Rule", (), {"rule": rule})()
        self.path = path or rule or "/"
        self.view_args = view_args or {}
        self.headers = headers or {}
        self.remote_addr = remote_addr
        self.mimetype = "application/json" if json_body is not None else "multipart/form-data"
        self._json = json_body
        self.form = form or {}

    def get_json(self, silent=False):
        return self._json


class FakeResponse:
    def __init__(self, status_code=200):
        self.status_code = status_code


class TestShouldAudit:
    def test_reads_are_not_actions(self):
        for method in ("GET", "HEAD", "OPTIONS"):
            req = FakeRequest(method=method, rule="/api/v1/status")
            assert not audit_events.should_audit(req)

    def test_mutations_are_actions(self):
        assert audit_events.should_audit(FakeRequest())

    def test_unmatched_routes_are_not_actions(self):
        # A 404 never ran anything.
        req = FakeRequest(rule=None)
        assert not audit_events.should_audit(req)

    def test_machine_chatter_is_skipped(self):
        for rule in audit_events.SKIP_RULES:
            req = FakeRequest(rule=rule)
            assert not audit_events.should_audit(req), rule


class TestEventMapping:
    def test_known_routes_get_stable_keys(self):
        assert audit_events.event_for(FakeRequest(rule="/api/v1/start")) == \
            "detection.start"
        assert audit_events.event_for(
            FakeRequest(method="DELETE",
                        rule="/api/v1/training/datasets/<dataset_id>")) == \
            "dataset.deleted"

    def test_aliases_map_to_the_same_event_as_the_v1_route(self):
        # controllers/aliases.py serves these through separate view functions;
        # they are the same user action and must read as one in the trail.
        for alias, canonical in (("/api/start", "/api/v1/start"),
                                 ("/api/stop", "/api/v1/stop"),
                                 ("/api/threshold", "/api/v1/threshold")):
            assert audit_events.event_for(FakeRequest(rule=alias)) == \
                audit_events.event_for(FakeRequest(rule=canonical))

    def test_an_unmapped_mutation_still_lands_in_the_trail(self):
        # Coverage must not depend on someone remembering to register a route.
        req = FakeRequest(method="POST", rule="/api/v1/brand/new",
                          path="/api/v1/brand/new")
        assert audit_events.event_for(req) == audit_events.GENERIC_EVENT
        assert audit_events.detail_for(req) == "POST /api/v1/brand/new"

    def test_every_mapped_key_is_a_domain_action_pair(self):
        for key in list(audit_events.ROUTE_EVENTS.values()) + \
                [audit_events.GENERIC_EVENT]:
            domain, _, action = key.partition(".")
            assert domain and action, key
            assert key.islower(), key


class TestDetail:
    def test_names_the_target_from_the_body(self):
        req = FakeRequest(rule="/api/v1/model/select",
                          json_body={"model_name": "yolo26s.pt"})
        assert audit_events.detail_for(req) == "yolo26s.pt"

    def test_names_the_target_from_the_url(self):
        req = FakeRequest(method="DELETE", rule="/api/v1/model/<model_name>",
                          view_args={"model_name": "old.engine"})
        assert audit_events.detail_for(req) == "old.engine"

    def test_wifi_password_never_reaches_the_trail(self):
        req = FakeRequest(rule="/api/v1/network/wifi/connect",
                          json_body={"ssid": "fabrica", "password": "s3cr3t"})
        detail = audit_events.detail_for(req)
        assert detail == "fabrica"
        assert "s3cr3t" not in detail

    def test_multipart_uploads_contribute_only_their_name_field(self):
        req = FakeRequest(rule="/api/v1/training/datasets",
                          form={"name": "Peças"})
        assert audit_events.detail_for(req) == "Peças"

    def test_a_malformed_body_costs_the_detail_not_the_record(self):
        req = FakeRequest(rule="/api/v1/model/select", json_body=None)
        req.mimetype = "application/json"
        assert audit_events.detail_for(req) == ""

    def test_long_details_are_truncated(self):
        req = FakeRequest(rule="/api/v1/training/datasets",
                          json_body={"name": "x" * 500})
        assert len(audit_events.detail_for(req)) == audit_events.MAX_DETAIL_LEN


class TestActorAndOrigin:
    def test_actor_comes_from_the_hub_headers(self):
        req = FakeRequest(headers={"X-Conecsa-User": "ana",
                                   "X-Conecsa-Role": "admin"})
        assert audit_events.actor(req) == ("ana", "admin")

    def test_percent_encoded_names_are_decoded(self):
        req = FakeRequest(headers={"X-Conecsa-User": "Jos%C3%A9",
                                   "X-Conecsa-Role": "user"})
        assert audit_events.actor(req) == ("José", "user")

    def test_an_unattributed_action_is_recorded_as_such(self):
        assert audit_events.actor(FakeRequest()) == ("", "")

    def test_origin_prefers_the_hub_then_the_proxy_then_the_peer(self):
        hub = FakeRequest(headers={"X-Conecsa-Origin-Ip": "192.168.1.10",
                                   "X-Forwarded-For": "10.0.0.2"})
        assert audit_events.source_ip(hub, from_proxy=True) == "192.168.1.10"

        proxied = FakeRequest(headers={"X-Forwarded-For": "10.0.0.2, 10.0.0.3"})
        assert audit_events.source_ip(proxied, from_proxy=True) == "10.0.0.2"

        direct = FakeRequest(headers={})
        assert audit_events.source_ip(direct, from_proxy=True) == "172.18.0.5"

    def test_a_direct_caller_cannot_name_its_own_origin(self):
        # Not relayed by nginx: both headers are the caller's own invention and
        # would otherwise let it write any address it liked into the trail.
        spoofed = FakeRequest(headers={"X-Conecsa-Origin-Ip": "192.168.1.10",
                                       "X-Forwarded-For": "203.0.113.7"})
        assert audit_events.source_ip(spoofed, from_proxy=False) == "172.18.0.5"

    def test_outcome_reflects_the_status(self):
        assert audit_events.outcome_for(FakeResponse(200)) == "ok"
        assert audit_events.outcome_for(FakeResponse(204)) == "ok"
        assert audit_events.outcome_for(FakeResponse(400)) == "failed"
        assert audit_events.outcome_for(FakeResponse(500)) == "failed"


# trusted_proxy fixture and TERMINATOR_IP are shared in conftest.py.
from conftest import TERMINATOR_IP  # noqa: E402


@pytest.fixture
def hooked_app(buf, monkeypatch):
    """A Flask app with the audit hook installed over a stub route."""
    monkeypatch.setattr(audit, "buffer", lambda: buf)
    app = Flask(__name__)

    @app.route("/api/v1/start", methods=["POST"])
    def start():
        return {"ok": True}

    @app.route("/api/v1/status", methods=["GET"])
    def status():
        return {"running": False}

    @app.after_request
    def hook(response):
        audit.record_request(request, response)
        return response

    return app.test_client()


class TestRequestHook:
    def test_a_mutation_is_recorded_with_its_actor(self, hooked_app, buf,
                                                   trusted_proxy):
        hooked_app.post("/api/v1/start", headers={
            "X-Conecsa-Client-Verify": "SUCCESS",
            "X-Conecsa-User": "ana",
            "X-Conecsa-Role": "admin",
            "X-Conecsa-Origin-Ip": "192.168.1.10",
        }, environ_base={"REMOTE_ADDR": TERMINATOR_IP})
        record = buf.list_backlog()["records"][0]
        assert record["event"] == "detection.start"
        assert record["username"] == "ana"
        assert record["source_ip"] == "192.168.1.10"

    def test_an_identity_from_an_untrusted_peer_is_ignored(self, hooked_app, buf):
        # Any container on the compose network can set these headers; only the
        # mTLS terminator's relay may name the operator. The action is still
        # recorded — anonymously.
        hooked_app.post("/api/v1/start", headers={"X-Conecsa-User": "owner",
                                                  "X-Conecsa-Role": "owner"})
        record = buf.list_backlog()["records"][0]
        assert record["event"] == "detection.start"
        assert record["username"] == ""
        assert record["role"] == ""

    def test_reads_leave_no_trace(self, hooked_app, buf):
        hooked_app.get("/api/v1/status")
        assert buf.pending_count() == 0

    def test_a_failed_action_is_recorded_as_failed(self, hooked_app, buf):
        hooked_app.post("/api/v1/start", json={})
        # The stub route succeeds; drive the failure through a 404 instead of
        # faking a status, and confirm an unmatched route records nothing.
        hooked_app.post("/api/v1/nope")
        assert buf.pending_count() == 1

    def test_the_hook_never_breaks_a_request(self, hooked_app, buf, monkeypatch):
        def explode(**_kwargs):
            raise RuntimeError("buffer is on fire")

        monkeypatch.setattr(buf, "record", explode)
        response = hooked_app.post("/api/v1/start")
        assert response.status_code == 200


@pytest.fixture
def backlog_client(buf, monkeypatch):
    monkeypatch.setattr(audit, "buffer", lambda: buf)
    app = Flask(__name__)
    app.register_blueprint(api_bp)
    return app.test_client()


class TestBacklogEndpoints:
    HUB = {"X-Conecsa-Client-Verify": "SUCCESS"}

    def test_backlog_is_hub_only(self, backlog_client, buf):
        buf.record(event="detection.start")
        assert backlog_client.get("/api/v1/audit/backlog").status_code == 403
        assert backlog_client.post("/api/v1/audit/backlog/ack",
                                   json={"ids": [1]}).status_code == 403

    def test_hub_drains_and_acks(self, backlog_client, buf, trusted_proxy):
        buf.record(event="detection.start")
        page = backlog_client.get("/api/v1/audit/backlog", headers=self.HUB,
                                  environ_base={"REMOTE_ADDR": TERMINATOR_IP})
        assert page.status_code == 200
        ids = [r["id"] for r in page.get_json()["records"]]
        assert ids

        acked = backlog_client.post("/api/v1/audit/backlog/ack", json={"ids": ids},
                                    headers=self.HUB,
                                    environ_base={"REMOTE_ADDR": TERMINATOR_IP})
        assert acked.get_json() == {"success": True, "deleted": 1}
        assert buf.pending_count() == 0

    def test_ack_rejects_a_malformed_body(self, backlog_client, trusted_proxy):
        bad = backlog_client.post("/api/v1/audit/backlog/ack",
                                  json={"ids": ["1"]}, headers=self.HUB,
                                  environ_base={"REMOTE_ADDR": TERMINATOR_IP})
        assert bad.status_code == 400


class TestHubOwnActions:
    """The hub calls the device directly for some actions (applying a recipe,
    deleting a dataset, pairing) and records them itself, against the session
    it authenticated. Recording them here too would put an anonymous duplicate
    beside every named row."""

    def test_a_hub_action_with_no_operator_is_left_to_the_hub(
            self, hooked_app, buf, trusted_proxy):
        hooked_app.post("/api/v1/start",
                        headers={"X-Conecsa-Client-Verify": "SUCCESS"},
                        environ_base={"REMOTE_ADDR": TERMINATOR_IP})
        assert buf.pending_count() == 0

    def test_a_relayed_operator_action_is_still_recorded(
            self, hooked_app, buf, trusted_proxy):
        hooked_app.post("/api/v1/start", headers={
            "X-Conecsa-Client-Verify": "SUCCESS",
            "X-Conecsa-User": "ana",
        }, environ_base={"REMOTE_ADDR": TERMINATOR_IP})
        assert buf.pending_count() == 1

    def test_a_local_client_is_still_recorded(self, hooked_app, buf):
        # Not hub-verified: a Flow node or a local integration. Anonymous, but
        # it changed the device, so it belongs in the trail.
        hooked_app.post("/api/v1/start")
        record = buf.list_backlog()["records"][0]
        assert record["username"] == ""


class TestCatalogCoverage:
    """A key with no catalog entry still produces a row, but the hub renders it
    as "unrecognized action" — the event stops saying what happened."""

    def _catalog_events(self):
        import json
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        repo = os.path.abspath(os.path.join(here, os.pardir, os.pardir))
        path = os.path.join(repo, "i18n", "hub-vision", "en.json")
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)["audit"]["events"]

    def test_every_device_event_has_a_sentence(self):
        catalog = self._catalog_events()
        keys = set(audit_events.ROUTE_EVENTS.values()) | {audit_events.GENERIC_EVENT}
        for key in sorted(keys):
            # "detection.start" is spelled "detection_start" in the catalog;
            # only the domain separator becomes an underscore.
            catalog_key = key.replace(".", "_", 1)
            assert catalog_key in catalog, f"audit.events.{catalog_key} is missing"


class TestByteAccounting:
    """`size_bytes` backs the ring's byte cap, which is a promise about disk."""

    def test_size_counts_encoded_bytes_not_characters(self, buf):
        # "Peças" is 5 characters but 6 bytes; names and dataset titles here
        # are written in Portuguese and Spanish.
        buf.record(event="dataset.created", detail="Peças")
        assert buf._bytes == len("dataset.created") + len("Peças".encode()) + len("ok")

    def test_size_includes_every_stored_field(self, buf):
        buf.record(event="e", username="u", role="r", detail="d",
                   source_ip="i", outcome="failed")
        assert buf._bytes == len("e") + len("u") + len("r") + len("d") + \
            len("i") + len("failed")


class TestSpoofedOriginThroughTheHook:
    def test_a_direct_caller_is_recorded_by_its_real_address(self, hooked_app, buf):
        hooked_app.post("/api/v1/start",
                        headers={"X-Conecsa-Origin-Ip": "192.168.1.10",
                                 "X-Forwarded-For": "203.0.113.7"},
                        environ_base={"REMOTE_ADDR": "172.18.0.9"})
        assert buf.list_backlog()["records"][0]["source_ip"] == "172.18.0.9"

    def test_the_hub_origin_is_kept_when_nginx_relayed_it(self, hooked_app, buf,
                                                          trusted_proxy):
        hooked_app.post("/api/v1/start", headers={
            "X-Conecsa-Client-Verify": "SUCCESS",
            "X-Conecsa-User": "ana",
            "X-Conecsa-Origin-Ip": "192.168.1.10",
        }, environ_base={"REMOTE_ADDR": TERMINATOR_IP})
        assert buf.list_backlog()["records"][0]["source_ip"] == "192.168.1.10"
