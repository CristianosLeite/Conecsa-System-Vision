"""Tests for the server-side role policy on mutating routes (gateway/authz.py).

The exhaustive url_map walk is the load-bearing test: it fails the moment a
new mutating route is added without a policy entry, which is what keeps the
default-deny branch in ``authz.enforce`` unreachable in production.
"""
import grpc
import pytest
from conftest import hub_request
from flask import Flask, request
from gateway import authz, grpc_clients


class FakeRpcError(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.UNAVAILABLE

    def details(self):
        return "inference down"


class UnavailableStub:
    """Any RPC on it fails UNAVAILABLE — enough to prove authz let it through."""

    def __getattr__(self, name):
        def call(*args, **kwargs):
            raise FakeRpcError()
        return call


class CountingStub(UnavailableStub):
    def __init__(self):
        self.reset_calls = 0

    def ResetStats(self, req):
        self.reset_calls += 1


@pytest.fixture
def unavailable_backends(monkeypatch):
    stub = UnavailableStub()
    for name in ("detection", "models", "training", "hardware"):
        if hasattr(grpc_clients.clients, name):
            monkeypatch.setattr(grpc_clients.clients, name, stub)
    return stub


class TestPolicyCoverage:
    def test_every_mutating_route_has_a_policy_or_is_exempt(self, real_app):
        missing = []
        for rule in real_app.url_map.iter_rules():
            if rule.endpoint == "static":
                continue
            blueprint = rule.endpoint.split(".")[0]
            if blueprint in authz.EXEMPT_BLUEPRINTS:
                continue
            for method in (rule.methods or set()) & authz.MUTATING_METHODS:
                if (method, rule.rule) not in authz.ROUTE_POLICIES:
                    missing.append((method, rule.rule))
        assert missing == [], (
            "mutating routes without a role policy (add them to "
            f"gateway/authz.py ROUTE_POLICIES): {sorted(missing)}")

    def test_every_policy_names_a_real_route(self, real_app):
        # The inverse direction: a stale policy entry means a route was
        # renamed/removed and the table no longer describes reality.
        registered = set()
        for rule in real_app.url_map.iter_rules():
            for method in (rule.methods or set()) & authz.MUTATING_METHODS:
                registered.add((method, rule.rule))
        stale = set(authz.ROUTE_POLICIES) - registered
        assert stale == set(), f"policies for routes that do not exist: {sorted(stale)}"


class TestRoleMatrix:
    def test_a_user_cannot_reach_an_admin_route(self, real_app,
                                                unavailable_backends):
        resp = real_app.test_client().post("/api/v1/model/select",
                                           json={"model_name": "x.engine"},
                                           **hub_request(role="user"))
        assert resp.status_code == 403
        assert resp.get_json() == {"error": "forbidden"}

    def test_an_admin_passes_authz_on_an_admin_route(self, real_app,
                                                     unavailable_backends):
        # The backend is down (503), which proves the request got past authz.
        resp = real_app.test_client().post("/api/v1/model/select",
                                           json={"model_name": "x.engine"},
                                           **hub_request(role="admin"))
        assert resp.status_code != 403

    def test_a_user_can_operate_the_detector(self, real_app, monkeypatch):
        stub = CountingStub()
        monkeypatch.setattr(grpc_clients.clients, "detection", stub)
        resp = real_app.test_client().post("/api/v1/stats/reset",
                                           **hub_request(role="user"))
        assert resp.status_code == 200
        assert stub.reset_calls == 1

    def test_an_owner_outranks_admin_routes(self, real_app,
                                            unavailable_backends):
        resp = real_app.test_client().post("/api/v1/system/power",
                                           json={"action": "reboot"},
                                           **hub_request(role="owner"))
        assert resp.status_code != 403

    def test_an_unknown_role_is_rejected(self, real_app):
        resp = real_app.test_client().post("/api/v1/stats/reset",
                                           **hub_request(role="root"))
        assert resp.status_code == 403

    def test_the_hub_itself_is_not_blocked(self, real_app, monkeypatch):
        # A hub-verified request without operator headers is the hub acting on
        # its own behalf (recipes, backlog drains) — its command layer already
        # authorized it. See the trust model in gateway/authz.py.
        stub = CountingStub()
        monkeypatch.setattr(grpc_clients.clients, "detection", stub)
        kwargs = hub_request()
        del kwargs["headers"]["X-Conecsa-User"]
        del kwargs["headers"]["X-Conecsa-Role"]
        resp = real_app.test_client().post("/api/v1/stats/reset", **kwargs)
        assert resp.status_code == 200

    def test_an_unverified_caller_is_untouched_by_the_policy(self, real_app,
                                                             monkeypatch):
        # Not relayed by the terminator: the pre-existing network boundary
        # applies, not the role policy — even with a forged role header.
        stub = CountingStub()
        monkeypatch.setattr(grpc_clients.clients, "detection", stub)
        resp = real_app.test_client().post(
            "/api/v1/stats/reset",
            headers={"X-Conecsa-Role": "user"})
        assert resp.status_code == 200

    def test_reads_are_never_role_gated(self, real_app,
                                        unavailable_backends):
        resp = real_app.test_client().get("/api/v1/status",
                                          **hub_request(role="user"))
        assert resp.status_code != 403


class TestDefaultDeny:
    def test_an_unpoliced_mutating_route_is_denied(self, trusted_proxy):
        # A standalone app proves the default-deny branch without mutating the
        # real app's route table after setup.
        app = Flask(__name__)

        @app.route("/api/v1/new-unregistered-thing", methods=["POST"])
        def new_thing():
            return {"ok": True}

        @app.before_request
        def hook():
            return authz.enforce(request)

        resp = app.test_client().post("/api/v1/new-unregistered-thing",
                                      **hub_request(role="owner"))
        assert resp.status_code == 403


class TestEnrollExemption:
    def test_enroll_routes_are_outside_the_role_policy(self, real_app,
                                                       trusted_proxy):
        # enforce() itself must step aside for the enroll blueprint — its
        # pairing-token policy decides, not the operator role.
        with real_app.test_request_context("/enroll/reset", method="POST",
                                           **hub_request(role="user")):
            assert authz.enforce(request) is None
