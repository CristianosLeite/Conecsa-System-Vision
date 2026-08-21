"""Smoke tests for the assembled gateway app (``gateway.app``).

Every other suite builds a bare per-blueprint Flask app, which never installs
the app-level hooks or error handlers — these tests prove the ``real_app``
fixture exercises the genuine application, so app-level behaviour (audit hook,
error handlers, authorization) is actually testable.
"""
import pytest
from conftest import hub_request
from gateway import grpc_clients


class FakeDetectionStub:
    def __init__(self):
        self.reset_requests = []

    def ResetStats(self, request):
        self.reset_requests.append(request)


@pytest.fixture
def detection_stub(monkeypatch):
    stub = FakeDetectionStub()
    monkeypatch.setattr(grpc_clients.clients, "detection", stub)
    return stub


class TestRealApp:
    def test_unknown_route_hits_the_real_404_handler(self, real_app):
        resp = real_app.test_client().get("/api/v1/definitely-not-a-route")
        assert resp.status_code == 404
        assert resp.get_json() == {"error": "Route not found"}

    def test_a_hub_relayed_mutation_flows_through_all_hooks(
            self, real_app, detection_stub):
        # One request through the full chain: before_request (clock, inert
        # without the hub-time header), the route, and the after_request audit
        # hook recording the actor the terminator stamped.
        from gateway import audit
        resp = real_app.test_client().post("/api/v1/stats/reset",
                                           **hub_request(role="admin"))
        assert resp.status_code == 200
        assert detection_stub.reset_requests
        record = audit.buffer().list_backlog()["records"][0]
        assert record["username"] == "ana"
        assert record["role"] == "admin"


def test_the_app_imports_in_a_pristine_interpreter():
    # The container's waitress entrypoint imports gateway.app in a cold
    # interpreter. In-process tests can mask import-order bugs (an import
    # sweep once hoisted a flat proto import above the sys.path shim and only
    # the container noticed), so this exercises the real cold path.
    import os
    import subprocess
    import sys

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os_base = os.path.join(os.path.dirname(here), "os-base")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([here, os_base])}
    result = subprocess.run(
        [sys.executable, "-c", "import gateway.app"],
        capture_output=True, text=True, env=env, cwd=here, timeout=60,
    )
    assert result.returncode == 0, result.stderr
