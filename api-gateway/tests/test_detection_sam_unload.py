"""Unit tests for the best-effort labeling-assistant unloads in POST /api/v1/start.

Training's assistants (SAM3 in the training-service, the labeling engine on
the inference-service's private worker) must never stay GPU-pinned once
detection runs, but an unreachable service must not block the start either.
"""
from types import SimpleNamespace

import grpc
import inference_pb2 as inf_pb
import pytest
from flask import Flask
from gateway.controllers import detection
from gateway.training import helpers as training_helpers


class FakeRpcError(grpc.RpcError):
    pass


@pytest.fixture
def app():
    return Flask(__name__)


def _wire(monkeypatch, sam_loaded=True, sam_status_raises=False,
          unload_raises=False, model_loaded=False):
    """Stub every gRPC surface start_detection touches; returns the call log."""
    calls = []

    def get_status(_):
        calls.append("get_status")
        return inf_pb.StatusResponse(is_running=False, camera_connected=True)

    def get_sam_status(_):
        calls.append("get_sam_status")
        if sam_status_raises:
            raise FakeRpcError()
        return SimpleNamespace(loaded=sam_loaded)

    def unload_sam(_):
        calls.append("unload_sam")
        if unload_raises:
            raise FakeRpcError()

    def get_label_model_status(_):
        calls.append("get_label_model_status")
        if sam_status_raises:
            raise FakeRpcError()
        return SimpleNamespace(loaded=model_loaded)

    def unload_label_model(_):
        calls.append("unload_label_model")
        if unload_raises:
            raise FakeRpcError()

    def start(_):
        calls.append("start")
        return SimpleNamespace(success=True, message="Detection started")

    # The GPU-busy probes (gateway.training.helpers) read their own `clients`
    # binding: an idle device here, so the assistant unloads stay the subject.
    def get_training(_):
        return SimpleNamespace(status="idle")

    def list_conversions(_):
        return SimpleNamespace(jobs=[])

    fake_clients = SimpleNamespace(
        detection=SimpleNamespace(GetStatus=get_status, Start=start),
        training=SimpleNamespace(GetSamStatus=get_sam_status,
                                 UnloadSam=unload_sam,
                                 GetTraining=get_training),
        model=SimpleNamespace(GetLabelModelStatus=get_label_model_status,
                              UnloadLabelModel=unload_label_model,
                              ListConversions=list_conversions))
    monkeypatch.setattr(detection, "clients", fake_clients)
    monkeypatch.setattr(training_helpers, "clients", fake_clients)
    # Keep the SSE side out of the unit: pass the response through untouched.
    monkeypatch.setattr(detection, "_publish_if_success",
                        lambda resp, *a, **kw: resp)
    return calls


def _post_start(app):
    with app.test_request_context("/api/v1/start", method="POST"):
        return detection.start_detection()


def test_loaded_sam_is_unloaded_before_start(app, monkeypatch):
    calls = _wire(monkeypatch, sam_loaded=True)
    resp = _post_start(app)
    assert resp.status_code == 200
    assert calls == ["get_status", "get_sam_status", "unload_sam",
                     "get_label_model_status", "start"]


def test_loaded_label_model_is_unloaded_before_start(app, monkeypatch):
    calls = _wire(monkeypatch, sam_loaded=False, model_loaded=True)
    resp = _post_start(app)
    assert resp.status_code == 200
    assert calls == ["get_status", "get_sam_status", "get_label_model_status",
                     "unload_label_model", "start"]


def test_unloaded_sam_is_left_alone(app, monkeypatch):
    # The status probe avoids a spurious sam_changed event from the
    # unconditional publish in SamService.unload().
    calls = _wire(monkeypatch, sam_loaded=False)
    resp = _post_start(app)
    assert resp.status_code == 200
    assert "unload_sam" not in calls and "unload_label_model" not in calls
    assert calls[-1] == "start"


def test_unreachable_training_service_does_not_block_start(app, monkeypatch):
    calls = _wire(monkeypatch, sam_status_raises=True)
    resp = _post_start(app)
    assert resp.status_code == 200
    assert "unload_sam" not in calls and "unload_label_model" not in calls
    assert calls[-1] == "start"


def test_failing_unload_does_not_block_start(app, monkeypatch):
    calls = _wire(monkeypatch, sam_loaded=True, unload_raises=True, model_loaded=True)
    resp = _post_start(app)
    assert resp.status_code == 200
    assert calls[-1] == "start"
