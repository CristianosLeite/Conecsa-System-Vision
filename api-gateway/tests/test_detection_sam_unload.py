# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for the labeling-assistant unloads in POST /api/v1/start.

Training's assistants (SAM3 in the training-service, the labeling engine on
the inference-service's private worker) must never stay GPU-pinned once
detection runs: an unload that fails, is refused or leaves the assistant
loaded refuses the start. Only a training-service nobody answers on
(``UNAVAILABLE``) counts as one without SAM3 — its worker is a child process
of that container.
"""
from types import SimpleNamespace

import grpc
import inference_pb2 as inf_pb
import pytest
from flask import Flask
from gateway import gpu
from gateway.controllers import detection


class FakeRpcError(grpc.RpcError):
    def __init__(self, code=grpc.StatusCode.UNAVAILABLE):
        self._code = code

    def code(self):
        return self._code


@pytest.fixture
def app():
    return Flask(__name__)


def _wire(monkeypatch, sam_loaded=True, sam_status_raises=None, unload_raises=False,
          unload_ok=True, sticky=False, model_loaded=False):
    """Stub every gRPC surface start_detection touches; returns the call log.

    ``sam_status_raises`` is a gRPC code the SAM status probe fails with;
    ``unload_ok=False`` answers an unsuccessful Result; ``sticky`` keeps the
    assistant loaded after a "successful" unload.
    """
    calls = []
    state = {"sam": sam_loaded, "model": model_loaded}

    def get_status(_):
        calls.append("get_status")
        return inf_pb.StatusResponse(is_running=False, camera_connected=True)

    def get_sam_status(_):
        calls.append("get_sam_status")
        if sam_status_raises is not None:
            raise FakeRpcError(sam_status_raises)
        return SimpleNamespace(loaded=state["sam"])

    def unload_sam(_):
        calls.append("unload_sam")
        if unload_raises:
            raise FakeRpcError()
        if unload_ok and not sticky:
            state["sam"] = False
        return SimpleNamespace(success=unload_ok, message="worker stuck")

    def get_label_model_status(_):
        calls.append("get_label_model_status")
        return SimpleNamespace(loaded=state["model"])

    def unload_label_model(_):
        calls.append("unload_label_model")
        if unload_raises:
            raise FakeRpcError()
        if unload_ok and not sticky:
            state["model"] = False
        return SimpleNamespace(success=unload_ok, message="worker stuck")

    def start(_):
        calls.append("start")
        return SimpleNamespace(success=True, message="Detection started")

    # The GPU-busy probes (gateway.gpu) read their own `clients` binding: an
    # idle device here, so the assistant unloads stay the subject.
    def get_training(_):
        return SimpleNamespace(status="idle", model_name="")

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
    monkeypatch.setattr(gpu, "clients", fake_clients)
    # Keep the SSE side out of the unit: pass the response through untouched.
    monkeypatch.setattr(detection, "_publish_if_success",
                        lambda resp, *a, **kw: resp)
    return calls


def _post_start(app):
    with app.test_request_context("/api/v1/start", method="POST"):
        return detection.start_detection()


def test_loaded_sam_is_unloaded_and_confirmed_before_start(app, monkeypatch):
    calls = _wire(monkeypatch, sam_loaded=True)
    resp = _post_start(app)
    assert resp.status_code == 200
    assert calls == ["get_status", "get_sam_status", "unload_sam", "get_sam_status",
                     "get_label_model_status", "start"]


def test_loaded_label_model_is_unloaded_and_confirmed_before_start(app, monkeypatch):
    calls = _wire(monkeypatch, sam_loaded=False, model_loaded=True)
    resp = _post_start(app)
    assert resp.status_code == 200
    assert calls == ["get_status", "get_sam_status", "get_label_model_status",
                     "unload_label_model", "get_label_model_status", "start"]


def test_unloaded_sam_is_left_alone(app, monkeypatch):
    # The status probe avoids a spurious sam_changed event from the
    # unconditional publish in SamService.unload().
    calls = _wire(monkeypatch, sam_loaded=False)
    resp = _post_start(app)
    assert resp.status_code == 200
    assert "unload_sam" not in calls and "unload_label_model" not in calls
    assert calls[-1] == "start"


def test_a_training_service_nobody_answers_on_has_no_sam(app, monkeypatch):
    calls = _wire(monkeypatch, sam_status_raises=grpc.StatusCode.UNAVAILABLE)
    resp = _post_start(app)
    assert resp.status_code == 200
    assert "unload_sam" not in calls
    assert calls[-1] == "start"


@pytest.mark.parametrize("code", [grpc.StatusCode.DEADLINE_EXCEEDED,
                                  grpc.StatusCode.INTERNAL])
def test_an_unreadable_sam_status_refuses_the_start(app, monkeypatch, code):
    # The service is up but cannot say whether SAM3 is resident.
    calls = _wire(monkeypatch, sam_status_raises=code)
    resp = _post_start(app)
    assert resp.status_code == 503
    assert "start" not in calls


@pytest.mark.parametrize("kw,status", [
    ({"unload_raises": True}, 503),
    ({"unload_ok": False}, 500),
    ({"sticky": True}, 500),
])
@pytest.mark.parametrize("assistant", ["sam", "model"])
def test_a_failed_unload_refuses_the_start(app, monkeypatch, kw, status, assistant):
    calls = _wire(monkeypatch, sam_loaded=assistant == "sam",
                  model_loaded=assistant == "model", **kw)
    resp = _post_start(app)
    assert resp.status_code == status
    assert resp.get_json()["gpu_busy"] is None
    assert "start" not in calls
