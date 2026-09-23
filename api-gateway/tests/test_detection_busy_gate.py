# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for the GPU-busy gate in POST /api/v1/start.

The single Jetson GPU belongs to a training job or a TensorRT engine build
while one runs; the gateway refuses to start detection (409) in both states
so that the hub, Node-RED and a second browser tab cannot bypass the device
UI's disabled Start button. The probes live in gateway.training.helpers.
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


def _wire(monkeypatch, job_status="idle", conversions=(), training_raises=False,
          conversions_raise=False, task=None):
    """Stub every gRPC surface start_detection touches; returns the call log.

    ``task=None`` leaves StatusResponse.task absent (a producer that predates
    application types); a string sets it, ``""`` meaning none chosen.
    """
    calls = []

    def get_status(_):
        calls.append("get_status")
        extra = {} if task is None else {"task": task}
        return inf_pb.StatusResponse(is_running=False, camera_connected=True, **extra)

    def get_training(_):
        calls.append("get_training")
        if training_raises:
            raise FakeRpcError()
        return SimpleNamespace(status=job_status)

    def list_conversions(_):
        calls.append("list_conversions")
        if conversions_raise:
            raise FakeRpcError()
        return SimpleNamespace(jobs=[SimpleNamespace(status=s) for s in conversions])

    def loaded_false(_):
        return SimpleNamespace(loaded=False)

    def start(_):
        calls.append("start")
        return SimpleNamespace(success=True, message="Detection started")

    fake_clients = SimpleNamespace(
        detection=SimpleNamespace(GetStatus=get_status, Start=start),
        training=SimpleNamespace(GetTraining=get_training, GetSamStatus=loaded_false),
        model=SimpleNamespace(ListConversions=list_conversions,
                              GetLabelModelStatus=loaded_false))
    monkeypatch.setattr(detection, "clients", fake_clients)
    monkeypatch.setattr(training_helpers, "clients", fake_clients)
    monkeypatch.setattr(detection, "_publish_if_success", lambda resp, *a, **kw: resp)
    return calls


def _post_start(app, headers=None):
    with app.test_request_context("/api/v1/start", method="POST", headers=headers):
        return detection.start_detection()


def test_no_application_type_refuses_start_before_the_gpu_probes(app, monkeypatch):
    calls = _wire(monkeypatch, task="")
    resp = _post_start(app)
    assert resp.status_code == 409
    assert "No application type" in resp.get_json()["error"]
    assert calls == ["get_status"]


@pytest.mark.parametrize("job_status", training_helpers.ACTIVE_JOB_STATUSES)
def test_active_training_job_refuses_start(app, monkeypatch, job_status):
    calls = _wire(monkeypatch, job_status=job_status)
    resp = _post_start(app)
    assert resp.status_code == 409
    assert "training is in progress" in resp.get_json()["error"]
    assert "start" not in calls


@pytest.mark.parametrize("status", training_helpers.ACTIVE_CONVERSION_STATUSES)
def test_active_conversion_refuses_start(app, monkeypatch, status):
    calls = _wire(monkeypatch, conversions=("done", status))
    resp = _post_start(app)
    assert resp.status_code == 409
    assert "conversion is in progress" in resp.get_json()["error"]
    assert "start" not in calls


def test_training_takes_precedence_in_the_message(app, monkeypatch):
    _wire(monkeypatch, job_status="training", conversions=("pending",))
    resp = _post_start(app)
    assert resp.status_code == 409
    assert "training is in progress" in resp.get_json()["error"]


@pytest.mark.parametrize("job_status", ["idle", "done", "failed", "cancelled", ""])
def test_idle_device_starts(app, monkeypatch, job_status):
    calls = _wire(monkeypatch, job_status=job_status, conversions=("done", "failed"))
    resp = _post_start(app)
    assert resp.status_code == 200
    assert calls[-1] == "start"


def test_unreachable_probes_do_not_block_start(app, monkeypatch):
    # A training-service or conversion registry that cannot be reached is not
    # busy; the start must go through (and the failures are only logged).
    calls = _wire(monkeypatch, training_raises=True, conversions_raise=True)
    resp = _post_start(app)
    assert resp.status_code == 200
    assert calls[-1] == "start"


def test_refusal_honours_protobuf_negotiation(app, monkeypatch):
    import detection_pb2 as det_pb

    _wire(monkeypatch, job_status="training")
    resp = _post_start(app, headers={"Accept": "application/x-protobuf"})
    assert resp.status_code == 409
    msg = det_pb.StartDetectionResponse()
    msg.ParseFromString(resp.get_data())
    assert not msg.success
    assert "training is in progress" in msg.message
