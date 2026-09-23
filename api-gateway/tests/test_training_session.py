# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for the shared exit-training-mode helper (session._do_exit)."""
from types import SimpleNamespace

import grpc
import pytest
from gateway.training import helpers as training_helpers
from gateway.training import session


class FakeRpcError(grpc.RpcError):
    pass


@pytest.fixture
def events(monkeypatch):
    """Capture SSE publishes; returns the (event, data) call list."""
    calls = []
    monkeypatch.setattr(
        session, "event_service",
        SimpleNamespace(publish=lambda event, keys=None, data=None:
                        calls.append((event, data))))
    return calls


def _wire(monkeypatch, resume_result=None, resume_raises=False,
          unload_raises=False, job_status="idle", resumes=None):
    """Stub the gRPC surfaces _do_exit touches; returns the call log.

    ``resumes`` (a list) collects every ResumeRuntime request.
    """
    calls = []

    def get_training(_):
        calls.append("get_training")
        return SimpleNamespace(status=job_status)

    def unload_sam(_):
        calls.append("unload_sam")
        if unload_raises:
            raise FakeRpcError()

    def unload_label_model(_):
        calls.append("unload_label_model")
        if unload_raises:
            raise FakeRpcError()

    def resume(request):
        calls.append("resume")
        if resumes is not None:
            resumes.append(request)
        if resume_raises:
            raise FakeRpcError()
        return resume_result

    fake_clients = SimpleNamespace(
        training=SimpleNamespace(UnloadSam=unload_sam, GetTraining=get_training),
        model=SimpleNamespace(UnloadLabelModel=unload_label_model),
        management=SimpleNamespace(ResumeRuntime=resume))
    monkeypatch.setattr(session, "clients", fake_clients)
    # The training-job probe lives in gateway.training.helpers.
    monkeypatch.setattr(training_helpers, "clients", fake_clients)
    return calls


def test_no_resume_ends_the_handover_but_keeps_detection_stopped(monkeypatch, events):
    # The post-training handoff: the conversion keeps the GPU (detection is
    # not restarted), but the handover ends so the application type can change
    # again.
    resumes = []
    calls = _wire(monkeypatch, resumes=resumes,
                  resume_result=SimpleNamespace(success=True, message="ended"))
    ok, message = session._do_exit(resume_detection=False)
    assert ok
    assert "conversion" in message
    assert calls == ["unload_sam", "unload_label_model", "get_training", "resume"]
    assert [r.keep_detection_stopped for r in resumes] == [True]
    assert events == [("detection_state_changed", {"is_running": False})]


def test_ending_the_handover_on_exit_is_best_effort(monkeypatch, events):
    _wire(monkeypatch, resume_raises=True)
    ok, message = session._do_exit(resume_detection=False)
    assert ok
    assert "conversion" in message
    assert events == [("detection_state_changed", {"is_running": False})]


def test_resume_success(monkeypatch, events):
    resumes = []
    calls = _wire(monkeypatch, resumes=resumes,
                  resume_result=SimpleNamespace(success=True, message="resumed"))
    ok, message = session._do_exit(resume_detection=True)
    assert (ok, message) == (True, "resumed")
    assert calls == ["unload_sam", "unload_label_model", "get_training", "resume"]
    assert [r.keep_detection_stopped for r in resumes] == [False]
    assert events == [("detection_state_changed", {"is_running": True})]


@pytest.mark.parametrize("resume_detection", [True, False])
@pytest.mark.parametrize("job_status", ["preparing", "training", "uploading"])
def test_resume_is_skipped_while_a_training_job_runs(monkeypatch, events, job_status,
                                                    resume_detection):
    # Leaving the training page mid-run must not restart detection on top of
    # the trainer, nor end the handover the trainer still holds (an application
    # switch would then pass the inference-service's check): the runtime stays
    # released, like the conversion handoff.
    calls = _wire(monkeypatch, job_status=job_status,
                  resume_result=SimpleNamespace(success=True, message="resumed"))
    ok, message = session._do_exit(resume_detection=resume_detection)
    assert ok
    assert "training" in message
    assert "resume" not in calls, "the runtime must stay released"
    assert events == [("detection_state_changed", {"is_running": False})]


def test_resume_refusal_reports_failure(monkeypatch, events):
    _wire(monkeypatch,
          resume_result=SimpleNamespace(success=False, message="busy"))
    ok, message = session._do_exit(resume_detection=True)
    assert (ok, message) == (False, "busy")
    assert events == [], "no state-change event on a failed resume"


def test_resume_rpc_error_propagates(monkeypatch, events):
    _wire(monkeypatch, resume_raises=True)
    with pytest.raises(grpc.RpcError):
        session._do_exit(resume_detection=True)
    assert events == []


def test_unload_failures_are_best_effort(monkeypatch, events):
    calls = _wire(monkeypatch, unload_raises=True,
                  resume_result=SimpleNamespace(success=True, message="ok"))
    ok, _ = session._do_exit(resume_detection=True)
    assert ok
    assert calls == ["unload_sam", "unload_label_model", "get_training", "resume"], \
        "both unloads are attempted and resume must still run"


def test_heartbeat_returns_json_ok():
    # The whole effect is the blueprint's before_request hook (tracker.touch,
    # covered by test_orphan.py); the route itself must only answer 200 with a
    # JSON body — the frontend transport parses every response as JSON.
    from flask import Flask

    with Flask(__name__).test_request_context("/api/v1/training/heartbeat",
                                              method="POST"):
        resp = session.training_heartbeat()
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}
