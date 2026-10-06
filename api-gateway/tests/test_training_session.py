# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for the shared exit-training-mode helper (session._do_exit)."""
from types import SimpleNamespace

import grpc
import pytest
from gateway import gpu
from gateway.training import session


class FakeRpcError(grpc.RpcError):
    def __init__(self, code=grpc.StatusCode.UNAVAILABLE):
        self._code = code

    def code(self):
        return self._code


@pytest.fixture
def events(monkeypatch):
    """Capture SSE publishes; returns the (event, data) call list."""
    calls = []
    monkeypatch.setattr(
        session, "event_service",
        SimpleNamespace(publish=lambda event, keys=None, data=None:
                        calls.append((event, data))))
    return calls


def _wire(monkeypatch, resume_result=None, resume_raises=False, unload_ok=True,
          sam_status_raises=None, job_status="idle", resumes=None):
    """Stub the gRPC surfaces _do_exit touches; returns the call log.

    Both assistants start loaded; an unload flips them unless ``unload_ok``
    is False (an unsuccessful Result). ``resumes`` (a list) collects every
    ResumeRuntime request.
    """
    calls = []
    state = {"sam": True, "label": True}

    def get_training(_):
        calls.append("get_training")
        return SimpleNamespace(status=job_status)

    def get_sam_status(_):
        calls.append("get_sam_status")
        if sam_status_raises is not None:
            raise FakeRpcError(sam_status_raises)
        return SimpleNamespace(loaded=state["sam"])

    def unload_sam(_):
        calls.append("unload_sam")
        if unload_ok:
            state["sam"] = False
        return SimpleNamespace(success=unload_ok, message="worker stuck")

    def get_label_status(_):
        calls.append("get_label_status")
        return SimpleNamespace(loaded=state["label"])

    def unload_label_model(_):
        calls.append("unload_label_model")
        if unload_ok:
            state["label"] = False
        return SimpleNamespace(success=unload_ok, message="worker stuck")

    def resume(request):
        calls.append("resume")
        if resumes is not None:
            resumes.append(request)
        if resume_raises:
            raise FakeRpcError()
        return resume_result

    fake_clients = SimpleNamespace(
        training=SimpleNamespace(UnloadSam=unload_sam, GetSamStatus=get_sam_status,
                                 GetTraining=get_training),
        model=SimpleNamespace(UnloadLabelModel=unload_label_model,
                              GetLabelModelStatus=get_label_status),
        management=SimpleNamespace(ResumeRuntime=resume))
    monkeypatch.setattr(session, "clients", fake_clients)
    # The verified unloads and the training-job probe live in gateway.gpu.
    monkeypatch.setattr(gpu, "clients", fake_clients)
    return calls


UNLOADED = ["get_sam_status", "unload_sam", "get_sam_status",
            "get_label_status", "unload_label_model", "get_label_status"]


def test_no_resume_ends_the_handover_but_keeps_detection_stopped(monkeypatch, events):
    # The post-training handoff: the conversion keeps the GPU (detection is
    # not restarted), but the handover ends.
    resumes = []
    calls = _wire(monkeypatch, resumes=resumes,
                  resume_result=SimpleNamespace(success=True, message="ended"))
    ok, message = session._do_exit(resume_detection=False)
    assert ok
    assert "conversion" in message
    assert calls == [*UNLOADED, "get_training", "resume"]
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
    assert calls == [*UNLOADED, "get_training", "resume"]
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


def test_an_unconfirmed_unload_refuses_the_resume(monkeypatch, events):
    # Detection must not be loaded beside an assistant that is still resident.
    calls = _wire(monkeypatch, unload_ok=False,
                  resume_result=SimpleNamespace(success=True, message="ok"))
    ok, message = session._do_exit(resume_detection=True)
    assert not ok and "Could not unload SAM3" in message
    assert "resume" not in calls
    assert events == []


def test_an_unconfirmed_unload_still_ends_a_handover_without_resume(monkeypatch, events):
    # Nothing is loaded without a resume, so the handover still ends.
    resumes = []
    _wire(monkeypatch, unload_ok=False, resumes=resumes,
          resume_result=SimpleNamespace(success=True, message="ended"))
    ok, _ = session._do_exit(resume_detection=False)
    assert ok
    assert [r.keep_detection_stopped for r in resumes] == [True]


def test_a_training_service_nobody_answers_on_does_not_block_the_resume(monkeypatch, events):
    calls = _wire(monkeypatch, sam_status_raises=grpc.StatusCode.UNAVAILABLE,
                  resume_result=SimpleNamespace(success=True, message="resumed"))
    ok, _ = session._do_exit(resume_detection=True)
    assert ok and calls[-1] == "resume"


def test_an_unreadable_sam_status_refuses_the_resume(monkeypatch, events):
    calls = _wire(monkeypatch, sam_status_raises=grpc.StatusCode.DEADLINE_EXCEEDED,
                  resume_result=SimpleNamespace(success=True, message="resumed"))
    ok, _ = session._do_exit(resume_detection=True)
    assert not ok and "resume" not in calls


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
