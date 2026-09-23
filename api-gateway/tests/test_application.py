# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""GET/PUT /api/v1/application and the application gates on /start and datasets.

The switch must refuse when the GPU is busy (409) and — unlike /start — also
when a busy probe cannot answer (503, fail-closed). Refusals from the
inference-service keep their meaning: FAILED_PRECONDITION → 409,
INVALID_ARGUMENT → 400, INTERNAL (stop or write failure) → 500.
"""
from types import SimpleNamespace

import grpc
import inference_pb2 as inf_pb
import pytest
from flask import Flask
from gateway.controllers import application, detection
from gateway.training import datasets
from gateway.training import helpers as training_helpers


class FakeRpcError(grpc.RpcError):
    def __init__(self, code=grpc.StatusCode.UNAVAILABLE, details="down"):
        self._code = code
        self._details = details

    def code(self):
        return self._code

    def details(self):
        return self._details


@pytest.fixture
def app():
    return Flask(__name__)


def _wire(monkeypatch, *, task="", job_status="idle", conversions=(), training_raises=False,
          conversions_raise=False, set_raises=None):
    calls = []

    def get_application(_):
        return inf_pb.ApplicationInfo(task=task, supported_tasks=["detect"], migrated=False)

    def set_application(req):
        calls.append(("set", req.task))
        if set_raises is not None:
            raise set_raises
        return inf_pb.ApplicationInfo(task=req.task, supported_tasks=["detect"])

    def get_training(_):
        if training_raises:
            raise FakeRpcError()
        return SimpleNamespace(status=job_status)

    def list_conversions(_):
        if conversions_raise:
            raise FakeRpcError()
        return SimpleNamespace(jobs=[SimpleNamespace(status=s) for s in conversions])

    fake = SimpleNamespace(
        management=SimpleNamespace(GetApplication=get_application,
                                   SetApplication=set_application),
        training=SimpleNamespace(GetTraining=get_training),
        model=SimpleNamespace(ListConversions=list_conversions))
    monkeypatch.setattr(application, "clients", fake)
    monkeypatch.setattr(training_helpers, "clients", fake)
    return calls


def _put(app, body):
    with app.test_request_context("/api/v1/application", method="PUT", json=body):
        return application.set_application()


class TestGet:
    def test_unset_is_null(self, app, monkeypatch):
        _wire(monkeypatch, task="")
        with app.test_request_context("/api/v1/application"):
            resp = application.get_application()
        assert resp.get_json() == {"task": None, "supported_tasks": ["detect"],
                                   "migrated": False}


class TestPut:
    def test_switches(self, app, monkeypatch):
        calls = _wire(monkeypatch)
        resp = _put(app, {"task": "detect"})
        assert resp.status_code == 200
        assert resp.get_json()["task"] == "detect"
        assert calls == [("set", "detect")]

    @pytest.mark.parametrize("body", [None, {}, {"task": ""}, {"task": 3}, ["detect"]])
    def test_a_task_is_required(self, app, monkeypatch, body):
        calls = _wire(monkeypatch)
        assert _put(app, body).status_code == 400
        assert calls == []

    @pytest.mark.parametrize("status", training_helpers.ACTIVE_JOB_STATUSES)
    def test_refused_while_training(self, app, monkeypatch, status):
        calls = _wire(monkeypatch, job_status=status)
        resp = _put(app, {"task": "detect"})
        assert resp.status_code == 409 and "training" in resp.get_json()["error"]
        assert calls == []

    def test_refused_while_converting(self, app, monkeypatch):
        calls = _wire(monkeypatch, conversions=("converting_to_engine",))
        resp = _put(app, {"task": "detect"})
        assert resp.status_code == 409 and "conversion" in resp.get_json()["error"]
        assert calls == []

    @pytest.mark.parametrize("kw", [{"training_raises": True}, {"conversions_raise": True}])
    def test_an_unanswered_probe_fails_closed(self, app, monkeypatch, kw):
        calls = _wire(monkeypatch, **kw)
        resp = _put(app, {"task": "detect"})
        assert resp.status_code == 503
        assert "Cannot verify" in resp.get_json()["error"]
        assert calls == []

    @pytest.mark.parametrize("code,status", [
        (grpc.StatusCode.FAILED_PRECONDITION, 409),
        (grpc.StatusCode.INVALID_ARGUMENT, 400),
        (grpc.StatusCode.INTERNAL, 500),
        (grpc.StatusCode.UNAVAILABLE, 503),
    ])
    def test_service_refusals_keep_their_meaning(self, app, monkeypatch, code, status):
        _wire(monkeypatch, set_raises=FakeRpcError(code, "the reason"))
        resp = _put(app, {"task": "classify"})
        assert resp.status_code == status
        if status != 503:
            assert resp.get_json()["error"] == "the reason"

    def test_the_route_never_publishes_the_change_itself(self, app, monkeypatch):
        # The inference-service publishes application_changed and the relay
        # forwards it; a second publish here would reach clients twice.
        published = []
        from gateway import helpers
        monkeypatch.setattr(helpers, "_publish_event",
                            lambda *a, **k: published.append(a))
        _wire(monkeypatch)
        assert _put(app, {"task": "detect"}).status_code == 200
        assert published == []


class TestStartGate:
    def _wire_start(self, monkeypatch, task):
        def get_status(_):
            extra = {} if task is None else {"task": task}
            return inf_pb.StatusResponse(is_running=False, camera_connected=True, **extra)

        started = []
        fake = SimpleNamespace(
            detection=SimpleNamespace(
                GetStatus=get_status,
                Start=lambda _: started.append(1) or SimpleNamespace(success=True, message="")),
            training=SimpleNamespace(GetTraining=lambda _: SimpleNamespace(status="idle"),
                                     GetSamStatus=lambda _: SimpleNamespace(loaded=False)),
            model=SimpleNamespace(ListConversions=lambda _: SimpleNamespace(jobs=[]),
                                  GetLabelModelStatus=lambda _: SimpleNamespace(loaded=False)))
        monkeypatch.setattr(detection, "clients", fake)
        monkeypatch.setattr(training_helpers, "clients", fake)
        monkeypatch.setattr(detection, "_publish_if_success", lambda resp, *a, **kw: resp)
        return started

    def test_no_application_refuses_start(self, app, monkeypatch):
        started = self._wire_start(monkeypatch, "")
        with app.test_request_context("/api/v1/start", method="POST"):
            resp = detection.start_detection()
        assert resp.status_code == 409
        assert "No application type" in resp.get_json()["error"]
        assert started == []

    @pytest.mark.parametrize("task", [None, "detect"])
    def test_a_chosen_or_unknown_application_lets_start_through(self, app, monkeypatch,
                                                                task):
        # None = a producer that predates application types: no gate.
        started = self._wire_start(monkeypatch, task)
        with app.test_request_context("/api/v1/start", method="POST"):
            resp = detection.start_detection()
        assert resp.status_code == 200
        assert started == [1]


class TestStatusTask:
    """The task in /api/v1/status, under both encodings (vocabulary table)."""

    def _status(self, app, monkeypatch, task, accept="", **settings):
        extra = {} if task is None else {"task": task}
        fake = SimpleNamespace(detection=SimpleNamespace(
            GetStatus=lambda _: inf_pb.StatusResponse(camera_connected=True, **extra,
                                                       **settings)))
        monkeypatch.setattr(detection, "clients", fake)
        headers = {"Accept": accept} if accept else {}
        with app.test_request_context("/api/v1/status", headers=headers):
            return detection.get_status()

    @pytest.mark.parametrize("task,expected", [("", None), ("detect", "detect"),
                                               ("pose", "pose")])
    def test_json(self, app, monkeypatch, task, expected):
        assert self._status(app, monkeypatch, task).get_json()["task"] == expected

    def test_json_omits_the_task_of_an_older_inference_service(self, app, monkeypatch):
        # The hub reads a missing key as older firmware and null as "unset".
        assert "task" not in self._status(app, monkeypatch, None).get_json()

    @pytest.mark.parametrize("task,expected", [(None, None), ("", ""), ("detect", "detect")])
    def test_protobuf(self, app, monkeypatch, task, expected):
        import detection_pb2 as det_pb
        resp = self._status(app, monkeypatch, task, accept="application/x-protobuf")
        msg = det_pb.StatusResponse()
        msg.ParseFromString(resp.get_data())
        # None = no presence: an older inference-service, not an unset device.
        assert (msg.task if msg.HasField("task") else None) == expected

    _PER_TASK = ("segment_max_instances", "face_match_threshold", "face_min_size_px",
                 "face_max_faces")

    def test_protobuf_carries_the_per_task_settings_the_service_reports(self, app,
                                                                        monkeypatch):
        import detection_pb2 as det_pb
        resp = self._status(app, monkeypatch, "face", accept="application/x-protobuf",
                            face_match_threshold=0.45, face_min_size_px=64, face_max_faces=4)
        msg = det_pb.StatusResponse()
        msg.ParseFromString(resp.get_data())
        assert [msg.HasField(k) for k in self._PER_TASK] == [False, True, True, True]
        assert (round(msg.face_match_threshold, 2), msg.face_min_size_px,
                msg.face_max_faces) == (0.45, 64, 4)

        resp = self._status(app, monkeypatch, "segment", accept="application/x-protobuf",
                            segment_max_instances=16)
        msg.ParseFromString(resp.get_data())
        assert [msg.HasField(k) for k in self._PER_TASK] == [True, False, False, False]
        assert msg.segment_max_instances == 16

    def test_protobuf_leaves_the_per_task_settings_unset_otherwise(self, app, monkeypatch):
        import detection_pb2 as det_pb
        resp = self._status(app, monkeypatch, "detect", accept="application/x-protobuf")
        msg = det_pb.StatusResponse()
        msg.ParseFromString(resp.get_data())
        assert not any(msg.HasField(k) for k in self._PER_TASK)


class TestDatasetCreate:
    def _wire_datasets(self, monkeypatch, task="detect", get_raises=None):
        created = []

        def get_application(_):
            if get_raises is not None:
                raise get_raises
            return inf_pb.ApplicationInfo(task=task)

        def create(req):
            created.append((req.name, req.task))
            return SimpleNamespace(dataset_id="d1", name=req.name, created_at=1.0,
                                   cover_image_id="", image_count=0, labeled_count=0,
                                   class_count=0, task=req.task)

        fake = SimpleNamespace(management=SimpleNamespace(GetApplication=get_application),
                               training=SimpleNamespace(CreateDataset=create))
        monkeypatch.setattr(datasets, "clients", fake)
        return created

    def _post(self, app, body):
        with app.test_request_context("/api/v1/training/datasets", method="POST", json=body):
            return datasets.training_dataset_create()

    def test_the_dataset_takes_the_device_task(self, app, monkeypatch):
        created = self._wire_datasets(monkeypatch)
        resp = self._post(app, {"name": "Parts"})
        assert resp.status_code == 201 and resp.get_json()["task"] == "detect"
        assert created == [("Parts", "detect")]

    def test_refused_on_a_device_without_application(self, app, monkeypatch):
        created = self._wire_datasets(monkeypatch, task="")
        assert self._post(app, {"name": "Parts"}).status_code == 409
        assert created == []

    def test_a_dataset_of_another_task_is_refused(self, app, monkeypatch):
        created = self._wire_datasets(monkeypatch)
        resp = self._post(app, {"name": "Parts", "task": "classify"})
        assert resp.status_code == 409 and "'detect' application" in resp.get_json()["error"]
        assert created == []

    def test_an_inference_service_without_application_types_is_detect(self, app,
                                                                       monkeypatch):
        created = self._wire_datasets(
            monkeypatch, get_raises=FakeRpcError(grpc.StatusCode.UNIMPLEMENTED))
        assert self._post(app, {"name": "Parts"}).status_code == 201
        assert created == [("Parts", "detect")]

    def test_an_unreachable_inference_service_is_503(self, app, monkeypatch):
        created = self._wire_datasets(monkeypatch, get_raises=FakeRpcError())
        assert self._post(app, {"name": "Parts"}).status_code == 503
        assert created == []
