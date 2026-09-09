"""Unit tests for the model-assisted labeling relays (gateway/training/label_model.py)
and the base_model passthrough on POST /api/v1/training/train."""
from types import SimpleNamespace

import grpc
import inference_pb2 as inf_pb
import pytest
import training_pb2 as trn_pb
from flask import Flask
from gateway.training import jobs, label_model, sam


@pytest.fixture
def app():
    return Flask(__name__)


class _FailedCall(grpc.RpcError, grpc.Call):
    """A failed unary call as grpcio raises it (an RpcError that is a Call)."""

    def __init__(self, code):
        super().__init__()
        self._code = code

    def code(self):
        return self._code

    def details(self):
        return self._code.name

    def initial_metadata(self):
        return ()

    def trailing_metadata(self):
        return ()

    def is_active(self):
        return False

    def time_remaining(self):
        return 0.0

    def cancel(self):
        return False

    def add_callback(self, callback):
        return False


def _rpc_error(code):
    return _FailedCall(code)


def _wire(monkeypatch, module, *, model=None, training=None):
    """Stub the model (inference) and training gRPC surfaces; returns the call log."""
    calls = []

    def make(name, reply):
        def call(msg, timeout=None):
            calls.append((name, msg, timeout))
            if isinstance(reply, Exception):
                raise reply
            return reply
        return call

    monkeypatch.setattr(module, "clients", SimpleNamespace(
        model=SimpleNamespace(**{n: make(n, r) for n, r in (model or {}).items()}),
        training=SimpleNamespace(**{n: make(n, r) for n, r in (training or {}).items()}),
    ))
    return calls


class TestStatus:
    def test_serializes_the_message(self, app, monkeypatch):
        _wire(monkeypatch, label_model, model={"GetLabelModelStatus": inf_pb.LabelModelStatus(
            loaded=True, model_name="Teste.engine", class_names=["logo", "person"])})
        with app.test_request_context("/api/v1/training/label-model"):
            resp = label_model.training_label_model_status()
        assert resp.get_json() == {"loaded": True, "model_name": "Teste.engine",
                                   "class_names": ["logo", "person"], "message": ""}


class TestLoad:
    def test_requires_model_name(self, app, monkeypatch):
        calls = _wire(monkeypatch, label_model,
                      model={"LoadLabelModel": inf_pb.Result(success=True)})
        with app.test_request_context("/api/v1/training/label-model/load", method="POST",
                                      json={}):
            resp = label_model.training_label_model_load()
        assert resp.status_code == 400 and calls == []

    def test_relays_to_the_inference_service_with_a_long_deadline(self, app, monkeypatch):
        calls = _wire(monkeypatch, label_model,
                      model={"LoadLabelModel": inf_pb.Result(success=True, message="loaded")},
                      training={"GetSamStatus": trn_pb.SamStatus(loaded=False)})
        with app.test_request_context("/api/v1/training/label-model/load", method="POST",
                                      json={"model_name": " Teste.engine "}):
            resp = label_model.training_label_model_load()
        assert resp.status_code == 200
        # SAM not loaded: probed, not unloaded.
        assert [c[0] for c in calls] == ["GetSamStatus", "LoadLabelModel"]
        assert calls[1][1].name == "Teste.engine" and calls[1][2] == 300

    def test_unloads_a_loaded_sam_first(self, app, monkeypatch):
        calls = _wire(monkeypatch, label_model,
                      model={"LoadLabelModel": inf_pb.Result(success=True)},
                      training={"GetSamStatus": trn_pb.SamStatus(available=True, loaded=True),
                                "UnloadSam": trn_pb.Result(success=True)})
        with app.test_request_context("/api/v1/training/label-model/load", method="POST",
                                      json={"model_name": "Teste.engine"}):
            resp = label_model.training_label_model_load()
        assert resp.status_code == 200
        assert [c[0] for c in calls] == ["GetSamStatus", "UnloadSam", "LoadLabelModel"]

    def test_sam_probe_failure_does_not_block_the_load(self, app, monkeypatch):
        calls = _wire(monkeypatch, label_model,
                      model={"LoadLabelModel": inf_pb.Result(success=True)},
                      training={"GetSamStatus": _rpc_error(grpc.StatusCode.UNAVAILABLE)})
        with app.test_request_context("/api/v1/training/label-model/load", method="POST",
                                      json={"model_name": "Teste.engine"}):
            resp = label_model.training_label_model_load()
        assert resp.status_code == 200
        assert [c[0] for c in calls] == ["GetSamStatus", "LoadLabelModel"]

    def test_sam_load_unloads_a_loaded_labeling_engine_first(self, app, monkeypatch):
        calls = _wire(monkeypatch, sam,
                      model={"GetLabelModelStatus": inf_pb.LabelModelStatus(loaded=True),
                             "UnloadLabelModel": inf_pb.Result(success=True)},
                      training={"LoadSam": trn_pb.Result(success=True)})
        with app.test_request_context("/api/v1/training/sam/load", method="POST", json={}):
            resp = sam.training_sam_load()
        assert resp.status_code == 200
        assert [c[0] for c in calls] == ["GetLabelModelStatus", "UnloadLabelModel", "LoadSam"]
        calls.clear()
        _wire(monkeypatch, sam,
              model={"GetLabelModelStatus": inf_pb.LabelModelStatus(loaded=False)},
              training={"LoadSam": trn_pb.Result(success=True)})

    def test_service_refusal_is_a_400(self, app, monkeypatch):
        _wire(monkeypatch, label_model,
              model={"LoadLabelModel": inf_pb.Result(success=False,
                                                     message="not a TensorRT engine")},
              training={"GetSamStatus": trn_pb.SamStatus(loaded=False)})
        with app.test_request_context("/api/v1/training/label-model/load", method="POST",
                                      json={"model_name": "w.pt"}):
            resp = label_model.training_label_model_load()
        assert resp.status_code == 400
        assert "TensorRT" in resp.get_json()["error"]


class TestDetect:
    def test_validates_ids(self, app, monkeypatch):
        calls = _wire(monkeypatch, label_model, model={"LabelDetect": inf_pb.LabelDetectResult()})
        with app.test_request_context("/api/v1/training/label-model/detect", method="POST",
                                      json={"dataset_id": "d"}):
            resp = label_model.training_label_model_detect()
        assert resp.status_code == 400 and calls == []

    def test_fetches_the_dataset_image_and_maps_the_detections(self, app, monkeypatch):
        calls = _wire(
            monkeypatch, label_model,
            training={"GetImage": trn_pb.ImageBlob(image_id="i", jpeg=b"jpegbytes")},
            model={"LabelDetect": inf_pb.LabelDetectResult(
                success=True,
                detections=[inf_pb.LabelDetection(class_id=1, class_name="person", score=0.9,
                                                  x1=0.25, y1=0.25, x2=0.5, y2=0.5)])})
        with app.test_request_context("/api/v1/training/label-model/detect", method="POST",
                                      json={"dataset_id": "d", "image_id": "i",
                                            "threshold": 0.4}):
            resp = label_model.training_label_model_detect()
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["class_names"] == ["person"]
        assert body["scores"] == pytest.approx([0.9])
        box = body["boxes"][0]
        assert box["class_id"] == 1
        assert (box["cx"], box["cy"], box["w"], box["h"]) == pytest.approx(
            (0.375, 0.375, 0.25, 0.25))
        assert [c[0] for c in calls] == ["GetImage", "LabelDetect"]
        assert calls[1][1].jpeg == b"jpegbytes"
        assert calls[1][1].threshold == pytest.approx(0.4)

    def test_service_failure_is_a_400(self, app, monkeypatch):
        _wire(monkeypatch, label_model,
              training={"GetImage": trn_pb.ImageBlob(jpeg=b"x")},
              model={"LabelDetect": inf_pb.LabelDetectResult(success=False, message="no model")})
        with app.test_request_context("/api/v1/training/label-model/detect", method="POST",
                                      json={"dataset_id": "d", "image_id": "i"}):
            resp = label_model.training_label_model_detect()
        assert resp.status_code == 400


class TestTrainBaseModel:
    def test_base_model_is_forwarded(self, app, monkeypatch):
        calls = _wire(monkeypatch, jobs, training={"StartTraining": trn_pb.TrainingJob(
            job_id="j1", status="preparing", model_name="Teste2", base_model="Teste.engine")})
        monkeypatch.setattr(jobs, "_release_runtime",
                            lambda: SimpleNamespace(success=True, message="ok"))
        monkeypatch.setattr(jobs, "tracker", SimpleNamespace(arm=lambda: None))
        with app.test_request_context("/api/v1/training/train", method="POST",
                                      json={"dataset_id": "d", "model_name": "Teste2",
                                            "base_model": " Teste.engine "}):
            resp = jobs.training_start()
        assert resp.status_code == 202
        assert calls[0][1].base_model == "Teste.engine"
        assert resp.get_json()["base_model"] == "Teste.engine"
