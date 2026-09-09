"""Tests for the model-assisted labeling RPCs (inference_grpc.ModelControlServicer)
and the labeling engine's teardown on ReleaseRuntime.

The servicers are thin adapters: they must relay to LabelingService, map its
exceptions to Result/LabelDetectResult failures (never gRPC errors, the
gateway turns them into 4xx) and stream the weights sidecar like a model.
"""
import os
from types import SimpleNamespace

import api.inference_grpc as ig
import inference_pb2 as pb
import pytest
from api.config import Config
from api.services.model_service import ModelService


class FakeLabeling:
    def __init__(self):
        self.loaded = ""
        self.unloads = 0
        self.fail_load = None
        self.detections = []

    def status(self):
        return {"loaded": bool(self.loaded), "model_name": self.loaded,
                "class_names": ["logo", "person"] if self.loaded else [], "message": ""}

    def load(self, name):
        if self.fail_load:
            raise self.fail_load
        self.loaded = name

    def unload(self):
        self.unloads += 1
        self.loaded = ""

    def detect(self, jpeg, threshold):
        if not self.loaded:
            raise RuntimeError("No labeling model is loaded")
        self.last = (jpeg, threshold)
        return self.detections


@pytest.fixture
def app(tmp_path):
    (tmp_path / "Teste.engine").write_bytes(b"engine")
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "Teste.pt").write_bytes(b"checkpoint-bytes")
    application = SimpleNamespace(
        config=Config(),
        model_service=ModelService(Config(), str(tmp_path)),
        labeling_service=FakeLabeling(),
        detection_service=SimpleNamespace(stop=lambda: True),
        conversion_service=SimpleNamespace(get_active_jobs=lambda: []),
        event_service=SimpleNamespace(publish=lambda *a, **k: None),
    )
    return application


class TestLabelRpcs:
    def test_status_serializes_the_service_state(self, app):
        svc = ig.ModelControlServicer(app)
        s = svc.GetLabelModelStatus(pb.Empty(), None)
        assert s.loaded is False and s.model_name == "" and list(s.class_names) == []
        app.labeling_service.loaded = "Teste.engine"
        s = svc.GetLabelModelStatus(pb.Empty(), None)
        assert s.loaded is True and s.model_name == "Teste.engine"
        assert list(s.class_names) == ["logo", "person"]

    def test_load_and_unload_relay_and_report_failures_as_results(self, app):
        svc = ig.ModelControlServicer(app)
        r = svc.LoadLabelModel(pb.ModelName(name="Teste.engine"), None)
        assert r.success and app.labeling_service.loaded == "Teste.engine"
        app.labeling_service.fail_load = FileNotFoundError("Model 'x.engine' not found")
        r = svc.LoadLabelModel(pb.ModelName(name="x.engine"), None)
        assert not r.success and "not found" in r.message
        r = svc.UnloadLabelModel(pb.Empty(), None)
        assert r.success and app.labeling_service.unloads == 1

    def test_detect_relays_the_image_and_threshold(self, app):
        svc = ig.ModelControlServicer(app)
        app.labeling_service.loaded = "Teste.engine"
        app.labeling_service.detections = [
            {"class_id": 1, "class_name": "person", "score": 0.9,
             "x1": 0.1, "y1": 0.2, "x2": 0.5, "y2": 0.6},
        ]
        r = svc.LabelDetect(pb.LabelDetectRequest(jpeg=b"jpg", threshold=0.4), None)
        assert r.success and len(r.detections) == 1
        d = r.detections[0]
        assert d.class_id == 1 and d.class_name == "person"
        assert d.score == pytest.approx(0.9) and d.x2 == pytest.approx(0.5)
        assert app.labeling_service.last == (b"jpg", pytest.approx(0.4))

    def test_detect_without_an_engine_is_a_failed_result(self, app):
        svc = ig.ModelControlServicer(app)
        r = svc.LabelDetect(pb.LabelDetectRequest(jpeg=b"jpg"), None)
        assert not r.success and "No labeling model" in r.message


class FakeContext:
    def __init__(self):
        self.code = None
        self.details = ""

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


class TestWeightsDownload:
    def test_streams_the_sidecar(self, app):
        svc = ig.ModelControlServicer(app)
        ctx = FakeContext()
        chunks = list(svc.DownloadModelWeights(pb.ModelName(name="Teste.engine"), ctx))
        assert b"".join(c.chunk for c in chunks) == b"checkpoint-bytes"
        assert ctx.code is None

    def test_model_without_sidecar_is_not_found(self, app):
        import grpc
        svc = ig.ModelControlServicer(app)
        os.remove(os.path.join(app.model_service.model_directory, "weights", "Teste.pt"))
        ctx = FakeContext()
        assert list(svc.DownloadModelWeights(pb.ModelName(name="Teste.engine"), ctx)) == []
        assert ctx.code == grpc.StatusCode.NOT_FOUND
        ctx = FakeContext()
        assert list(svc.DownloadModelWeights(pb.ModelName(name="../x.engine"), ctx)) == []
        assert ctx.code == grpc.StatusCode.NOT_FOUND

    def test_list_models_reports_has_weights(self, app):
        svc = ig.ModelControlServicer(app)
        ml = svc.ListModels(pb.Empty(), None)
        assert {m.name: m.has_weights for m in ml.models} == {"Teste.engine": True}


class TestReleaseRuntime:
    def test_unloads_the_labeling_engine_before_the_workers(self, app, monkeypatch):
        order = []
        import api.runtime_management.worker_client as wc
        monkeypatch.setattr(wc, "release_all_workers", lambda: order.append("workers"))
        app.labeling_service.loaded = "Teste.engine"
        original_unload = app.labeling_service.unload

        def unload():
            order.append("labeling")
            original_unload()

        app.labeling_service.unload = unload
        r = ig.ManagementControlServicer(app).ReleaseRuntime(pb.Empty(), None)
        assert r.success, r.message
        assert order == ["labeling", "workers"]
        assert app.labeling_service.loaded == ""
