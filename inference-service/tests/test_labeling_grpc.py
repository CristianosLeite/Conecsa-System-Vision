# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

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
        self.image_class = None
        self.candidates = []
        self.task = "detect"

    @property
    def is_loaded(self):
        # A property, as on LabelingService: calling it would be a TypeError.
        return bool(self.loaded)

    def status(self):
        return {"loaded": bool(self.loaded), "model_name": self.loaded,
                "class_names": ["logo", "person"] if self.loaded else [], "message": "",
                "task": self.task if self.loaded else ""}

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
        return {"detections": self.detections, "image_class": self.image_class,
                "candidates": self.candidates}


@pytest.fixture
def app(tmp_path):
    (tmp_path / "Teste.engine").write_bytes(b"engine")
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "Teste.pt").write_bytes(b"checkpoint-bytes")
    models = ModelService(Config(), str(tmp_path))
    application = SimpleNamespace(
        config=Config(),
        model_service=models,
        labeling_service=FakeLabeling(),
        detection_service=SimpleNamespace(is_running=False, stop=lambda: True,
                                          unload_runtime=lambda: None),
        conversion_service=SimpleNamespace(get_active_jobs=lambda: []),
        event_service=SimpleNamespace(publish=lambda *a, **k: None),
    )
    models.attach_detection_service(application.detection_service)
    models.attach_conversion_service(application.conversion_service)
    models.attach_labeling_service(application.labeling_service)
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
        assert s.task == "detect"

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
        assert not r.HasField("image_class") and list(r.candidates) == []

    def test_detect_maps_segmentation_rings_as_flat_points(self, app):
        svc = ig.ModelControlServicer(app)
        app.labeling_service.loaded = "Parts.engine"
        app.labeling_service.detections = [
            {"class_id": 0, "class_name": "bolt", "score": 0.8,
             "x1": 0.1, "y1": 0.2, "x2": 0.5, "y2": 0.6,
             "rings": [[[0.1, 0.2], [0.5, 0.2], [0.5, 0.6]]]},
        ]
        r = svc.LabelDetect(pb.LabelDetectRequest(jpeg=b"jpg"), None)
        (ring,) = r.detections[0].rings
        assert list(ring.points) == pytest.approx([0.1, 0.2, 0.5, 0.2, 0.5, 0.6])

    def test_detect_maps_a_classification_suggestion(self, app):
        svc = ig.ModelControlServicer(app)
        app.labeling_service.loaded = "Pets.engine"
        app.labeling_service.image_class = {"class_id": 0, "class_name": "cat", "score": 0.8}
        app.labeling_service.candidates = [
            {"class_id": 0, "class_name": "cat", "score": 0.8},
            {"class_id": 1, "class_name": "dog", "score": 0.2},
        ]
        r = svc.LabelDetect(pb.LabelDetectRequest(jpeg=b"jpg"), None)
        assert r.success and list(r.detections) == []
        # Class 0 is a real class: presence, not the value, says "has a class".
        assert r.HasField("image_class") and r.image_class.class_id == 0
        assert r.image_class.class_name == "cat"
        assert [c.class_name for c in r.candidates] == ["cat", "dog"]
        assert r.candidates[1].score == pytest.approx(0.2)

    def test_detect_below_the_threshold_has_no_image_class(self, app):
        svc = ig.ModelControlServicer(app)
        app.labeling_service.loaded = "Pets.engine"
        app.labeling_service.candidates = [{"class_id": 0, "class_name": "cat", "score": 0.4}]
        r = svc.LabelDetect(pb.LabelDetectRequest(jpeg=b"jpg"), None)
        assert r.success and not r.HasField("image_class") and len(r.candidates) == 1

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
        # The handover is marked, so leaving training mode can resume detection.
        assert app.model_service.runtime_released is True

    def test_refused_while_a_conversion_runs(self, app, monkeypatch):
        import api.runtime_management.worker_client as wc
        monkeypatch.setattr(wc, "release_all_workers", lambda: None)
        app.conversion_service.get_active_jobs = lambda: [SimpleNamespace(job_id="job")]
        r = ig.ManagementControlServicer(app).ReleaseRuntime(pb.Empty(), None)
        assert not r.success and "conversion" in r.message
        assert app.model_service.runtime_released is False


class TestFreeGpu:
    @pytest.fixture
    def released(self, monkeypatch):
        calls = []
        import api.runtime_management.worker_client as wc
        monkeypatch.setattr(wc, "release_all_workers", lambda: calls.append("workers"))
        return calls

    def test_frees_every_idle_holder(self, app, released):
        app.labeling_service.loaded = "Teste.engine"
        r = ig.ManagementControlServicer(app).FreeGpu(pb.FreeGpuRequest(), None)
        assert r.success and r.busy == pb.GPU_HOLDER_NONE
        assert app.labeling_service.loaded == ""
        assert released == ["workers"]

    def test_names_a_running_detection(self, app, released):
        app.detection_service.is_running = True
        r = ig.ManagementControlServicer(app).FreeGpu(pb.FreeGpuRequest(), None)
        assert not r.success and r.busy == pb.GPU_HOLDER_DETECTION
        assert "stop it first" in r.message
        assert released == []

    def test_stops_detection_on_request(self, app, released):
        app.detection_service.is_running = True

        def stop():
            was, app.detection_service.is_running = app.detection_service.is_running, False
            return was

        app.detection_service.stop = stop
        r = ig.ManagementControlServicer(app).FreeGpu(
            pb.FreeGpuRequest(stop_detection=True), None)
        assert r.success and app.detection_service.is_running is False
        assert released == ["workers"]

    def test_names_a_running_conversion(self, app, released):
        app.conversion_service.get_active_jobs = lambda: [SimpleNamespace(job_id="job")]
        r = ig.ManagementControlServicer(app).FreeGpu(
            pb.FreeGpuRequest(stop_detection=True), None)
        assert not r.success and r.busy == pb.GPU_HOLDER_CONVERSION
        assert released == []

    def test_ends_a_leftover_handover_and_says_so(self, app, released):
        events = []
        app.event_service = SimpleNamespace(
            publish=lambda event_type, keys=None, data=None: events.append((event_type, data)))
        servicer = ig.ManagementControlServicer(app)
        assert servicer.ReleaseRuntime(pb.Empty(), None).success
        events.clear()
        assert servicer.FreeGpu(pb.FreeGpuRequest(), None).success
        assert app.model_service.runtime_released is False
        assert events == [("runtime_changed", {"runtime_released": False})]
        # Nothing to end the second time: no event.
        events.clear()
        assert servicer.FreeGpu(pb.FreeGpuRequest(), None).success
        assert events == []

    def test_begins_a_handover_in_the_same_call(self, app, released):
        events = []
        app.event_service = SimpleNamespace(
            publish=lambda event_type, keys=None, data=None: events.append((event_type, data)))
        app.detection_service.is_running = True

        def stop():
            was, app.detection_service.is_running = app.detection_service.is_running, False
            return was

        app.detection_service.stop = stop
        servicer = ig.ManagementControlServicer(app)
        r = servicer.FreeGpu(pb.FreeGpuRequest(stop_detection=True, begin_handover=True), None)
        assert r.success and app.model_service.runtime_released is True
        assert app.detection_service.is_running is False
        assert released == ["workers"]
        assert events == [("runtime_changed", {"runtime_released": True})]
        # Re-entering keeps the handover: no second event.
        events.clear()
        assert servicer.FreeGpu(
            pb.FreeGpuRequest(stop_detection=True, begin_handover=True), None).success
        assert events == []

    def test_a_worker_that_will_not_exit_fails_the_release(self, app, monkeypatch):
        import api.runtime_management.worker_client as wc

        def stuck():
            raise RuntimeError("Could not terminate the TensorRT workers on ports 5501")

        monkeypatch.setattr(wc, "release_all_workers", stuck)
        servicer = ig.ManagementControlServicer(app)
        r = servicer.FreeGpu(pb.FreeGpuRequest(begin_handover=True), None)
        assert not r.success and r.busy == pb.GPU_HOLDER_NONE and "5501" in r.message
        assert app.model_service.runtime_released is False

    def test_a_failed_release_restarts_the_detection_it_stopped(self, app, monkeypatch):
        import api.runtime_management.worker_client as wc

        def stuck():
            raise RuntimeError("Could not terminate the TensorRT workers on ports 5501")

        monkeypatch.setattr(wc, "release_all_workers", stuck)
        # The worker did exit after all: nothing is left beside a restart.
        monkeypatch.setattr(wc, "live_worker_ports", lambda: [])
        detection = app.detection_service
        detection.is_running = True
        calls = []

        def stop():
            calls.append("stop")
            was, detection.is_running = detection.is_running, False
            return was

        def start():
            calls.append("start")
            detection.is_running = True
            return True

        detection.stop, detection.start = stop, start
        detection.unload_runtime = lambda: calls.append("unload")
        r = ig.ManagementControlServicer(app).FreeGpu(
            pb.FreeGpuRequest(stop_detection=True, begin_handover=True), None)
        assert not r.success and "5501" in r.message
        # Entering training mode was refused: detection runs again, as before.
        assert detection.is_running is True
        assert calls[-2:] == ["unload", "start"]
        assert app.model_service.runtime_released is False

    def test_a_failed_release_leaves_a_stopped_detection_stopped(self, app, monkeypatch):
        import api.runtime_management.worker_client as wc

        def stuck():
            raise RuntimeError("Could not terminate the TensorRT workers on ports 5501")

        monkeypatch.setattr(wc, "release_all_workers", stuck)
        started = []
        app.detection_service.start = lambda: started.append(True)
        r = ig.ManagementControlServicer(app).FreeGpu(
            pb.FreeGpuRequest(stop_detection=True), None)
        assert not r.success
        assert started == []

    def test_a_resident_labeling_engine_keeps_detection_stopped(self, app, monkeypatch):
        import api.runtime_management.worker_client as wc
        monkeypatch.setattr(wc, "release_all_workers", lambda: None)
        labeling = app.labeling_service
        labeling.loaded = "Teste.engine"

        def stuck_unload():
            raise RuntimeError("labeling worker did not exit")

        labeling.unload = stuck_unload
        detection = app.detection_service
        detection.is_running = True
        started = []

        def stop():
            was, detection.is_running = detection.is_running, False
            return was

        detection.stop = stop
        detection.start = lambda: started.append(True)
        r = ig.ManagementControlServicer(app).FreeGpu(
            pb.FreeGpuRequest(stop_detection=True, begin_handover=True), None)
        assert not r.success and "did not exit" in r.message
        # Its worker still holds CUDA memory: no detection is started beside it.
        assert started == [] and detection.is_running is False

    def test_a_surviving_worker_keeps_detection_stopped(self, app, monkeypatch):
        import api.runtime_management.worker_client as wc

        def stuck():
            raise RuntimeError("Could not terminate the TensorRT workers on ports 5517")

        monkeypatch.setattr(wc, "release_all_workers", stuck)
        # E.g. the face embedder's private worker, which the labeling
        # service knows nothing about.
        monkeypatch.setattr(wc, "live_worker_ports", lambda: [5517])
        detection = app.detection_service
        detection.is_running = True
        started = []

        def stop():
            was, detection.is_running = detection.is_running, False
            return was

        detection.stop = stop
        detection.start = lambda: started.append(True)
        r = ig.ManagementControlServicer(app).FreeGpu(
            pb.FreeGpuRequest(stop_detection=True, begin_handover=True), None)
        assert not r.success and "5517" in r.message
        assert started == [] and detection.is_running is False
