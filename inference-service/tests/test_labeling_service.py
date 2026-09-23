# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for LabelingService with the TensorRT manager and detector faked.

The service must reuse the live pipeline's building blocks (ModelManager on a
private port, YOLODetector over the model's classes sidecar) and return the
detections normalized on the submitted image.
"""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from api.config import Config
from api.models.detection_models import Detection
from api.postprocess import detect as pp_detect
from api.services import labeling_service as ls
from api.services.labeling_service import LabelingService, label_worker_port
from api.services.model_service import ModelService


class FakeManager:
    """Single-tile manager: records how it was built, returns a canned output."""

    instances = []

    def __init__(self, config, port=None, task="detect"):
        self.config = config
        self.port = port
        self.task = task
        self.tiling_active = False
        self.inferred = 0
        FakeManager.instances.append(self)

    def preprocess_tiles(self, frame):
        h, w = frame.shape[:2]
        # A square frame letterboxed to the 8 px input: scale = h / 8.
        scale = h / 8.0 if self.task == "segment" else 1.0
        return [np.zeros((1, 3, 8, 8), np.float32)], [
            SimpleNamespace(scale=scale, border_top=0, input_size=8, ox=0, oy=0,
                            width=w, height=h)]

    def run_inference(self, tensor):
        self.inferred += 1
        if self.task == "classify":
            return [np.array([[0.1, 0.8, 0.1]], np.float32)], 0.001
        if self.task == "segment":
            # One "nut" row over input px 2..6 with a full mask (nm = 1).
            rows = np.zeros((1, 4, 7), np.float32)
            rows[0, 0] = [2, 2, 6, 6, 0.9, 1, 1.0]
            return [rows, np.full((1, 1, 2, 2), 10.0, np.float32)], 0.001
        return [np.zeros((1, 6, 1))], 0.001


class FakeDetector:
    def __init__(self, class_labels, config):
        self.class_labels = list(class_labels)
        self.config = config
        self.areas = None
        self.calls = []

    def set_areas(self, areas):
        self.areas = areas

    def process_detections(self, output, frame, scale, border_top, actual_input_size):
        self.calls.append(self.config.CONFIDENCE_THRESHOLD)
        h, w = frame.shape[:2]
        det = Detection(class_id=1, class_name=self.class_labels[1], confidence=0.9,
                        bbox=(w // 4, h // 4, w // 2, h // 2), center=(0, 0))
        return frame, 1, [det]


def _fake_detector(svc) -> FakeDetector:
    """The FakeDetector behind the labeling service's detect postprocessor."""
    return svc._detector.detector  # type: ignore[union-attr]


@pytest.fixture
def wired(monkeypatch, tmp_path):
    FakeManager.instances = []
    monkeypatch.setattr(ls, "ModelManager", FakeManager)
    monkeypatch.setattr(pp_detect, "YOLODetector", FakeDetector)
    (tmp_path / "Teste.engine").write_bytes(b"engine")
    (tmp_path / "Teste.txt").write_text("logo\nperson\n")
    models = ModelService(Config(), str(tmp_path))
    events = []
    svc = LabelingService(Config(), models, port=5599,
                          event_service=SimpleNamespace(
                              publish=lambda ev, keys=None, source=None, data=None:
                              events.append((ev, data))))
    return svc, events, tmp_path


def _jpeg(w=64, h=32):
    ok, buf = cv2.imencode(".jpg", np.zeros((h, w, 3), np.uint8))
    assert ok
    return buf.tobytes()


class TestPort:
    def test_defaults_past_the_context_lanes(self, monkeypatch):
        monkeypatch.setenv("TENSORRT_WORKER_PORT", "5501")
        monkeypatch.delenv("TENSORRT_LABEL_WORKER_PORT", raising=False)
        assert label_worker_port() == 5517

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("TENSORRT_LABEL_WORKER_PORT", "6000")
        assert label_worker_port() == 6000


class TestLoad:
    def test_builds_the_manager_on_the_private_port_with_the_sidecar_classes(self, wired):
        svc, events, tmp_path = wired
        svc.load("Teste.engine")
        manager = FakeManager.instances[-1]
        assert manager.port == 5599
        assert manager.config.MODEL_PATH == str(tmp_path / "Teste.engine")
        assert svc.status() == {"loaded": True, "model_name": "Teste.engine",
                                "class_names": ["logo", "person"], "message": "",
                                "task": "detect"}
        assert events[-1] == ("label_model_changed", svc.status())

    def test_rejects_unknown_and_non_engine_models(self, wired, tmp_path):
        svc, _, _ = wired
        with pytest.raises(FileNotFoundError):
            svc.load("Other.engine")
        (tmp_path / "w.pt").write_bytes(b"pt")
        with pytest.raises(ValueError, match="TensorRT"):
            svc.load("w.pt")
        # An .onnx is loadable by the runtime but would build an engine on
        # the private worker (minutes): prebuilt engines only.
        (tmp_path / "w.onnx").write_bytes(b"onnx")
        with pytest.raises(ValueError, match="TensorRT"):
            svc.load("w.onnx")
        with pytest.raises(ValueError):
            svc.load("../Teste.engine")
        assert svc.status()["loaded"] is False

    def test_same_engine_is_a_no_op_and_the_live_config_is_untouched(self, wired):
        svc, _, _ = wired
        svc.load("Teste.engine")
        svc.load("Teste.engine")
        assert len(FakeManager.instances) == 1
        assert svc._config.MODEL_PATH != FakeManager.instances[0].config.MODEL_PATH

    def test_failed_load_closes_the_spawned_worker(self, wired, monkeypatch):
        svc, _, _ = wired
        closed = []
        import api.runtime_management.worker_client as wc
        monkeypatch.setattr(wc, "get_worker_client",
                            lambda port=None: SimpleNamespace(close=lambda: closed.append(port)))

        def broken(config, port=None, task="detect"):
            raise RuntimeError("engine deserialization failed")

        monkeypatch.setattr(ls, "ModelManager", broken)
        with pytest.raises(RuntimeError, match="deserialization"):
            svc.load("Teste.engine")
        assert closed == [5599]
        assert svc.status()["loaded"] is False

    def test_unload_closes_the_private_worker(self, wired, monkeypatch):
        svc, events, _ = wired
        closed = []
        import api.runtime_management.worker_client as wc
        monkeypatch.setattr(wc, "get_worker_client",
                            lambda port=None: SimpleNamespace(close=lambda: closed.append(port)))
        svc.load("Teste.engine")
        svc.unload()
        assert closed == [5599]
        assert svc.status()["loaded"] is False
        assert events[-1][1]["loaded"] is False
        # Nothing loaded: no worker to close, still publishes the state.
        svc.unload()
        assert closed == [5599]


class TestDetect:
    def test_requires_a_loaded_engine(self, wired):
        svc, _, _ = wired
        with pytest.raises(RuntimeError, match="No labeling model"):
            svc.detect(_jpeg())

    def test_returns_normalized_corners_and_the_sidecar_class_name(self, wired):
        svc, _, _ = wired
        svc.load("Teste.engine")
        out = svc.detect(_jpeg(64, 32), threshold=0.4)
        assert out["detections"] == [{"class_id": 1, "class_name": "person", "score": 0.9,
                                      "x1": 0.25, "y1": 0.25, "x2": 0.5, "y2": 0.5,
                                      "rings": []}]
        # A detection engine suggests boxes only.
        assert out["image_class"] is None and out["candidates"] == []
        assert FakeManager.instances[-1].inferred == 1
        assert _fake_detector(svc).calls == [pytest.approx(0.4)]
        assert _fake_detector(svc).areas == []

    def test_zero_threshold_falls_back_to_the_default(self, wired):
        svc, _, _ = wired
        svc.load("Teste.engine")
        svc.detect(_jpeg(), threshold=0.0)
        assert _fake_detector(svc).calls == [ls.DEFAULT_THRESHOLD]

    def test_undecodable_image_is_rejected(self, wired):
        svc, _, _ = wired
        svc.load("Teste.engine")
        with pytest.raises(ValueError, match="decode"):
            svc.detect(b"not a jpeg")


class TestClassify:
    """A classification engine suggests the image's class, not boxes."""

    @pytest.fixture
    def classify(self, wired):
        svc, events, tmp_path = wired
        (tmp_path / "Pets.engine").write_bytes(b"engine")
        (tmp_path / "Pets.txt").write_text("cat\ndog\nbird\n")
        (tmp_path / "Pets.settings.json").write_text('{"task": "classify"}')
        svc.load("Pets.engine")
        return svc

    def test_loads_the_classification_strategy_and_reports_its_task(self, classify):
        assert FakeManager.instances[-1].task == "classify"
        assert classify.loaded_task() == "classify"
        assert classify.status()["task"] == "classify"

    def test_suggests_the_top_class_and_the_candidates(self, classify):
        out = classify.detect(_jpeg(), threshold=0.5)
        assert out["detections"] == []
        assert out["image_class"] == {"class_id": 1, "class_name": "dog",
                                      "score": pytest.approx(0.8)}
        # Highest first; the 0.1 tie keeps the lower class index first.
        assert [c["class_name"] for c in out["candidates"]] == ["dog", "cat", "bird"]

    def test_below_the_threshold_there_is_no_class_but_still_candidates(self, classify):
        out = classify.detect(_jpeg(), threshold=0.9)
        assert out["image_class"] is None
        assert len(out["candidates"]) == 3

    def test_each_image_is_judged_on_its_own(self, classify):
        # The live transition state must not leak into labeling: the same
        # class twice is suggested twice.
        assert classify.detect(_jpeg())["image_class"] is not None
        assert classify.detect(_jpeg())["image_class"] is not None


class TestSegment:
    """A segmentation engine suggests boxes with their rings."""

    @pytest.fixture
    def segment(self, wired):
        svc, _, tmp_path = wired
        (tmp_path / "Parts.engine").write_bytes(b"engine")
        (tmp_path / "Parts.txt").write_text("bolt\nnut\n")
        (tmp_path / "Parts.settings.json").write_text('{"task": "segment"}')
        svc.load("Parts.engine")
        return svc

    def test_loads_the_segmentation_strategy(self, segment):
        assert FakeManager.instances[-1].task == "segment"
        assert segment.loaded_task() == "segment"

    def test_suggests_boxes_with_normalized_rings(self, segment):
        out = segment.detect(_jpeg(64, 64), threshold=0.5)
        (det,) = out["detections"]
        assert det["class_name"] == "nut"
        assert (det["x1"], det["y1"], det["x2"], det["y2"]) == (0.25, 0.25, 0.75, 0.75)
        (ring,) = det["rings"]
        xs, ys = [p[0] for p in ring], [p[1] for p in ring]
        assert min(xs) == pytest.approx(0.25, abs=0.02) and max(xs) == pytest.approx(0.75, abs=0.02)
        assert min(ys) == pytest.approx(0.25, abs=0.02) and max(ys) == pytest.approx(0.75, abs=0.02)
        assert out["image_class"] is None and out["candidates"] == []
