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
from api.services import labeling_service as ls
from api.services.labeling_service import LabelingService, label_worker_port
from api.services.model_service import ModelService


class FakeManager:
    """Single-tile manager: records how it was built, returns a canned output."""

    instances = []

    def __init__(self, config, port=None):
        self.config = config
        self.port = port
        self.tiling_active = False
        self.inferred = 0
        FakeManager.instances.append(self)

    def preprocess_tiles(self, frame):
        h, w = frame.shape[:2]
        return [np.zeros((1, 3, 8, 8), np.float32)], [
            SimpleNamespace(scale=1.0, border_top=0, input_size=8, ox=0, oy=0,
                            width=w, height=h)]

    def run_inference(self, tensor):
        self.inferred += 1
        return np.zeros((1, 6, 1)), 0.001


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


@pytest.fixture
def wired(monkeypatch, tmp_path):
    FakeManager.instances = []
    monkeypatch.setattr(ls, "ModelManager", FakeManager)
    monkeypatch.setattr(ls, "YOLODetector", FakeDetector)
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
                                "class_names": ["logo", "person"], "message": ""}
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

        def broken(config, port=None):
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
        assert out == [{"class_id": 1, "class_name": "person", "score": 0.9,
                        "x1": 0.25, "y1": 0.25, "x2": 0.5, "y2": 0.5}]
        assert FakeManager.instances[-1].inferred == 1
        assert svc._detector.calls == [pytest.approx(0.4)]
        assert svc._detector.areas == []

    def test_zero_threshold_falls_back_to_the_default(self, wired):
        svc, _, _ = wired
        svc.load("Teste.engine")
        svc.detect(_jpeg(), threshold=0.0)
        assert svc._detector.calls == [ls.DEFAULT_THRESHOLD]

    def test_undecodable_image_is_rejected(self, wired):
        svc, _, _ = wired
        svc.load("Teste.engine")
        with pytest.raises(ValueError, match="decode"):
            svc.detect(b"not a jpeg")
