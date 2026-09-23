# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""DetectionService under an application type:
the start gate, the model-task gate, the engine output contract, the
snapshot's task, and the sidecar keeping the model's task."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from api.config import Config
from api.services import detection_service as mod
from api.services.application_service import ApplicationService
from api.services.detection_service import DetectionService
from api.services.errors import PreconditionFailed
from api.services.model_settings_service import ModelSettingsService


def _app(tmp_path, task):
    (tmp_path / "application.json").write_text(json.dumps({"task": task}))
    app = ApplicationService(str(tmp_path), ["detect"])
    app.load_or_migrate()
    return app


def _config(tmp_path, task=None):
    (tmp_path / "m.engine").write_bytes(b"engine")
    if task is not None:
        (tmp_path / "m.settings.json").write_text(json.dumps({"task": task}))
    config = Config()
    config.MODEL_PATH = str(tmp_path / "m.engine")
    return config


def _manager_with_outputs(*shapes):
    return lambda cfg, **kw: SimpleNamespace(
        tiling_active=False,
        output_details=[{"shape": list(s)} for s in shapes],
        input_details=[{"shape": [1, 3, 640, 640]}],
    )


class TestGates:
    def test_start_is_refused_without_an_application(self, tmp_path):
        svc = DetectionService(Config(), application_service=_app(tmp_path, None))
        with pytest.raises(PreconditionFailed, match="No application type"):
            svc.start()
        assert svc.is_running is False

    def test_a_model_of_another_task_is_refused_before_loading(self, tmp_path, monkeypatch):
        def never(cfg, **kw):
            raise AssertionError("no engine may load for a model of another task")

        monkeypatch.setattr(mod, "ModelManager", never)
        svc = DetectionService(_config(tmp_path, "classify"),
                               application_service=_app(tmp_path, "detect"))
        with pytest.raises(PreconditionFailed, match="'classify' model"):
            svc.initialize()
        assert svc.model_manager is None

    def test_a_model_without_a_recorded_task_is_a_detection_model(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "ModelManager", _manager_with_outputs((1, 300, 6)))
        monkeypatch.setattr(mod, "load_class_labels", lambda cfg: ["a"])
        svc = DetectionService(_config(tmp_path), application_service=_app(tmp_path, "detect"))
        assert svc.initialize() is True
        assert svc.postprocessor is not None and svc.postprocessor.task == "detect"

    def test_an_engine_of_another_layout_is_refused_and_nothing_is_swapped(self, tmp_path,
                                                                           monkeypatch):
        monkeypatch.setattr(mod, "ModelManager", _manager_with_outputs((1, 5)))
        monkeypatch.setattr(mod, "load_class_labels", lambda cfg: ["a"])
        svc = DetectionService(_config(tmp_path, "detect"),
                               application_service=_app(tmp_path, "detect"))
        before = svc.generation
        with pytest.raises(RuntimeError, match="looks like a 'classify' model"):
            svc.initialize()
        assert svc.model_manager is None and svc.postprocessor is None
        assert svc.generation == before


class TestResults:
    def test_the_snapshot_carries_the_application_task(self, tmp_path):
        # Without an application service the device behaves as detection.
        assert DetectionService(Config()).detections_snapshot()["task"] == "detect"
        blank = DetectionService(Config(), application_service=_app(tmp_path, None))
        assert blank.detections_snapshot()["task"] is None

    def test_the_snapshot_names_no_model_without_a_runtime(self):
        # Regression: a blank device or a deselect reported the config's
        # default "weights.engine".
        svc = DetectionService(Config())
        assert svc.detections_snapshot()["model"] == ""
        svc.model_manager = SimpleNamespace(  # type: ignore[assignment]
            acceleration_type="TensorRT", runtime_api="TensorRT")
        assert svc.detections_snapshot()["model"] == svc.config.MODEL_PATH.split("/")[-1]
        svc.unload_runtime()
        assert svc.detections_snapshot()["model"] == ""

    def test_reset_results_drops_the_last_result(self):
        svc = DetectionService(Config())
        svc.last_detection_result = SimpleNamespace()  # type: ignore[assignment]
        svc.reset_results()
        assert svc.last_detection_result is None

    def test_unload_runtime_forgets_the_model_and_rejects_older_frames(self):
        svc = DetectionService(Config())
        svc.model_manager = SimpleNamespace()  # type: ignore[assignment]
        before = svc.generation
        svc.unload_runtime()
        assert svc.model_manager is None and svc.postprocessor is None
        assert svc.generation == before + 1


class TestSidecarTask:
    def test_a_threshold_snapshot_keeps_the_task(self, tmp_path):
        path = tmp_path / "m.settings.json"
        path.write_text(json.dumps({"task": "classify", "imgsz": 224}))
        settings = ModelSettingsService(Config(), None)
        settings.switch_model(str(path))  # no thresholds yet → seeded by save()
        data = json.loads(path.read_text())
        assert data["task"] == "classify"
        assert data["imgsz"] == 224
        assert "thresholds" in data
        assert ModelSettingsService.task_of(str(path)) == "classify"

    def test_a_missing_sidecar_reads_as_detect(self, tmp_path):
        assert ModelSettingsService.task_of(str(tmp_path / "none.settings.json")) == "detect"


class TestClassifyActivation:
    """A classification engine must output probabilities (softmax in the graph)."""

    @staticmethod
    def _service(tmp_path, monkeypatch, probs):
        (tmp_path / "application.json").write_text(json.dumps({"task": "classify"}))
        app = ApplicationService(str(tmp_path), ["detect", "classify"])
        app.load_or_migrate()
        monkeypatch.setattr(mod, "ModelManager", lambda cfg, **kw: SimpleNamespace(
            tiling_active=False, input_size=224,
            output_details=[{"shape": [1, len(probs)]}],
            input_details=[{"shape": [1, 3, 224, 224]}],
            preprocess_tiles=lambda frame: ([frame], [None]),
            run_inference=lambda tensor: ([np.array([probs], np.float32)], 0.0),
        ))
        monkeypatch.setattr(mod, "load_class_labels", lambda cfg: ["cat", "dog"])
        return DetectionService(_config(tmp_path, "classify"), application_service=app)

    def test_a_softmax_engine_activates_as_classification(self, tmp_path, monkeypatch):
        svc = self._service(tmp_path, monkeypatch, [0.3, 0.7])
        assert svc.initialize() is True
        assert svc.postprocessor is not None and svc.postprocessor.task == "classify"

    def test_a_logits_engine_is_refused_and_nothing_is_swapped(self, tmp_path, monkeypatch):
        svc = self._service(tmp_path, monkeypatch, [2.5, -1.0])
        before = svc.generation
        with pytest.raises(RuntimeError, match="sum to 1"):
            svc.initialize()
        assert svc.model_manager is None and svc.postprocessor is None
        assert svc.generation == before


class TestSegmentActivation:
    """Only the end-to-end segmentation head (rows + prototypes) activates."""

    @staticmethod
    def _service(tmp_path, monkeypatch, *shapes):
        (tmp_path / "application.json").write_text(json.dumps({"task": "segment"}))
        app = ApplicationService(str(tmp_path), ["detect", "classify", "segment"])
        app.load_or_migrate()
        monkeypatch.setattr(mod, "ModelManager", _manager_with_outputs(*shapes))
        monkeypatch.setattr(mod, "load_class_labels", lambda cfg: ["bolt", "nut"])
        return DetectionService(_config(tmp_path, "segment"), application_service=app)

    def test_an_end_to_end_engine_activates_as_segmentation(self, tmp_path, monkeypatch):
        svc = self._service(tmp_path, monkeypatch, [1, 300, 38], [1, 32, 160, 160])
        assert svc.initialize() is True
        assert svc.postprocessor is not None and svc.postprocessor.task == "segment"

    def test_a_legacy_engine_is_refused_and_nothing_is_swapped(self, tmp_path, monkeypatch):
        svc = self._service(tmp_path, monkeypatch, [1, 116, 8400], [1, 32, 160, 160])
        before = svc.generation
        with pytest.raises(RuntimeError):
            svc.initialize()
        assert svc.model_manager is None and svc.postprocessor is None
        assert svc.generation == before


class TestFaceStrategyBinding:
    def test_a_strategy_that_fails_to_bind_frees_its_worker(self, tmp_path, monkeypatch):
        """The face strategy opens its embedder worker in its constructor; a
        refused output binding must close it, since the strategy never
        becomes ``self.postprocessor`` for the task-switch cleanup to reach."""
        closed = []

        class FakeFace:
            task = "face"

            def __init__(self, *args, **kwargs):
                pass

            def bind_outputs(self, details):
                raise mod.ContractError("the face detector has no 'cls_8' output")

            def close(self):
                closed.append(True)

        monkeypatch.setattr(mod, "ModelManager", _manager_with_outputs((1, 5)))
        monkeypatch.setattr(mod.DetectionService, "_check_contract", staticmethod(lambda m, t: None))
        monkeypatch.setattr(mod, "load_class_labels", lambda cfg: ["Ana"])
        monkeypatch.setattr(mod.postprocess, "FacePostprocessor", FakeFace)
        monkeypatch.setattr(mod.postprocess, "create", lambda task, labels, cfg: FakeFace())
        (tmp_path / "application.json").write_text(json.dumps({"task": "face"}))
        app = ApplicationService(str(tmp_path), ["detect", "face"])
        app.load_or_migrate()
        svc = DetectionService(_config(tmp_path, "face"), application_service=app)

        with pytest.raises(RuntimeError, match="no 'cls_8' output"):
            svc.initialize()

        assert closed == [True]
        assert svc.postprocessor is None and svc.model_manager is None

    def test_switching_between_face_models_hands_the_worker_over(self, tmp_path, monkeypatch):
        """One SFace worker per device: the second face model reuses the
        first one's embedder instead of opening another on the same port."""
        created, closed = [], []

        class FakeEmbedder:
            def close(self):
                closed.append(self)

        class FakeFace:
            task = "face"

            def __init__(self, labels, cfg, embedder=None):
                self.embedder = embedder if embedder is not None else FakeEmbedder()
                created.append(self.embedder)

            def bind_outputs(self, details):
                pass

            def close(self):
                self.embedder.close()

        monkeypatch.setattr(mod, "ModelManager", _manager_with_outputs((1, 5)))
        monkeypatch.setattr(mod.DetectionService, "_check_contract", staticmethod(lambda m, t: None))
        monkeypatch.setattr(mod, "load_class_labels", lambda cfg: ["Ana"])
        monkeypatch.setattr(mod.postprocess, "FacePostprocessor", FakeFace)
        monkeypatch.setattr(mod.postprocess, "supported_tasks", lambda: ["detect", "face"])
        monkeypatch.setattr(mod.postprocess, "create", lambda task, labels, cfg: FakeFace(labels, cfg))
        (tmp_path / "application.json").write_text(json.dumps({"task": "face"}))
        app = ApplicationService(str(tmp_path), ["detect", "face"])
        app.load_or_migrate()
        svc = DetectionService(_config(tmp_path, "face"), application_service=app)

        assert svc.initialize() is True
        first = svc.postprocessor
        assert svc.initialize() is True

        assert svc.postprocessor is not first
        assert len(created) == 2 and created[0] is created[1] and closed == []

    def test_a_handed_over_worker_survives_a_failed_binding(self, tmp_path, monkeypatch):
        closed = []

        class FakeEmbedder:
            def close(self):
                closed.append(self)

        class FakeFace:
            task = "face"
            fail = False

            def __init__(self, labels, cfg, embedder=None):
                self.embedder = embedder if embedder is not None else FakeEmbedder()

            def bind_outputs(self, details):
                if FakeFace.fail:
                    raise mod.ContractError("no 'cls_8' output")

            def close(self):
                self.embedder.close()

        monkeypatch.setattr(mod, "ModelManager", _manager_with_outputs((1, 5)))
        monkeypatch.setattr(mod.DetectionService, "_check_contract", staticmethod(lambda m, t: None))
        monkeypatch.setattr(mod, "load_class_labels", lambda cfg: ["Ana"])
        monkeypatch.setattr(mod.postprocess, "FacePostprocessor", FakeFace)
        monkeypatch.setattr(mod.postprocess, "supported_tasks", lambda: ["detect", "face"])
        monkeypatch.setattr(mod.postprocess, "create", lambda task, labels, cfg: FakeFace(labels, cfg))
        (tmp_path / "application.json").write_text(json.dumps({"task": "face"}))
        app = ApplicationService(str(tmp_path), ["detect", "face"])
        app.load_or_migrate()
        svc = DetectionService(_config(tmp_path, "face"), application_service=app)
        assert svc.initialize() is True
        first = svc.postprocessor

        FakeFace.fail = True
        with pytest.raises(RuntimeError):
            svc.initialize()

        # The previous strategy, restored by the rollback, still owns a live worker.
        assert svc.postprocessor is first and closed == []
