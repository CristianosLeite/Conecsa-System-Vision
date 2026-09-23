# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Whole-frame classification: the strict threshold,
the top-k, the clean processed frame, and the transition truth table checked against the
counter increment, the snapshot, the offline-buffer rows and the
Node-RED-visible ``total``."""
from types import SimpleNamespace

import api.inference_grpc as ig
import api.services.detection_buffer as buffer_mod
import inference_pb2 as pb
import numpy as np
import pytest
from api.config import Config
from api.models.detection_models import DetectionResult
from api.postprocess.base import PostprocessResult
from api.postprocess.classify import ClassifyPostprocessor, classify_topk_from_env
from api.services.detection_service import DetectionService

LABELS = ["cat", "dog", "bird #00ff00"]
FRAME = np.full((72, 128, 3), 40, np.uint8)

#: One probability row per scene; "none" stays below the 0.5 threshold.
SCENES = {
    "none": [0.40, 0.35, 0.25],
    "A": [0.90, 0.05, 0.05],
    "B": [0.05, 0.90, 0.05],
}
CLASS_OF = {"A": "cat", "B": "dog"}

#: The thresholded class sequence → what each frame adds to the counter.
TRUTH_TABLE = [
    (["none", "A"], [0, 1]),
    (["A", "B"], [1, 1]),
    (["A", "none"], [1, 0]),
    (["A", "none", "A"], [1, 0, 1]),
    (["A", "A"], [1, 0]),
]


def _outputs(probs):
    """One frame's engine outputs: a single (untiled) entry with a [1, nc] row."""
    return [[np.array([probs], np.float32)]]


def _config(threshold=0.5):
    cfg = Config()
    cfg.CONFIDENCE_THRESHOLD = threshold
    return cfg


def _process(pp, scene_or_probs) -> PostprocessResult:
    probs = SCENES[scene_or_probs] if isinstance(scene_or_probs, str) else scene_or_probs
    return pp.process(_outputs(probs), FRAME, [None], False)


def _cands(out: PostprocessResult) -> list:
    """A classification frame always reports its candidates."""
    assert out.candidates is not None
    return out.candidates


@pytest.fixture
def pp():
    return ClassifyPostprocessor(LABELS, _config())


class TestTopK:
    def test_candidates_are_sorted_and_ties_keep_the_lower_index(self):
        out = _process(ClassifyPostprocessor(LABELS, _config(0.3)), [0.4, 0.2, 0.4])
        assert [c["class_id"] for c in _cands(out)] == [0, 2, 1]
        assert _cands(out)[0] == {"class_id": 0, "class_name": "cat", "confidence": 0.4}
        assert out.items[0].class_name == "cat"

    def test_top_k_is_clamped_to_the_class_count(self, monkeypatch):
        monkeypatch.setenv("CLASSIFY_TOPK", "10")
        assert len(_cands(_process(ClassifyPostprocessor(LABELS, _config()), "A"))) == 3
        monkeypatch.setenv("CLASSIFY_TOPK", "1")
        out = _process(ClassifyPostprocessor(LABELS, _config()), "B")
        assert [c["class_name"] for c in _cands(out)] == ["dog"]

    @pytest.mark.parametrize("raw", ["0", "-2", "many"])
    def test_an_invalid_top_k_falls_back_to_five(self, monkeypatch, raw):
        monkeypatch.setenv("CLASSIFY_TOPK", raw)
        assert classify_topk_from_env() == 5

    def test_an_engine_with_more_classes_than_its_sidecar_gets_generic_names(self):
        out = _process(ClassifyPostprocessor(["cat"], _config()), [0.1, 0.9])
        assert out.items[0].class_name == "Class-1"
        assert out.items[0].color is not None


class TestThreshold:
    @pytest.mark.parametrize("threshold, probs", [
        (0.5, [0.5, 0.25, 0.25]),        # exactly at the gate: not above it
        (0.75, [0.75, 0.125, 0.125]),
    ])
    def test_the_gate_is_strict(self, threshold, probs):
        out = _process(ClassifyPostprocessor(LABELS, _config(threshold)), probs)
        assert out.count == 0 and out.items == []
        # The candidates are reported whatever the gate says.
        assert _cands(out)[0]["class_name"] == "cat"

    def test_the_threshold_is_read_on_every_frame(self, pp):
        assert _process(pp, "A").count == 1
        pp.config.CONFIDENCE_THRESHOLD = 0.95
        assert _process(pp, "A").count == 0

    def test_the_overlay_threshold_does_not_apply(self):
        cfg = _config()
        cfg.OVERLAY_THRESHOLD = 0.99
        assert _process(ClassifyPostprocessor(LABELS, cfg), "A").count == 1

    def test_the_item_has_no_geometry_and_the_class_colour(self, pp):
        item = _process(pp, [0.05, 0.05, 0.9]).items[0]
        assert item.class_name == "bird" and item.color == "#00ff00"
        assert item.bbox is None and item.center is None and item.area is None


class TestProcessedFrame:
    def test_a_class_is_not_drawn_on_the_frame(self, pp):
        # Nothing is drawn: the UI panel names the class; the stream stays the clean frame.
        out = _process(pp, "A")
        assert out.count == 1
        assert np.array_equal(out.image, FRAME)

    def test_no_class_leaves_the_frame_untouched(self, pp):
        assert np.array_equal(_process(pp, "none").image, FRAME)

    def test_areas_are_ignored(self, pp):
        pp.set_areas([SimpleNamespace(is_editing=True, x=0, y=0, width=1, height=1)])
        assert np.array_equal(_process(pp, "none").image, FRAME)


class TestTransitions:
    @pytest.mark.parametrize("scenes, increments", TRUTH_TABLE)
    def test_counter_increments(self, pp, scenes, increments):
        assert [_process(pp, s).count_increment for s in scenes] == increments

    def test_frames_with_a_class_count_one(self, pp):
        assert [_process(pp, s).count for s in ["none", "A", "A", "none"]] == [0, 1, 1, 0]

    def test_reset_state_makes_the_same_class_count_again(self, pp):
        assert _process(pp, "A").count_increment == 1
        pp.reset_state()
        assert _process(pp, "A").count_increment == 1


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _service(buffer=None) -> DetectionService:
    """A DetectionService running the classification strategy, no engine."""
    svc = DetectionService(_config(), buffer_service=buffer,
                           application_service=SimpleNamespace(task="classify"))
    svc.postprocessor = ClassifyPostprocessor(LABELS, svc.config)
    svc.model_manager = SimpleNamespace(  # type: ignore[assignment]
        tiling_active=False, acceleration_type="GPU", runtime_api="TensorRT")
    return svc


def _offline_buffer(tmp_path) -> buffer_mod.DetectionBufferService:
    """A buffer whose hub was seen once and then went silent: it records changes."""
    clock = FakeClock()
    buffer = buffer_mod.DetectionBufferService(
        str(tmp_path / "buffer.db"), max_records=100, max_bytes=10_000_000,
        offline_threshold_s=5.0, clock=clock,
        sample_interval_s=0.0)  # every finished frame is a sample
    buffer.note_snapshot_pull()
    clock.now += 6.0
    return buffer


def _finish(svc, scene) -> DetectionResult:
    result = svc.finish(_outputs(SCENES[scene]), FRAME, [None])
    assert result is not None
    return result


class TestBufferedTask:
    def test_a_buffered_record_carries_the_strategy_task(self, tmp_path):
        buffer = _offline_buffer(tmp_path)
        _finish(_service(buffer), "A")
        assert [r["task"] for r in buffer.list_backlog()["records"]] == ["classify"]


class TestTruthTableEndToEnd:
    @pytest.mark.parametrize("scenes, increments", TRUTH_TABLE)
    def test_snapshot_buffer_rows_and_total(self, tmp_path, scenes, increments):
        buffer = _offline_buffer(tmp_path)
        svc = _service(buffer)

        rows, counted = [], []
        for scene in scenes:
            before = buffer.pending_count()
            counted.append(_finish(svc, scene).count_increment)
            rows.append(buffer.pending_count() - before)

            snap = svc.detections_snapshot(include_frame=False)
            has_class = scene != "none"
            assert snap["task"] == "classify"
            # Node-RED's detection node emits on total > 0.
            assert snap["total"] == (1 if has_class else 0)
            assert [d["class_name"] for d in snap["detections"]] == (
                [CLASS_OF[scene]] if has_class else [])
            assert all("bbox" not in d and d["area"] is None for d in snap["detections"])
            assert [c["class_name"] for c in snap["candidates"]][:1] == [
                "dog" if scene == "B" else "cat"]

        assert counted == increments
        # One buffered row per transition to a class; nothing for a static
        # scene or for a class going away.
        assert rows == increments

    def test_the_buffered_payload_carries_no_candidates(self, tmp_path):
        buffer = _offline_buffer(tmp_path)
        svc = _service(buffer)
        _finish(svc, "A")
        records = buffer.list_backlog(10)["records"]
        assert len(records) == 1
        record = records[0]
        assert record["total"] == 1 and "candidates" not in record
        assert [d["class_name"] for d in record["detections"]] == ["cat"]
        assert all("bbox" not in d for d in record["detections"])


class TestStateResets:
    def test_stop_start_and_stats_reset_restart_from_none(self):
        svc = _service()
        assert _finish(svc, "A").count_increment == 1
        assert _finish(svc, "A").count_increment == 0

        svc.stop()
        assert _finish(svc, "A").count_increment == 1

        svc.start()
        assert _finish(svc, "A").count_increment == 1

        app = SimpleNamespace(detection_service=svc,
                              stats_service=SimpleNamespace(reset=lambda: None))
        r = ig.DetectionControlServicer(app).ResetStats(pb.Empty(), None)
        assert r.success
        assert _finish(svc, "A").count_increment == 1

    def test_an_application_change_drops_the_result_and_the_state(self):
        svc = _service()
        _finish(svc, "A")
        svc.reset_results()
        assert svc.last_detection_result is None
        assert _finish(svc, "A").count_increment == 1
