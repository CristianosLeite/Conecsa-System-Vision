# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""IP camera mosaic on the inference side: one input set per camera, results
placed back in the mosaic and named by camera in the snapshot."""
from types import SimpleNamespace

import numpy as np
import pytest
from api.config import Config
from api.model_manager import ModelManager, TileMeta, input_size_from_shape
from api.models.detection_models import Detection, DetectionResult
from api.postprocess.base import PostprocessResult
from api.postprocess.classify import ClassifyPostprocessor
from api.services.detection_service import DetectionService
from api.services.processing_pipeline import ProcessingPipelineService
from conecsa_shm.mosaic import Rect


def _manager(size=64, task="detect", tiling="off"):
    mm = ModelManager.__new__(ModelManager)
    mm._configure_preprocessing(task)
    mm._tiling_mode = tiling
    mm.input_details = [{"index": 0, "name": "images", "shape": (1, 3, size, size),
                         "dtype": np.float32}]
    mm.input_size = input_size_from_shape(mm.input_details[0]["shape"])
    return mm


MOSAIC = np.zeros((72, 256, 3), np.uint8)  # two 128×72 cells side by side
CELLS = [Rect(0, 0, 128, 72), Rect(128, 0, 128, 72)]


class TestPreprocessRegions:
    def test_each_camera_is_one_input_offset_to_its_cell(self):
        tensors, metas = _manager().preprocess_tiles(MOSAIC, CELLS)
        assert len(tensors) == 2
        assert [(m.ox, m.oy, m.width, m.height) for m in metas] == [
            (0, 0, 128, 72), (128, 0, 128, 72)]

    def test_tiling_applies_inside_each_camera(self):
        _, metas = _manager(tiling="grid").preprocess_tiles(MOSAIC, CELLS)
        # A 16:9 camera gives two tiles; none of them crosses into the other cell.
        assert len(metas) == 4
        for meta in metas:
            cell = CELLS[0] if meta.ox < 128 else CELLS[1]
            assert cell.x0 <= meta.ox and meta.ox + meta.width <= cell.x1

    def test_a_letterboxed_camera_only_feeds_its_picture(self):
        _, metas = _manager().preprocess_tiles(MOSAIC, [Rect(140, 4, 96, 64)])
        assert [(m.ox, m.oy, m.width, m.height) for m in metas] == [(140, 4, 96, 64)]

    def test_no_live_camera_means_no_input(self):
        assert _manager().preprocess_tiles(MOSAIC, []) == ([], [])

    def test_without_regions_nothing_changes(self):
        _, metas = _manager().preprocess_tiles(MOSAIC)
        assert [(m.ox, m.oy, m.width, m.height) for m in metas] == [(0, 0, 256, 72)]


def _meta(ox, oy=0):
    return TileMeta(1.0, 0, 64, ox, oy, 128, 72)


def _probs(*rows):
    return [[np.array([row], np.float32)] for row in rows]


class TestClassifyAfterTheMosaic:
    def test_a_single_camera_gets_its_top_level_candidates_back(self):
        cfg = Config()
        cfg.CONFIDENCE_THRESHOLD = 0.5
        pp = ClassifyPostprocessor(["cat", "dog"], cfg)
        pp.process(_probs([0.9, 0.1], [0.2, 0.8]), MOSAIC, [_meta(0), _meta(128)], True)
        # DetectionService resets the strategy when the source leaves IP.
        pp.reset_state()
        out = pp.process(_probs([0.9, 0.1]), MOSAIC, [_meta(0)], False)
        assert out.candidates is not None and out.candidates[0]["class_name"] == "cat"


class TestClassifyPerCamera:
    @pytest.fixture
    def pp(self):
        cfg = Config()
        cfg.CONFIDENCE_THRESHOLD = 0.5
        return ClassifyPostprocessor(["cat", "dog"], cfg)

    def test_each_camera_has_its_class_and_its_own_transitions(self, pp):
        metas = [_meta(0), _meta(128)]
        out = pp.process(_probs([0.9, 0.1], [0.2, 0.8]), MOSAIC, metas, True)
        assert [(d.class_name, d.origin) for d in out.items] == [
            ("cat", (0, 0)), ("dog", (128, 0))]
        assert out.count_increment == 2
        assert out.candidates is None
        assert [r["origin"] for r in out.region_candidates or []] == [(0, 0), (128, 0)]

        # Camera 1 stays a cat (no transition), camera 2 turns into a cat (one).
        out = pp.process(_probs([0.9, 0.1], [0.9, 0.1]), MOSAIC, metas, True)
        assert out.count_increment == 1

    def test_a_camera_coming_back_counts_again(self, pp):
        both = [_meta(0), _meta(128)]
        pp.process(_probs([0.9, 0.1], [0.9, 0.1]), MOSAIC, both, True)
        pp.process(_probs([0.9, 0.1], [0.9, 0.1]), MOSAIC, both, True)
        # Camera 2 offline for a frame, then back with the same class.
        out = pp.process(_probs([0.9, 0.1]), MOSAIC, [_meta(0)], True)
        assert out.count_increment == 0
        out = pp.process(_probs([0.9, 0.1], [0.9, 0.1]), MOSAIC, both, True)
        assert out.count_increment == 1


CAMERAS = [{"id": "cam-00000001", "name": "Dock 1", "rect": CELLS[0]},
           {"id": "cam-00000002", "name": "Dock 2", "rect": CELLS[1]}]


class TestSnapshot:
    def _result(self, detections, **extra):
        return DetectionResult(detections=detections, processed_image=MOSAIC,
                               inference_time=0.0, num_detections=len(detections),
                               cameras=CAMERAS, **extra)

    def test_items_name_the_camera_of_their_center(self):
        boxes = [Detection(0, "box", 0.9, bbox=(10, 10, 30, 30), center=(20, 20)),
                 Detection(0, "box", 0.8, bbox=(140, 10, 170, 30), center=(155, 20))]
        items, _ = DetectionService._snapshot_items(self._result(boxes))
        assert [(i["camera"], i["camera_name"]) for i in items] == [
            ("cam-00000001", "Dock 1"), ("cam-00000002", "Dock 2")]
        # The box stays normalized to the whole mosaic.
        assert items[1]["bbox"][0] == pytest.approx(140 / 256, abs=1e-4)

    def test_a_classification_is_placed_by_its_input_origin(self):
        items, _ = DetectionService._snapshot_items(
            self._result([Detection(1, "dog", 0.9, origin=(128, 0))]))
        assert items[0]["camera"] == "cam-00000002"

    def test_a_single_camera_snapshot_has_no_camera_keys(self):
        result = DetectionResult(
            detections=[Detection(0, "box", 0.9, bbox=(1, 1, 5, 5), center=(3, 3))],
            processed_image=MOSAIC, inference_time=0.0, num_detections=1)
        items, _ = DetectionService._snapshot_items(result)
        assert "camera" not in items[0]

    def test_the_layout_and_per_camera_candidates(self):
        result = self._result([], region_candidates=[
            {"origin": (128, 0), "candidates": [{"class_id": 1}]}])
        assert DetectionService._camera_layout(result) == [
            {"id": "cam-00000001", "name": "Dock 1", "rect": [0.0, 0.0, 0.5, 1.0]},
            {"id": "cam-00000002", "name": "Dock 2", "rect": [0.5, 0.0, 1.0, 1.0]}]
        # Every configured camera has its row; an offline one has no candidates.
        assert DetectionService._per_camera_candidates(result) == [
            {"camera": "cam-00000001", "camera_name": "Dock 1", "candidates": []},
            {"camera": "cam-00000002", "camera_name": "Dock 2",
             "candidates": [{"class_id": 1}]}]


class _RecordingPostprocessor:
    task = "detect"

    def __init__(self):
        self.calls = []
        self.resets = 0

    def reset_state(self):
        self.resets += 1

    def process(self, outputs, frame, metas, tiled):
        self.calls.append((len(metas), tiled))
        return PostprocessResult(frame, 0, [])


class TestFinish:
    def _service(self, recorder):
        # No runtime: only what finish() reads is set.
        service = DetectionService.__new__(DetectionService)
        vars(service).update(postprocessor=recorder, generation=0, buffer_service=None,
                             model_manager=SimpleNamespace(tiling_active=False),
                             last_detection_result=None)
        return service

    def test_an_offset_camera_takes_the_tiled_decode(self):
        recorder = _RecordingPostprocessor()
        result = self._service(recorder).finish([[0]], MOSAIC, [_meta(128)], cameras=CAMERAS)
        assert recorder.calls == [(1, True)]
        assert result is not None and result.cameras == CAMERAS

    def test_task_state_resets_when_the_camera_layout_changes(self):
        recorder = _RecordingPostprocessor()
        service = self._service(recorder)
        service.finish([[0]], MOSAIC, [_meta(0)], cameras=CAMERAS)
        service.finish([[0]], MOSAIC, [_meta(0)], cameras=CAMERAS)
        assert recorder.resets == 1, "entering the IP source"
        reordered = [dict(CAMERAS[1], rect=CELLS[0]), dict(CAMERAS[0], rect=CELLS[1])]
        service.finish([[0]], MOSAIC, [_meta(0)], cameras=reordered)
        assert recorder.resets == 2, "a reordered list keys positions differently"
        service.finish([[0]], MOSAIC, [_meta(0)])
        assert recorder.resets == 3, "leaving the IP source"
        service.finish([[0]], MOSAIC, [_meta(0)])
        assert recorder.resets == 3

    def test_every_camera_offline_yields_the_bare_mosaic(self):
        recorder = _RecordingPostprocessor()
        service = self._service(recorder)
        service.finish([[0]], MOSAIC, [_meta(0)], cameras=CAMERAS)
        resets = recorder.resets
        result = service.finish([], MOSAIC, [], cameras=CAMERAS)
        assert recorder.calls == [(1, False)]
        assert result is not None and result.num_detections == 0
        # Nobody is present: a camera coming back counts as an arrival again.
        assert recorder.resets == resets + 1


class TestCompose:
    def test_only_live_cameras_get_a_region_and_every_cell_a_name(self):
        pipeline = ProcessingPipelineService.__new__(ProcessingPipelineService)
        pipeline._cell = (128, 72)
        pipeline._video = SimpleNamespace(ip_camera_identities=lambda: [
            {"id": "cam-00000001", "name": "Dock 1"}])
        frame = np.full((72, 128, 3), 255, np.uint8)
        mosaic, regions, cameras = pipeline._compose_cameras([frame, None, frame])
        assert mosaic.shape == (144, 256, 3)
        assert regions == [Rect(0, 0, 128, 72), Rect(0, 72, 128, 72)]
        assert [(c["id"], c["name"]) for c in cameras] == [
            ("cam-00000001", "Dock 1"), ("ip1", "Camera 2"), ("ip2", "Camera 3")]
