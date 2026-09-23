# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""DetectionService stage plumbing for tiled inference (fake manager/detector).

``prepare``/``infer``/``finish`` now move parallel lists (one entry per tile,
a single full-frame entry when tiling is off); these tests pin the routing in
``finish`` and the per-input fan-out in ``infer`` without a TensorRT stack.
"""
import numpy as np
import pytest
from api.config import Config
from api.model_manager import TileMeta
from api.postprocess.detect import DetectPostprocessor
from api.services.detection_service import DetectionService

_META1 = TileMeta(scale=720 / 640, border_top=0, input_size=640,
                  ox=0, oy=0, width=720, height=720)
_META2 = TileMeta(scale=720 / 640, border_top=0, input_size=640,
                  ox=560, oy=0, width=720, height=720)
_FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)


class _FakeManager:
    def __init__(self, tiling_active: bool):
        self.tiling_active = tiling_active
        self.inputs_seen = []

    def run_inference(self, input_data):
        self.inputs_seen.append(input_data)
        return [np.full((1, 1, 6), len(self.inputs_seen), dtype=np.float32)], 0.01


class _FakeDetector:
    def __init__(self):
        self.tiled_args: tuple | None = None
        self.plain_args: tuple | None = None

    def process_tiled_detections(self, outputs, frame, metas):
        self.tiled_args = (outputs, metas)
        return frame, 0, []

    def process_detections(self, output, frame, scale, border_top, actual_input_size):
        self.plain_args = (output, scale, border_top, actual_input_size)
        return frame, 0, []


def _service(tiling_active: bool):
    """Build a service with fakes; returns them too (the service attributes
    are typed Optional[ModelManager]/Optional[Postprocessor])."""
    service = DetectionService(Config())
    manager = _FakeManager(tiling_active)
    detector = _FakeDetector()
    service.model_manager = manager  # type: ignore[assignment]
    service.postprocessor = DetectPostprocessor(
        [], Config(), detector=detector)  # type: ignore[arg-type]
    return service, manager, detector


class TestInfer:
    def test_runs_one_inference_per_input_and_sums_the_time(self):
        service, _manager, _detector = _service(False)
        outputs, total = service.infer([np.zeros(1), np.zeros(1), np.zeros(1)])
        assert len(outputs) == 3
        assert total == pytest.approx(0.03)
        # Every input yields the list of all engine outputs.
        assert [int(o[0][0, 0, 0]) for o in outputs] == [1, 2, 3]


class TestFinishRouting:
    def test_tiling_off_uses_the_plain_path_with_the_single_meta(self):
        service, _manager, detector = _service(False)
        output = np.zeros((1, 300, 6), dtype=np.float32)
        result = service.finish([[output]], _FRAME, [_META1], 0.01)
        assert result is not None
        assert detector.tiled_args is None
        assert detector.plain_args is not None
        seen_output, scale, border_top, input_size = detector.plain_args
        assert seen_output is output
        assert (scale, border_top, input_size) == (_META1.scale, 0, 640)

    def test_tiling_grid_routes_all_tiles_to_the_tiled_path(self):
        service, _manager, detector = _service(True)
        outputs = [np.zeros((1, 300, 6), dtype=np.float32) for _ in range(2)]
        result = service.finish([[o] for o in outputs], _FRAME, [_META1, _META2], 0.02)
        assert result is not None
        assert detector.plain_args is None
        assert detector.tiled_args is not None
        seen_outputs, seen_metas = detector.tiled_args
        assert len(seen_outputs) == 2
        assert all(seen is sent for seen, sent in zip(seen_outputs, outputs, strict=True))
        assert [m.ox for m in seen_metas] == [0, 560]

    def test_the_detection_head_is_picked_by_rank_not_by_binding_index(self):
        service, _manager, detector = _service(False)
        head = np.zeros((1, 300, 6), dtype=np.float32)
        service.finish([[np.zeros((1, 2), np.float32), head]], _FRAME, [_META1], 0.01)
        assert detector.plain_args is not None
        assert detector.plain_args[0] is head
