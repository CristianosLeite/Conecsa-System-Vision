# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for the pure snapshot helpers of the detection service."""
import numpy as np
import pytest
from api.config import Config
from api.models.detection_models import Detection, DetectionResult
from api.services.detection_service import DetectionService, normalized_bbox


class TestNormalizedBbox:
    def test_maps_pixel_corners_to_unit_range(self):
        assert normalized_bbox((160, 90, 320, 180), 640, 360) == [
            0.25, 0.25, 0.5, 0.5,
        ]

    def test_full_frame_box(self):
        assert normalized_bbox((0, 0, 640, 360), 640, 360) == [0.0, 0.0, 1.0, 1.0]

    def test_clamps_out_of_frame_corners(self):
        # NMS/decoding can produce corners slightly outside the frame.
        assert normalized_bbox((-10, -5, 700, 400), 640, 360) == [
            0.0, 0.0, 1.0, 1.0,
        ]

    def test_rounds_to_four_decimals(self):
        out = normalized_bbox((1, 1, 2, 2), 3, 3)
        assert out == [
            pytest.approx(0.3333), pytest.approx(0.3333),
            pytest.approx(0.6667), pytest.approx(0.6667),
        ]
        assert all(v == round(v, 4) for v in out)


def _snapshot(detections, candidates=None):
    svc = DetectionService(Config())
    svc.last_detection_result = DetectionResult(
        detections=detections, processed_image=np.zeros((360, 640, 3), np.uint8),
        inference_time=0.0, num_detections=len(detections), candidates=candidates)
    return svc.detections_snapshot(include_frame=False)


class TestSnapshotItems:
    def test_a_detection_is_unchanged_and_has_no_candidates_key(self):
        snap = _snapshot([Detection(0, "cat", 0.91234, bbox=(160, 90, 320, 180),
                                    center=(240, 135), color="#ff0000")])
        item = snap["detections"][0]
        # Key order is part of the byte-identical backlog/snapshot contract.
        assert list(item) == ["class_name", "color", "confidence", "area", "bbox"]
        assert item["bbox"] == [0.25, 0.25, 0.5, 0.5] and item["confidence"] == 0.9123
        assert "candidates" not in snap

    def test_a_classification_has_no_bbox_and_carries_its_candidates(self):
        candidates = [{"class_id": 1, "class_name": "dog", "confidence": 0.8},
                      {"class_id": 0, "class_name": "cat", "confidence": 0.2}]
        snap = _snapshot([Detection(1, "dog", 0.8, color="#00ff00")], candidates)
        assert snap["detections"] == [{"class_name": "dog", "color": "#00ff00",
                                       "confidence": 0.8, "area": None}]
        assert snap["total"] == 1
        assert snap["candidates"] == candidates

    def test_a_frame_without_a_class_keeps_its_candidates(self):
        snap = _snapshot([], [{"class_id": 0, "class_name": "cat", "confidence": 0.3}])
        assert snap["detections"] == [] and snap["total"] == 0
        assert snap["candidates"][0]["class_name"] == "cat"


class TestSnapshotPolygons:
    """Segmentation rings in the snapshot and the polygon payload cap."""

    BIG = [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]
    SMALL = [[0.6, 0.6], [0.61, 0.6], [0.61, 0.61], [0.6, 0.61]]

    def _item(self, rings, confidence=0.9):
        return Detection(0, "cat", confidence, bbox=(0, 0, 320, 180), center=(160, 90),
                         color="#ff0000", polygons=rings)

    def test_a_segmentation_item_carries_rounded_rings_after_its_bbox(self):
        ring = [[0.25, 0.25], [0.512345, 0.25], [0.5, 0.5]]
        snap = _snapshot([self._item([ring])])
        item = snap["detections"][0]
        assert list(item) == ["class_name", "color", "confidence", "area", "bbox", "polygons"]
        assert item["polygons"] == [[[0.25, 0.25], [0.5123, 0.25], [0.5, 0.5]]]
        assert "polygons_truncated" not in snap

    def test_an_instance_without_rings_keeps_an_empty_list(self):
        assert _snapshot([self._item([])])["detections"][0]["polygons"] == []

    def test_the_payload_cap_drops_the_smallest_rings_first(self, monkeypatch):
        monkeypatch.setattr(DetectionService, "SNAPSHOT_POLYGON_BYTES", 60)
        snap = _snapshot([self._item([self.SMALL]), self._item([self.BIG], 0.8)])
        assert snap["detections"][0]["polygons"] == []
        assert snap["detections"][1]["polygons"] == [self.BIG]
        assert snap["polygons_truncated"] is True

    def test_the_default_cap_leaves_a_normal_scene_alone(self):
        snap = _snapshot([self._item([self.BIG, self.SMALL]) for _ in range(32)])
        assert all(len(d["polygons"]) == 2 for d in snap["detections"])
        assert "polygons_truncated" not in snap

    def test_offline_records_use_the_same_capped_items(self, monkeypatch):
        monkeypatch.setattr(DetectionService, "SNAPSHOT_POLYGON_BYTES", 60)
        result = DetectionResult(
            detections=[self._item([self.SMALL]), self._item([self.BIG], 0.8)],
            processed_image=np.zeros((360, 640, 3), np.uint8), inference_time=0.0,
            num_detections=2)
        svc = DetectionService(Config())
        svc.last_detection_result = result
        snap = svc.detections_snapshot(include_frame=False)
        assert DetectionService._detection_dicts(result) == snap["detections"]
