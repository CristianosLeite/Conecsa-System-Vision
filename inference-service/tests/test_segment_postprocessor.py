# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Instance segmentation postprocess: masks from synthetic prototypes, the
letterbox undo, the strict gate, the mask caps, the seam stitching (and the
documented cut with stitching off), the reusable label buffer and the
normalized rings."""
from types import SimpleNamespace

import numpy as np
import pytest
from api.config import Config
from api.model_manager import TileMeta
from api.postprocess import segment as seg
from api.postprocess.segment import SegmentPostprocessor, segment_settings_from_env

LABELS = ["bolt", "nut #00ff00"]
NM = 2
FULL = (1.0, 0.0)   # coefficient on the all-positive prototype → a full box mask
EMPTY = (0.0, 1.0)  # coefficient on the all-negative prototype → no mask


def _rows(*dets, n=10):
    """``[1, n, 6 + NM]`` rows: (box in input px, conf, class, coeffs), zero-padded."""
    rows = np.zeros((1, n, 6 + NM), np.float32)
    for i, (box, conf, cls, coeffs) in enumerate(dets):
        rows[0, i, :4] = box
        rows[0, i, 4] = conf
        rows[0, i, 5] = cls
        rows[0, i, 6:] = coeffs
    return rows


def _protos(ph=16, left_half=False):
    protos = np.full((1, NM, ph, ph), -10.0, np.float32)
    protos[0, 0] = 10.0
    if left_half:
        protos[0, 0, :, ph // 2:] = -10.0
    return protos


def _meta(width, height, size=64, ox=0, oy=0):
    """The TileMeta ModelManager builds for a crop letterboxed to ``size``."""
    resized_h = int(size * height / width)
    return TileMeta(height / resized_h, int((size - resized_h) / 2), size, ox, oy, width, height)


def _config(threshold=0.5, overlay=0.45):
    cfg = Config()
    cfg.CONFIDENCE_THRESHOLD = threshold
    cfg.OVERLAY_THRESHOLD = overlay
    return cfg


def _pp(**config):
    return SegmentPostprocessor(LABELS, _config(**config))


def _rings(det):
    rings = det.resolve_polygons()
    assert rings is not None
    return rings


def _bbox(det):
    assert det.bbox is not None
    return det.bbox


def _ring_bbox_px(det, width, height):
    pts = np.asarray([p for ring in _rings(det) for p in ring])
    return (pts[:, 0].min() * width, pts[:, 1].min() * height,
            pts[:, 0].max() * width, pts[:, 1].max() * height)


class TestDecode:
    def test_a_full_mask_fills_the_box_and_becomes_one_ring(self):
        frame = np.zeros((64, 64, 3), np.uint8)
        out = _pp().process([[_rows(([8, 8, 40, 40], 0.9, 1, FULL)), _protos()]], frame,
                            [_meta(64, 64)], False)
        (det,) = out.items
        assert out.count == 1
        assert det.bbox == (8, 8, 40, 40) and det.class_name == "nut"
        assert det.color == "#00ff00"
        assert len(_rings(det)) == 1
        assert _ring_bbox_px(det, 64, 64) == pytest.approx((8, 8, 39, 39), abs=1)

    def test_outputs_are_picked_by_rank_not_order(self):
        frame = np.zeros((64, 64, 3), np.uint8)
        rows = _rows(([8, 8, 40, 40], 0.9, 0, FULL))
        a = _pp().process([[rows, _protos()]], frame, [_meta(64, 64)], False)
        b = _pp().process([[_protos(), rows]], frame, [_meta(64, 64)], False)
        assert [d.resolve_polygons() for d in a.items] == [d.resolve_polygons() for d in b.items]
        np.testing.assert_array_equal(a.image, b.image)

    def test_the_mask_follows_the_prototypes(self):
        frame = np.zeros((64, 64, 3), np.uint8)
        out = _pp().process([[_rows(([0, 8, 64, 40], 0.9, 0, FULL)),
                              _protos(left_half=True)]], frame, [_meta(64, 64)], False)
        (det,) = out.items
        x1, _, x2, _ = _ring_bbox_px(det, 64, 64)
        assert x1 == pytest.approx(0, abs=1) and x2 == pytest.approx(31, abs=1)

    def test_an_empty_mask_keeps_the_instance_without_rings(self):
        frame = np.zeros((64, 64, 3), np.uint8)
        out = _pp().process([[_rows(([8, 8, 40, 40], 0.9, 0, EMPTY)), _protos()]], frame,
                            [_meta(64, 64)], False)
        (det,) = out.items
        assert det.resolve_polygons() == []

    def test_rings_are_extracted_on_first_use_only(self, monkeypatch):
        from conecsa_common import polygons

        calls = []
        real = polygons.mask_rings
        monkeypatch.setattr(polygons, "mask_rings", lambda *a, **k: calls.append(1) or real(*a, **k))
        frame = np.zeros((64, 64, 3), np.uint8)
        out = _pp().process([[_rows(([8, 8, 40, 40], 0.9, 0, FULL)), _protos()]], frame,
                            [_meta(64, 64)], False)
        (det,) = out.items
        assert calls == [] and det.polygons is None
        first = det.resolve_polygons()
        assert first is not None and len(first) == 1
        assert det.resolve_polygons() is first and len(calls) == 1

    def test_the_letterbox_is_undone(self):
        # 128×72 → width 64, height 36, border 14, scale 2.
        frame = np.zeros((72, 128, 3), np.uint8)
        meta = _meta(128, 72)
        assert (meta.scale, meta.border_top) == (2.0, 14)
        out = _pp().process([[_rows(([16, 22, 48, 40], 0.9, 0, FULL)), _protos()]], frame,
                            [meta], False)
        (det,) = out.items
        assert det.bbox == (32, 16, 96, 52)
        assert _ring_bbox_px(det, 128, 72) == pytest.approx((32, 16, 95, 51), abs=1)

    def test_normalized_boxes_are_accepted(self):
        frame = np.zeros((64, 64, 3), np.uint8)
        out = _pp().process([[_rows(([0.125, 0.125, 0.625, 0.625], 0.9, 0, FULL)), _protos()]],
                            frame, [_meta(64, 64)], False)
        assert out.items[0].bbox == (8, 8, 40, 40)

    def test_the_confidence_gate_is_strict(self):
        frame = np.zeros((64, 64, 3), np.uint8)
        out = _pp(threshold=0.5).process(
            [[_rows(([8, 8, 40, 40], 0.5, 0, FULL)), _protos()]], frame, [_meta(64, 64)], False)
        assert out.count == 0 and out.items == []

    def test_tiny_boxes_are_dropped_like_detection(self):
        frame = np.zeros((64, 64, 3), np.uint8)
        out = _pp().process([[_rows(([8, 8, 13, 40], 0.9, 0, FULL)), _protos()]], frame,
                            [_meta(64, 64)], False)
        assert out.items == []

    def test_garbage_outputs_yield_nothing(self):
        frame = np.zeros((64, 64, 3), np.uint8)
        out = _pp().process([[np.zeros((1, 10, 6), np.float32)]], frame, [_meta(64, 64)], False)
        assert out.count == 0
        np.testing.assert_array_equal(out.image, frame)


class TestOverlay:
    def test_the_clean_frame_is_never_drawn_on(self):
        frame = np.full((64, 64, 3), 40, np.uint8)
        before = frame.copy()
        out = _pp().process([[_rows(([8, 8, 40, 40], 0.9, 1, FULL)), _protos()]], frame,
                            [_meta(64, 64)], False)
        np.testing.assert_array_equal(frame, before)
        assert out.image is not frame
        # Inside the mask the fill moved the pixel towards the class colour (green).
        assert out.image[24, 24, 1] > 40 and out.image[24, 24, 0] < 40

    def test_the_label_buffer_is_reused_and_left_clean(self):
        pp = _pp()
        frame = np.zeros((64, 64, 3), np.uint8)
        outputs = [[_rows(([8, 8, 40, 40], 0.9, 0, FULL)), _protos()]]
        pp.process(outputs, frame, [_meta(64, 64)], False)
        buffer = pp._labels
        assert buffer is not None and not buffer.any()
        pp.process(outputs, frame, [_meta(64, 64)], False)
        assert pp._labels is buffer

    def test_overlapping_masks_paint_the_best_instance_on_top(self):
        # IoU 0.09, under the overlay threshold: both instances are kept.
        frame = np.full((64, 64, 3), 40, np.uint8)
        out = _pp().process([[_rows(([0, 0, 40, 40], 0.9, 1, FULL), ([24, 24, 64, 64], 0.8, 0, FULL)),
                              _protos()]], frame, [_meta(64, 64)], False)
        assert out.count == 2
        best_only, other_only, overlap = out.image[10, 10], out.image[54, 54], out.image[32, 32]
        np.testing.assert_array_equal(overlap, best_only)
        assert not np.array_equal(best_only, other_only)
        # Outside every box (and away from the labels) the frame is untouched.
        np.testing.assert_array_equal(out.image[60, 5], frame[60, 5])

    def test_the_union_fill_matches_one_blend_per_box(self):
        # One blend over the union of the instance boxes replaced a blend per
        # box; the image must not change. The union here is narrower than the
        # frame and one mask hangs off its top-left corner.
        import cv2

        rng = np.random.default_rng(7)
        frame = rng.integers(0, 256, (48, 96, 3), dtype=np.uint8)
        pp = _pp()
        placed = [(-4, -3, 20, 18, 1), (10, 6, 30, 24, 0), (22, 20, 18, 20, 1), (60, 30, 12, 10, 0)]
        kept = []
        for x0, y0, w, h, class_id in placed:
            mask = (rng.random((h, w)) > 0.3).astype(np.uint8)
            kept.append((SimpleNamespace(class_id=class_id),
                         SimpleNamespace(mask=mask, x0=x0, y0=y0)))

        expected = frame.copy()
        winner = np.zeros(frame.shape[:2], np.int32)
        for index in range(len(kept) - 1, -1, -1):
            inst = kept[index][1]
            h, w = inst.mask.shape
            ys, xs = np.nonzero(inst.mask)
            ok = (ys + inst.y0 >= 0) & (xs + inst.x0 >= 0) & (ys + inst.y0 < 48) & (xs + inst.x0 < 96)
            winner[ys[ok] + inst.y0, xs[ok] + inst.x0] = index + 1
        for index, (det, _) in enumerate(kept):
            won = winner == index + 1
            color = np.empty_like(expected)
            color[:] = pp.detector._get_class_color(det.class_id)
            blended = cv2.addWeighted(expected, 1.0 - seg.MASK_ALPHA, color, seg.MASK_ALPHA, 0.0)
            expected[won] = blended[won]

        out = frame.copy()
        pp._fill_masks(out, kept)
        np.testing.assert_array_equal(out, expected)
        assert pp._labels is not None and not pp._labels.any()

    def test_areas_filter_on_the_instance_centre(self):
        pp = _pp()
        pp.set_areas([SimpleNamespace(id="a1", label="left", shape="rectangle", is_editing=False,
                                      x=0.0, y=0.0, width=0.5, height=1.0)])
        frame = np.zeros((64, 64, 3), np.uint8)
        out = pp.process([[_rows(([2, 8, 20, 40], 0.9, 0, FULL), ([40, 8, 60, 40], 0.8, 0, FULL)),
                           _protos()]], frame, [_meta(64, 64)], False)
        (det,) = out.items
        assert _bbox(det)[0] == 2 and det.area == {"id": "a1", "label": "left", "shape": "rectangle"}


class TestCaps:
    DETS = [([0, 0, 14, 14], 0.6, 0, FULL), ([20, 0, 34, 14], 0.9, 0, FULL),
            ([40, 0, 54, 14], 0.7, 0, FULL), ([0, 30, 14, 44], 0.8, 0, FULL)]

    def test_per_tile_cap_keeps_the_highest_scores(self, monkeypatch):
        monkeypatch.setenv("SEGMENT_MAX_MASKS_PER_TILE", "2")
        out = _pp().process([[_rows(*self.DETS), _protos()]], np.zeros((64, 64, 3), np.uint8),
                            [_meta(64, 64)], False)
        assert sorted(d.confidence for d in out.items) == pytest.approx([0.8, 0.9])

    def test_a_duplicate_row_gets_no_mask(self, monkeypatch):
        # Three rows on one object (IoU > the overlay threshold) and one apart:
        # the overlay NMS would drop the duplicates, so they never reach the mask math.
        calls = []
        real = seg._instance_mask
        monkeypatch.setattr(seg, "_instance_mask", lambda *a: calls.append(1) or real(*a))
        dets = [([8, 8, 40, 40], 0.9, 0, FULL), ([9, 9, 41, 41], 0.8, 0, FULL),
                ([8, 10, 40, 42], 0.7, 1, FULL), ([46, 46, 62, 62], 0.6, 0, FULL)]
        out = _pp().process([[_rows(*dets), _protos()]], np.zeros((64, 64, 3), np.uint8),
                            [_meta(64, 64)], False)
        assert [d.confidence for d in out.items] == pytest.approx([0.9, 0.6])
        assert len(calls) == 2

    def test_the_per_tile_cap_counts_distinct_objects(self, monkeypatch):
        # Duplicates do not use up the cap: with a cap of 2 the second object is kept.
        monkeypatch.setenv("SEGMENT_MAX_MASKS_PER_TILE", "2")
        dets = [([8, 8, 40, 40], 0.9, 0, FULL), ([9, 9, 41, 41], 0.8, 0, FULL),
                ([46, 46, 62, 62], 0.6, 0, FULL)]
        out = _pp().process([[_rows(*dets), _protos()]], np.zeros((64, 64, 3), np.uint8),
                            [_meta(64, 64)], False)
        assert [d.confidence for d in out.items] == pytest.approx([0.9, 0.6])

    def test_per_frame_cap_keeps_the_highest_scores(self, monkeypatch):
        monkeypatch.setenv("SEGMENT_MAX_MASKS", "3")
        out = _pp().process([[_rows(*self.DETS), _protos()]], np.zeros((64, 64, 3), np.uint8),
                            [_meta(64, 64)], False)
        assert [d.confidence for d in out.items] == pytest.approx([0.9, 0.8, 0.7])
        assert out.count == 3

    def test_the_operator_limit_overrides_both_env_caps(self, monkeypatch):
        # The per-model instance limit caps per tile and per frame, read live.
        monkeypatch.setenv("SEGMENT_MAX_MASKS", "32")
        monkeypatch.setenv("SEGMENT_MAX_MASKS_PER_TILE", "32")
        pp = _pp()
        frame = np.zeros((64, 64, 3), np.uint8)
        outputs = [[_rows(*self.DETS), _protos()]]
        assert pp.process(outputs, frame, [_meta(64, 64)], False).count == 4
        pp.config.SEGMENT_MAX_INSTANCES = 2
        out = pp.process(outputs, frame, [_meta(64, 64)], False)
        assert [d.confidence for d in out.items] == pytest.approx([0.9, 0.8])
        assert pp.caps() == (2, 2)


class TestSeam:
    """Two 64 px tiles over a 104×64 frame; the overlap band is x 40..64."""

    FRAME = np.zeros((64, 104, 3), np.uint8)
    METAS = [_meta(64, 64, ox=0), _meta(64, 64, ox=40)]

    def _run(self, left, right, monkeypatch=None, stitch=True):
        if monkeypatch is not None and not stitch:
            monkeypatch.setenv("SEGMENT_TILE_STITCH", "0")
        outputs = [[_rows(*left), _protos()], [_rows(*right), _protos()]]
        return _pp().process(outputs, self.FRAME, self.METAS, True)

    # An object at x 20..90: tile 0 sees 20..64, tile 1 sees 40..90 (local 0..50).
    STRADDLING = ([([20, 10, 64, 50], 0.9, 0, FULL)], [([0, 10, 50, 50], 0.8, 0, FULL)])

    def test_an_object_across_the_band_is_one_instance_and_one_ring(self):
        out = self._run(*self.STRADDLING)
        (det,) = out.items
        assert det.bbox == (20, 10, 90, 50)
        assert det.confidence == pytest.approx(0.9)
        assert len(_rings(det)) == 1
        assert _ring_bbox_px(det, 104, 64) == pytest.approx((20, 10, 89, 49), abs=1)

    def test_with_stitching_off_the_object_stays_cut(self, monkeypatch):
        out = self._run(*self.STRADDLING, monkeypatch=monkeypatch, stitch=False)
        assert len(out.items) == 2

    def test_two_nearby_objects_across_the_band_stay_separate(self):
        # A at x 20..50, B at 55..90, each seen (cut) by both tiles.
        left = [([20, 10, 50, 50], 0.9, 0, FULL), ([55, 10, 64, 50], 0.7, 0, FULL)]
        right = [([0, 10, 10, 50], 0.6, 0, FULL), ([15, 10, 50, 50], 0.85, 0, FULL)]
        out = self._run(left, right)
        assert sorted(_bbox(d) for d in out.items) == [(20, 10, 50, 50), (55, 10, 90, 50)]

    def test_an_object_inside_one_tile_is_unchanged(self):
        out = self._run([([4, 10, 30, 50], 0.9, 0, FULL)], [])
        (det,) = out.items
        assert det.bbox == (4, 10, 30, 50)


class TestSettings:
    def test_defaults(self, monkeypatch):
        for name in ("SEGMENT_TILE_STITCH", "SEGMENT_MAX_MASKS", "SEGMENT_MAX_MASKS_PER_TILE",
                     "SEGMENT_MIN_COMPONENT_AREA", "TILING_MERGE_IOS"):
            monkeypatch.delenv(name, raising=False)
        assert segment_settings_from_env() == seg.SegmentSettings()

    @pytest.mark.parametrize(("name", "raw"), [
        ("SEGMENT_TILE_STITCH", "maybe"), ("SEGMENT_MAX_MASKS", "0"),
        ("SEGMENT_MAX_MASKS", "256"), ("SEGMENT_MAX_MASKS_PER_TILE", "many"),
        ("SEGMENT_MIN_COMPONENT_AREA", "2"), ("TILING_MERGE_IOS", "-0.1"),
    ])
    def test_invalid_values_fall_back(self, monkeypatch, name, raw):
        monkeypatch.setenv(name, raw)
        assert segment_settings_from_env() == seg.SegmentSettings()

    def test_overrides(self, monkeypatch):
        monkeypatch.setenv("SEGMENT_TILE_STITCH", "off")
        monkeypatch.setenv("SEGMENT_MAX_MASKS", "8")
        monkeypatch.setenv("TILING_MERGE_IOS", "0.3")
        settings = segment_settings_from_env()
        assert (settings.stitch, settings.max_masks, settings.merge_ios) == (False, 8, 0.3)
