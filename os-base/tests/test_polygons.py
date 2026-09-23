# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for conecsa_common.polygons — the rasterise-and-re-extract rule.

Host-side with OpenCV (the dev venv and CI both ship it). Every producer of
segmentation polygons goes through these helpers, so the topology cases
(concave, hole, disconnected, tile split) and the round trips through the
YOLO label row are pinned here once.
"""
import numpy as np
import pytest
from conecsa_common.polygons import (
    MAX_RINGS,
    clip_polygon_rows,
    mask_rings,
    normalize_rings,
    rasterize_rings,
    ring_area,
    rings_bbox,
)
from conecsa_common.tiling import Tile, auto_tile, tile_grid

W, H = 640, 480
SQUARE = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]]
# A "U" open at the bottom: concave, two legs.
U_SHAPE = [
    [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.7, 0.9],
    [0.7, 0.3], [0.3, 0.3], [0.3, 0.9], [0.1, 0.9],
]


def _px(rings, w=W, h=H):
    return [np.asarray(r, dtype=np.float64) * [w, h] for r in rings]


def _iou(a, b, w=W, h=H):
    ma = rasterize_rings(_px(a, w, h), w, h)
    mb = rasterize_rings(_px(b, w, h), w, h)
    return (ma & mb).sum() / max(1, (ma | mb).sum())


def _row_rings(row: str):
    values = [float(v) for v in row.split()[1:]]
    return np.asarray(values).reshape(-1, 2)


class TestNormalizeRings:
    def test_axis_aligned_square_is_exact(self):
        (ring,) = normalize_rings([SQUARE], W, H)
        assert len(ring) == 4
        assert rings_bbox([ring]) == pytest.approx((0.1, 0.1, 0.5, 0.5), abs=1.0 / W)

    def test_is_idempotent(self):
        once = normalize_rings([U_SHAPE], W, H)
        assert normalize_rings(once, W, H) == once

    def test_accepts_a_flat_point_list(self):
        flat = [v for p in SQUARE for v in p]
        assert normalize_rings([flat], W, H) == normalize_rings([SQUARE], W, H)

    def test_concave_polygon_keeps_its_shape(self):
        (ring,) = normalize_rings([U_SHAPE], W, H)
        assert len(ring) == 8
        assert _iou([U_SHAPE], [ring]) > 0.99

    def test_rings_are_clockwise_with_y_down(self):
        counter_clockwise = list(reversed(SQUARE))
        assert ring_area(counter_clockwise) < 0
        (ring,) = normalize_rings([counter_clockwise], W, H)
        assert ring_area(ring) > 0

    def test_hole_is_dropped(self):
        outer = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
        mask = rasterize_rings(_px([outer]), W, H)
        mask[180:300, 200:440] = 0  # punch a hole
        (ring,) = mask_rings(mask, W, H)
        assert len(ring) == 4
        assert rings_bbox([ring]) == pytest.approx((0.1, 0.1, 0.9, 0.9), abs=1.0 / W)

    def test_disconnected_components_become_separate_rings_largest_first(self):
        big = [[0.05, 0.05], [0.45, 0.05], [0.45, 0.45], [0.05, 0.45]]
        small = [[0.6, 0.6], [0.7, 0.6], [0.7, 0.7], [0.6, 0.7]]
        rings = normalize_rings([small, big], W, H)
        assert len(rings) == 2
        assert abs(ring_area(rings[0])) > abs(ring_area(rings[1]))

    def test_overlapping_rings_of_one_instance_union(self):
        a = [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]]
        b = [[0.3, 0.3], [0.6, 0.3], [0.6, 0.6], [0.3, 0.6]]
        rings = normalize_rings([a, b], W, H)
        assert len(rings) == 1  # one outline, not an even-odd XOR

    def test_tiny_components_are_removed(self):
        speck = [[0.8, 0.8], [0.805, 0.8], [0.805, 0.805], [0.8, 0.805]]
        assert normalize_rings([speck], W, H) == []
        (ring,) = normalize_rings([SQUARE, speck], W, H)
        assert rings_bbox([ring]) == pytest.approx((0.1, 0.1, 0.5, 0.5), abs=1.0 / W)

    def test_repeated_saves_of_a_concave_ring_never_drift(self):
        ring = normalize_rings([U_SHAPE], W, H)
        for _ in range(5):
            ring = normalize_rings(ring, W, H)
        assert ring == [U_SHAPE]

    def test_self_intersecting_input_is_re_extracted(self):
        bowtie = [[0.1, 0.1], [0.5, 0.5], [0.5, 0.1], [0.1, 0.5]]
        rings = normalize_rings([bowtie], W, H)
        assert rings and rings != [bowtie]
        assert normalize_rings(rings, W, H) == rings

    def test_jagged_ring_is_simplified_not_collapsed(self):
        angles = np.linspace(0, 2 * np.pi, 400, endpoint=False)
        jagged = [
            [0.5 + (0.3 + 0.01 * (i % 2)) * np.cos(a), 0.5 + (0.3 + 0.01 * (i % 2)) * np.sin(a)]
            for i, a in enumerate(angles)
        ]
        (ring,) = normalize_rings([jagged], 1000, 1000)
        assert 3 <= len(ring) <= 200
        assert _iou([jagged], [ring], 1000, 1000) > 0.9

    def test_degenerate_rings_are_rejected(self):
        assert normalize_rings([[[0.1, 0.1], [0.5, 0.5]]], W, H) == []
        assert normalize_rings([[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]], W, H) == []
        assert normalize_rings([], W, H) == []

    def test_a_collinear_ring_rasterises_to_nothing(self):
        # fillPoly draws a zero-area ring as a line on some OpenCV builds.
        line = [[64.0, 48.0], [128.0, 96.0], [192.0, 144.0]]
        assert not rasterize_rings([line], W, H).any()

    def test_vertex_cap(self):
        angles = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
        star = [
            [0.5 + (0.4 if i % 2 else 0.2) * np.cos(a), 0.5 + (0.4 if i % 2 else 0.2) * np.sin(a)]
            for i, a in enumerate(angles)
        ]
        (ring,) = normalize_rings([star], 2000, 2000, max_vertices=50)
        assert 3 <= len(ring) <= 50

    def test_ring_cap_drops_the_smallest(self):
        squares = [
            [[x, 0.1], [x + s, 0.1], [x + s, 0.1 + s], [x, 0.1 + s]]
            for x, s in zip(np.linspace(0.0, 0.9, 10), np.linspace(0.02, 0.08, 10), strict=True)
        ]
        rings = normalize_rings(squares, W, H)
        assert len(rings) == MAX_RINGS
        areas = [abs(ring_area(r)) for r in rings]
        assert areas == sorted(areas, reverse=True)

    def test_coordinates_stay_inside_the_image(self):
        outside = [[-0.2, -0.2], [0.5, -0.2], [0.5, 0.5], [-0.2, 0.5]]
        (ring,) = normalize_rings([outside], W, H)
        pts = np.asarray(ring)
        assert pts.min() >= 0.0 and pts.max() <= 1.0

    def test_rejects_bad_sizes(self):
        with pytest.raises(ValueError):
            normalize_rings([SQUARE], 0, H)


class TestMaskRings:
    def test_crop_offset_maps_into_image_coordinates(self):
        crop = np.ones((40, 80), dtype=np.uint8)
        (ring,) = mask_rings(crop, W, H, offset=(100, 200))
        box = rings_bbox([ring])
        assert box is not None
        x1, y1, x2, y2 = box
        assert (x1 * W, y1 * H) == pytest.approx((100, 200))
        assert (x2 * W, y2 * H) == pytest.approx((179, 239))

    def test_empty_mask(self):
        assert mask_rings(np.zeros((10, 10), np.uint8), W, H) == []


class TestClipPolygonRows:
    def test_concave_polygon_split_by_the_tile_edge_gives_one_ring_per_piece(self):
        # The window below the U's bridge only sees its two legs.
        tile = Tile(0, 300, W, H)
        rows, touched = clip_polygon_rows([3], [_px([U_SHAPE])], tile, min_visible=0.1)
        assert touched == 1
        assert len(rows) == 2
        assert all(r.startswith("3 ") for r in rows)
        for row in rows:
            pts = _row_rings(row)
            assert len(pts) >= 3
            assert pts.min() >= 0.0 and pts.max() <= 1.0

    def test_rows_are_normalised_to_the_tile(self):
        tile = Tile(32, 24, 352, 264)
        rows, _ = clip_polygon_rows([0], [_px([SQUARE])], tile)
        (row,) = rows
        pts = _row_rings(row)
        # The square spans x 64..320, y 48..240 in the frame.
        assert pts[:, 0].min() * tile.width == pytest.approx(64 - 32, abs=1)
        assert pts[:, 0].max() * tile.width == pytest.approx(320 - 32, abs=1)
        assert pts[:, 1].min() * tile.height == pytest.approx(48 - 24, abs=1)

    def test_fragment_below_min_visible_is_counted_but_not_kept(self):
        tile = Tile(300, 0, W, H)  # sees only a thin strip of the square
        rows, touched = clip_polygon_rows([0], [_px([SQUARE])], tile, min_visible=0.25)
        assert (rows, touched) == ([], 1)

    def test_untouched_tile_is_a_clean_negative(self):
        rows, touched = clip_polygon_rows([0], [_px([SQUARE])], Tile(400, 300, W, H))
        assert (rows, touched) == ([], 0)

    def test_round_trip_through_the_label_row_keeps_the_shape(self):
        tile = Tile(0, 0, W, H)
        rows, _ = clip_polygon_rows([1], [_px([U_SHAPE])], tile)
        (row,) = rows
        assert _iou([U_SHAPE], [_row_rings(row).tolist()]) > 0.99

    def test_tiles_of_a_k2_grid_cover_the_whole_instance(self):
        fw, fh = 1280, 720
        wide = [[400, 200], [900, 200], [900, 500], [400, 500]]
        tiles = tile_grid(fw, fh, auto_tile(fw, fh))
        kept = sum(len(clip_polygon_rows([0], [[wide]], t, min_visible=0.1)[0]) for t in tiles)
        assert kept == 2

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            clip_polygon_rows([0], [], Tile(0, 0, 10, 10))
        with pytest.raises(ValueError):
            clip_polygon_rows([0], [_px([SQUARE])], Tile(0, 0, 10, 10), min_visible=0.0)
