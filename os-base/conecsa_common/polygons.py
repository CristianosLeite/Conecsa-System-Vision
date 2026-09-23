# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: Apache-2.0

"""Segmentation polygon normalization shared by every producer.

One rule turns any polygon source — a model mask on the device, a SAM mask,
a ring drawn in the label editor, a label row clipped to a training tile, an
imported dataset — into the same topology: the shape is **rasterized** at the
label resolution and its exterior contours are **re-extracted**. That makes
the result independent of how the input was built:

- self-intersections cannot come out of contour extraction;
- a concave polygon cut by a tile edge yields the right number of rings,
  with no bridge edges;
- holes are dropped (exterior rings only — a documented lossy policy);
- components smaller than ``min_area_frac`` of the image are removed;
- each ring is simplified with ``approxPolyDP`` (epsilon a fraction of its
  perimeter) and capped at ``max_vertices``; an instance keeps at most
  ``max_rings`` rings, the smallest dropped first;
- rings are oriented clockwise in image coordinates (y down) and closed
  implicitly; a ring with fewer than 3 vertices is rejected.

A ring is a list of ``[x, y]`` pairs normalized 0..1 to the image the label
belongs to; an instance is a list of rings.

Unlike :mod:`conecsa_common.tiling`, these helpers need OpenCV. It is imported
at call time, so importing this module stays free of cv2 for every service
that never touches polygons.
"""
from collections.abc import Sequence

import numpy as np

from conecsa_common.tiling import Tile

__all__ = [
    "MAX_RINGS",
    "MAX_VERTICES",
    "MIN_AREA_FRAC",
    "clip_polygon_rows",
    "mask_rings",
    "normalize_rings",
    "rasterize_rings",
    "ring_area",
    "rings_bbox",
]

MIN_AREA_FRAC = 0.0005
"""Smallest kept component, as a fraction of the image area (0.05 %)."""
EPS_FRAC = 0.005
"""``approxPolyDP`` epsilon as a fraction of each ring's perimeter (0.5 %)."""
MAX_VERTICES = 200
MAX_RINGS = 8
#: Rings enclosing less than this many square pixels are not rasterized.
MIN_RING_AREA_PX = 0.5

Ring = list[list[float]]

# Sub-pixel precision for cv2.fillPoly (fixed point with 4 fractional bits).
_SHIFT = 4
_SCALE = 1 << _SHIFT


def _cv2():
    import cv2  # noqa: PLC0415 — optional dependency, see the module docstring

    return cv2


def ring_area(ring) -> float:
    """Signed shoelace area of a ring; positive when clockwise with y down."""
    pts = np.asarray(ring, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    # The same rotation np.roll builds, without its generic overhead: this
    # runs for every ring of every segmented instance in the live pipeline.
    x_next = np.concatenate((x[1:], x[:1]))
    y_next = np.concatenate((y[1:], y[:1]))
    return float(np.dot(x, y_next) - np.dot(x_next, y)) / 2.0


def rings_bbox(rings) -> tuple[float, float, float, float] | None:
    """Bounding rectangle ``(x1, y1, x2, y2)`` of every vertex, or ``None``."""
    pts = [np.asarray(r, dtype=np.float64).reshape(-1, 2) for r in rings]
    pts = [p for p in pts if len(p)]
    if not pts:
        return None
    allp = np.concatenate(pts)
    x1, y1 = allp.min(axis=0)
    x2, y2 = allp.max(axis=0)
    return float(x1), float(y1), float(x2), float(y2)


def _fan_area(pts: np.ndarray) -> float:
    """Sum of the unsigned areas of the triangles fanned from the first vertex.

    Zero exactly when every vertex lies on one line; unlike the shoelace
    area, the lobes of a self-crossing ring (a bowtie) do not cancel.
    """
    d = pts[1:] - pts[0]
    return float(np.abs(d[:-1, 0] * d[1:, 1] - d[:-1, 1] * d[1:, 0]).sum()) / 2.0


def rasterize_rings(rings_px, width: int, height: int, offset=(0.0, 0.0)) -> np.ndarray:
    """Fill pixel-space rings into a ``height``×``width`` uint8 mask (0/1).

    ``offset`` is subtracted from every vertex first, so a crop of a larger
    image can be rasterized on its own small canvas. Every ring is filled in
    its own call: the union of an instance's rings, never an even-odd XOR.
    A ring enclosing less than half a pixel (collinear vertices) is skipped:
    how ``fillPoly`` draws such a ring differs between OpenCV builds (a line
    on some, nothing on others), and a label must not depend on that.
    """
    cv2 = _cv2()
    mask = np.zeros((max(1, int(height)), max(1, int(width))), dtype=np.uint8)
    ox, oy = float(offset[0]), float(offset[1])
    for ring in rings_px:
        pts = np.asarray(ring, dtype=np.float64).reshape(-1, 2)
        if len(pts) < 3 or _fan_area(pts) < MIN_RING_AREA_PX:
            continue
        # fillPoly includes the boundary pixels, and mask_rings reads a pixel
        # index back as the vertex, so rasterize → re-extract is exact for
        # axis-aligned edges and idempotent in general.
        fixed = np.round((pts - (ox, oy)) * _SCALE).astype(np.int32)
        cv2.fillPoly(mask, [fixed.reshape(-1, 1, 2)], 1, lineType=cv2.LINE_8, shift=_SHIFT)
    return mask


def _simplify(contour, max_vertices: int, eps_frac: float):
    """``approxPolyDP`` a contour down to ``max_vertices``, never below 3 vertices.

    The epsilon is ``eps_frac`` of the ring's perimeter, capped at the
    perimeter of its bounding rectangle: a jagged mask edge (SAM, a noisy
    model mask) can have a perimeter many times its outline, and the uncapped
    epsilon would collapse the ring. Should the epsilon still overshoot while
    enforcing the vertex cap, the last approximation that kept a polygon is
    subsampled uniformly instead of dropping a component that passed the
    area filter.
    """
    cv2 = _cv2()
    _, _, w, h = cv2.boundingRect(contour)
    perimeter = min(cv2.arcLength(contour, True), 2.0 * (w + h))
    eps = max(eps_frac * perimeter, 0.5)
    previous = contour
    approx = cv2.approxPolyDP(contour, eps, True)
    while len(approx) > max_vertices:
        previous = approx
        eps *= 1.25
        approx = cv2.approxPolyDP(contour, eps, True)
    if len(approx) >= 3:
        return approx.reshape(-1, 2)
    pts = previous.reshape(-1, 2)
    if len(pts) > max_vertices:
        pts = pts[np.linspace(0, len(pts), max_vertices, endpoint=False).astype(int)]
    return pts


def _dedupe(pts: np.ndarray) -> np.ndarray:
    """Drop consecutive duplicate vertices (the closing one included)."""
    if len(pts) < 2:
        return pts
    keep = np.any(pts != np.roll(pts, -1, axis=0), axis=1)
    return pts[keep]


def _self_intersects(pts: np.ndarray) -> bool:
    """Whether two non-adjacent edges of a closed ring properly cross."""
    n = len(pts)
    if n < 4:
        return False
    a = pts[:, None, :]
    b = np.roll(pts, -1, axis=0)[:, None, :]
    c = pts[None, :, :]
    d = np.roll(pts, -1, axis=0)[None, :, :]

    def orient(p, q, r):
        return np.sign((q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1])
                       - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0]))

    crossing = (orient(a, b, c) * orient(a, b, d) < 0) & (orient(c, d, a) * orient(c, d, b) < 0)
    idx = np.arange(n)
    gap = np.abs(idx[:, None] - idx[None, :])
    crossing &= (gap > 1) & (gap < n - 1)
    return bool(crossing.any())


def _ring_perimeter(pts: np.ndarray) -> float:
    return float(np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1).sum())


def mask_rings(
    mask: np.ndarray,
    width: int,
    height: int,
    *,
    offset=(0, 0),
    min_area_frac: float = MIN_AREA_FRAC,
    eps_frac: float = EPS_FRAC,
    max_vertices: int = MAX_VERTICES,
    max_rings: int = MAX_RINGS,
) -> list[Ring]:
    """Exterior rings of a binary mask, normalized to a ``width``×``height`` image.

    ``mask`` may be a crop placed at pixel ``offset`` inside that image (a
    box-sized instance mask, a tile window). Components below
    ``min_area_frac`` of the *image* area are dropped. Returns the rings
    largest first.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"image size must be positive, got {width}x{height}")
    cv2 = _cv2()
    binary = np.ascontiguousarray((np.asarray(mask) > 0).astype(np.uint8))
    if binary.ndim != 2 or not binary.any():
        return []
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_frac * float(width) * float(height)
    kept = []
    for contour in contours:
        # Pixel-count area: contourArea of a 1-px-wide blob is 0.
        x, y, w, h = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour)) + 0.5 * cv2.arcLength(contour, True) + 1.0
        if area < max(min_area, 1.0) or w < 1 or h < 1:
            continue
        pts = _simplify(contour, max_vertices, eps_frac)
        if len(pts) < 3:
            continue
        kept.append((area, pts))
    kept.sort(key=lambda item: -item[0])
    ox, oy = float(offset[0]), float(offset[1])
    rings: list[Ring] = []
    for _, pts in kept[:max_rings]:
        # Contour points are pixel indices (see rasterize_rings).
        fx = (pts[:, 0].astype(np.float64) + ox) / float(width)
        fy = (pts[:, 1].astype(np.float64) + oy) / float(height)
        ring = np.stack([np.clip(fx, 0.0, 1.0), np.clip(fy, 0.0, 1.0)], axis=1)
        area = ring_area(ring)
        if area == 0.0:
            continue
        if area < 0:
            ring = ring[::-1]
        rings.append(ring.tolist())
    return rings


def normalize_rings(
    rings: Sequence,
    width: int,
    height: int,
    *,
    min_area_frac: float = MIN_AREA_FRAC,
    eps_frac: float = EPS_FRAC,
    max_vertices: int = MAX_VERTICES,
    max_rings: int = MAX_RINGS,
) -> list[Ring]:
    """Normalize one instance's rings (normalized coordinates) at ``width``×``height``.

    Each ring is a sequence of ``(x, y)`` pairs or a flat ``x1, y1, x2, …``
    list. Rings are rasterized in pixel space over the instance's bounding
    rectangle only (never a full-size canvas per instance) and re-extracted
    by :func:`mask_rings`. An instance whose every ring is degenerate or too
    small normalizes to ``[]``.

    The function is idempotent: when the given rings are already valid (no
    self-intersection, within the vertex cap, one ring per extracted
    component) and their raster agrees with the re-extraction to within a
    one-pixel band, they are returned as given (oriented, largest first), so
    a ring saved again and again by the label editor never drifts.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"image size must be positive, got {width}x{height}")
    scale = np.array([float(width), float(height)])
    inputs = []
    for ring in rings:
        pts = np.asarray(ring, dtype=np.float64).reshape(-1, 2)
        if len(pts) and np.all(np.isfinite(pts)):
            pts = _dedupe(np.clip(pts, 0.0, 1.0))
            if len(pts) >= 3:
                inputs.append(pts)
    rings_px = [pts * scale for pts in inputs]
    box = rings_bbox(rings_px)
    if box is None:
        return []
    x0 = max(0, int(np.floor(box[0])) - 1)
    y0 = max(0, int(np.floor(box[1])) - 1)
    x1 = min(int(width), int(np.ceil(box[2])) + 1)
    y1 = min(int(height), int(np.ceil(box[3])) + 1)
    if x1 <= x0 or y1 <= y0:
        return []
    mask = rasterize_rings(rings_px, x1 - x0, y1 - y0, offset=(x0, y0))
    extracted = mask_rings(
        mask, width, height, offset=(x0, y0), min_area_frac=min_area_frac,
        eps_frac=eps_frac, max_vertices=max_vertices, max_rings=max_rings,
    )
    # Fixed point: contour tracing trims every concave corner by a pixel, so
    # re-extracting an already normal ring would creep on each save. Rings
    # that are valid as given and whose raster matches the re-extraction to
    # within a one-pixel band are kept verbatim.
    if len(inputs) != len(extracted) or any(
        len(pts) > max_vertices or _self_intersects(px) for pts, px in zip(
            inputs, rings_px, strict=True)
    ):
        return extracted
    redrawn = rasterize_rings(
        [np.asarray(r) * scale for r in extracted], x1 - x0, y1 - y0, offset=(x0, y0),
    )
    band = sum(_ring_perimeter(px) for px in rings_px)
    if np.count_nonzero(mask != redrawn) > band:
        return extracted
    kept = [pts if ring_area(pts) > 0 else pts[::-1] for pts in inputs]
    kept.sort(key=lambda pts: -abs(ring_area(pts)))
    return [[[float(x), float(y)] for x, y in pts] for pts in kept]


def clip_polygon_rows(
    class_ids,
    instances_px,
    tile: Tile,
    min_visible: float = 0.25,
) -> tuple[list[str], int]:
    """Ground-truth polygons of one frame rewritten as YOLO rows for ``tile``.

    The polygon counterpart of :func:`conecsa_common.tiling.tile_label_rows`.
    ``instances_px[i]`` is the list of rings (frame pixels) of an instance of
    class ``class_ids[i]``. Each instance is rasterized over its bounding
    rectangle, the tile window is cut out and the rings are re-extracted, so
    a concave polygon split by the tile edge becomes as many rings as there
    are visible pieces. An instance is kept when at least ``min_visible`` of
    its *area* lies inside the tile; each surviving ring is one
    ``"class x1 y1 … xn yn"`` row normalized to the tile. Returns the rows and
    the number of instances that touch the tile at all (see ``tile_label_rows``
    for why the caller needs both).
    """
    if not 0.0 < min_visible <= 1.0:
        raise ValueError(f"min_visible must be in (0, 1], got {min_visible}")
    ids = list(class_ids)
    instances = list(instances_px)
    if len(ids) != len(instances):
        raise ValueError(f"class_ids and instances_px disagree on N: {len(ids)}, {len(instances)}")
    rows: list[str] = []
    touched = 0
    for class_id, rings in zip(ids, instances, strict=True):
        box = rings_bbox(rings)
        if box is None:
            continue
        bx0, by0 = int(np.floor(box[0])), int(np.floor(box[1]))
        bx1, by1 = int(np.ceil(box[2])) + 1, int(np.ceil(box[3])) + 1
        if bx1 <= tile.x0 or by1 <= tile.y0 or bx0 >= tile.x1 or by0 >= tile.y1:
            continue
        mask = rasterize_rings(rings, bx1 - bx0, by1 - by0, offset=(bx0, by0))
        total = int(mask.sum())
        if total == 0:
            continue
        cx0, cy0 = max(bx0, tile.x0), max(by0, tile.y0)
        cx1, cy1 = min(bx1, tile.x1), min(by1, tile.y1)
        window = mask[cy0 - by0:cy1 - by0, cx0 - bx0:cx1 - bx0]
        visible = int(window.sum())
        if visible == 0:
            continue
        touched += 1
        if visible / total < min_visible:
            continue
        for ring in mask_rings(
            window, tile.width, tile.height, offset=(cx0 - tile.x0, cy0 - tile.y0),
        ):
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in ring)
            rows.append(f"{int(class_id)} {coords}")
    return rows, touched
