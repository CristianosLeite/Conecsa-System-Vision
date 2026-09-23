# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: Apache-2.0

"""SAHI-style tiled inference helpers (pure numpy).

A detector trained at 640 px sees a 1920×1080 frame downscaled roughly 3×, so
an object that is 20 px wide in the frame reaches the network at 7 px — below
what the smallest detection head resolves. Slicing Aided Hyper Inference
(SAHI, Akyon et al. 2022) sidesteps this without retraining: run the model on
overlapping square crops at native resolution, shift each crop's boxes back
into frame coordinates, and merge the duplicates that fall in the overlap
bands with non-maximum suppression (optionally alongside one full-frame pass
so large objects that straddle several tiles are still seen whole).

This module holds the geometry and the merge step, nothing else — no model,
no I/O — so that the dataset crops in ``training-service`` and the on-device
pipeline in ``inference-service`` share one implementation and the tile
layout trained on is byte-for-byte the layout deployed:

- :func:`auto_tile` — the resolution-agnostic default tile side: the frame's
  short side, so any 16:9 frame yields two columns whatever its pixel count.
- :func:`tile_grid` — deterministic, row-major square tiles clamped to the
  frame (the trailing tile in each axis slides back to the frame edge; frames
  are never padded, so the model never sees synthetic borders).
- :func:`tile_crop` / :func:`shift_boxes` — zero-copy crop and the inverse
  coordinate shift.
- :func:`clip_box` / :func:`tile_label_rows` — the training-side counterpart:
  ground-truth boxes clipped into a tile and rewritten as tile-normalized
  YOLO rows, so a model is trained on exactly the crops it will be run on.
- :func:`iou_matrix` / :func:`merge_tiles` — pairwise IoU and greedy NMS over
  the union of per-tile detections.
- :func:`tile_overlap` / :func:`merge_tiles_grouped` — the segmentation
  counterpart of the merge: instead of discarding the lower-scoring
  duplicate it groups the fragments of one object seen by different tiles,
  so their masks can be stitched into one instance.

The polygon helpers that need OpenCV live in :mod:`conecsa_common.polygons`.

NMS is deliberately re-implemented here in numpy although both consumers
have faster kernels at hand (``cv2.dnn.NMSBoxes`` in the inference pipeline,
``torchvision.ops.nms`` in the trainer). ``conecsa_common`` is the Apache-2.0
layer shared by every service and must stay importable without cv2 or torch;
a few hundred boxes per frame is far below the point where the numpy loop
matters, and one implementation on both sides keeps the offline metrics
honest about what the device will produce.
"""
from dataclasses import dataclass

import numpy as np

__all__ = [
    "Tile",
    "auto_tile",
    "clip_box",
    "iou_matrix",
    "merge_tiles",
    "merge_tiles_grouped",
    "shift_boxes",
    "tile_overlap",
    "tile_crop",
    "tile_grid",
    "tile_label_rows",
]


@dataclass(frozen=True)
class Tile:
    """A pixel box inside a frame; ``x1``/``y1`` are exclusive."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        if self.x0 < 0 or self.y0 < 0:
            raise ValueError(f"tile origin must be non-negative, got ({self.x0}, {self.y0})")
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(
                f"tile must have positive size, got ({self.x0}, {self.y0}, {self.x1}, {self.y1})"
            )

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def auto_tile(width: int, height: int) -> int:
    """The default tile side for a ``width``×``height`` frame: its short side.

    Tiles are square, so a tile spanning the short side turns the frame into
    a single row (or column) of overlapping crops whose count depends only on
    the aspect ratio, never on the pixel count: a 1280×720, 1920×1080 or
    3840×2160 frame all give two columns (K=2), 4:3 gives two heavily
    overlapping ones, wider than 1.8:1 gives three, a square frame gives
    one. Training crops and inference crops computed this way stay the same
    geometry on every camera the device may be fitted with.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"frame size must be positive, got {width}x{height}")
    return min(width, height)


def _axis_starts(length: int, tile: int, stride: int) -> list[int]:
    """Tile origins along one axis: fixed stride, last one flush with the edge."""
    if length <= tile:
        return [0]
    starts: list[int] = []
    start = 0
    while start + tile < length:
        starts.append(start)
        start += stride
    # The next tile would run past the edge: slide it back so it ends exactly
    # at the frame border instead of padding. It is never a duplicate of the
    # previous start because that one satisfied start + tile < length.
    starts.append(length - tile)
    return starts


def tile_grid(
    width: int,
    height: int,
    tile: int,
    overlap: float = 0.2,
    include_full: bool = False,
) -> list[Tile]:
    """Square tiles of side ``tile`` covering a ``width``×``height`` frame.

    Adjacent tiles overlap by ``round(overlap * tile)`` pixels
    (``0 <= overlap < 1``). Tiles are clamped to the frame: the last tile in
    each axis is shifted back so it ends at the frame edge, never padded. If
    the frame is smaller than ``tile`` in an axis, a single tile spans that
    axis. Order is deterministic and row-major (left to right, then top to
    bottom). ``include_full=True`` appends ``Tile(0, 0, width, height)`` as
    the *last* element so callers can run one whole-frame pass for large
    objects.

    Raises :class:`ValueError` on non-positive sizes or an overlap outside
    ``[0, 1)``.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"frame size must be positive, got {width}x{height}")
    if tile <= 0:
        raise ValueError(f"tile side must be positive, got {tile}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    # overlap < 1 keeps the stride at least 1 except for pathological
    # round-ups on tiny tiles; the max() guarantees the loop terminates.
    stride = max(1, tile - int(round(overlap * tile)))
    xs = _axis_starts(width, tile, stride)
    ys = _axis_starts(height, tile, stride)
    tile_w = min(tile, width)
    tile_h = min(tile, height)

    tiles = [Tile(x, y, x + tile_w, y + tile_h) for y in ys for x in xs]
    if include_full:
        tiles.append(Tile(0, 0, width, height))
    return tiles


def tile_crop(frame: np.ndarray, tile: Tile) -> np.ndarray:
    """The ``frame[y0:y1, x0:x1]`` view for ``tile`` (no copy)."""
    return frame[tile.y0:tile.y1, tile.x0:tile.x1]


def clip_box(box_xyxy, tile: Tile) -> tuple[list[float], float]:
    """Intersect one pixel ``xyxy`` box with ``tile``.

    Returns the clipped box in tile-relative pixels and the fraction of the
    original box area that lies inside the tile; ``([], 0.0)`` when the box
    is degenerate or disjoint from the tile.
    """
    bx0, by0, bx1, by1 = (float(v) for v in box_xyxy)
    x0, y0 = max(bx0, tile.x0), max(by0, tile.y0)
    x1, y1 = min(bx1, tile.x1), min(by1, tile.y1)
    area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if area <= 0.0 or inter <= 0.0:
        return [], 0.0
    return [x0 - tile.x0, y0 - tile.y0, x1 - tile.x0, y1 - tile.y0], inter / area


def tile_label_rows(
    class_ids,
    boxes_xyxy,
    tile: Tile,
    min_visible: float = 0.25,
) -> tuple[list[str], int]:
    """Ground-truth boxes of one frame rewritten as YOLO rows for ``tile``.

    ``class_ids`` and ``boxes_xyxy`` (frame pixels) are parallel sequences.
    A box is kept when at least ``min_visible`` of its area lies inside the
    tile, clipped to the tile and normalized to the tile's size as
    ``"class cx cy w h"``. Returns the rows and the number of boxes that
    *touch* the tile at all, so the caller can tell a genuinely empty tile
    (a valid negative) from one whose only content was a discarded fragment
    (which would teach the model that a visible piece of an object is
    background and should be skipped).
    """
    if not 0.0 < min_visible <= 1.0:
        raise ValueError(f"min_visible must be in (0, 1], got {min_visible}")
    ids = list(class_ids)
    boxes = list(boxes_xyxy)
    if len(ids) != len(boxes):
        raise ValueError(f"class_ids and boxes_xyxy disagree on N: {len(ids)}, {len(boxes)}")
    rows: list[str] = []
    touched = 0
    for class_id, box in zip(ids, boxes, strict=True):
        clipped, visible = clip_box(box, tile)
        if visible <= 0.0:
            continue
        touched += 1
        if visible < min_visible:
            continue
        cx = (clipped[0] + clipped[2]) / 2.0 / tile.width
        cy = (clipped[1] + clipped[3]) / 2.0 / tile.height
        w = (clipped[2] - clipped[0]) / tile.width
        h = (clipped[3] - clipped[1]) / tile.height
        rows.append(f"{int(class_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return rows, touched


def _as_boxes(boxes: np.ndarray, name: str) -> np.ndarray:
    """Validate an ``(N, 4)`` box array and return it as a float array."""
    arr = np.asarray(boxes)
    if arr.size == 0:
        arr = arr.reshape(0, 4)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"{name} must have shape (N, 4), got {arr.shape}")
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)
    return arr


def shift_boxes(boxes_xyxy: np.ndarray, ox: int, oy: int) -> np.ndarray:
    """Translate ``(N, 4)`` xyxy boxes from tile space into frame space.

    Adds the tile origin ``(ox, oy)`` to both corners. Returns a new float
    array of the same shape; an empty input yields an empty ``(0, 4)`` array.
    """
    boxes = _as_boxes(boxes_xyxy, "boxes_xyxy")
    offset = np.array([ox, oy, ox, oy], dtype=boxes.dtype)
    return boxes + offset


def iou_matrix(a_xyxy: np.ndarray, b_xyxy: np.ndarray) -> np.ndarray:
    """Pairwise intersection-over-union: ``(N, 4)`` × ``(M, 4)`` → ``(N, M)``.

    Degenerate (zero-area) pairs yield 0 rather than NaN.
    """
    a = _as_boxes(a_xyxy, "a_xyxy")
    b = _as_boxes(b_xyxy, "b_xyxy")

    # Broadcast (N, 1, 2) against (1, M, 2) for the intersection corners.
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter

    out = np.zeros_like(inter)
    np.divide(inter, union, out=out, where=union > 0)
    return out


def merge_tiles(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_threshold: float = 0.5,
    class_aware: bool = True,
) -> np.ndarray:
    """Greedy NMS over detections already shifted into frame space.

    Returns the indices to *keep* as an ``int64`` array sorted by descending
    score. A box is suppressed when its IoU with an already-kept, higher
    scoring box exceeds ``iou_threshold``; with ``class_aware=True`` only
    boxes of the same class id suppress each other, so two classes may share
    a location. Ties are broken by input order, which keeps the result
    deterministic. Running the merge again on the kept set is a no-op.
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold}")

    boxes = _as_boxes(boxes_xyxy, "boxes_xyxy")
    score_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    class_arr = np.asarray(classes).reshape(-1)
    n = boxes.shape[0]
    if score_arr.shape[0] != n or class_arr.shape[0] != n:
        raise ValueError(
            f"boxes, scores and classes disagree on N: {n}, {score_arr.shape[0]}, "
            f"{class_arr.shape[0]}"
        )
    if n == 0:
        return np.empty(0, dtype=np.int64)

    order = np.argsort(-score_arr, kind="stable")
    ious = iou_matrix(boxes, boxes)
    if class_aware:
        # Cross-class pairs never suppress: zero their IoU up front.
        same_class = class_arr[:, None] == class_arr[None, :]
        ious = np.where(same_class, ious, 0.0)

    alive = np.ones(n, dtype=bool)
    keep: list[int] = []
    for idx in order:
        if not alive[idx]:
            continue
        keep.append(int(idx))
        alive &= ~(ious[idx] > iou_threshold)
        alive[idx] = False
    return np.asarray(keep, dtype=np.int64)


def tile_overlap(a: Tile, b: Tile) -> "Tile | None":
    """The overlap rectangle of two tiles, or ``None`` when they do not overlap."""
    x0, y0 = max(a.x0, b.x0), max(a.y0, b.y0)
    x1, y1 = min(a.x1, b.x1), min(a.y1, b.y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return Tile(x0, y0, x1, y1)


def merge_tiles_grouped(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    tile_ids,
    tiles: "list[Tile]",
    iou_threshold: float = 0.5,
    ios_threshold: float = 0.5,
) -> list[list[int]]:
    """Group the per-tile fragments of one object (frame-space boxes).

    Returns groups of input indices; each group's first index is its
    highest-scoring member (the survivor) and groups are ordered by that
    score. Greedy by descending score, ties by input order. A candidate joins
    a group when it matches **any** member under all of:

    - same class;
    - a different tile of origin (``tile_ids`` indexes ``tiles``) — and the
      group holds no member of the candidate's own tile yet, since two
      detections from one tile are the model's own distinct predictions;
    - both boxes intersect the overlap rectangle of their two tiles;
    - ``IoU >= iou_threshold`` **or** ``IoS >= ios_threshold``, where IoS is
      the intersection over the smaller box with **both boxes clipped to the
      overlap rectangle** first. Fragments of one object cut by the tile
      edges cover the same part of the band, so their band IoS is close to 1
      however long the object is; whole-box IoS (and IoU) falls below any
      useful threshold as soon as an object is much wider than the band.

    Matching against any member makes the rule transitive, so an object
    spanning three tiles ends up in one group although the outer tiles never
    overlap. A candidate that matches several groups joins the one with the
    highest-scoring survivor and merges into it every other matching group
    whose tiles are disjoint from the ones gathered so far, so the order of
    the fragment scores never leaves one object split; everything else starts
    its own group.
    """
    for name, value in (("iou_threshold", iou_threshold), ("ios_threshold", ios_threshold)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    boxes = _as_boxes(boxes_xyxy, "boxes_xyxy")
    score_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    class_arr = np.asarray(classes).reshape(-1)
    tile_arr = np.asarray(tile_ids, dtype=np.int64).reshape(-1)
    n = boxes.shape[0]
    if not score_arr.shape[0] == class_arr.shape[0] == tile_arr.shape[0] == n:
        raise ValueError(
            f"boxes, scores, classes and tile_ids disagree on N: {n}, {score_arr.shape[0]}, "
            f"{class_arr.shape[0]}, {tile_arr.shape[0]}"
        )
    if n == 0:
        return []
    if tile_arr.min() < 0 or tile_arr.max() >= len(tiles):
        raise ValueError("tile_ids must index tiles")

    match = _fragment_matches(boxes, class_arr, tile_arr, tiles, iou_threshold, ios_threshold)

    # Each group keeps the set of its members' tiles next to it.
    groups: list[list[int]] = []
    group_tiles: list[set[int]] = []
    for idx in np.argsort(-score_arr, kind="stable"):
        i = int(idx)
        tile_i = int(tile_arr[i])
        row = match[i]
        joined: "int | None" = None
        for g, group in enumerate(groups):
            if joined is None:
                if tile_i in group_tiles[g] or not row[group].any():
                    continue
                group.append(i)
                group_tiles[g].add(tile_i)
                joined = g
            elif (group_tiles[g].isdisjoint(group_tiles[joined]) and group
                  and row[group].any()):
                # The candidate bridges two groups formed before it arrived
                # (a middle fragment scoring below both outer ones).
                groups[joined].extend(group)
                group_tiles[joined] |= group_tiles[g]
                group.clear()
                group_tiles[g] = set()
        if joined is None:
            groups.append([i])
            group_tiles.append({tile_i})
        else:
            kept = [g for g, group in enumerate(groups) if group]
            groups = [groups[g] for g in kept]
            group_tiles = [group_tiles[g] for g in kept]
    return groups


def _fragment_matches(boxes: np.ndarray, class_arr: np.ndarray, tile_arr: np.ndarray,
                      tiles: "list[Tile]", iou_threshold: float,
                      ios_threshold: float) -> np.ndarray:
    """``match[i, j]``: fragments ``i`` and ``j`` may belong to one object.

    The pairwise rule of :func:`merge_tiles_grouped` (same class, different
    tiles, both boxes touching their tiles' overlap, IoU or band IoS over its
    threshold), evaluated once per tile pair on arrays. A box touches the
    overlap when it intersects it with positive area; band IoS clips both
    boxes to the overlap and divides their intersection by the smaller
    clipped area (0 when that area is 0).
    """
    n = boxes.shape[0]
    match = np.zeros((n, n), dtype=bool)
    ious = iou_matrix(boxes, boxes)
    for ta in range(len(tiles)):
        for tb in range(ta + 1, len(tiles)):
            band = tile_overlap(tiles[ta], tiles[tb])
            if band is None:
                continue
            touch = ((boxes[:, 0] < band.x1) & (boxes[:, 2] > band.x0)
                     & (boxes[:, 1] < band.y1) & (boxes[:, 3] > band.y0))
            ia = np.flatnonzero((tile_arr == ta) & touch)
            ib = np.flatnonzero((tile_arr == tb) & touch)
            if ia.size == 0 or ib.size == 0:
                continue
            a = boxes[ia][:, None, :]
            b = boxes[ib][None, :, :]
            ca = (np.maximum(a[..., 0], band.x0), np.maximum(a[..., 1], band.y0),
                  np.minimum(a[..., 2], band.x1), np.minimum(a[..., 3], band.y1))
            cb = (np.maximum(b[..., 0], band.x0), np.maximum(b[..., 1], band.y0),
                  np.minimum(b[..., 2], band.x1), np.minimum(b[..., 3], band.y1))
            area_a = np.maximum(0.0, ca[2] - ca[0]) * np.maximum(0.0, ca[3] - ca[1])
            area_b = np.maximum(0.0, cb[2] - cb[0]) * np.maximum(0.0, cb[3] - cb[1])
            inter = (np.maximum(0.0, np.minimum(ca[2], cb[2]) - np.maximum(ca[0], cb[0]))
                     * np.maximum(0.0, np.minimum(ca[3], cb[3]) - np.maximum(ca[1], cb[1])))
            smaller = np.minimum(area_a, area_b)
            ios = np.divide(inter, smaller, out=np.zeros_like(inter), where=smaller > 0)
            pair = ((class_arr[ia][:, None] == class_arr[ib][None, :])
                    & ((ious[np.ix_(ia, ib)] >= iou_threshold) | (ios >= ios_threshold)))
            match[np.ix_(ia, ib)] = pair
            match[np.ix_(ib, ia)] = pair.T
    return match
