# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Materialise one dataset image as the tile crops the inference-service runs on.

The inference-service slices every frame into overlapping square tiles
(``TILING_MODE=grid``, geometry from ``conecsa_common.tiling``) and a model
only performs at the scale it was trained at, so the training split has to
be built from the same crops: :func:`materialize_tiles` decodes one stored
dataset image, lays the same grid over it, writes ``<stem>_t<k>.jpg`` per
tile and rewrites the image's YOLO rows per tile (clipped and re-normalized
by ``conecsa_common.tiling.tile_label_rows``; a segmentation image's polygon
rows are rasterized, cut to the tile and re-extracted by
``conecsa_common.polygons.clip_polygon_rows``, so a concave outline split by
the tile edge becomes one ring per visible piece).

Label rules, validated offline before they became the default:

- a box is kept in a tile when at least ``min_visible`` of its area lies
  inside it (a small sliver would teach the wrong shape);
- a tile whose only content was such a discarded sliver is *skipped*
  (keeping it would teach that a visible piece of an object is background);
- a tile no box touches is kept as a genuine negative.

When the grid degenerates to a single tile spanning the whole image (a
legacy 640×640 letterboxed dataset, a square image, or a pixel tile at
least as large as the image) nothing is written and the caller keeps its
whole-frame path (a symlink); ``TileSplitStats.whole`` counts those.

cv2 is imported lazily so the module stays importable on a bare host; the
training image ships it (ultralytics depends on it).
"""
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

TileSpec = Union[None, str, int]
"""``None`` = whole frames, ``"auto"`` = the image's short side, ``int`` = pixels."""

LabelRow = Tuple[int, float, float, float, float]
"""``(class_id, cx, cy, w, h)`` normalized to the stored image."""

PolygonRow = Tuple[int, Sequence[Sequence[float]]]
"""``(class_id, [[x, y], …])`` — one ring normalized to the stored image."""

JPEG_QUALITY = 95


class TileSplitError(Exception):
    """An image could not be decoded or a crop could not be written."""


@dataclass
class TileSplitStats:
    """Counters accumulated over a split (all zero when tiling is off)."""

    tiles: int = 0        # crops written
    boxes: int = 0        # labels written across those crops
    fragments: int = 0    # boxes dropped for being < min_visible inside a tile
    background: int = 0   # crops written with no label (genuine negatives)
    skipped: int = 0      # tiles omitted because their only content was a fragment
    whole: int = 0        # images kept whole (grid degenerated to one full tile)

    def add(self, other: "TileSplitStats") -> None:
        """Accumulate ``other`` into this instance."""
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))


def resolve_tile(tile: TileSpec, width: int, height: int) -> Optional[int]:
    """The tile side in pixels for one image, or ``None`` when tiling is off."""
    if tile is None:
        return None
    if tile == "auto":
        from conecsa_common.tiling import auto_tile
        return auto_tile(width, height)
    if isinstance(tile, int) and tile > 0:
        return tile
    raise ValueError(f"tile must be None, 'auto' or a positive int, got {tile!r}")


def materialize_tiles(
    image_path: str,
    rows: Sequence[LabelRow],
    images_dir: str,
    labels_dir: str,
    stem: str,
    *,
    tile: TileSpec,
    overlap: float = 0.2,
    min_visible: float = 0.25,
    polygons: Optional[Sequence[PolygonRow]] = None,
) -> Tuple[List[Tuple[str, str]], TileSplitStats]:
    """Write the tile crops + labels of one image; returns ``(pairs, stats)``.

    ``pairs`` are the ``(image, label)`` paths written, in grid order, and
    are empty when the image was kept whole (``stats.whole == 1``). With
    ``polygons`` (a segmentation image) those rings are the labels and
    ``rows`` is ignored; each ring is its own instance, as in the label file,
    and ``min_visible`` applies to its visible area.
    """
    import cv2  # lazy: see the module docstring
    from conecsa_common.tiling import tile_crop, tile_grid, tile_label_rows

    stats = TileSplitStats()
    frame = cv2.imread(image_path)
    if frame is None:
        raise TileSplitError(f"could not decode dataset image {image_path}")
    height, width = frame.shape[:2]
    side = resolve_tile(tile, width, height)
    if side is None:
        stats.whole = 1
        return [], stats
    grid = tile_grid(width, height, side, overlap)
    if len(grid) == 1 and (grid[0].width, grid[0].height) == (width, height):
        stats.whole = 1
        return [], stats

    class_ids = [int(r[0]) for r in rows]
    boxes = [
        [(cx - w / 2.0) * width, (cy - h / 2.0) * height,
         (cx + w / 2.0) * width, (cy + h / 2.0) * height]
        for _, cx, cy, w, h in rows
    ]
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    rings_px = [
        (int(class_id), [[float(x) * width, float(y) * height] for x, y in points])
        for class_id, points in (polygons or ())
    ]
    written: List[Tuple[str, str]] = []
    for index, region in enumerate(grid):
        if polygons is not None:
            from conecsa_common.polygons import clip_polygon_rows

            lines, touched, kept = [], 0, 0
            for class_id, ring in rings_px:
                ring_lines, ring_touched = clip_polygon_rows(
                    [class_id], [[ring]], region, min_visible)
                lines.extend(ring_lines)
                touched += ring_touched
                kept += 1 if ring_lines else 0
            stats.fragments += touched - kept
        else:
            lines, touched = tile_label_rows(class_ids, boxes, region, min_visible)
            stats.fragments += touched - len(lines)
        if touched and not lines:
            stats.skipped += 1
            continue
        dst_image = os.path.join(images_dir, f"{stem}_t{index}.jpg")
        dst_label = os.path.join(labels_dir, f"{stem}_t{index}.txt")
        if not cv2.imwrite(dst_image, tile_crop(frame, region),
                           [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
            raise TileSplitError(f"could not write tile {dst_image}")
        with open(dst_label, "w") as f:
            f.write("".join(f"{line}\n" for line in lines))
        stats.tiles += 1
        stats.boxes += len(lines)
        if not lines:
            stats.background += 1
        written.append((dst_image, dst_label))
    return written, stats
