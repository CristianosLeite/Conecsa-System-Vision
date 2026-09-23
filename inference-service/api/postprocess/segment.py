# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Instance segmentation postprocess.

The YOLO26 end-to-end segmentation head has two outputs, picked by rank: the
rows ``[1, N ≤ 300, 6 + nm]`` (``x1, y1, x2, y2, conf, cls`` in model-input
pixels, then ``nm`` mask coefficients) and the prototypes
``[1, nm, S/4, S/4]``. An instance's mask is ``coeffs · protos > 0``, the
logit form of ``sigmoid > 0.5`` that ultralytics thresholds too.

Per frame:

1. Per tile, only rows strictly above ``CONFIDENCE_THRESHOLD`` (the one
   confidence gate) that the overlay NMS keeps within the tile get a mask,
   at most ``SEGMENT_MAX_MASKS_PER_TILE`` of them, highest scores first. The
   mask is computed at prototype resolution on the box's window only,
   resized bilinearly to the box in frame pixels and thresholded at 0: a
   uint8 crop, never a full-frame mask per row.
2. Tiled frames merge across tiles. With ``SEGMENT_TILE_STITCH=1`` (default)
   the fragments of one object are grouped by ``merge_tiles_grouped`` and
   their masks OR-ed into one instance (bbox = bounding rectangle of the
   union, score = the best member's); with ``0`` the IoU-only ``merge_tiles``
   of detection keeps the best fragment with its own mask, so an object on
   the seam stays cut — the documented comparison mode.
3. The overlay NMS (``OVERLAY_THRESHOLD``, the same class-agnostic
   suppression as detection), then at most ``SEGMENT_MAX_MASKS`` instances,
   then the detection-area filter on the recomputed center.
4. One frame-sized uint8 label buffer, allocated once and reused, collects
   every kept mask; each instance's box gets one alpha fill where its label
   won (the best instance on top), then the boxes and
   class labels exactly as detection draws them, then the areas. Each
   instance's outline becomes normalized exterior rings
   (``conecsa_common.polygons.mask_rings``).

Stage C is single-threaded, so one strategy instance (and its buffer) serves
every lane; the labeling assistant builds its own.
"""
import functools
import logging
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import cv2
import numpy as np

from ..models.detection_models import Detection
from ..utils import bgr_to_hex
from ..views.detection_boxes import nms_indices
from ..yolo_detector import YOLODetector
from .base import PostprocessResult

logger = logging.getLogger(__name__)

#: Opacity of the mask fill on the processed frame.
MASK_ALPHA = 0.45
#: Smallest box side (frame pixels) kept, as detection does.
_MIN_BOX_SIDE = 5


@dataclass(frozen=True)
class SegmentSettings:
    """The ``SEGMENT_*`` / ``TILING_MERGE_IOS`` knobs."""

    stitch: bool = True
    max_masks: int = 32
    max_masks_per_tile: int = 32
    min_component_area: float = 0.0005
    merge_ios: float = 0.5


def _env_number(name: str, default, low, high, cast):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = cast(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", name, raw, default)
        return default
    if not low <= value <= high:
        logger.warning("%s=%s out of %s..%s; using %s", name, value, low, high, default)
        return default
    return value


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    logger.warning("%s=%r is not a flag; using %s", name, raw, default)
    return default


def segment_settings_from_env() -> SegmentSettings:
    """Read the segmentation knobs; invalid values fall back to the defaults.

    ``SEGMENT_MAX_MASKS`` is bounded by 255, the label buffer's uint8 range.
    """
    d = SegmentSettings()
    return SegmentSettings(
        stitch=_env_flag("SEGMENT_TILE_STITCH", d.stitch),
        max_masks=_env_number("SEGMENT_MAX_MASKS", d.max_masks, 1, 255, int),
        max_masks_per_tile=_env_number(
            "SEGMENT_MAX_MASKS_PER_TILE", d.max_masks_per_tile, 1, 300, int),
        min_component_area=_env_number(
            "SEGMENT_MIN_COMPONENT_AREA", d.min_component_area, 0.0, 1.0, float),
        merge_ios=_env_number("TILING_MERGE_IOS", d.merge_ios, 0.0, 1.0, float),
    )


def segmentation_outputs(outputs: Sequence[np.ndarray]):
    """``(rows, protos)`` among an engine's outputs, picked by rank (3 and 4)."""
    rows = next((o for o in outputs if getattr(o, "ndim", 0) == 3), None)
    protos = next((o for o in outputs if getattr(o, "ndim", 0) == 4), None)
    return rows, protos


@dataclass
class _Instance:
    """One candidate: frame-space box, its mask crop and the crop's origin."""

    box: tuple  # float (x1, y1, x2, y2) in frame pixels
    score: float
    class_id: int
    tile: int
    mask: np.ndarray  # uint8 0/1, (height, width) of the crop
    x0: int
    y0: int


def _instance_mask(protos: np.ndarray, coeffs: np.ndarray, box_in, size: float, meta,
                   rect) -> np.ndarray:
    """The 0/1 mask of one row over its integer box ``rect`` (tile pixels).

    ``box_in`` is the row's box in model-input pixels, ``size`` the model
    input side. The mask is evaluated on the prototype cells the box covers,
    resized bilinearly to that window's extent in tile pixels (undoing the
    letterbox: X spans the input width, Y is offset by ``border_top`` and
    scaled by ``meta.scale``), cut to the box and thresholded at 0.5.
    """
    _, ph, pw = protos.shape
    ix0, iy0, ix1, iy1 = rect
    out = np.zeros((iy1 - iy0, ix1 - ix0), dtype=np.uint8)
    ratio = ph / size
    px0 = max(int(np.floor(box_in[0] * ratio)), 0)
    py0 = max(int(np.floor(box_in[1] * ratio)), 0)
    px1 = min(int(np.ceil(box_in[2] * ratio)), pw)
    py1 = min(int(np.ceil(box_in[3] * ratio)), ph)
    if px1 <= px0 or py1 <= py0 or out.size == 0:
        return out
    # Logits, not probabilities: sigmoid(x) > 0.5 is x > 0, and ultralytics'
    # process_mask also resizes the logits and thresholds them at 0, so the
    # per-pixel exp is skipped.
    # The product np.tensordot(coeffs, window, axes=1) reduces to, without its
    # per-call bookkeeping (32 calls a tile in a dense scene).
    window = protos[:, py0:py1, px0:px1]
    logits = np.dot(coeffs.astype(np.float32).reshape(1, -1),
                    window.reshape(window.shape[0], -1)).reshape(window.shape[1:])

    sx = meta.width / size
    wx0, wx1 = px0 / ratio * sx, px1 / ratio * sx
    wy0 = (py0 / ratio - meta.border_top) * meta.scale
    wy1 = (py1 / ratio - meta.border_top) * meta.scale
    win_w = max(1, int(round(wx1 - wx0)))
    win_h = max(1, int(round(wy1 - wy0)))
    resized = cv2.resize(logits, (win_w, win_h), interpolation=cv2.INTER_LINEAR)

    ox = int(round(ix0 - wx0))
    oy = int(round(iy0 - wy0))
    sx0, sy0 = max(ox, 0), max(oy, 0)
    sx1, sy1 = min(ox + out.shape[1], win_w), min(oy + out.shape[0], win_h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - oy:sy1 - oy, sx0 - ox:sx1 - ox] = resized[sy0:sy1, sx0:sx1] > 0.0
    return out


def _union(members: List[_Instance]) -> _Instance:
    """One instance from a group: OR of the masks, bounding rectangle of the union."""
    survivor = members[0]
    if len(members) == 1:
        return survivor
    x0 = min(m.x0 for m in members)
    y0 = min(m.y0 for m in members)
    x1 = max(m.x0 + m.mask.shape[1] for m in members)
    y1 = max(m.y0 + m.mask.shape[0] for m in members)
    canvas = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    for m in members:
        view = canvas[m.y0 - y0:m.y0 - y0 + m.mask.shape[0], m.x0 - x0:m.x0 - x0 + m.mask.shape[1]]
        np.maximum(view, m.mask, out=view)
    bx, by, bw, bh = cv2.boundingRect(canvas)
    if bw > 0 and bh > 0:
        box = (float(x0 + bx), float(y0 + by), float(x0 + bx + bw), float(y0 + by + bh))
    else:
        box = (min(m.box[0] for m in members), min(m.box[1] for m in members),
               max(m.box[2] for m in members), max(m.box[3] for m in members))
    return _Instance(box, max(m.score for m in members), survivor.class_id, survivor.tile,
                     canvas, x0, y0)


class SegmentPostprocessor:
    """Segmentation strategy: masks, tile stitching, NMS, area filter, overlay."""

    task = "segment"

    def __init__(self, class_labels: List[str], config: Any,
                 detector: Optional[YOLODetector] = None):
        self.config = config
        # Class names/colors, the overlay drawing, the area filter and the
        # tile-merge IoU are detection's, so both tasks look and filter alike.
        self.detector = detector if detector is not None else YOLODetector(class_labels, config)
        self.settings = segment_settings_from_env()
        self._labels: Optional[np.ndarray] = None

    @property
    def class_labels(self) -> List[str]:
        return list(self.detector.class_labels)

    def set_class_labels(self, class_labels: List[str]) -> None:
        """Adopt renamed labels live (see the protocol)."""
        self.detector.set_class_labels(class_labels)

    def set_areas(self, areas: Sequence[Any]) -> None:
        self.detector.set_areas(areas)

    def reset_state(self) -> None:
        """Segmentation keeps no per-stream state (the label buffer is scratch)."""

    def caps(self) -> tuple:
        """``(per frame, per tile)`` instance caps, read on every frame.

        The operator's per-model limit (``Config.SEGMENT_MAX_INSTANCES``)
        sets both; without one the ``SEGMENT_MAX_MASKS*`` knobs apply.
        """
        limit = getattr(self.config, "SEGMENT_MAX_INSTANCES", None)
        if limit:
            return int(limit), int(limit)
        return self.settings.max_masks, self.settings.max_masks_per_tile

    # ── decode ────────────────────────────────────────────────────────────────

    def _decode_tile(self, outputs: Sequence[np.ndarray], meta, tile: int) -> List[_Instance]:
        rows, protos = segmentation_outputs(outputs)
        if rows is None or protos is None:
            return []
        protos = np.asarray(protos, dtype=np.float32)[0]
        nm, ph, _ = protos.shape
        rows = np.asarray(rows, dtype=np.float32).reshape(-1, rows.shape[-1])
        if rows.shape[1] < 6 + nm or rows.shape[0] == 0:
            return []
        conf = rows[:, 4]
        gated = np.flatnonzero(conf > float(self.config.CONFIDENCE_THRESHOLD))
        if gated.size == 0:
            return []
        order = gated[np.argsort(-conf[gated], kind="stable")][:self.detector.nms_top_k]

        size = float(meta.input_size) if meta.input_size else float(4 * ph)
        boxes_in = rows[order, :4].astype(np.float64)
        if float(np.max(np.abs(boxes_in))) <= 2.0:  # normalized-coordinate export
            boxes_in = boxes_in * size
        sx = meta.width / size

        candidates = []
        for box_in, row in zip(boxes_in, order, strict=True):
            if not (box_in[0] < box_in[2] and box_in[1] < box_in[3]):
                continue
            x1 = min(max(box_in[0] * sx, 0.0), float(meta.width))
            x2 = min(max(box_in[2] * sx, 0.0), float(meta.width))
            y1 = min(max((box_in[1] - meta.border_top) * meta.scale, 0.0), float(meta.height))
            y2 = min(max((box_in[3] - meta.border_top) * meta.scale, 0.0), float(meta.height))
            if x2 - x1 <= _MIN_BOX_SIDE or y2 - y1 <= _MIN_BOX_SIDE:
                continue
            candidates.append((box_in, row, (x1, y1, x2, y2)))
        if not candidates:
            return []
        # The mask math is the expensive step, so rows the overlay NMS drops
        # as duplicates of a better row in this tile never get a mask, and
        # the per-tile cap counts distinct objects. Candidates are in
        # score order, so sorted indices keep the best first.
        keep = nms_indices([c[2] for c in candidates], [float(conf[c[1]]) for c in candidates],
                           self.config.OVERLAY_THRESHOLD)
        keep = sorted(int(k) for k in keep)[:self.caps()[1]]

        instances = []
        for box_in, row, (x1, y1, x2, y2) in (candidates[k] for k in keep):
            rect = (int(np.floor(x1)), int(np.floor(y1)), int(np.ceil(x2)), int(np.ceil(y2)))
            mask = _instance_mask(protos, rows[row, 6:6 + nm], box_in, size, meta, rect)
            instances.append(_Instance(
                (x1 + meta.ox, y1 + meta.oy, x2 + meta.ox, y2 + meta.oy),
                float(conf[row]), int(rows[row, 5]), tile, mask,
                rect[0] + meta.ox, rect[1] + meta.oy,
            ))
        return instances

    def _merge(self, instances: List[_Instance], metas: Sequence[Any]) -> List[_Instance]:
        if len(instances) < 2:
            return instances
        from conecsa_common.tiling import Tile, merge_tiles, merge_tiles_grouped

        boxes = np.asarray([i.box for i in instances], dtype=np.float64)
        scores = np.asarray([i.score for i in instances], dtype=np.float64)
        classes = np.asarray([i.class_id for i in instances])
        if self.settings.stitch:
            tiles = [Tile(m.ox, m.oy, m.ox + m.width, m.oy + m.height) for m in metas]
            groups = merge_tiles_grouped(
                boxes, scores, classes, [i.tile for i in instances], tiles,
                iou_threshold=self.detector.tile_merge_iou,
                ios_threshold=self.settings.merge_ios,
            )
            return [_union([instances[k] for k in group]) for group in groups]
        keep = merge_tiles(boxes, scores, classes, iou_threshold=self.detector.tile_merge_iou)
        return [instances[int(k)] for k in keep]

    def _suppress(self, instances: List[_Instance]) -> List[_Instance]:
        if not instances:
            return []
        order = sorted(range(len(instances)), key=lambda k: -instances[k].score)
        order = order[:self.detector.nms_top_k]
        keep = nms_indices([instances[k].box for k in order], [instances[k].score for k in order],
                           self.config.OVERLAY_THRESHOLD)
        kept = sorted((instances[order[int(k)]] for k in keep), key=lambda i: -i.score)
        max_masks = self.caps()[0]
        if len(kept) > max_masks:
            logger.debug("segment: %d instances over the limit of %d dropped",
                         len(kept) - max_masks, max_masks)
            kept = kept[:max_masks]
        return kept

    # ── draw ──────────────────────────────────────────────────────────────────

    def _label_buffer(self, height: int, width: int) -> np.ndarray:
        if self._labels is None or self._labels.shape != (height, width):
            self._labels = np.zeros((height, width), dtype=np.uint8)
        return self._labels

    def _fill_masks(self, img: np.ndarray, kept: List[tuple]) -> None:
        """One alpha fill of every kept mask, through the reusable label buffer."""
        if not kept:
            return
        height, width = img.shape[:2]
        labels = self._label_buffer(height, width)
        painted = []
        # Lowest score first, so the best instance ends on top where masks overlap.
        for index in range(len(kept) - 1, -1, -1):
            det, inst = kept[index]
            h, w = inst.mask.shape
            x0, y0 = max(inst.x0, 0), max(inst.y0, 0)
            x1, y1 = min(inst.x0 + w, width), min(inst.y0 + h, height)
            if x1 <= x0 or y1 <= y0:
                continue
            crop = inst.mask[y0 - inst.y0:y1 - inst.y0, x0 - inst.x0:x1 - inst.x0]
            np.copyto(labels[y0:y1, x0:x1], index + 1, where=crop > 0)
            painted.append((index, det.class_id, x0, y0, x1, y1))
        if not painted:
            return
        # One uint8 blend over the union of the boxes, each pixel against the
        # color of the label that won it (a per-label lookup table), copied
        # back where any label won. Per-box blends repeated the work wherever
        # boxes overlapped: 59 ms of stage C on the device for 32 dense
        # instances.
        ux0 = min(p[2] for p in painted)
        uy0 = min(p[3] for p in painted)
        ux1 = max(p[4] for p in painted)
        uy1 = max(p[5] for p in painted)
        region = labels[uy0:uy1, ux0:ux1]
        lut = np.zeros((256, 3), dtype=np.uint8)
        for index, class_id, *_ in painted:
            lut[index + 1] = self.detector._get_class_color(class_id)
        color = cv2.merge([cv2.LUT(region, np.ascontiguousarray(lut[:, c])) for c in range(3)])
        view = np.ascontiguousarray(img[uy0:uy1, ux0:ux1])
        blended = cv2.addWeighted(view, 1.0 - MASK_ALPHA, color, MASK_ALPHA, 0.0)
        cv2.copyTo(blended, (region > 0).view(np.uint8), view)
        img[uy0:uy1, ux0:ux1] = view
        region[:] = 0  # the buffer is scratch: clear what this frame painted

    # ── entry point ───────────────────────────────────────────────────────────

    def process(self, tile_outputs: Sequence[Sequence[np.ndarray]], frame: np.ndarray,
                metas: Sequence[Any], tiled: bool) -> PostprocessResult:
        from conecsa_common.polygons import mask_rings

        img = frame.copy()
        frame_h, frame_w = img.shape[:2]
        instances: List[_Instance] = []
        for tile, (outputs, meta) in enumerate(zip(tile_outputs, metas, strict=True)):
            instances.extend(self._decode_tile(outputs, meta, tile))
        if tiled and len(metas) > 1:
            instances = self._merge(instances, metas)
        instances = self._suppress(instances)

        names = self.detector.class_labels
        detections = []
        by_detection = {}
        for inst in instances:
            x1, y1, x2, y2 = (int(v) for v in inst.box)
            det = Detection(
                class_id=inst.class_id,
                class_name=names[inst.class_id] if inst.class_id < len(names)
                else f"Class-{inst.class_id}",
                confidence=inst.score,
                bbox=(x1, y1, x2, y2),
                center=((x1 + x2) // 2, (y1 + y2) // 2),
                color=bgr_to_hex(self.detector._get_class_color(inst.class_id)),
            )
            by_detection[id(det)] = inst
            detections.append(det)
        detections = self.detector._filter_detections_by_areas(detections, frame_w, frame_h)

        kept = [(det, by_detection[id(det)]) for det in detections]
        for det, inst in kept:
            # Extracted when a consumer asks (Detection.resolve_polygons).
            det.ring_source = functools.partial(
                mask_rings, inst.mask, frame_w, frame_h, offset=(inst.x0, inst.y0),
                min_area_frac=self.settings.min_component_area)
        self._fill_masks(img, kept)
        for det, _ in kept:
            x1, y1, x2, y2 = det.bbox  # type: ignore[misc]
            self.detector._draw_detection_box(img, x1, y1, x2, y2, det.class_id, det.confidence)
        return PostprocessResult(self.detector._draw_areas(img), len(detections), detections)
