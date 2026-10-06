# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Whole-frame image classification postprocess.

The engine outputs one probability row ``[1, nc]`` (softmax is part of the
exported graph; activation checks it). A frame "has a class" when its top-1
probability is strictly above ``CONFIDENCE_THRESHOLD`` — the same strict gate
as detection's ``_confidence_mask`` — otherwise its class is ``none``.
``OVERLAY_THRESHOLD`` does not apply: there is no NMS.

The accumulated counter counts class *transitions*, not frames: it grows by
one when the thresholded class changes to a class (``none→A``, ``A→B``,
``A→none→A``), never on ``A→A`` or ``A→none``. The state lives here and is
reset by ``reset_state()`` (stats reset, start, stop); a model or application
change builds a new strategy.

Nothing is drawn on the processed frame: the device UI's
classification panel names the class, so the stream stays the clean frame.
"""
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..models.detection_models import Detection
from ..utils import bgr_to_hex, generate_colors, resolve_class_colors
from .base import PostprocessResult

logger = logging.getLogger(__name__)

DEFAULT_TOPK = 5


def classify_topk_from_env() -> int:
    """How many candidates a frame reports, from ``CLASSIFY_TOPK`` (default 5).

    Clamped to the model's class count per frame; non-numeric or
    non-positive values fall back to the default.
    """
    raw = os.environ.get("CLASSIFY_TOPK", str(DEFAULT_TOPK)).strip()
    try:
        value = int(raw)
    except ValueError:
        logger.warning("CLASSIFY_TOPK=%r is not an int; using %d", raw, DEFAULT_TOPK)
        return DEFAULT_TOPK
    if value < 1:
        logger.warning("CLASSIFY_TOPK=%d must be at least 1; using %d", value, DEFAULT_TOPK)
        return DEFAULT_TOPK
    return value


def probabilities_output(outputs: Sequence[np.ndarray]) -> np.ndarray:
    """The classification head among an engine's outputs (picked by rank, not index)."""
    for output in outputs:
        if getattr(output, "ndim", 0) == 2:
            return output
    return outputs[0]


class ClassifyPostprocessor:
    """Classification strategy: top-k, strict threshold, transition counter."""

    task = "classify"

    def __init__(self, class_labels: List[str], config: Any):
        self.config = config
        self._names, self._colors = resolve_class_colors(class_labels)
        self.topk = classify_topk_from_env()
        # The current thresholded class index; None is the "none" class.
        self._current: Optional[int] = None
        # The same per input origin when several inputs (IP cameras) are
        # classified; each camera counts its own transitions.
        self._currents: Dict[Tuple[int, int], Optional[int]] = {}

    @property
    def class_labels(self) -> List[str]:
        return list(self._names)

    def set_class_labels(self, class_labels: List[str]) -> None:
        """Adopt renamed labels live (see the protocol)."""
        self._names, self._colors = resolve_class_colors(class_labels)

    def set_areas(self, areas: Sequence[Any]) -> None:
        """Whole-frame classification: detection areas do not apply."""

    def reset_state(self) -> None:
        """Back to the ``none`` class: the next class counts as a transition."""
        self._current = None
        self._currents = {}

    def _fit_classes(self, num_classes: int) -> None:
        """Pad names/colors when the engine has more classes than its sidecar."""
        if len(self._names) >= num_classes:
            return
        palette = generate_colors(num_classes)
        while len(self._names) < num_classes:
            index = len(self._names)
            self._names.append(f"Class-{index}")
            self._colors.append(palette[index])

    def _classify(self, outputs: Sequence[np.ndarray], previous: Optional[int]):
        """``(candidates, current class or None, its probability, increment)`` of one input."""
        probs = np.asarray(probabilities_output(outputs), dtype=np.float32).reshape(-1)
        self._fit_classes(int(probs.shape[0]))

        # A stable sort of the negated scores keeps the lower index first on ties.
        order = np.argsort(-probs, kind="stable")[:min(self.topk, int(probs.shape[0]))]
        candidates = [
            {"class_id": int(i), "class_name": self._names[int(i)],
             "confidence": round(float(probs[i]), 4)}
            for i in order
        ]
        top = int(order[0])
        top_prob = float(probs[top])
        current = top if top_prob > float(self.config.CONFIDENCE_THRESHOLD) else None
        increment = 1 if current is not None and current != previous else 0
        return candidates, current, top_prob, increment

    def _item(self, current: int, prob: float, origin=None) -> Detection:
        return Detection(class_id=current, class_name=self._names[current], confidence=prob,
                         color=bgr_to_hex(self._colors[current]), origin=origin)

    def process(self, tile_outputs: Sequence[Sequence[np.ndarray]], frame: np.ndarray,
                metas: Sequence[Any], tiled: bool) -> PostprocessResult:
        # Classification is never tiled: one entry per input — the whole
        # frame, or each camera of an IP camera mosaic.
        # Nothing is drawn either way: the clean frame is the processed frame.
        origin = (int(getattr(metas[0], "ox", 0)), int(getattr(metas[0], "oy", 0))) \
            if metas else (0, 0)
        # Per camera as soon as the inputs are cameras of a mosaic: several of
        # them, one away from the origin, or one left of a set seen before.
        if len(tile_outputs) != 1 or origin != (0, 0) or self._currents:
            return self._process_regions(tile_outputs, frame, metas)
        candidates, current, top_prob, increment = self._classify(tile_outputs[0], self._current)
        self._current = current
        # Also by input, so a one-camera mosaic reads like a larger one; a
        # single camera's snapshot only uses ``candidates``.
        per_region = [{"origin": origin, "candidates": candidates}]
        if current is None:
            return PostprocessResult(frame, 0, [], count_increment=0, candidates=candidates,
                                     region_candidates=per_region)
        return PostprocessResult(frame, 1, [self._item(current, top_prob, origin)],
                                 count_increment=increment, candidates=candidates,
                                 region_candidates=per_region)

    def _process_regions(self, tile_outputs: Sequence[Sequence[np.ndarray]],
                         frame: np.ndarray, metas: Sequence[Any]) -> PostprocessResult:
        """One classification per IP camera; transitions counted per camera."""
        items: List[Detection] = []
        per_region = []
        increment = 0
        currents: Dict[Tuple[int, int], Optional[int]] = {}
        for outputs, meta in zip(tile_outputs, metas, strict=True):
            origin = (int(meta.ox), int(meta.oy))
            # The first time several inputs show up, the camera at the origin
            # carries on from the single-input state.
            previous = self._currents.get(origin) if self._currents else (
                self._current if origin == (0, 0) else None)
            candidates, current, prob, step = self._classify(outputs, previous)
            currents[origin] = current
            increment += step
            per_region.append({"origin": origin, "candidates": candidates})
            if current is not None:
                items.append(self._item(current, prob, origin))
        # A camera that went offline forgets its class: back online, its first
        # class is an arrival again.
        self._currents = currents
        return PostprocessResult(frame, len(items), items, count_increment=increment,
                                 region_candidates=per_region)
