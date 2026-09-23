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
from typing import Any, List, Optional, Sequence

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

    def _fit_classes(self, num_classes: int) -> None:
        """Pad names/colors when the engine has more classes than its sidecar."""
        if len(self._names) >= num_classes:
            return
        palette = generate_colors(num_classes)
        while len(self._names) < num_classes:
            index = len(self._names)
            self._names.append(f"Class-{index}")
            self._colors.append(palette[index])

    def process(self, tile_outputs: Sequence[Sequence[np.ndarray]], frame: np.ndarray,
                metas: Sequence[Any], tiled: bool) -> PostprocessResult:
        # Classification is never tiled: one entry, the whole frame.
        probs = np.asarray(probabilities_output(tile_outputs[0]), dtype=np.float32).reshape(-1)
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
        increment = 1 if current is not None and current != self._current else 0
        self._current = current

        # Nothing is drawn either way: the clean frame is the processed frame.
        if current is None:
            return PostprocessResult(frame, 0, [], count_increment=0, candidates=candidates)

        item = Detection(class_id=current, class_name=self._names[current],
                         confidence=top_prob, color=bgr_to_hex(self._colors[current]))
        return PostprocessResult(frame, 1, [item], count_increment=increment,
                                 candidates=candidates)
