# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Object detection postprocess, backed by ``YOLODetector``."""
from typing import Any, List, Optional, Sequence

import numpy as np

from ..yolo_detector import YOLODetector
from .base import PostprocessResult


def detection_output(outputs: Sequence[np.ndarray]) -> np.ndarray:
    """The detection head among an engine's outputs (picked by rank, not index)."""
    for output in outputs:
        if getattr(output, "ndim", 0) == 3:
            return output
    return outputs[0]


class DetectPostprocessor:
    """Detection strategy: decode, cross-tile merge, NMS, area filter, overlay."""

    task = "detect"

    def __init__(self, class_labels: List[str], config: Any,
                 detector: Optional[YOLODetector] = None):
        self.detector = detector if detector is not None else YOLODetector(class_labels, config)

    @property
    def class_labels(self) -> List[str]:
        return list(self.detector.class_labels)

    def set_class_labels(self, class_labels: List[str]) -> None:
        """Adopt renamed labels live (see the protocol)."""
        self.detector.set_class_labels(class_labels)

    def set_areas(self, areas: Sequence[Any]) -> None:
        self.detector.set_areas(areas)

    def process(self, tile_outputs: Sequence[Sequence[np.ndarray]], frame: np.ndarray,
                metas: Sequence[Any], tiled: bool) -> PostprocessResult:
        outputs = [detection_output(o) for o in tile_outputs]
        if tiled:
            image, count, items = self.detector.process_tiled_detections(outputs, frame, metas)
        else:
            meta = metas[0]
            image, count, items = self.detector.process_detections(
                outputs[0], frame, scale=meta.scale, border_top=meta.border_top,
                actual_input_size=meta.input_size,
            )
        return PostprocessResult(image, count, items)

    def reset_state(self) -> None:
        """Detection keeps no per-stream state."""
