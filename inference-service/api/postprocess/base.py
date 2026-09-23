# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""The interface every task's postprocess strategy implements."""
from dataclasses import dataclass
from typing import Any, List, Optional, Protocol, Sequence

import numpy as np

from ..models.detection_models import Detection


@dataclass
class PostprocessResult:
    """One decoded frame.

    ``count`` is the number of results in the frame (detections, or 1/0 for
    whether a classification frame has a class); it feeds
    ``frames_with_detections`` and the snapshot's ``total``.
    ``count_increment`` is what the frame adds to the accumulated counter
    when that differs from ``count`` (classification counts class
    transitions); ``None`` means ``count``. ``candidates`` is the
    classification top-k (``[{class_id, class_name, confidence}]``), ``None``
    for tasks without one.
    """

    image: np.ndarray
    count: int
    items: List[Detection]
    count_increment: Optional[int] = None
    candidates: Optional[List[dict]] = None


class Postprocessor(Protocol):
    """Turns one frame's engine outputs into an annotated frame and results.

    ``DetectionService`` picks the strategy from the model's task when it
    initializes; the pipeline stages only ever call this interface.
    """

    #: The task id this strategy serves ("detect", "classify", "segment").
    task: str

    @property
    def class_labels(self) -> List[str]:
        """The model's class names, by index."""
        ...

    def set_areas(self, areas: Sequence[Any]) -> None:
        """Detection areas for the next frame (a no-op for whole-frame tasks)."""
        ...

    def set_class_labels(self, class_labels: List[str]) -> None:
        """Adopt renamed labels without reloading the engine.

        ``SetClasses`` rewrites the model's sibling ``.txt`` while the stream
        runs. The strategy caches the names (and the colors parsed out of
        them) when it is built, so without this the burned-in overlay and the
        snapshot would keep the old names until the next model load — most
        visible on ``face``, where the name *is* the result.
        """
        ...

    def process(self, tile_outputs: Sequence[Sequence[np.ndarray]], frame: np.ndarray,
                metas: Sequence[Any], tiled: bool) -> PostprocessResult:
        """Decode one frame.

        ``tile_outputs`` holds every engine output for each tile (a single
        entry when tiling is off), ``metas`` the matching ``TileMeta``
        geometry. Never draw on ``frame``: it stays the clean frame the
        snapshot and the offline buffer keep, so an overlay goes on a copy. A
        strategy that draws nothing (classification) returns ``frame``
        itself, so the result's image may alias the clean frame; every
        consumer (encode stage, snapshot, buffer) only reads it.
        """
        ...

    def reset_state(self) -> None:
        """Forget per-stream state (stats reset, start/stop, model change)."""
        ...
