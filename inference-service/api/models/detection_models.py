# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Data models for object detection system.
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

# noinspection PyPackageRequirements
import numpy as np  # ships in conecsa-os-base:base


@dataclass
class Detection:
    """One result: a detected object, or the class of a whole frame.

    A classification result has no geometry: ``bbox`` and ``center`` stay
    ``None`` and the snapshot item carries no ``bbox``.
    """
    class_id: int
    class_name: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    center: Optional[Tuple[int, int]] = None  # (center_x, center_y)
    # Box color as "#rrggbb", resolved from the class entry's "name #hex"
    # suffix or, absent one, from the generated palette. Travels with the
    # detection so client-side overlays match the burned-in one.
    color: Optional[str] = None
    # Saved detection area whose shape contains this detection's center, as
    # {"id", "label", "shape"}; None when no saved area contains it. Assigned
    # in YOLODetector._filter_detections_by_areas. A center may fall inside
    # several overlapping areas — the first match (by area order) wins.
    area: Optional[dict] = None
    # Segmentation instance outline: exterior rings as [[x, y], …] normalized
    # 0..1 on the frame; None for tasks without masks, [] for
    # a segmentation instance whose mask left no ring. Read it through
    # resolve_polygons(): segmentation fills it on first use.
    polygons: Optional[List[List[List[float]]]] = None
    # Extracts ``polygons`` from the instance mask the segmentation postprocess
    # kept. Ring extraction costs as much as the mask decode, and only
    # the snapshot, an offline record and the labeling assistant read rings,
    # so it runs on demand rather than on every frame.
    ring_source: Optional[Callable[[], List[List[List[float]]]]] = field(
        default=None, repr=False, compare=False)

    def resolve_polygons(self) -> Optional[List[List[List[float]]]]:
        """The outline rings, extracted from ``ring_source`` on first use.

        Safe from two threads (the snapshot RPC and the pipeline's offline
        buffer): both would extract the same rings from the same mask.
        """
        source = self.ring_source
        if self.polygons is None and source is not None:
            self.polygons = source()
        return self.polygons


@dataclass
class DetectionResult:
    """Result of a detection operation."""
    detections: List[Detection]
    processed_image: np.ndarray
    inference_time: float
    num_detections: int
    # Pristine frame (no overlay), held by reference — safe because every
    # pipeline frame is a freshly allocated buffer and the detector draws on a
    # copy. Classification draws nothing, so there processed_image IS
    # this array; both are read-only downstream. Must become a copy if the
    # pipeline ever reuses frame buffers.
    raw_image: Optional[np.ndarray] = None
    # What the frame adds to the accumulated counter when that is not
    # num_detections (classification counts class transitions); None means
    # num_detections.
    count_increment: Optional[int] = None
    # Classification top-k ([{class_id, class_name, confidence}]); None for
    # tasks without one, and then absent from the snapshot.
    candidates: Optional[List[dict]] = None


@dataclass
class SystemStats:
    """System performance statistics."""
    fps: float = 0.0
    inference_time: float = 0.0
    detections: int = 0
    frames_with_detections: int = 0
    # Pipeline service times in ms (benchmark protocol; see StageTimer).
    finish_mean_ms: float = 0.0
    finish_p95_ms: float = 0.0
    finish_p99_ms: float = 0.0
    encode_mean_ms: float = 0.0
    encode_p95_ms: float = 0.0
    frame_age_p95_ms: float = 0.0


@dataclass
class ModelInfo:
    """Information about a model file."""
    name: str
    path: str
    size: int
    modified: float
    is_active: bool = False
    # A training checkpoint sidecar (the .pt the engine was built from) exists.
    has_weights: bool = False
    # The model's task from its settings sidecar ("detect" when unrecorded).
    task: str = "detect"

