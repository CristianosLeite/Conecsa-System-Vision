# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Face recognition postprocess: YuNet faces named from the model's gallery.

The live engine is the bundled YuNet detector; this strategy decodes its
twelve outputs (bound by name, see ``bind_outputs``), keeps faces whose score
is strictly above ``CONFIDENCE_THRESHOLD`` and whose sides reach
``FACE_MIN_SIZE_PX``, suppresses overlaps at ``OVERLAY_THRESHOLD`` IoU,
applies the saved detection areas like detection does, and embeds at most
``FACE_MAX_FACES`` of them, largest first, on the SFace worker. A face takes
the name of the enrolled person whose best photo is more similar than
``FACE_MATCH_THRESHOLD``; otherwise it is ``unknown``. ``confidence`` is that
similarity.

The accumulated counter counts *arrivals* of known people: a person adds one
when they are in this frame but were not in the previous one; ``unknown``
never counts. The processed frame gets boxes and names on a copy.
"""
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from conecsa_common.tasks import UNKNOWN_FACE

from ..models.detection_models import Detection
from ..utils import bgr_to_hex, generate_colors, resolve_class_colors
from ..views.area_overlay import draw_areas
from ..views.detection_boxes import nms_indices
from . import _face_assets, _face_embedder, _face_gallery, _yunet
from ._face_align import align_crop
from .base import PostprocessResult
from .contract import ContractError

logger = logging.getLogger(__name__)

#: The class name of a face no enrolled person matches (reserved as a person name).
UNKNOWN = UNKNOWN_FACE
UNKNOWN_CLASS_ID = -1
_UNKNOWN_BGR = (128, 128, 128)


def gallery_file_for_model(model_path: str) -> str:
    """``<model>.gallery.npz`` beside the model file."""
    base, _ = os.path.splitext(model_path)
    return f"{base}.gallery.npz"


class FacePostprocessor:
    """Face recognition strategy: decode, align, embed, match, arrival counter."""

    task = "face"

    def __init__(self, class_labels: List[str], config: Any, embedder: Any = None):
        self.config = config
        self._areas: List[Any] = []
        self._indices: Optional[Dict[str, int]] = None
        self._present: set = set()
        try:
            self._gallery = _face_gallery.load(gallery_file_for_model(config.MODEL_PATH))
        except _face_gallery.GalleryError as exc:
            raise ContractError(str(exc)) from None
        expected = _face_assets.embedder_sha256()
        if expected and self._gallery.embedder_sha256 != expected:
            raise ContractError(
                "the face gallery was built with another face embedder; rebuild it "
                "from its dataset")
        stamp = self._gallery.labels_sha256
        if stamp and stamp != _face_gallery.labels_stamp(class_labels):
            # An interrupted publication (or a sidecar of another model under
            # the same name): these embeddings would take the wrong names.
            raise ContractError(
                "the model's class names are out of step with its face gallery; save "
                "its class names again or rebuild it from its dataset")
        # Names and colours travel as one tuple: the frame thread reads it once
        # per frame, so a live rename can never show it a half-built list.
        self._people: Tuple[List[str], List[Any]] = self._fit_classes(class_labels)
        self._embedder = embedder if embedder is not None else _face_embedder.create_embedder(config)

    @property
    def class_labels(self) -> List[str]:
        return list(self._people[0])

    def _fit_classes(self, class_labels: List[str]) -> Tuple[List[str], List[Any]]:
        """``(names, colors)`` from the sidecar entries, padded from the gallery
        when it has more people than the sidecar lists."""
        names, colors = resolve_class_colors(class_labels)
        num_classes = len(self._gallery.names)
        palette = generate_colors(max(num_classes, 1))
        while len(names) < num_classes:
            index = len(names)
            names.append(self._gallery.names[index])
            colors.append(palette[index])
        return names, colors

    def bind_outputs(self, output_details: Sequence[Dict[str, Any]]) -> None:
        """Locate the twelve YuNet outputs by name (``ContractError`` when one is missing)."""
        try:
            self._indices = _yunet.output_indices(output_details)
        except KeyError as exc:
            raise ContractError(f"the face detector has no '{exc.args[0]}' output") from None

    def set_class_labels(self, class_labels: List[str]) -> None:
        """Adopt renamed people live: the gallery keeps the indices, the
        sidecar the names, so a rename never needs a rebuild. Built aside
        and swapped in one assignment (see ``_people``)."""
        self._people = self._fit_classes(class_labels)

    @property
    def embedder(self) -> Any:
        """The private SFace worker: one per device, handed from one face
        model to the next at activation rather than opened twice on one port."""
        return self._embedder

    def close(self) -> None:
        """Free the private embedder worker (a strategy that never went live)."""
        close = getattr(self._embedder, "close", None)
        if close is not None:
            close()

    def set_areas(self, areas: Sequence[Any]) -> None:
        self._areas = list(areas) if areas else []

    def reset_state(self) -> None:
        """Nobody is present: the next known face counts as an arrival."""
        self._present = set()

    # ── per frame ──

    def _faces(self, outputs: Sequence[np.ndarray], meta: Any, frame: np.ndarray):
        if self._indices is None:
            raise ContractError("the face detector outputs were not bound")
        size = int(getattr(meta, "input_size", 0) or 640)
        faces = _yunet.to_frame(
            _yunet.decode(outputs, self._indices, size, float(self.config.CONFIDENCE_THRESHOLD)),
            meta)
        if len(faces.scores) == 0:
            return faces
        height, width = frame.shape[:2]
        boxes = faces.boxes.copy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height - 1)
        sides = np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
        keep = np.flatnonzero(sides >= int(self.config.FACE_MIN_SIZE_PX))
        if keep.size:
            kept = nms_indices(boxes[keep].astype(np.int32), faces.scores[keep],
                               float(self.config.OVERLAY_THRESHOLD))
            keep = keep[kept]
        return _yunet.Faces(boxes[keep], faces.scores[keep], faces.landmarks[keep])

    def _in_areas(self, box: np.ndarray, width: int, height: int):
        """``(kept, area summary)`` for a face center under the saved areas."""
        saved = [a for a in self._areas if not getattr(a, "is_editing", False)]
        if not saved:
            return True, None
        # Lazy: yolo_detector is the detection strategy's module.
        from ..yolo_detector import YOLODetector

        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        area = YOLODetector._first_area_containing(cx, cy, saved, width, height)
        if area is None:
            return False, None
        return True, YOLODetector._area_summary(area)

    def process(self, tile_outputs: Sequence[Sequence[np.ndarray]], frame: np.ndarray,
                metas: Sequence[Any], tiled: bool) -> PostprocessResult:
        # Faces are found on the whole frame: one entry, never tiled.
        meta = metas[0] if metas else None
        faces = self._faces(tile_outputs[0], meta, frame)
        height, width = frame.shape[:2]

        chosen = []
        for i in np.argsort(-(faces.boxes[:, 2] - faces.boxes[:, 0])
                            * (faces.boxes[:, 3] - faces.boxes[:, 1]), kind="stable"):
            kept, area = self._in_areas(faces.boxes[i], width, height)
            if kept:
                chosen.append((int(i), area))
            if len(chosen) >= int(self.config.FACE_MAX_FACES):
                break

        items: List[Detection] = []
        names, colors = self._people
        if chosen:
            crops = [align_crop(frame, faces.landmarks[i]) for i, _ in chosen]
            classes, sims = _face_gallery.match(self._gallery, self._embedder.embed(crops))
            threshold = float(self.config.FACE_MATCH_THRESHOLD)
            for (i, area), c, sim in zip(chosen, classes, sims, strict=True):
                x1, y1, x2, y2 = (int(v) for v in faces.boxes[i])
                known = float(sim) > threshold
                items.append(Detection(
                    class_id=int(c) if known else UNKNOWN_CLASS_ID,
                    class_name=names[int(c)] if known else UNKNOWN,
                    confidence=round(float(sim), 4),
                    bbox=(x1, y1, x2, y2),
                    center=((x1 + x2) // 2, (y1 + y2) // 2),
                    color=bgr_to_hex(colors[int(c)] if known else _UNKNOWN_BGR),
                    area=area,
                ))

        # Keyed by the gallery slot: a live rename must not read as an arrival.
        present = {d.class_id for d in items if d.class_id != UNKNOWN_CLASS_ID}
        increment = len(present - self._present)
        self._present = present
        return PostprocessResult(self._draw(frame, items, colors), len(items), items,
                                 count_increment=increment)

    def _draw(self, frame: np.ndarray, items: List[Detection],
              colors: Sequence[Any]) -> np.ndarray:
        img = frame.copy()
        for d in items:
            assert d.bbox is not None
            x1, y1, x2, y2 = d.bbox
            color = colors[d.class_id] if d.class_id != UNKNOWN_CLASS_ID else _UNKNOWN_BGR
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            label = f"{d.class_name} {d.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - th - 15), (x1 + tw + 10, y1), color, -1)
            cv2.putText(img, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
        return draw_areas(img, self._areas)
