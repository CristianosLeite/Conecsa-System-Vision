# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""YuNet face detector output decode (pure numpy).

The bundled ``face_detection_yunet_2023mar`` graph has twelve outputs, one
``cls``/``obj``/``bbox``/``kps`` quadruple per stride (8, 16, 32), each
``[1, (S / stride)², k]`` for a square ``S`` input. The decode transcribes
OpenCV's ``FaceDetectorYN::postProcess``: the score is
``sqrt(clip(cls) · clip(obj))``, a box is a center offset plus an
exponential size in stride units, and each of the five landmarks (right eye,
left eye, nose tip, right and left mouth corner) is an offset from its anchor
cell. Outputs are matched by name, never by binding index: ``cls_*`` and
``obj_*`` have the same shape.
"""
from typing import Any, Dict, List, NamedTuple, Optional, Sequence

import numpy as np

STRIDES = (8, 16, 32)
KINDS = ("cls", "obj", "bbox", "kps")
#: Last dimension of each output kind.
KIND_WIDTH = {"cls": 1, "obj": 1, "bbox": 4, "kps": 10}
#: Every output name the graph must expose.
OUTPUT_NAMES = tuple(f"{kind}_{stride}" for kind in KINDS for stride in STRIDES)


class Faces(NamedTuple):
    """Decoded faces in model-input pixels.

    ``boxes`` is ``[N, 4]`` ``x1, y1, x2, y2``; ``scores`` ``[N]``;
    ``landmarks`` ``[N, 5, 2]``.
    """

    boxes: np.ndarray
    scores: np.ndarray
    landmarks: np.ndarray


def empty_faces() -> Faces:
    return Faces(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                 np.zeros((0, 5, 2), np.float32))


def output_indices(output_details: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Position of each YuNet output among an engine's outputs, by name.

    Raises ``KeyError`` naming the first missing output.
    """
    by_name = {str(detail.get("name")): i for i, detail in enumerate(output_details)}
    missing = [name for name in OUTPUT_NAMES if name not in by_name]
    if missing:
        raise KeyError(missing[0])
    return {name: by_name[name] for name in OUTPUT_NAMES}


def decode(outputs: Sequence[np.ndarray], indices: Dict[str, int], input_size: int,
           score_threshold: float) -> Faces:
    """Every anchor whose score is strictly above ``score_threshold``, unsuppressed."""
    boxes: List[np.ndarray] = []
    scores: List[np.ndarray] = []
    landmarks: List[np.ndarray] = []
    for stride in STRIDES:
        cols = input_size // stride
        cls = np.asarray(outputs[indices[f"cls_{stride}"]], np.float32).reshape(-1)
        obj = np.asarray(outputs[indices[f"obj_{stride}"]], np.float32).reshape(-1)
        score = np.sqrt(np.clip(cls, 0.0, 1.0) * np.clip(obj, 0.0, 1.0))
        keep = np.flatnonzero(score > score_threshold)
        if keep.size == 0:
            continue
        bbox = np.asarray(outputs[indices[f"bbox_{stride}"]], np.float32).reshape(-1, 4)[keep]
        kps = np.asarray(outputs[indices[f"kps_{stride}"]], np.float32).reshape(-1, 5, 2)[keep]
        col = (keep % cols).astype(np.float32)
        row = (keep // cols).astype(np.float32)
        cx = (col + bbox[:, 0]) * stride
        cy = (row + bbox[:, 1]) * stride
        w = np.exp(bbox[:, 2]) * stride
        h = np.exp(bbox[:, 3]) * stride
        boxes.append(np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1))
        scores.append(score[keep])
        anchor = np.stack([col, row], axis=1)[:, None, :]
        landmarks.append((kps + anchor) * stride)
    if not boxes:
        return empty_faces()
    return Faces(np.concatenate(boxes).astype(np.float32),
                 np.concatenate(scores).astype(np.float32),
                 np.concatenate(landmarks).astype(np.float32))


def to_frame(faces: Faces, meta: Optional[Any]) -> Faces:
    """Map faces from model-input pixels to frame pixels through a ``TileMeta``.

    The inverse of ``ModelManager._preprocess_face``: the pad bands come off
    both axes (``border_left`` is 0 for the Y-only letterbox the other tasks
    use) and what remains scales by ``meta.scale``, the one factor from
    resized pixels back to original ones. ``meta.ox`` / ``meta.oy`` then shift
    a tile into frame space. ``None`` means the input already is the frame.
    """
    if meta is None or len(faces.scores) == 0:
        return faces
    scale = float(meta.scale) or 1.0
    band = np.array([float(getattr(meta, "border_left", 0)), float(meta.border_top)], np.float32)
    origin = np.array([float(meta.ox), float(meta.oy)], np.float32)

    def mapped(points: np.ndarray) -> np.ndarray:
        out = points.reshape(-1, 2).astype(np.float32)
        return (out - band) * scale + origin

    return Faces(mapped(faces.boxes).reshape(-1, 4), faces.scores,
                 mapped(faces.landmarks).reshape(-1, 5, 2))
