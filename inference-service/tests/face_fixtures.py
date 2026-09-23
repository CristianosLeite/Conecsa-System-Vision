# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Synthetic YuNet outputs, galleries and a fake embedder for the face tests.

No real faces: engine outputs are built from the decode formulas, so a face
is placed exactly where a test wants it.
"""
import math
from typing import List, Sequence, Tuple

import numpy as np
from api.postprocess import _face_gallery, _yunet

SIZE = 640


def yunet_details(order: Sequence[str] = _yunet.OUTPUT_NAMES) -> List[dict]:
    """Engine output details (name + shape) in ``order``."""
    details = []
    for name in order:
        kind, stride = name.split("_")
        rows = (SIZE // int(stride)) ** 2
        details.append({"name": name, "shape": [1, rows, _yunet.KIND_WIDTH[kind]]})
    return details


def face_landmarks(box: Tuple[float, float, float, float]) -> np.ndarray:
    """Five plausible landmarks inside ``box`` (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    rel = [(0.3, 0.4), (0.7, 0.4), (0.5, 0.6), (0.35, 0.8), (0.65, 0.8)]
    return np.array([[x1 + rx * w, y1 + ry * h] for rx, ry in rel], np.float32)


def yunet_outputs(faces: Sequence[Tuple[Tuple[float, float, float, float], float]],
                  order: Sequence[str] = _yunet.OUTPUT_NAMES, stride: int = 32) -> List[np.ndarray]:
    """The twelve outputs, in ``order``, for ``(box, score)`` faces in input pixels.

    Each face sits on the stride-32 anchor cell under its center (tests keep
    faces in distinct cells).
    """
    arrays = {d["name"]: np.zeros(d["shape"], np.float32) for d in yunet_details()}
    cols = SIZE // stride
    for (x1, y1, x2, y2), score in faces:
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        col, row = int(cx // stride), int(cy // stride)
        idx = row * cols + col
        arrays[f"cls_{stride}"][0, idx, 0] = score
        arrays[f"obj_{stride}"][0, idx, 0] = score
        arrays[f"bbox_{stride}"][0, idx] = [cx / stride - col, cy / stride - row,
                                            math.log(w / stride), math.log(h / stride)]
        marks = face_landmarks((x1, y1, x2, y2)) / stride - np.array([col, row], np.float32)
        arrays[f"kps_{stride}"][0, idx] = marks.reshape(-1)
    return [arrays[name] for name in order]


def unit(*values: float) -> np.ndarray:
    v = np.asarray(values, np.float32)
    return v / np.linalg.norm(v)


def write_gallery(path: str, names: Sequence[str], vectors: Sequence[np.ndarray],
                  labels: Sequence[int], sha: str = "sha-test") -> None:
    _face_gallery.save(path, _face_gallery.Gallery(
        embeddings=np.stack(vectors), labels=np.asarray(labels, np.int32), names=list(names),
        image_ids=[f"img{i}" for i in range(len(vectors))], embedder_sha256=sha))


class FakeEmbedder:
    """Returns queued embeddings in call order and records the crops."""

    def __init__(self, vectors: Sequence[np.ndarray] = ()):
        self.queue = [np.asarray(v, np.float32) for v in vectors]
        self.calls: List[int] = []

    def embed(self, crops):
        self.calls.append(len(crops))
        rows = [self.queue.pop(0) for _ in crops]
        return np.stack(rows) if rows else np.zeros((0, 2), np.float32)
