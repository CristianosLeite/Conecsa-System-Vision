# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Five-point face alignment to the 112×112 recognition crop.

The destination landmarks are the ArcFace template OpenCV's
``FaceRecognizerSF::alignCrop`` uses; the similarity transform (rotation,
uniform scale, translation) is Umeyama's least-squares estimate, the same
closed form as OpenCV's ``getSimilarityTransformMatrix``.
"""
import cv2
import numpy as np

CROP_SIZE = 112

#: Right eye, left eye, nose tip, right and left mouth corner in the crop.
TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], np.float64)


def similarity_matrix(src: np.ndarray, dst: np.ndarray = TEMPLATE) -> np.ndarray:
    """The ``2×3`` similarity transform mapping ``src`` points onto ``dst`` (Umeyama)."""
    src = np.asarray(src, np.float64).reshape(-1, 2)
    dst = np.asarray(dst, np.float64).reshape(-1, 2)
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - src_mean, dst - dst_mean
    cov = dst_c.T @ src_c / len(src)
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(cov) < 0:
        d[1] = -1.0
    rotation = u @ np.diag(d) @ vt
    variance = (src_c ** 2).sum() / len(src)
    scale = float((s * d).sum() / variance) if variance > 0 else 1.0
    matrix = np.zeros((2, 3), np.float64)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = dst_mean - scale * rotation @ src_mean
    return matrix


def align_crop(frame_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """The aligned ``112×112`` BGR face for five frame-space landmarks."""
    matrix = similarity_matrix(landmarks)
    return cv2.warpAffine(frame_bgr, matrix, (CROP_SIZE, CROP_SIZE), flags=cv2.INTER_LINEAR)


def embedder_input(crops_bgr) -> np.ndarray:
    """Aligned BGR crops → the SFace input batch.

    ``blobFromImage(crop, 1, (112, 112), 0, swapRB=True)``: RGB, float32
    0..255, NCHW.
    """
    batch = np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops_bgr])
    return np.ascontiguousarray(batch.transpose(0, 3, 1, 2), dtype=np.float32)
