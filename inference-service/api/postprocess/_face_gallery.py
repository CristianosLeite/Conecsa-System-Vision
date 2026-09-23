# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""The enrolled-faces gallery of a ``face`` model.

``<model>.gallery.npz`` holds one L2-normalized embedding per enrolled photo,
the class (person) of each, the class names and the embedder it was built
with. Recognition takes, per person, the best cosine similarity over that
person's photos; the best person wins when it beats ``FACE_MATCH_THRESHOLD``.
The file stays on the device: it is never downloadable, federated or sent to
the hub.
"""
import hashlib
import io
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from conecsa_common import atomic_write_bytes

FORMAT = 1


class GalleryError(ValueError):
    """The gallery file is missing, unreadable or inconsistent."""


@dataclass
class Gallery:
    embeddings: np.ndarray      # [M, D] float32, L2-normalized
    labels: np.ndarray          # [M] int32 class index per embedding
    names: List[str]            # class names by index
    image_ids: List[str]        # dataset image id per embedding
    embedder_sha256: str
    #: ``labels_stamp`` of the model's ``.txt`` entries this gallery was
    #: published or last renamed with; "" when unknown (test fixtures). The
    #: strategy refuses a mismatch: a build interrupted between the two files
    #: would otherwise pair these embeddings with another model's names.
    labels_sha256: str = ""

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])


def labels_stamp(entries: Sequence[str]) -> str:
    """The stamp of a class-label list as the ``.txt`` sidecar carries it."""
    return hashlib.sha256("\n".join(e.strip() for e in entries).encode("utf-8")).hexdigest()


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Rows scaled to unit length (a zero row stays zero)."""
    vectors = np.asarray(vectors, np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def save(path: str, gallery: Gallery) -> None:
    """Write the gallery atomically."""
    buffer = io.BytesIO()
    np.savez(buffer,
             format=np.int32(FORMAT),
             embeddings=normalize(gallery.embeddings),
             labels=np.asarray(gallery.labels, np.int32),
             names=np.asarray(gallery.names, dtype=np.str_),
             image_ids=np.asarray(gallery.image_ids, dtype=np.str_),
             embedder_sha256=np.asarray(gallery.embedder_sha256),
             labels_sha256=np.asarray(gallery.labels_sha256))
    atomic_write_bytes(path, buffer.getvalue(), mode=0o600)


def load(path: str) -> Gallery:
    """Read and validate a gallery; :class:`GalleryError` otherwise."""
    try:
        with np.load(path, allow_pickle=False) as data:
            if int(data["format"]) != FORMAT:
                raise GalleryError(f"unsupported gallery format {int(data['format'])}")
            gallery = Gallery(
                embeddings=np.asarray(data["embeddings"], np.float32),
                labels=np.asarray(data["labels"], np.int32),
                names=[str(n) for n in data["names"]],
                image_ids=[str(i) for i in data["image_ids"]],
                embedder_sha256=str(data["embedder_sha256"]),
                labels_sha256=str(data["labels_sha256"]) if "labels_sha256" in data else "",
            )
    except FileNotFoundError:
        raise GalleryError("the model has no face gallery; rebuild it from its dataset") from None
    except (OSError, KeyError, ValueError) as exc:
        if isinstance(exc, GalleryError):
            raise
        raise GalleryError(f"the face gallery cannot be read: {exc}") from None
    emb, labels = gallery.embeddings, gallery.labels
    if emb.ndim != 2 or emb.shape[0] == 0 or emb.shape[0] != labels.shape[0]:
        raise GalleryError(f"the face gallery is inconsistent ({emb.shape} vs {labels.shape})")
    if labels.min() < 0 or labels.max() >= len(gallery.names):
        raise GalleryError("the face gallery names fewer people than its embeddings use")
    return gallery


def match(gallery: Gallery, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Best person and its similarity for each query embedding.

    Returns ``(class_index [N] int, similarity [N] float32)``; a person is
    scored by the best of their photos.
    """
    queries = normalize(np.asarray(embeddings, np.float32).reshape(-1, gallery.dim))
    if queries.shape[0] == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.float32)
    sims = queries @ gallery.embeddings.T                      # [N, M]
    per_class = np.full((queries.shape[0], len(gallery.names)), -1.0, np.float32)
    for c in np.unique(gallery.labels):
        per_class[:, c] = sims[:, gallery.labels == c].max(axis=1)
    best = per_class.argmax(axis=1)
    return best, per_class[np.arange(len(best)), best]
