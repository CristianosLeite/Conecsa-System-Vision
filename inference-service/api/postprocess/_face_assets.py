# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""The bundled face models and where their engines and workers live.

The YuNet detector (MIT) and the SFace embedder (Apache-2.0) ship as ONNX
graphs in ``FACE_MODELS_DIR`` (fetched and checksummed at image build, see
``scripts/face-models.pin``). Their TensorRT engines are built on the device
the first time a gallery is built and kept in ``<models dir>/face/``, a
subdirectory ``ModelService.list_models`` does not list.
"""
import hashlib
import os
from functools import lru_cache
from pathlib import Path

from ..worker_ports import (
    FACE_BUILD_DETECTOR_SLOT,
    FACE_BUILD_EMBEDDER_SLOT,
    FACE_EMBED_SLOT,
    private_worker_port,
)

DETECTOR_ONNX = "face_detection_yunet_2023mar.onnx"
EMBEDDER_ONNX = "face_recognition_sface_2021dec.onnx"
#: Subdirectory of the model directory holding the shared face engines.
ENGINE_SUBDIR = "face"

#: The face workers take the slots after the labeling worker in the shared
#: private-worker scheme (``api.worker_ports``): +17, +18 and +19 from
#: ``TENSORRT_WORKER_PORT`` by default, later when more lanes are configured.

_DEFAULT_DIR = Path(__file__).resolve().parents[2] / "assets" / "face"


def models_dir() -> str:
    """Directory of the bundled ONNX graphs (``FACE_MODELS_DIR``)."""
    return os.environ.get("FACE_MODELS_DIR", str(_DEFAULT_DIR))


def onnx_path(name: str) -> str:
    return os.path.join(models_dir(), name)


def assets_available() -> bool:
    """True when both bundled graphs are present: the build can serve ``face``."""
    return all(os.path.isfile(onnx_path(n)) for n in (DETECTOR_ONNX, EMBEDDER_ONNX))


def engine_path(model_directory: str, onnx_name: str) -> str:
    """Where the shared engine built from ``onnx_name`` lives.

    The name carries the graph's hash: the model directory is a persistent
    volume, so an image that ships a newer graph must not reuse the engine
    built from the old one (a gallery built on it would then not match the
    live embedder either). A stale engine is simply never found again.
    """
    stem = os.path.splitext(onnx_name)[0]
    digest = onnx_sha256(onnx_name)
    suffix = f"-{digest[:12]}" if digest else ""
    return os.path.join(model_directory, ENGINE_SUBDIR, f"{stem}{suffix}.engine")


def embed_worker_port() -> int:
    """The live embedder's private worker (``TENSORRT_FACE_WORKER_PORT``)."""
    return private_worker_port(FACE_EMBED_SLOT, "TENSORRT_FACE_WORKER_PORT")


def build_worker_ports() -> tuple:
    """The gallery builder's detector and embedder workers."""
    return (private_worker_port(FACE_BUILD_DETECTOR_SLOT, "TENSORRT_FACE_BUILD_DETECTOR_PORT"),
            private_worker_port(FACE_BUILD_EMBEDDER_SLOT, "TENSORRT_FACE_BUILD_EMBEDDER_PORT"))


@lru_cache(maxsize=8)
def _sha256(path: str, size: int, mtime: float) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def onnx_sha256(onnx_name: str) -> str:
    """SHA-256 of a bundled graph, "" when it is missing."""
    path = onnx_path(onnx_name)
    try:
        stat = os.stat(path)
    except OSError:
        return ""
    return _sha256(path, stat.st_size, stat.st_mtime)


def embedder_sha256() -> str:
    """SHA-256 of the bundled embedder graph, "" when it is missing.

    A gallery records it: embeddings of another embedder are not comparable,
    so activation refuses a gallery built with a different one.
    """
    return onnx_sha256(EMBEDDER_ONNX)
