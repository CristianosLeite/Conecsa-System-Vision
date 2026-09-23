# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""The SFace embedder on a private TensorRT worker.

Runs beside the live YuNet engine the way the labeling engine does: its own
``ModelManager`` pinned to one worker port, so it never displaces the
pipeline's contexts. ``ReleaseRuntime`` frees it with every other worker; the
next ``DetectionService.initialize()`` builds a new strategy and a new
embedder.
"""
import copy
import logging
from typing import Any, Optional, Sequence

import numpy as np

from . import _face_assets
from ._face_align import CROP_SIZE, embedder_input
from ._face_gallery import normalize
from .contract import ContractError

logger = logging.getLogger(__name__)


class FaceEmbedder:
    """Aligned 112×112 face crops → L2-normalized embeddings."""

    def __init__(self, config: Any, engine_path: str, port: int):
        # Lazy: ModelManager pulls in cv2 and the runtime registry.
        from ..model_manager import ModelManager

        cfg = copy.copy(config)
        cfg.MODEL_PATH = engine_path
        self._port = port
        try:
            self._manager = ModelManager(cfg, port=port, task="face")
            self._check_io()
        except Exception:
            self.close()
            raise

    def _check_io(self) -> None:
        inputs = self._manager.input_details
        outputs = self._manager.output_details
        shape = list(inputs[0].get("shape", [])) if inputs else []
        if shape != [1, 3, CROP_SIZE, CROP_SIZE]:
            raise ContractError(
                f"the face embedder takes [1, 3, {CROP_SIZE}, {CROP_SIZE}], this one {shape}")
        if len(outputs) != 1 or len(outputs[0].get("shape", [])) != 2:
            raise ContractError("the face embedder has one [1, D] output")

    def embed(self, crops_bgr: Sequence[np.ndarray]) -> np.ndarray:
        """One normalized embedding row per crop (batch-1 engine, one call each)."""
        rows = []
        for crop in crops_bgr:
            outputs, _ = self._manager.run_inference(embedder_input([crop]))
            rows.append(np.asarray(outputs[0], np.float32).reshape(-1))
        if not rows:
            return np.zeros((0, 0), np.float32)
        return normalize(np.stack(rows))

    def close(self) -> None:
        """Terminate the private worker (frees its CUDA context)."""
        close_worker(self._port)


def close_worker(port: int) -> None:
    from ..runtime_management.worker_client import get_worker_client

    try:
        get_worker_client(port).close()
    except Exception as exc:  # noqa: BLE001 - freeing must never fail the caller
        logger.warning("Could not close the face worker on port %d: %s", port, exc)


def create_embedder(config: Any, model_directory: Optional[str] = None,
                    port: Optional[int] = None) -> FaceEmbedder:
    """The live embedder for a face model under ``config.MODEL_PATH``.

    Refused (``ContractError``) when the shared engine was never built: it is
    built with the gallery, never under an activation deadline.
    """
    import os

    directory = model_directory or os.path.dirname(config.MODEL_PATH)
    engine = _face_assets.engine_path(directory, _face_assets.EMBEDDER_ONNX)
    if not os.path.isfile(engine):
        raise ContractError(
            "the face embedder engine is missing; rebuild the gallery from its dataset")
    return FaceEmbedder(config, engine,
                        port if port is not None else _face_assets.embed_worker_port())
