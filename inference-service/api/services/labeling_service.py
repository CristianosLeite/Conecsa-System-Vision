# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Model-assisted labeling on the device's own TensorRT runtime.

The training page can pre-label a dataset image with an engine that already
exists on the device. It runs here, not in the training-service, so the
suggestions come from exactly the pipeline live detection uses — the same
``ModelManager`` preprocessing (letterbox, ``TILING_MODE`` grid), the same
task postprocessor (``api.postprocess``: decode/merge/NMS for detection) and
the model's own classes sidecar. The
engine is pinned to a private worker (``TENSORRT_LABEL_WORKER_PORT``) so it
never displaces the live model on the base ports; ``ReleaseRuntime`` drops it
with every other worker, and the training page unloads it on exit.
"""
import copy
import logging
import os
import threading
from typing import Any, Dict, Optional

import cv2
import numpy as np

from api import postprocess
from api.model_manager import ModelManager
from api.model_paths import ENGINE_FILE_EXTENSIONS, validate_model_filename
from api.postprocess.base import Postprocessor
from api.repositories.class_labels_repository import ClassLabelsRepository
from api.worker_ports import LABEL_SLOT, private_worker_port

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.5
#: Offset from ``TENSORRT_WORKER_PORT`` for the private labeling worker —
#: past every ``TENSORRT_CONTEXTS`` lane the live pipeline can occupy
#: (``worker_ports``: +16 by default, later when more lanes are configured).


def label_worker_port() -> int:
    """Port of the private labeling worker (``TENSORRT_LABEL_WORKER_PORT``)."""
    return private_worker_port(LABEL_SLOT, "TENSORRT_LABEL_WORKER_PORT")


class LabelingService:
    """One engine at a time on a private worker, serving per-image detections."""

    def __init__(self, config, model_service, event_service=None,
                 port: Optional[int] = None):
        self._config = config
        self._models = model_service
        self._events = event_service
        self._port = port if port is not None else label_worker_port()
        self._lock = threading.RLock()
        self._model_name = ""
        self._label_config: Any = None
        self._manager: Optional[ModelManager] = None
        self._detector: Optional[Postprocessor] = None
        self._task: Optional[str] = None

    # ── status ────────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._manager is not None

    @property
    def model_name(self) -> str:
        with self._lock:
            return self._model_name if self._manager is not None else ""

    def loaded_task(self) -> Optional[str]:
        """Task of the loaded engine, ``None`` when nothing is loaded."""
        with self._lock:
            return self._task if self._manager is not None else None

    def status(self) -> Dict[str, Any]:
        """Status."""
        with self._lock:
            loaded = self._manager is not None
            names = list(self._detector.class_labels) if loaded and self._detector else []
            return {
                "loaded": loaded,
                "model_name": self._model_name if loaded else "",
                "class_names": names,
                "message": "",
                "task": (self._task or "") if loaded else "",
            }

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def load(self, model_name: str) -> None:
        """Load a listed engine on the private worker (a different one replaces it)."""
        path, error = validate_model_filename(model_name, self._models.model_directory)
        if error:
            raise ValueError(error)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model '{model_name}' not found")
        # Prebuilt engines only: an .onnx would make the private worker build
        # an engine (minutes), far past the caller's deadline.
        if not path.lower().endswith(ENGINE_FILE_EXTENSIONS):
            raise ValueError(f"Model '{model_name}' is not a TensorRT engine")
        with self._lock:
            if self._manager is not None and self._model_name == model_name:
                return
            self._drop()
            cfg = self._config_for(path)
            task = self._models.task_of(model_name)
            if task == "face":
                raise ValueError("Face recognition models cannot assist labeling")
            labels = ClassLabelsRepository(cfg.CLASSES_FILE_PATH).load_labels()
            try:
                detector = postprocess.create(task, labels, cfg)
            except postprocess.ContractError as ex:
                raise ValueError(str(ex)) from None
            try:
                manager = ModelManager(cfg, port=self._port, task=task)
            except Exception:
                # The worker subprocess is spawned before the engine is
                # validated; a failed load must not leave it (and its CUDA
                # context) around until the next runtime release.
                self._close_worker()
                raise
            detector.set_areas([])  # every object counts when labeling
            self._label_config = cfg
            self._manager = manager
            self._detector = detector
            self._model_name = model_name
            self._task = task
            logger.info("Labeling engine loaded on port %d: %s (%d classes)",
                        self._port, model_name, len(labels))
        self._publish()

    def unload(self) -> None:
        """Drop the engine and terminate the private worker (frees its CUDA memory)."""
        with self._lock:
            had_engine = self._manager is not None
            self._drop()
        if had_engine:
            self._close_worker()
            logger.info("Labeling engine unloaded")
        self._publish()

    def _drop(self) -> None:
        self._manager = None
        self._detector = None
        self._label_config = None
        self._model_name = ""
        self._task = None

    def _close_worker(self) -> None:
        # The cached client stays (like every other worker); only its
        # subprocess goes away, so the next load restarts it cleanly.
        from api.runtime_management.worker_client import get_worker_client

        try:
            get_worker_client(self._port).close()
        except Exception as exc:  # noqa: BLE001 - freeing must never fail the caller
            logger.warning("Could not close the labeling worker on port %d: %s",
                           self._port, exc)

    def _config_for(self, model_path: str):
        """A private copy of the runtime config pointed at ``model_path``.

        The manager and detector read the model path, classes file and
        thresholds from it; a copy keeps the live detector's settings
        untouched while labeling uses its own confidence gate.
        """
        cfg = copy.copy(self._config)
        cfg.MODEL_PATH = model_path
        cfg.CLASSES_FILE_PATH = self._models.classes_file_for_model(model_path)
        cfg.CONFIDENCE_THRESHOLD = DEFAULT_THRESHOLD
        return cfg

    # ── detection ─────────────────────────────────────────────────────────────

    def detect(self, jpeg: bytes, threshold: float = 0.0) -> Dict[str, Any]:
        """Suggestions for one encoded image.

        Returns ``{"detections", "image_class", "candidates"}``: a detection
        engine fills ``detections`` (normalized corners on that image, and
        for a segmentation engine each detection's ``rings``); a
        classification engine fills ``image_class`` (the top-1 when above the
        threshold, else ``None``) and ``candidates`` (its top-k).
        """
        with self._lock:
            manager, detector, cfg = self._manager, self._detector, self._label_config
            if manager is None or detector is None:
                raise RuntimeError("No labeling model is loaded")
            frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Could not decode the image")
            cfg.CONFIDENCE_THRESHOLD = float(threshold) if threshold > 0.0 else DEFAULT_THRESHOLD

            tensors, metas = manager.preprocess_tiles(frame)
            outputs = [manager.run_inference(tensor)[0] for tensor in tensors]
            # Each image is judged on its own: no transition state carries over.
            detector.reset_state()
            decoded = detector.process(outputs, frame, metas, manager.tiling_active)
            height, width = frame.shape[:2]

        def norm(value: float, extent: int) -> float:
            return min(1.0, max(0.0, float(value) / float(extent)))

        detections = [
            {
                "class_id": int(det.class_id),
                "class_name": str(det.class_name),
                "score": float(det.confidence),
                "x1": norm(det.bbox[0], width),
                "y1": norm(det.bbox[1], height),
                "x2": norm(det.bbox[2], width),
                "y2": norm(det.bbox[3], height),
                "rings": [[[float(x), float(y)] for x, y in ring]
                          for ring in (det.resolve_polygons() or ())],
            }
            for det in decoded.items
            if det.bbox is not None
        ]
        image_class = None
        if decoded.candidates is not None and decoded.items:
            top = decoded.items[0]
            image_class = {"class_id": int(top.class_id), "class_name": str(top.class_name),
                           "score": float(top.confidence)}
        candidates = [
            {"class_id": int(c["class_id"]), "class_name": str(c["class_name"]),
             "score": float(c["confidence"])}
            for c in (decoded.candidates or [])
        ]
        return {"detections": detections, "image_class": image_class, "candidates": candidates}

    def _publish(self) -> None:
        if self._events is None:
            return
        try:
            self._events.publish("label_model_changed", keys=["label_model"],
                                 source="labeling", data=self.status())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not publish label-model event: %s", exc)
