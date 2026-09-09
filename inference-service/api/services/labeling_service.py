"""Model-assisted labeling on the device's own TensorRT runtime.

The training page can pre-label a dataset image with an engine that already
exists on the device. It runs here, not in the training-service, so the
suggestions come from exactly the pipeline live detection uses — the same
``ModelManager`` preprocessing (letterbox, ``TILING_MODE`` grid), the same
``YOLODetector`` decode/merge/NMS and the model's own classes sidecar. The
engine is pinned to a private worker (``TENSORRT_LABEL_WORKER_PORT``) so it
never displaces the live model on the base ports; ``ReleaseRuntime`` drops it
with every other worker, and the training page unloads it on exit.
"""
import copy
import logging
import os
import threading
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from api.model_manager import ModelManager
from api.model_paths import ENGINE_FILE_EXTENSIONS, validate_model_filename
from api.repositories.class_labels_repository import ClassLabelsRepository
from api.yolo_detector import YOLODetector

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.5
#: Offset from ``TENSORRT_WORKER_PORT`` for the private labeling worker —
#: past any ``TENSORRT_CONTEXTS`` lane the live pipeline can occupy.
_LABEL_PORT_OFFSET = 16


def label_worker_port() -> int:
    """Port of the private labeling worker (``TENSORRT_LABEL_WORKER_PORT``)."""
    base = int(os.environ.get("TENSORRT_WORKER_PORT", "5501"))
    try:
        return int(os.environ.get("TENSORRT_LABEL_WORKER_PORT", str(base + _LABEL_PORT_OFFSET)))
    except ValueError:
        return base + _LABEL_PORT_OFFSET


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
        self._detector: Optional[YOLODetector] = None

    # ── status ────────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._manager is not None

    @property
    def model_name(self) -> str:
        with self._lock:
            return self._model_name if self._manager is not None else ""

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
            try:
                manager = ModelManager(cfg, port=self._port)
            except Exception:
                # The worker subprocess is spawned before the engine is
                # validated; a failed load must not leave it (and its CUDA
                # context) around until the next runtime release.
                self._close_worker()
                raise
            labels = ClassLabelsRepository(cfg.CLASSES_FILE_PATH).load_labels()
            detector = YOLODetector(labels, cfg)
            detector.set_areas([])  # every object counts when labeling
            self._label_config = cfg
            self._manager = manager
            self._detector = detector
            self._model_name = model_name
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

    def detect(self, jpeg: bytes, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Detections on one encoded image, as normalized corners on that image."""
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
            if manager.tiling_active:
                _, _, detections = detector.process_tiled_detections(outputs, frame, metas)
            else:
                meta = metas[0]
                _, _, detections = detector.process_detections(
                    outputs[0], frame, scale=meta.scale, border_top=meta.border_top,
                    actual_input_size=meta.input_size,
                )
            height, width = frame.shape[:2]

        def norm(value: float, extent: int) -> float:
            return min(1.0, max(0.0, float(value) / float(extent)))

        return [
            {
                "class_id": int(det.class_id),
                "class_name": str(det.class_name),
                "score": float(det.confidence),
                "x1": norm(det.bbox[0], width),
                "y1": norm(det.bbox[1], height),
                "x2": norm(det.bbox[2], width),
                "y2": norm(det.bbox[3], height),
            }
            for det in detections
        ]

    def _publish(self) -> None:
        if self._events is None:
            return
        try:
            self._events.publish("label_model_changed", keys=["label_model"],
                                 source="labeling", data=self.status())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not publish label-model event: %s", exc)
