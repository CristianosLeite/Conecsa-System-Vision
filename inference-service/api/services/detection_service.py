# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Detection service - Manages detection operations.
"""
import logging
from threading import Lock
from typing import List, Optional

# noinspection PyPackageRequirements
import numpy as np  # ships in conecsa-os-base:base
from conecsa_common.tasks import DEFAULT_TASK

from .. import postprocess
from ..config import Config
from ..model_manager import ModelManager
from ..models.detection_models import DetectionResult
from ..postprocess.base import Postprocessor
from ..postprocess.classify import probabilities_output
from ..postprocess.contract import ContractError, check, check_probabilities
from ..utils import load_class_labels
from .detection_area_service import DetectionAreaService
from .errors import NO_APPLICATION_MESSAGE, PreconditionFailed
from .model_settings_service import ModelSettingsService

logger = logging.getLogger(__name__)

# How long ``stop()`` waits for the pipeline to finish the frames it already
# holds before a runtime swap/release proceeds. A wedged TensorRT worker times
# out its own request after WORKER_REQUEST_TIMEOUT_SEC (8 s) — this sits above.
_DRAIN_TIMEOUT_S = 10.0


class StaleGeneration(RuntimeError):
    """A pipeline item was prepared against a runtime that has since been swapped.

    Every runtime swap (``initialize``) advances ``DetectionService.generation``;
    ``prepare`` stamps the current value on its output and ``infer``/``finish``
    refuse items carrying another one, so a frame preprocessed for one engine is
    never submitted to another or decoded with the wrong labels/tiling.
    """


def normalized_bbox(bbox, width: int, height: int) -> List[float]:
    """Pixel corners (x1, y1, x2, y2) → normalized [x1, y1, x2, y2] in 0..1."""
    x1, y1, x2, y2 = bbox
    def _clamp(v: float) -> float:
        return min(1.0, max(0.0, v))
    return [
        round(_clamp(x1 / width), 4),
        round(_clamp(y1 / height), 4),
        round(_clamp(x2 / width), 4),
        round(_clamp(y2 / height), 4),
    ]


def _encode_frame_b64(image: np.ndarray) -> Optional[str]:
    """JPEG-encode a frame (quality 80) and return it base64-encoded."""
    import base64

    # noinspection PyPackageRequirements
    import cv2  # ships in conecsa-os-base:base
    ok, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return None
    # buf.data is a zero-copy memoryview onto the encoded JPEG; b64encode reads it
    # directly, avoiding the extra bytes copy tobytes() would make. memoryview(buf)
    # would also work but trips older numpy stubs (ndarray predates PEP 688's Buffer).
    return base64.b64encode(buf.data).decode('ascii')


class DetectionService:
    """Service for managing object detection operations."""

    def __init__(
        self,
        config: Config,
        area_service: Optional[DetectionAreaService] = None,
        video_service=None,
        buffer_service=None,
        application_service=None,
    ):
        """
        Initialize the detection service.

        Args:
            config: Configuration instance
            area_service: Optional detection-area repository. When provided,
                its current snapshot is pushed to the YOLO detector before
                each frame so the detector can spatially filter and overlay.
            video_service: Optional VideoService, used to refuse starting
                detection while no camera is streaming (the webcam-server
                publishes no frames at all in that case).
            buffer_service: Optional DetectionBufferService that persists
                on-change results while the hub is not polling (offline
                store-and-forward); observed from finish().
            application_service: Optional ApplicationService. When provided,
                detection refuses to start while no application type is
                chosen and refuses a model of another task; without one the
                device behaves as object detection.
        """
        self.config: Config = config
        self._application_service = application_service
        self.area_service: Optional[DetectionAreaService] = area_service
        self.video_service = video_service
        self.buffer_service = buffer_service
        self.model_manager: Optional[ModelManager] = None
        # The model task's postprocess strategy (api.postprocess), rebuilt by
        # every initialize().
        self.postprocessor: Optional[Postprocessor] = None
        self.class_labels: List[str] = []
        self.is_running: bool = False
        self.lock: Lock = Lock()
        self.trigger_enabled: bool = True
        self._detection_count: int = 0
        self._count_lock: Lock = Lock()
        self.last_detection_result: Optional[DetectionResult] = None
        # Runtime generation — advanced by every initialize(); see StaleGeneration.
        self.generation: int = 0
        # The processing pipeline, once it registers itself (attach_pipeline):
        # stop() quiesces it before the caller swaps or releases the runtime.
        self._pipeline = None

    def attach_pipeline(self, pipeline) -> None:
        """Register the processing pipeline so runtime transitions can quiesce it.

        Called by ``ProcessingPipelineService`` at construction. ``pipeline``
        must expose ``drain(timeout) -> bool`` and ``resume()``.
        """
        self._pipeline = pipeline

    def _drain_pipeline(self) -> None:
        """Close the detection path and wait for in-flight frames to finish."""
        if self._pipeline is None:
            return
        if not self._pipeline.drain(_DRAIN_TIMEOUT_S):
            logger.error(
                "Pipeline did not quiesce within %.0fs; continuing with the runtime "
                "swap (stale frames are rejected by generation)", _DRAIN_TIMEOUT_S)

    def _resume_pipeline(self) -> None:
        """Reopen the detection path after a runtime swap."""
        if self._pipeline is not None:
            self._pipeline.resume()

    @property
    def application_task(self) -> Optional[str]:
        """The device's application task (``None`` while none is chosen)."""
        app = self._application_service
        return app.task if app is not None else DEFAULT_TASK

    def _model_task(self) -> str:
        """The configured model's task, refused unless it is the application's."""
        app_task = self.application_task
        if app_task is None:
            raise PreconditionFailed(NO_APPLICATION_MESSAGE)
        # Lazy: model_service imports this module's siblings at wiring time.
        from .model_service import ModelService
        declared = ModelSettingsService.task_of(
            ModelService.settings_file_for_model(self.config.MODEL_PATH))
        if declared != app_task:
            raise PreconditionFailed(
                f"Model '{self._model_name()}' is a '{declared}' model, but the device "
                f"runs the '{app_task}' application")
        return declared

    @staticmethod
    def _check_contract(model_manager, task: str) -> None:
        """Verify the loaded engine's outputs against the declared task."""
        outputs = [detail.get("shape") for detail in getattr(model_manager, "output_details", [])]
        inputs = getattr(model_manager, "input_details", [])
        input_shape = inputs[0].get("shape") if inputs else None
        try:
            check(task, outputs, input_shape)
        except ContractError as ex:
            raise ContractError(
                f"{ex}. Re-upload the model declaring the task it was trained for.") from None

    @staticmethod
    def _check_probabilities(model_manager) -> None:
        """A classification engine must output probabilities.

        One inference on a blank frame through the real preprocessing: its
        output row must sum to 1, which softmax in the exported graph
        guarantees and logits almost never do.
        """
        size = int(getattr(model_manager, "input_size", 0) or 224)
        tensors, _ = model_manager.preprocess_tiles(np.zeros((size, size, 3), dtype=np.uint8))
        outputs, _ = model_manager.run_inference(tensors[0])
        values = np.asarray(probabilities_output(outputs), dtype=np.float64).reshape(-1)
        check_probabilities(values.tolist())

    def initialize(self) -> bool:
        """
        Initialize the detection model and components.

        The model's task (from its settings sidecar) must be the device's
        application task, and the engine's outputs must match that task's
        layout; otherwise nothing is swapped and the error propagates, so
        ``ModelService.activate_model`` rolls back to the previous model.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            task = self._model_task()

            import os
            if not os.path.exists(self.config.MODEL_PATH):
                logger.error(f"Model file not found: {self.config.MODEL_PATH}")
                raise FileNotFoundError(
                    "No model loaded. Please upload and select a model before starting detection."
                )

            logger.info("Initializing model manager...")
            model_manager = ModelManager(self.config, task=task)
            self._check_contract(model_manager, task)
            if task == "classify":
                self._check_probabilities(model_manager)

            logger.info("Loading class labels...")
            class_labels = load_class_labels(self.config)

            logger.info("Initializing %s postprocess...", task)
            previous = self.postprocessor
            if (task == "face" and isinstance(previous, postprocess.FacePostprocessor)
                    and "face" in postprocess.supported_tasks()):
                # Face to face: the SFace worker is one per device on one
                # port, so the new strategy takes over the running one instead
                # of opening a second beside it (or on top of it).
                postprocessor = postprocess.FacePostprocessor(
                    class_labels, self.config, embedder=previous.embedder)
                owns_embedder = False
            else:
                postprocessor = postprocess.create(task, class_labels, self.config)
                owns_embedder = True
            if isinstance(postprocessor, postprocess.FacePostprocessor):
                # YuNet's cls/obj outputs share shapes: bind them by name. A
                # strategy that opened its own worker must free it when the
                # binding is refused, since it never becomes
                # ``self.postprocessor`` for the task-switch cleanup to close;
                # a handed-over worker still belongs to the previous strategy,
                # which the activation rollback restores.
                try:
                    postprocessor.bind_outputs(model_manager.output_details)
                except Exception:
                    if owns_embedder:
                        postprocessor.close()
                    raise

            self.model_manager = model_manager
            self.class_labels = class_labels
            self.postprocessor = postprocessor

            # The runtime changed: frames prepared before this point must not
            # reach it (StaleGeneration), and the pipeline may run again.
            self.generation += 1
            self._resume_pipeline()

            logger.info("Detection service initialized successfully")
            return True
        except FileNotFoundError as ex:
            logger.error(f"Model file error: {ex}")
            raise
        except PreconditionFailed as ex:
            logger.error(f"Model not allowed on this device: {ex}")
            raise
        except Exception as ex:
            logger.error(f"Error initializing detection service: {ex}")
            raise RuntimeError(f"Failed to initialize detection service: {str(ex)}") from ex

    def start(self) -> bool:
        """
        Start detection processing.

        Returns:
            bool: True if started successfully, False if already running

        Raises:
            PreconditionFailed: if no application type is chosen.
            RuntimeError: if no camera is streaming.
        """
        with self.lock:
            if self.is_running:
                return False

            if self.application_task is None:
                raise PreconditionFailed(NO_APPLICATION_MESSAGE)

            if self.video_service and not self.video_service.camera_connected():
                raise RuntimeError(
                    "No camera connected. Connect a camera before starting detection."
                )

            if not self.model_manager:
                self.initialize()  # This will raise an exception if it fails

            # A start begins a new stream: its first class is a transition.
            self.reset_task_state()
            self.is_running = True
            logger.info("Detection started")
        self._resume_pipeline()
        return True

    def stop(self) -> bool:
        """
        Stop detection processing and quiesce the pipeline.

        Flipping ``is_running`` alone would leave frames in the pipeline's
        stage queues and a worker mid-inference; a caller that then swaps the
        model (``ModelService.activate_model``) or releases the TensorRT
        workers (``ReleaseRuntime``) would race them. So this also drains the
        pipeline — the detection path stays closed until ``initialize()`` or
        ``start()`` reopens it — and only returns once nothing is in flight
        (or after ``_DRAIN_TIMEOUT_S``, logged).

        Returns:
            bool: True if stopped successfully, False if not running
        """
        with self.lock:
            was_running = self.is_running
            if was_running:
                self.is_running = False
                logger.info("Detection stopped")
        # Drain even when nothing was running: it is a no-op then, and it keeps
        # the detection path closed for a swap/release that follows.
        self._drain_pipeline()
        # Nothing is in flight any more: forget the stream's task state.
        self.reset_task_state()
        return was_running

    def unload_runtime(self) -> None:
        """Forget the loaded model's runtime (after an application change).

        The next ``start()`` initializes whatever model is then configured;
        advancing the generation rejects any frame still stamped with the
        dropped runtime. Call ``stop()`` first.
        """
        with self.lock:
            self.model_manager = None
            self.postprocessor = None
            self.class_labels = []
            self.generation += 1

    def reset_results(self) -> None:
        """Drop the last result so no client pulls one of a previous task."""
        self.last_detection_result = None
        self.reset_task_state()

    def reset_task_state(self) -> None:
        """Forget per-stream task state: classification restarts from ``none``.

        Called on start, stop, stats reset and application change; a model
        change builds a new strategy, which starts from ``none`` anyway.
        """
        postprocessor = self.postprocessor
        if postprocessor is not None:
            postprocessor.reset_state()

    # ── Pipeline stages: prepare / infer / finish, run on separate pipeline threads ──

    def prepare(self, frame: np.ndarray):
        """Stage A: snapshot detection areas + preprocess.

        Returns ``(generation, inputs, metas)`` — the runtime generation this
        frame was prepared against plus parallel lists holding one preprocessed
        tensor and one ``TileMeta`` per tile (a single full-frame entry when
        ``TILING_MODE=off``) — or ``None`` if detection is not ready.
        """
        # Snapshot the generation before touching the runtime: a swap that
        # lands during preprocessing is then caught by infer/finish.
        generation = self.generation
        model_manager = self.model_manager
        postprocessor = self.postprocessor
        if not self.is_running or not model_manager or not postprocessor:
            return None

        # Snapshot the current detection areas onto the postprocessor so they
        # apply to this frame's filtering + overlay.
        if self.area_service is not None:
            postprocessor.set_areas(self.area_service.list())

        inputs, metas = model_manager.preprocess_tiles(frame)
        return generation, inputs, metas

    def _check_generation(self, generation: Optional[int]) -> None:
        """Raise ``StaleGeneration`` for an item prepared against an older runtime."""
        if generation is not None and generation != self.generation:
            raise StaleGeneration(
                f"frame prepared for runtime generation {generation}, "
                f"current is {self.generation}")

    def infer(self, inputs, generation: Optional[int] = None):
        """Stage B: run inference on each prepared tensor, in order.

        Returns ``(outputs, inference_time)`` — per input, the list of every
        engine output tensor, and the summed GPU seconds. Tiling off means a single entry; with
        ``TILING_MODE=grid`` the tiles of one frame run sequentially on this
        lane while other lanes interleave on the context pool. Only reachable
        once ``prepare`` returned a non-None result, which already implies a
        model manager; the guard exists so a stopped detector fails loudly
        instead of raising AttributeError on None. ``generation`` (from
        ``prepare``) is checked first so a stale frame never reaches a worker.
        """
        self._check_generation(generation)
        model_manager = self.model_manager
        if model_manager is None:
            raise RuntimeError("Detection is not running: no model manager")
        outputs = []
        total_time = 0.0
        for input_data in inputs:
            output_data, seconds = model_manager.run_inference(input_data)
            outputs.append(output_data)
            total_time += seconds
        return outputs, total_time

    def finish(self, outputs, frame: np.ndarray, metas, inference_time: float = 0.0,
               generation: Optional[int] = None) -> Optional[DetectionResult]:
        """Stage C: postprocess detections + draw overlay. Returns DetectionResult.

        ``outputs``/``metas`` are the parallel lists produced by ``infer`` and
        ``prepare``; the model task's postprocessor decodes them. With tiling
        off the single entry goes through the exact pre-tiling decode path;
        with ``TILING_MODE=grid`` each tile is decoded and duplicates across
        the overlap bands are merged. A stale ``generation`` raises before the
        outputs meet the wrong labels/tiling.
        """
        self._check_generation(generation)
        postprocessor = self.postprocessor
        if not postprocessor:
            return None
        tiled = self.model_manager is not None and self.model_manager.tiling_active
        decoded = postprocessor.process(outputs, frame, metas, tiled)
        result = DetectionResult(
            detections=decoded.items,
            processed_image=decoded.image,
            inference_time=inference_time,
            num_detections=decoded.count,
            raw_image=frame,
            count_increment=decoded.count_increment,
            candidates=decoded.candidates,
        )
        self.last_detection_result = result
        if self.buffer_service is not None:
            # Offline store-and-forward: never let the buffer take down the
            # pipeline. Runs on the single pipeline-finish thread.
            try:
                # The change signature needs no rings; the full items (rings
                # included, byte-identical to a snapshot) are built only when
                # a record is written.
                self.buffer_service.observe(
                    self._snapshot_items(result, with_polygons=False)[0],
                    result.num_detections,
                    self._model_name(),
                    result.raw_image,
                    result.processed_image,
                    task=postprocessor.task,
                    record_detections=lambda: self._detection_dicts(result),
                )
            except Exception:
                logger.exception("detection buffer observe failed")
        return result

    def refresh_class_labels(self) -> List[str]:
        """Re-read the active model's labels and hand them to the live strategy.

        Called after ``SetClasses``/``ClearClasses`` so a rename shows in the
        overlay, the snapshot and the records at once, instead of waiting for
        the next model load.
        """
        labels = load_class_labels(self.config)
        with self.lock:
            self.class_labels = labels
            postprocessor = self.postprocessor
        if postprocessor is not None:
            postprocessor.set_class_labels(labels)
        return labels

    def set_confidence_threshold(self, threshold: float) -> bool:
        """
        Set the confidence threshold for detections.

        Args:
            threshold: Confidence threshold (0.0 to 1.0)

        Returns:
            bool: True if set successfully
        """
        with self.lock:
            return self.config.set_confidence_threshold(threshold)

    def is_model_loaded(self) -> bool:
        """
        Check if model is loaded.

        Returns:
            bool: True if model is loaded
        """
        return self.model_manager is not None

    def acceleration_type(self) -> str:
        """
        Get the hardware acceleration type.

        Returns:
            str: "GPU", "CPU", "Disabled", or "None"
        """
        if self.model_manager is not None:
            return self.model_manager.acceleration_type
        return "None"

    def runtime_api(self) -> str:
        """
        Get the runtime API being used.

        Returns:
            str: Active runtime name. TensorRT is the only supported runtime.
        """
        if self.model_manager is not None:
            return self.model_manager.runtime_api
        return "Unknown"

    def _model_name(self) -> str:
        """Basename of the active model file (snapshot/backlog metadata)."""
        return self.config.MODEL_PATH.split('/')[-1]

    #: Resource bound: the compact-JSON size the polygon rings of one snapshot
    #: (and so of one offline record) may take; beyond it rings are dropped
    #: smallest first.
    SNAPSHOT_POLYGON_BYTES = 64 * 1024

    @staticmethod
    def _detection_dicts(result: DetectionResult) -> List[dict]:
        """Per-detection dicts in the snapshot wire format.

        Shared by detections_snapshot() and the offline-buffer hook so
        buffered backlog records stay byte-identical to live snapshots. A
        result without geometry (a classification) has no ``bbox`` key; a
        segmentation item adds ``polygons`` (its exterior rings as normalized
        ``[[x, y], …]``, 4 decimals), capped by :meth:`_cap_polygons`.
        """
        return DetectionService._snapshot_items(result)[0]

    @staticmethod
    def _snapshot_items(result: DetectionResult,
                        with_polygons: bool = True) -> tuple[List[dict], bool]:
        """:meth:`_detection_dicts` plus whether rings were dropped by the cap.

        ``with_polygons=False`` leaves the rings out (and never extracts them):
        the per-frame offline-buffer signature does not read them.
        """
        height, width = result.processed_image.shape[:2]
        items = []
        for d in result.detections:
            item = {
                "class_name": d.class_name,
                "color": d.color,
                "confidence": round(float(d.confidence), 4),
                "area": d.area,
            }
            if d.bbox is not None:
                item["bbox"] = normalized_bbox(d.bbox, width, height)
            rings = d.resolve_polygons() if with_polygons else None
            if rings is not None:
                item["polygons"] = [
                    [[round(float(x), 4), round(float(y), 4)] for x, y in ring]
                    for ring in rings
                ]
            items.append(item)
        if not with_polygons:
            return items, False
        return items, DetectionService._cap_polygons(
            items, DetectionService.SNAPSHOT_POLYGON_BYTES)

    @staticmethod
    def _cap_polygons(items: List[dict], limit: int) -> bool:
        """Drop rings, smallest area first, until they fit ``limit`` bytes of compact JSON.

        Sizes are counted from the rounded coordinates' ``str`` form, which is
        what ``json.dumps`` writes for a float. Returns whether any ring was
        dropped; an instance that loses every ring keeps an empty list.
        """
        rings = []
        total = 0
        for i, item in enumerate(items):
            for j, ring in enumerate(item.get("polygons") or []):
                size = sum(len(str(x)) + len(str(y)) + 4 for x, y in ring) + 1
                area = abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1)
                               in zip(ring, ring[1:] + ring[:1], strict=True))) / 2.0
                rings.append((area, i, j, size))
                total += size
        if total <= limit:
            return False
        dropped = set()
        for _, i, j, size in sorted(rings):
            if total <= limit:
                break
            dropped.add((i, j))
            total -= size
        for i, item in enumerate(items):
            if "polygons" in item:
                item["polygons"] = [ring for j, ring in enumerate(item["polygons"])
                                    if (i, j) not in dropped]
        return True

    def _pending_backlog(self) -> int:
        """Buffered offline records awaiting a hub drain (0 without a buffer)."""
        return self.buffer_service.pending_count() if self.buffer_service else 0

    def detections_snapshot(self, include_frame: bool = True,
                            include_raw_frame: bool = False) -> dict:
        """Build the latest-detections snapshot.

        Owns the business logic for `/api/v1/detections/snapshot` and the gRPC
        `Snapshot` RPC — both are thin adapters over this. Returns a dict with
        the per-detection list (each tagged with the saved area its center
        falls in, or None, plus its normalized bbox corners), totals,
        model/runtime metadata, the processed frame as a base64 JPEG (when
        ``include_frame``) and the clean frame — no overlay — as ``raw_frame``
        (when ``include_raw_frame``; used by the hub for dataset ingest).
        """
        result = self.last_detection_result
        meta = {
            "task": self.application_task,
            # No runtime means no model — not the default path MODEL_PATH
            # falls back to after a deselect or on a blank device.
            "model": self._model_name() if self.model_manager is not None else "",
            "acceleration_type": self.acceleration_type(),
            "runtime_type": self.runtime_api(),
            "pending_backlog": self._pending_backlog(),
        }

        if result is None:
            return {"detections": [], "total": 0, "frame": None,
                    "raw_frame": None, **meta}

        detections, polygons_truncated = self._snapshot_items(result)

        frame_b64 = None
        if include_frame and result.processed_image is not None:
            frame_b64 = _encode_frame_b64(result.processed_image)

        raw_b64 = None
        if include_raw_frame and result.raw_image is not None:
            raw_b64 = _encode_frame_b64(result.raw_image)

        snapshot = {
            "detections": detections,
            "total": result.num_detections,
            "frame": frame_b64,
            "raw_frame": raw_b64,
            **meta,
        }
        # Classification top-k; absent for other tasks so their snapshot is
        # unchanged. Neither the offline buffer nor the hub reads it.
        if result.candidates is not None:
            snapshot["candidates"] = result.candidates
        # Only present when the polygon payload cap dropped rings.
        if polygons_truncated:
            snapshot["polygons_truncated"] = True
        return snapshot

    # ── Trigger control ────────────────────────────────────────────────────────

    def enable_trigger(self) -> None:
        """Enable frame processing trigger."""
        with self.lock:
            self.trigger_enabled = True
        logger.info("Trigger enabled")

    def disable_trigger(self) -> None:
        """Disable frame processing trigger (freeze last frame)."""
        with self.lock:
            self.trigger_enabled = False
        logger.info("Trigger disabled")

    def get_trigger_status(self) -> bool:
        """Return current trigger state."""
        return self.trigger_enabled

    # ── Detection counter ──────────────────────────────────────────────────────

    def get_detection_count(self) -> int:
        """Return accumulated detection count."""
        with self._count_lock:
            return self._detection_count

    def increment_detection_count(self, n: int = 1) -> None:
        """Increment the detection counter by *n*."""
        with self._count_lock:
            self._detection_count += n

    def reset_detection_count(self) -> None:
        """Reset the detection counter to zero."""
        with self._count_lock:
            self._detection_count = 0
