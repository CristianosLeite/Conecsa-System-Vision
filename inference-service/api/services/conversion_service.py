# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Conversion service - Manages async model conversion jobs.

Supports:
  .pt  → .onnx   (via subprocess → api._pt_onnx_converter)
  .onnx → .engine (via TensorRT worker IPC → _trt_engine_builder.build_engine)

The .pt -> .onnx step runs in a short-lived subprocess, not in the long-lived
service process, so the PyTorch caching allocator and ultralytics global state
are fully reclaimed by the OS when the converter exits.
"""
import contextlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, NamedTuple, Optional

logger = logging.getLogger(__name__)

# Timeout for the ONNX graph inspection subprocess (reads the output shapes).
_ONNX_INSPECT_TIMEOUT_SEC = 120


class ConverterOutput(NamedTuple):
    """What the .pt → .onnx converter subprocess reports on its last stdout line."""

    class_names: List[str]
    task: Optional[str] = None                       # ultralytics ``model.task``
    output_shapes: Optional[List[List[int]]] = None  # the exported graph's outputs
    imgsz: Optional[int] = None                      # the size the export used


def _output_stem(original_filename: str) -> str:
    """The stem conversion outputs are named after.

    Reduced to a basename so a crafted upload name (e.g. ``../../x.pt``) can
    never steer the .onnx/.engine outputs outside the model directory —
    save_model validates the name first, but the output naming must not be
    the one unguarded path.
    """
    return os.path.splitext(os.path.basename(original_filename))[0]

# Timeout for the .pt -> .onnx subprocess. YOLO export on Orin Nano is
# usually under 2 min; 10 min is a wide safety margin.
_PT_ONNX_TIMEOUT_SEC = int(os.environ.get("PT_ONNX_TIMEOUT_SEC", "600"))


class ConversionStatus(str, Enum):
    """Lifecycle states of a `.pt`→`.onnx`→`.engine` conversion job."""

    PENDING = "pending"
    CONVERTING_TO_ONNX = "converting_to_onnx"
    CONVERTING_TO_ENGINE = "converting_to_engine"
    DONE = "done"
    FAILED = "failed"


@dataclass
class ConversionJob:
    """State of one asynchronous model-conversion job (paths, status, progress)."""

    job_id: str
    original_filename: str
    pt_path: str
    onnx_path: str
    engine_path: str
    status: ConversionStatus = ConversionStatus.PENDING
    progress: int = 0          # 0–100
    message: str = ""
    error: Optional[str] = None
    engine_filename: Optional[str] = None  # basename after conversion
    # Export image size of a .pt → .onnx → .engine job (640 or 1280); None for
    # .onnx → .engine-only jobs, whose size
    # is baked into the ONNX graph. Informational: lets the gateway/UI/logs tell
    # a 640 build from a 1280 build.
    imgsz: Optional[int] = None
    # True when the upload named no size: the export uses the checkpoint's own
    # training size and ``imgsz`` (the task's default) only when it records
    # none; ``imgsz`` then becomes the size the export used.
    imgsz_from_checkpoint: bool = False
    # Geometry the weights were trained on, as declared by the uploader
    # ("frames", "tiles:auto", "tiles:<px>"; see ModelUploadMeta). Recorded in
    # the settings sidecar next to imgsz so activation can warn when it
    # disagrees with TILING_MODE. None = unknown (browser / federated uploads).
    train_geometry: Optional[str] = None
    # Task the model was uploaded as (resolved against the device's
    # application). The job fails when the converter discovers another task;
    # the result is recorded in the settings sidecar. None = unrecorded.
    task: Optional[str] = None
    # The enrollment package of a face gallery build (``.faces``); "" for
    # model conversions.
    faces_path: str = ""
    started_at: float = field(default_factory=time.time)  # UNIX timestamp (seconds)
    # Age is reported from the monotonic clock, never from ``started_at``: the
    # hub steps CLOCK_REALTIME on drift (see os-base/agent/time_agent.py).
    started_monotonic: float = field(default_factory=time.monotonic)

    @property
    def elapsed_secs(self) -> float:
        """Seconds since the job started, immune to wall-clock steps."""
        return max(0.0, time.monotonic() - self.started_monotonic)


def _convert_pt_to_onnx(pt_path: str, onnx_path: str, imgsz: int = 640,
                        from_checkpoint: bool = False) -> ConverterOutput:
    """
    Spawn api._pt_onnx_converter as a short-lived subprocess.
    PyTorch / ultralytics are imported only in that child, so their footprint
    (caching allocator, global state) dies with the child instead of pinning
    memory in the long-lived service process forever.

    Returns:
        The class names from model.names (empty on the torch fallback), the
        model's task and the exported graph's output shapes (``None`` when
        the child could not tell).
    """
    logger.info(f"Spawning _pt_onnx_converter subprocess: {pt_path} → {onnx_path} (imgsz={imgsz})")

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "api._pt_onnx_converter",
                "--pt", pt_path,
                "--onnx", onnx_path,
                "--imgsz", str(imgsz),
                *(["--imgsz-from-checkpoint"] if from_checkpoint else []),
            ],
            capture_output=True,
            text=True,
            timeout=_PT_ONNX_TIMEOUT_SEC,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f".pt -> .onnx converter timed out after {_PT_ONNX_TIMEOUT_SEC}s"
        ) from exc

    # Surface child stderr in our logs regardless of outcome (it carries the
    # ultralytics/torch progress + tracebacks).
    if result.stderr:
        for line in result.stderr.rstrip().splitlines():
            logger.info("[pt_onnx] %s", line)

    if result.returncode != 0:
        # Try to extract a structured error from the last stdout line.
        err_msg = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
        try:
            last = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
            parsed = json.loads(last)
            if isinstance(parsed, dict) and parsed.get("error"):
                err_msg = parsed["error"]
        except (ValueError, IndexError):
            pass
        raise RuntimeError(f".pt -> .onnx conversion failed: {err_msg}")

    # Parse the last stdout line (machine-readable JSON contract).
    parsed = _last_json_line(result.stdout) or {}
    cn = parsed.get("class_names")
    class_names = [str(n) for n in cn] if isinstance(cn, list) else []
    task = parsed.get("task") if isinstance(parsed.get("task"), str) else None
    size = parsed.get("imgsz")
    exported_imgsz = size if type(size) is int and size > 0 else None

    if not os.path.exists(onnx_path):
        raise RuntimeError(
            f"_pt_onnx_converter reported success but {onnx_path} not found"
        )

    logger.info(f"ONNX conversion complete: {onnx_path} ({len(class_names)} class names, "
                f"task {task or 'unknown'}, imgsz {exported_imgsz or imgsz})")
    return ConverterOutput(class_names, task, _shapes(parsed.get("output_shapes")),
                           exported_imgsz)


def _last_json_line(stdout: str) -> Optional[dict]:
    """The JSON object on the last stdout line of a converter child, if any."""
    stripped = stdout.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped.splitlines()[-1])
    except ValueError:
        logger.warning("_pt_onnx_converter stdout did not end with a JSON line")
        return None
    return parsed if isinstance(parsed, dict) else None


def _shapes(value) -> Optional[List[List[int]]]:
    """Validate an ``output_shapes`` payload: a list of int lists, else ``None``."""
    if not isinstance(value, list):
        return None
    try:
        return [[int(d) for d in shape] for shape in value]
    except (TypeError, ValueError):
        return None


def _inspect_onnx(onnx_path: str) -> Optional[List[List[int]]]:
    """Output shapes of an uploaded ONNX graph, read in a short-lived child.

    ``None`` when the graph cannot be read (the ``onnx`` package missing, a
    malformed file): the engine is then verified at activation only.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "api._pt_onnx_converter",
             "--inspect-onnx", onnx_path],
            capture_output=True, text=True, timeout=_ONNX_INSPECT_TIMEOUT_SEC, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(f"Could not inspect {onnx_path}: {exc}")
        return None
    if result.returncode != 0:
        logger.warning(f"Could not inspect {onnx_path}: {result.stderr.strip()[-500:]}")
        return None
    return _shapes((_last_json_line(result.stdout) or {}).get("output_shapes"))


def _discovered_task(task: Optional[str], shapes: Optional[List[List[int]]]) -> Optional[str]:
    """The task a converted model really is: ultralytics' word, else the graph's shape."""
    from conecsa_common.tasks import is_task

    if is_task(task):
        return task
    if shapes:
        from api.postprocess.contract import infer_task
        return infer_task(shapes)
    return None


def _build_engine_from_onnx(onnx_path: str, engine_path: str) -> None:
    """
    Request a TensorRT .engine build from an already-exported .onnx via the
    IPC worker. The builder logic lives in
    api.runtime_management._trt_engine_builder.build_engine.
    """
    logger.info(f"Requesting TensorRT engine build via worker: {onnx_path} → {engine_path}")

    workspace_mb = int(os.environ.get("TENSORRT_WORKSPACE_MB", "256"))

    from api.runtime_management.worker_client import get_worker_client  # type: ignore

    client = get_worker_client()
    client.build_engine(onnx_path, engine_path, workspace_mb=workspace_mb)

    if not os.path.exists(engine_path):
        raise RuntimeError(
            f"Worker reported success but engine file not found: {engine_path}"
        )

    logger.info(f"TensorRT engine saved: {engine_path}")


def _keep_weights_sidecar(pt_path: str, engine_path: str) -> None:
    """Move the source .pt into the model's weights sidecar slot (best-effort).

    A failure only costs the fine-tune/labeling option for this model, never
    the conversion — the .pt is removed either way so it cannot linger as a
    phantom list_models entry.
    """
    from api.services.model_service import ModelService

    dest = ModelService.weights_file_for_model(engine_path)
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        os.replace(pt_path, dest)
        logger.info(f"Kept training checkpoint as {dest}")
    except OSError as exc:
        logger.warning(f"Could not keep {pt_path} as {dest}: {exc}")
        _remove_file_safe(pt_path)


def _remove_file_safe(path: str) -> None:
    """Remove a file, logging a warning on failure instead of raising."""
    try:
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"Removed intermediate file: {path}")
    except OSError as exc:
        logger.warning(f"Could not remove {path}: {exc}")


class ConversionService:
    """Thread-safe async conversion service."""

    def __init__(self, event_service=None) -> None:
        self._jobs: Dict[str, ConversionJob] = {}
        self._lock = threading.Lock()
        self._event_service = event_service
        self._publication_guard = lambda _engine_filename: contextlib.nullcontext()

    # ── Public API ──

    def attach_publication_guard(self, guard) -> None:
        """Inject ``ModelService.publication``: a face gallery is published
        inside it, so a rebuild of the active model is reloaded before the job
        reports done."""
        self._publication_guard = guard

    @staticmethod
    def to_dict(job: ConversionJob) -> dict:
        """Serialize a job to the JSON dict the gateway returns to clients."""
        return {
            "job_id": job.job_id,
            "original_filename": job.original_filename,
            "status": job.status.value,
            "progress": job.progress,
            "message": job.message,
            "error": job.error,
            "engine_filename": job.engine_filename,
            "imgsz": job.imgsz,
            "train_geometry": job.train_geometry,
            "task": job.task,
            "started_at": job.started_at,
            "elapsed_secs": job.elapsed_secs,
        }

    def start_onnx_conversion(
        self,
        onnx_path: str,
        original_filename: str,
        model_directory: str,
        task: Optional[str] = None,
    ) -> ConversionJob:
        """
        Enqueue an async .onnx → .engine conversion (skips the .pt → .onnx step).

        Returns the ConversionJob immediately (status=pending).
        """
        job_id = str(uuid.uuid4())
        base = _output_stem(original_filename)
        engine_path = os.path.join(model_directory, f"{base}.engine")

        job = ConversionJob(
            job_id=job_id,
            original_filename=original_filename,
            pt_path="",          # not applicable
            onnx_path=onnx_path,
            engine_path=engine_path,
            task=task,
        )

        with self._lock:
            self._jobs[job_id] = job
            pending = self.to_dict(job)
        # Announce the job before the worker starts: the UI's overlay attaches
        # on this event, and it should not depend on thread scheduling.
        self._publish_event("conversion_changed", ["conversion"], data=pending)

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id,),
            daemon=True,
            name=f"conversion-{job_id[:8]}",
        )
        thread.start()
        logger.info(f"ONNX conversion job {job_id} started for {original_filename}")
        return job

    def start_pt_conversion(
        self,
        pt_path: str,
        original_filename: str,
        model_directory: str,
        imgsz: int = 640,
        train_geometry: Optional[str] = None,
        task: Optional[str] = None,
        imgsz_from_checkpoint: bool = False,
    ) -> ConversionJob:
        """
        Enqueue an async .pt → .onnx → .engine conversion.

        Returns the ConversionJob immediately (status=pending).
        """
        job_id = str(uuid.uuid4())
        base = _output_stem(original_filename)
        onnx_path = os.path.join(model_directory, f"{base}.onnx")
        engine_path = os.path.join(model_directory, f"{base}.engine")

        job = ConversionJob(
            job_id=job_id,
            original_filename=original_filename,
            pt_path=pt_path,
            onnx_path=onnx_path,
            engine_path=engine_path,
            imgsz=imgsz,
            imgsz_from_checkpoint=imgsz_from_checkpoint,
            train_geometry=train_geometry,
            task=task,
        )

        with self._lock:
            self._jobs[job_id] = job
            pending = self.to_dict(job)
        # Announce the job before the worker starts: the UI's overlay attaches
        # on this event, and it should not depend on thread scheduling.
        self._publish_event("conversion_changed", ["conversion"], data=pending)

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id,),
            daemon=True,
            name=f"conversion-{job_id[:8]}",
        )
        thread.start()
        logger.info(f"Conversion job {job_id} started for {original_filename} "
                    f"(imgsz={imgsz}, from_checkpoint={imgsz_from_checkpoint}, "
                    f"train_geometry={train_geometry})")
        return job

    def start_face_gallery(
        self,
        faces_path: str,
        original_filename: str,
        model_directory: str,
    ) -> ConversionJob:
        """
        Enqueue an async face gallery build from a ``.faces`` enrollment package.

        Reported through the same job states and events as a conversion (the
        engine step is the gallery build). Returns the job immediately.
        """
        job_id = str(uuid.uuid4())
        base = _output_stem(original_filename)
        job = ConversionJob(
            job_id=job_id,
            original_filename=original_filename,
            pt_path="",          # not applicable
            onnx_path="",        # not applicable
            engine_path=os.path.join(model_directory, f"{base}.engine"),
            task="face",
            faces_path=faces_path,
        )

        with self._lock:
            self._jobs[job_id] = job
            pending = self.to_dict(job)
        self._publish_event("conversion_changed", ["conversion"], data=pending)

        thread = threading.Thread(
            target=self._run_face_job,
            args=(job_id,),
            daemon=True,
            name=f"face-gallery-{job_id[:8]}",
        )
        thread.start()
        logger.info(f"Face gallery job {job_id} started for {original_filename}")
        return job

    @staticmethod
    def _check_declared_task(job: ConversionJob, discovered: Optional[str]) -> None:
        """Fail the job before the engine build when the model is another task.

        The converter's discovery wins over the upload's declaration:
        building an engine the device would then refuse to run only wastes
        minutes of GPU time.
        """
        if job.task and discovered and discovered != job.task:
            raise RuntimeError(
                f"'{job.original_filename}' is a '{discovered}' model, but it was "
                f"uploaded as a '{job.task}' model")

    def get_job(self, job_id: str) -> Optional[ConversionJob]:
        """Return a job by id, or ``None`` if unknown."""
        with self._lock:
            return self._jobs.get(job_id)

    def get_active_jobs(self) -> List[ConversionJob]:
        """Return all jobs that have not yet reached a terminal state."""
        terminal = {ConversionStatus.DONE, ConversionStatus.FAILED}
        with self._lock:
            return [j for j in self._jobs.values() if j.status not in terminal]

    # ── Internal – conversion steps ──

    # ── Internal – job status helper ──

    def _set_status(
        self,
        job_id: str,
        status: ConversionStatus,
        progress: int,
        message: str,
        error: Optional[str] = None,
        engine_filename: Optional[str] = None,
    ) -> None:
        """Update a job's status/progress and publish a ``conversion_changed`` event."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = status
            job.progress = progress
            job.message = message
            if error is not None:
                job.error = error
            if engine_filename is not None:
                job.engine_filename = engine_filename
            event_data = self.to_dict(job)
        self._publish_event(
            "conversion_changed",
            ["conversion"],
            data=event_data,
        )

    def _publish_event(self, event_type: str, keys: List[str], data: Optional[dict] = None) -> None:
        """Publish an event via the EventService (no-op if none is wired)."""
        if self._event_service is None:
            return
        try:
            self._event_service.publish(event_type, keys=keys, source="conversion", data=data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not publish conversion event '%s': %s", event_type, exc)

    # ── Internal – worker thread ──

    def _run_job(self, job_id: str) -> None:
        """
        Worker thread for all conversion paths.

        job.imgsz=None  → .onnx → .engine only
        job.imgsz=<int> → .pt  → .onnx → .engine
        """
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return
        imgsz = job.imgsz

        try:
            # ── Step 1 (PT path only): .pt → .onnx ────────────────────
            if imgsz is not None:
                self._set_status(job_id, ConversionStatus.CONVERTING_TO_ONNX,
                                 progress=5, message="Converting .pt to ONNX…")
                exported = _convert_pt_to_onnx(job.pt_path, job.onnx_path, imgsz,
                                               from_checkpoint=job.imgsz_from_checkpoint)
                if exported.imgsz is not None:
                    # The size the export really used (the checkpoint's own
                    # when the upload named none) is what the job and the
                    # settings sidecar report.
                    imgsz = exported.imgsz
                    with self._lock:
                        job.imgsz = imgsz
                discovered = _discovered_task(exported.task, exported.output_shapes)
                self._check_declared_task(job, discovered)
                class_names = exported.class_names
                if class_names:
                    from api.repositories.class_labels_repository import ClassLabelsRepository
                    from api.services.model_service import ModelService
                    classes_path = ModelService.classes_file_for_model(job.engine_path)
                    ClassLabelsRepository(classes_path).save_labels(class_names)
                    logger.info(
                        f"Auto-saved {len(class_names)} class labels to {classes_path}: {class_names}"
                    )
                    self._publish_event(
                        "classes_changed",
                        ["classes"],
                        data={
                            "model": os.path.basename(job.engine_path),
                            "count": len(class_names),
                        },
                    )
                self._set_status(job_id, ConversionStatus.CONVERTING_TO_ONNX,
                                 progress=40, message="ONNX export complete. Building TensorRT engine…")
                tmp_files = [job.onnx_path]
                engine_progress = 45
            else:
                discovered = _discovered_task(None, _inspect_onnx(job.onnx_path))
                self._check_declared_task(job, discovered)
                tmp_files = [job.onnx_path]
                engine_progress = 10

            # ── Step 2: .onnx → .engine ───────────────────────────────
            self._set_status(job_id, ConversionStatus.CONVERTING_TO_ENGINE,
                             progress=engine_progress,
                             message="Building TensorRT .engine (this may take several minutes)…")
            _build_engine_from_onnx(job.onnx_path, job.engine_path)

            # Record the export size, the training geometry and the task next
            # to the engine (the settings file is otherwise created on first
            # activation, which is also where geometry and task are checked).
            task = job.task or discovered
            if imgsz is not None or task:
                from api.services.model_service import ModelService
                from api.services.model_settings_service import ModelSettingsService
                ModelSettingsService.record_training(
                    ModelService.settings_file_for_model(job.engine_path), imgsz,
                    job.train_geometry, task=task)

            # ── Keep the checkpoint, drop the intermediates ────────────
            # The .pt becomes the model's weights sidecar (fine-tune base /
            # labeling assistant for the training-service); it must not stay
            # beside the engine, where list_models would show it as a model.
            if imgsz is not None and job.pt_path:
                _keep_weights_sidecar(job.pt_path, job.engine_path)
            for path in tmp_files:
                _remove_file_safe(path)

            engine_filename = os.path.basename(job.engine_path)
            self._set_status(job_id, ConversionStatus.DONE, progress=100,
                             message=f"Conversion complete. Engine saved as '{engine_filename}'.",
                             engine_filename=engine_filename)
            self._publish_event(
                "models_changed",
                ["models"],
                data={"model": engine_filename},
            )
            logger.info(f"Job {job_id} completed → {engine_filename}")

        except Exception as exc:
            logger.exception(f"Job {job_id} failed: {exc}")
            self._set_status(job_id, ConversionStatus.FAILED, progress=0,
                             message="Conversion failed.", error=str(exc))
            # A failed job must not leave intermediates behind: orphaned
            # .pt/.onnx files reappear in list_models as phantom entries.
            for path in (job.pt_path, job.onnx_path):
                if path:
                    _remove_file_safe(path)

    def _run_face_job(self, job_id: str) -> None:
        """Worker thread of a face gallery build (see ``face_gallery_builder``)."""
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return

        def progress(percent: int, message: str) -> None:
            self._set_status(job_id, ConversionStatus.CONVERTING_TO_ENGINE,
                             progress=percent, message=message)

        try:
            from api.config import Config
            from api.services.face_gallery_builder import FaceGalleryBuilder

            progress(2, "Reading the enrollment photos…")
            builder = FaceGalleryBuilder(Config(), os.path.dirname(job.engine_path),
                                         _build_engine_from_onnx, progress,
                                         publish_guard=self._publication_guard)
            # The builder publishes the engine, the gallery, the names and the
            # settings sidecar (task "face") as one set; when that model is the
            # active one the guard reloads it, so "done" below means live.
            summary = builder.build(job.faces_path, job.engine_path)
            _remove_file_safe(job.faces_path)

            engine_filename = os.path.basename(job.engine_path)
            self._set_status(job_id, ConversionStatus.DONE, progress=100,
                             message=summary.message(), engine_filename=engine_filename)
            self._publish_event("classes_changed", ["classes"],
                                data={"model": engine_filename, "count": summary.people})
            self._publish_event("models_changed", ["models"], data={"model": engine_filename})
            logger.info(f"Face gallery job {job_id} completed → {engine_filename} "
                        f"({summary.message()})")
        except Exception as exc:
            logger.exception(f"Face gallery job {job_id} failed: {exc}")
            self._set_status(job_id, ConversionStatus.FAILED, progress=0,
                             message="Face gallery build failed.", error=str(exc))
            _remove_file_safe(job.faces_path)
            # Only what this job staged: a model of the same name that was
            # already published keeps every one of its files.
            from api.services.face_gallery_builder import discard_staging
            discard_staging(job.engine_path)
