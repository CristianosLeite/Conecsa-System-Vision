# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Model service - Manages model loading and switching.
"""
import logging
import os
from contextlib import contextmanager
from threading import Lock, RLock
from typing import Callable, Generator, List, NamedTuple, Optional, Sequence, Tuple

from conecsa_common import atomic_write_bytes

from ..config import Config
from ..model_paths import ALLOWED_MODEL_EXTENSIONS, validate_model_filename
from ..models.detection_models import ModelInfo
from ..runtime_management import RuntimeFactory
from .errors import PreconditionFailed
from .model_settings_service import ModelSettingsService

logger = logging.getLogger(__name__)

# Extensions that require async conversion before they can be used
_PT_EXTENSIONS = {'.pt'}
# Subdirectory of the model directory holding each model's training checkpoint.
WEIGHTS_SUBDIR = "weights"
_ONNX_EXTENSIONS = {'.onnx'}
# A face enrollment package (training-service): built into a face model.
_FACES_EXTENSIONS = {'.faces'}
FACE_TASK = "face"


class ModelService:
    """Service for managing ML models."""

    # Filename (within the model directory) used to remember the last-selected
    # model across restarts. The TensorRT worker warm-up in composition.py reads
    # it too, so it does not pre-load a different engine over the restored one.
    STATE_FILENAME = ".current_model"

    def __init__(self, config: Config, model_directory: str):
        """
        Initialize the model service.

        Args:
            config: Configuration instance
            model_directory: Directory containing models
        """
        self.config = config
        self.model_directory = model_directory
        # Empty until a model is activated (or the boot fallback loads the
        # legacy default engine): status must not name a model that is not loaded.
        self.current_model = ""
        self.lock = Lock()
        # Serializes whole model-lifecycle operations (upload, conversion
        # enqueue, select, activate, delete) against each other, so e.g. a
        # delete cannot interleave with an activation of the same file.
        # Reentrant because process_upload nests save_model/activate_model.
        self._op_lock = RLock()
        self._detection_service = None  # Wired via attach_detection_service()
        self._area_service = None       # Wired via attach_area_service()
        self._settings_service = None   # Wired via attach_settings_service()
        self._conversion_service = None  # Wired via attach_conversion_service()
        self._application_service = None  # Wired via attach_application_service()
        self._labeling_service = None   # Wired via attach_labeling_service()
        self._event_service = None      # Wired via attach_event_service()
        # True between a successful release_runtime() and the end of the
        # handover — resume_runtime(), with or without restarting detection,
        # or an explicit start_detection(): the GPU belongs to the
        # training-service, so the application type cannot change. Read and
        # written under _op_lock, which also serializes the release against
        # set_task() and the start.
        self._runtime_released = False
        self._state_file = os.path.join(model_directory, self.STATE_FILENAME)

        # Ensure model directory exists
        os.makedirs(model_directory, exist_ok=True)

    # ── Helpers ──

    @staticmethod
    def classes_file_for_model(model_path: str) -> str:
        """Return the per-model classes.txt file path (sibling of the model file).

        Example: /data/models/weights.engine -> /data/models/weights.txt
        """
        base, _ = os.path.splitext(model_path)
        return f"{base}.txt"

    @staticmethod
    def areas_file_for_model(model_path: str) -> str:
        """Return the per-model detection-areas file path (sibling of the model).

        Example: /data/models/weights.engine -> /data/models/weights.areas.json
        """
        base, _ = os.path.splitext(model_path)
        return f"{base}.areas.json"

    @staticmethod
    def settings_file_for_model(model_path: str) -> str:
        """Return the per-model settings file path (thresholds + camera config).

        Example: /data/models/weights.engine -> /data/models/weights.settings.json
        """
        base, _ = os.path.splitext(model_path)
        return f"{base}.settings.json"

    @staticmethod
    def weights_file_for_model(model_path: str) -> str:
        """Return the per-model training-checkpoint sidecar path.

        The .pt a conversion started from is kept under a ``weights/``
        subdirectory rather than beside the engine: ``list_models`` matches
        files by extension in the model directory itself, so a sibling
        ``.pt`` would show up as a second, unselectable model. The
        training-service fetches it to fine-tune from the model or to run it
        as a labeling assistant.

        Example: /data/models/weights.engine -> /data/models/weights/weights.pt
        """
        directory, filename = os.path.split(model_path)
        base, _ = os.path.splitext(filename)
        return os.path.join(directory, WEIGHTS_SUBDIR, f"{base}.pt")

    @staticmethod
    def gallery_file_for_model(model_path: str) -> str:
        """Return the face-recognition gallery sidecar path.

        Example: /data/models/staff.engine -> /data/models/staff.gallery.npz
        """
        from ..postprocess.face import gallery_file_for_model
        return gallery_file_for_model(model_path)

    @classmethod
    def model_artifacts(cls, model_path: str) -> List[str]:
        """Every file a model owns: the binary plus its sidecars.

        Deletion must remove the whole set — sidecars are found by basename,
        so a leftover .txt/.areas.json/.settings.json would be silently
        adopted by a later model uploaded under the same name (wrong labels
        on live detections), and a stale weights sidecar would let a fine-tune
        start from a model that no longer exists.

        A ``.faces`` package owns nothing but itself: its sidecars belong to
        the ``.engine`` of the same name, which must survive the package's
        removal after an interrupted or failed build.
        """
        if os.path.splitext(model_path)[1].lower() in _FACES_EXTENSIONS:
            return [model_path]
        return [
            model_path,
            cls.classes_file_for_model(model_path),
            cls.areas_file_for_model(model_path),
            cls.settings_file_for_model(model_path),
            cls.weights_file_for_model(model_path),
            cls.gallery_file_for_model(model_path),
        ]

    def attach_detection_service(self, detection_service) -> None:
        """Inject DetectionService for the activate_model lifecycle.

        Done post-construction to avoid a constructor-time circular dep
        with the Application container.
        """
        self._detection_service = detection_service

    def attach_area_service(self, area_service) -> None:
        """Inject DetectionAreaService so model selection switches the
        per-model detection-areas file."""
        self._area_service = area_service

    def attach_settings_service(self, settings_service) -> None:
        """Inject ModelSettingsService so model selection switches the
        per-model thresholds + camera settings file."""
        self._settings_service = settings_service

    @property
    def op_lock(self):
        """The model lifecycle lock, for other services that change per-model state."""
        return self._op_lock

    SETTING_SAVED = "saved"
    SETTING_INVALID = "invalid"
    SETTING_UNSAVED = "unsaved"

    def update_setting(self, name: str, set_value: Callable[[], bool]) -> str:
        """Change one per-model ``Config`` value and save it with the active model.

        ``update_settings`` for a single field; see there.
        """
        outcome, _ = self.update_settings([(name, set_value)])
        return outcome

    def update_settings(
        self, changes: Sequence[Tuple[str, Callable[[], bool]]],
    ) -> Tuple[str, Optional[str]]:
        """Change several per-model ``Config`` values as one transaction.

        Runs under ``_op_lock`` so a concurrent select or activation cannot
        swap the settings file between the changes and the save, which would
        store the values with the wrong model. Each ``set_value`` validates
        and assigns ``config.<name>``; they are applied in order and saved in
        one write. Returns ``(SETTING_INVALID, name)`` when ``name``'s value
        was refused (every field is left as it was), ``(SETTING_UNSAVED,
        None)`` when the file could not be written (every previous value is
        restored, so live state matches the file), and ``(SETTING_SAVED,
        None)`` otherwise.
        """
        with self._op_lock:
            previous = [(name, getattr(self.config, name)) for name, _ in changes]

            def restore() -> None:
                for name, value in previous:
                    setattr(self.config, name, value)

            for name, set_value in changes:
                if not set_value():
                    restore()
                    return self.SETTING_INVALID, name
            if self._settings_service is None or self._settings_service.save(
                    only=[name for name, _ in changes]):
                return self.SETTING_SAVED, None
            restore()
            return self.SETTING_UNSAVED, None

    def attach_conversion_service(self, conversion_service) -> None:
        """Inject ConversionService so process_upload can enqueue the async
        .pt/.onnx → .engine conversion jobs."""
        self._conversion_service = conversion_service

    @contextmanager
    def publication(self, engine_filename: str) -> Generator[None, None, None]:
        """Publish a model's files as one lifecycle operation.

        The body replaces ``engine_filename`` and its sidecars on disk. It
        runs under ``_op_lock``, and when that model is the active one it is
        reloaded before the lock is released: the running strategy would
        otherwise keep the previous gallery and names while a class rename
        already wrote to the new files.

        Raises:
            RuntimeError: the rebuilt active model could not be reloaded
                (``activate_model`` restored what it could).
        """
        with self._op_lock:
            yield
            if engine_filename != self.current_model or self._detection_service is None:
                return
            success, result, _was_running = self.activate_model(engine_filename)
            if not success:
                raise RuntimeError(
                    f"'{engine_filename}' was rebuilt but could not be reloaded: {result}")
            logger.info(f"Reloaded the active model '{engine_filename}' after its rebuild")

    def attach_application_service(self, application_service) -> None:
        """Inject ApplicationService: activation refuses models of another
        task, and set_task() persists the device's application through it."""
        self._application_service = application_service

    def attach_labeling_service(self, labeling_service) -> None:
        """Inject LabelingService so the runtime release and an application
        change can drop the labeling engine."""
        self._labeling_service = labeling_service

    def attach_event_service(self, event_service) -> None:
        """Inject EventService so an application change reaches every client."""
        self._event_service = event_service

    @property
    def runtime_released(self) -> bool:
        """True while the GPU is handed over to the training-service."""
        return self._runtime_released

    # ── Last-selected-model persistence ──

    def _persist_current_model(self, model_name: str) -> None:
        """Durably write the last-selected model name to disk (best-effort).

        Called only after the runtime proved it can load the model, so a
        broken selection can never become the boot default; atomic so a power
        cut can never leave a truncated state file.
        """
        try:
            atomic_write_bytes(self._state_file,
                               model_name.strip().encode("utf-8"), mode=0o644)
        except OSError as ex:
            logger.warning(f"Could not persist current model '{model_name}': {ex}")

    def load_persisted_current_model(self) -> str:
        """Read the last-selected model name from disk, or '' if none."""
        try:
            with open(self._state_file, "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            return ""
        except OSError as ex:
            logger.warning(f"Could not read persisted current model: {ex}")
            return ""

    def _forget_persisted_current_model(self) -> None:
        """Remove the boot-default marker (its absence is a valid state)."""
        try:
            os.remove(self._state_file)
        except FileNotFoundError:
            pass
        except OSError as ex:
            logger.warning(f"Could not remove the persisted model selection: {ex}")

    def heal_persisted_selection(self) -> bool:
        """Drop a boot-default selection that can no longer be activated.

        Runs once at boot, after the application type is known and before
        anything reads ``.current_model`` (the TensorRT warm-up thread and
        ``Application.initialize``). The selection is cleared, with a
        warning, when no application type is chosen, when the model file is
        gone, or when the model's task differs from the application's — so
        the device neither auto-starts a model it may not run nor crashes
        decoding one task's tensors with another task's postprocess. Returns
        True when the marker was removed.
        """
        name = self.load_persisted_current_model()
        if not name:
            return False
        app = self._application_service
        path = os.path.join(self.model_directory, name)
        reason = ""
        if app is not None and app.task is None:
            reason = "no application type is selected"
        elif not os.path.exists(path):
            reason = "its file is missing"
        elif app is not None and self.task_of(name) != app.task:
            reason = (f"it is a '{self.task_of(name)}' model and the device runs "
                      f"the '{app.task}' application")
        if not reason:
            return False
        logger.warning("Clearing the persisted model selection '%s': %s", name, reason)
        self._forget_persisted_current_model()
        return True

    def list_models(self) -> List[ModelInfo]:
        """
        List all available models.

        Returns:
            List of ModelInfo objects
        """
        models: List[ModelInfo] = []

        if not os.path.exists(self.model_directory):
            logger.warning(f"Model directory does not exist: {self.model_directory}")
            return models

        converting = self._converting_tasks()
        for filename in os.listdir(self.model_directory):
            # Case-insensitive, like validate_model_filename: an accepted
            # upload named "yard.ENGINE" must not vanish from the list.
            if filename.lower().endswith(ALLOWED_MODEL_EXTENSIONS):
                try:
                    file_path = os.path.join(self.model_directory, filename)
                    models.append(ModelInfo(
                        name=filename,
                        path=file_path,
                        size=os.path.getsize(file_path),
                        modified=os.path.getmtime(file_path),
                        is_active=(filename == self.current_model),
                        has_weights=(
                            not filename.lower().endswith(tuple(_FACES_EXTENSIONS))
                            and os.path.isfile(self.weights_file_for_model(file_path))),
                        task=converting.get(filename) or self.task_of(filename),
                    ))
                except Exception as ex:
                    logger.warning(f"Error reading model file {filename}: {ex}")
                    continue

        return models

    def _converting_tasks(self) -> dict[str, str]:
        """Declared task of the files an active conversion job owns, by basename.

        The task reaches the settings sidecar only when the engine is built, so
        until then a ``.pt``/``.onnx`` being converted would list as the
        default ``detect``; the job already knows the task it was uploaded as.
        """
        if self._conversion_service is None:
            return {}
        try:
            jobs = self._conversion_service.get_active_jobs()
        except Exception as ex:  # noqa: BLE001 - listing must not fail on it
            logger.warning(f"Could not read active conversions: {ex}")
            return {}
        tasks: dict[str, str] = {}
        for job in jobs:
            if not job.task:
                continue
            for path in (job.pt_path, job.onnx_path, getattr(job, "faces_path", "")):
                if path:
                    tasks[os.path.basename(path)] = job.task
        return tasks

    def _job_owned_names(self) -> set:
        """Basenames an active conversion job reads or will write.

        The inputs (a ``.pt``/``.onnx`` being converted, a ``.faces``
        package being enrolled) and the ``.engine`` the job publishes at the
        end: deleting or re-uploading any of them under a running job would
        make it fail, build from another upload's bytes, or write a model
        back after its delete.
        """
        if self._conversion_service is None:
            return set()
        try:
            jobs = self._conversion_service.get_active_jobs()
        except Exception as ex:  # noqa: BLE001 - a guard must not fail on it
            logger.warning(f"Could not read active conversions: {ex}")
            return set()
        names = set()
        for job in jobs:
            for path in (job.pt_path, job.onnx_path, getattr(job, "faces_path", ""),
                         getattr(job, "engine_path", "")):
                if path:
                    names.add(os.path.basename(path))
        return names

    def _owned_by_active_job(self, filename: str) -> bool:
        """True when ``filename`` or the engine its upload would publish is a job's."""
        owned = self._job_owned_names()
        stem = os.path.splitext(filename)[0]
        return filename in owned or f"{stem}.engine" in owned

    def model_file_path(self, model_name: str) -> str:
        """
        Resolve a model name to its file path for download.

        Returns "" when the name is not a plain model filename (path
        traversal), has a disallowed extension, or the file is missing.

        A ``.faces`` enrollment package is never resolved: it is accepted for
        upload and listed while its gallery builds, but it holds the photos
        the operator enrolled, and those must not leave the device — the
        gallery built from them does not either.
        """
        path, error = validate_model_filename(model_name, self.model_directory)
        if error:
            return ""
        if os.path.splitext(path)[1].lower() in _FACES_EXTENSIONS:
            return ""
        return path if os.path.isfile(path) else ""

    def weights_file_path(self, model_name: str) -> str:
        """Resolve a model name to its training-checkpoint sidecar for download.

        Returns "" when the name is invalid or the model has no sidecar. A
        ``.faces`` package is not a model and never has one: resolving it
        would hand out a same-stem checkpoint of an earlier model.
        """
        path, error = validate_model_filename(model_name, self.model_directory)
        if error:
            return ""
        if os.path.splitext(path)[1].lower() in _FACES_EXTENSIONS:
            return ""
        weights = self.weights_file_for_model(path)
        return weights if os.path.isfile(weights) else ""

    def task_of(self, model_name: str) -> str:
        """The task recorded in a model's settings sidecar (``"detect"`` when none)."""
        path = os.path.join(self.model_directory, os.path.basename(model_name))
        return ModelSettingsService.task_of(self.settings_file_for_model(path))

    def _check_task(self, model_name: str) -> None:
        """Refuse a model whose task is not the device's application task.

        A task mismatch would make the postprocess decode one task's tensors
        with another task's rules, so it is refused outright,
        before anything is stopped. Unknown or missing files are left to
        ``select_model``'s own validation.
        """
        app = self._application_service
        if app is None:
            return
        path, error = validate_model_filename(model_name, self.model_directory)
        if error or not os.path.exists(path):
            return
        device_task = app.task
        if device_task is None:
            raise PreconditionFailed(
                "No application type is selected; choose one before activating a model")
        model_task = self.task_of(model_name)
        if model_task != device_task:
            raise PreconditionFailed(
                f"Model '{model_name}' is a '{model_task}' model, but the device runs "
                f"the '{device_task}' application")

    def save_model(self, filename: str, file_data) -> Tuple[bool, str, str]:
        """
        Save an uploaded model file.

        Args:
            filename: Name of the model file
            file_data: File object with model data

        Returns:
            Tuple of (success, model_path, error_message)
        """
        with self._op_lock:
            model_path, error = validate_model_filename(filename, self.model_directory)
            if error:
                return False, "", error

            # Never overwrite the file the live detector is reading from; select
            # another model first, or upload under a different name. (Conversion
            # outputs replacing the active engine remain allowed — that is the
            # training handover flow, and the detector is reloaded afterwards. A
            # default current_model whose file never existed protects nothing.)
            if filename == self.current_model and os.path.exists(model_path):
                return False, "", "Cannot overwrite the currently active model"

            try:
                file_data.save(model_path)
                logger.info(f"Model saved successfully: {model_path}")
                # An engine/ONNX replacing a model that was converted from a
                # .pt must not inherit that checkpoint: has_weights would then
                # point a fine-tune at weights the new engine never came from.
                # A .pt upload keeps it until its own conversion replaces it.
                if os.path.splitext(filename)[1].lower() not in _PT_EXTENSIONS:
                    self._discard_weights_sidecar(model_path)
                return True, model_path, ""
            except Exception as ex:
                logger.error(f"Error saving model: {ex}")
                return False, "", str(ex)

    @classmethod
    def _discard_weights_sidecar(cls, model_path: str) -> None:
        """Remove the model's training-checkpoint sidecar, if any (best-effort)."""
        weights = cls.weights_file_for_model(model_path)
        try:
            os.remove(weights)
            logger.info(f"Dropped stale training checkpoint {weights}")
        except FileNotFoundError:
            pass
        except OSError as ex:
            logger.warning(f"Could not remove stale training checkpoint {weights}: {ex}")

    def process_upload(self, filename: str, file_data, imgsz: int = 640,
                       train_geometry: Optional[str] = None,
                       task: Optional[str] = None,
                       imgsz_from_checkpoint: bool = False) -> Tuple[dict, int]:
        """Full upload lifecycle — owns the business logic so both the REST
        controller and the gRPC servicer are thin adapters over it.

        Saves the uploaded file, then branches on extension:
          - .pt   → enqueue async .pt → .onnx → .engine conversion (202)
          - .onnx → enqueue async .onnx → .engine conversion (202)
          - other → activate immediately (load into the live detector) (200)

        Args:
            filename: Uploaded file name (drives the extension branch).
            file_data: File-like object exposing ``.save(path)`` (Flask
                FileStorage over HTTP, or a bytes adapter over gRPC).
            imgsz: Image size for .pt conversion (ignored otherwise).
            train_geometry: Declared training geometry (``"frames"``,
                ``"tiles:auto"``, ``"tiles:<px>"``) recorded in the model's
                settings sidecar by the .pt conversion; ``None`` = unknown.
            task: The model's task, already resolved against the device's
                application (``ApplicationService.resolve_upload_task``).
                Recorded in the settings sidecar; a conversion fails when the
                model turns out to be another task, an engine is verified
                when it is activated. ``None`` = unrecorded (``"detect"``).
            imgsz_from_checkpoint: The upload named no size: a .pt is exported
                at the size it was trained at, ``imgsz`` only when the
                checkpoint records none.

        Returns:
            (body, http_status) — body is the JSON-serializable response the
            gateway relays verbatim; http_status is the intended HTTP status
            (202 converting / 200 loaded / 4xx-5xx error).
        """
        if not filename:
            return {"error": "No file selected"}, 400

        if self._conversion_service is None:
            raise RuntimeError(
                "ModelService.process_upload requires attach_conversion_service() "
                "to be called during application wiring."
            )

        # A face model is only ever built from an enrollment package, and a
        # package only builds a face model.
        is_package = os.path.splitext(filename)[1].lower() in _FACES_EXTENSIONS
        if is_package and task != FACE_TASK:
            return {"error": "A .faces enrollment package builds a face recognition model; "
                             f"this device runs the '{task}' application"}, 400
        if task == FACE_TASK and not is_package:
            return {"error": "Face recognition models are built from an enrollment dataset "
                             "on the device, not uploaded"}, 400

        logger.info(f"Uploading model: {filename}")
        # One lifecycle operation: nothing (a delete, another upload) may
        # interleave between the save and the activation/conversion enqueue.
        with self._op_lock:
            # A second upload of a name a running job reads would overwrite
            # the bytes under it; one that publishes the same engine would
            # race it on the outputs.
            if self._owned_by_active_job(filename):
                return {"error": f"'{filename}' is being built into a model; upload it "
                                 "again after that finishes"}, 409
            success, model_path, error_message = self.save_model(filename, file_data)
            if not success:
                logger.error(error_message)
                return {"error": error_message}, 500

            file_ext = os.path.splitext(filename)[1].lower()

            # ── .pt model: start async .pt → .onnx → .engine conversion ──────
            if file_ext in _PT_EXTENSIONS:
                job = self._conversion_service.start_pt_conversion(
                    pt_path=model_path,
                    original_filename=filename,
                    model_directory=self.model_directory,
                    imgsz=imgsz,
                    train_geometry=train_geometry,
                    task=task,
                    imgsz_from_checkpoint=imgsz_from_checkpoint,
                )
                logger.info(f"Async conversion job {job.job_id} started for {filename}")
                return {
                    "status": "converting",
                    "message": (
                        f"'{filename}' received. Converting to TensorRT engine "
                        "in the background. Poll the conversion status endpoint for progress."
                    ),
                    "job_id": job.job_id,
                    "filename": filename,
                }, 202

            # ── .onnx model: start async .onnx → .engine conversion ──────────
            if file_ext in _ONNX_EXTENSIONS:
                job = self._conversion_service.start_onnx_conversion(
                    onnx_path=model_path,
                    original_filename=filename,
                    model_directory=self.model_directory,
                    task=task,
                )
                logger.info(f"Async ONNX conversion job {job.job_id} started for {filename}")
                return {
                    "status": "converting",
                    "message": (
                        f"'{filename}' received. Building TensorRT engine "
                        "in the background. Poll the conversion status endpoint for progress."
                    ),
                    "job_id": job.job_id,
                    "filename": filename,
                }, 202

            # ── .faces package: build the face gallery ───────────────────────
            if file_ext in _FACES_EXTENSIONS:
                job = self._conversion_service.start_face_gallery(
                    faces_path=model_path,
                    original_filename=filename,
                    model_directory=self.model_directory,
                )
                logger.info(f"Face gallery job {job.job_id} started for {filename}")
                return {
                    "status": "converting",
                    "message": (
                        f"'{filename}' received. Building the face gallery "
                        "in the background. Poll the conversion status endpoint for progress."
                    ),
                    "job_id": job.job_id,
                    "filename": filename,
                }, 202

            # ── Other formats: load immediately ──────────────────────────────
            # A prebuilt engine cannot be inspected before it loads: record
            # the declared task now, and let activation verify the engine's
            # outputs against it.
            if task:
                ModelSettingsService.record_training(
                    self.settings_file_for_model(model_path), None, task=task)
            try:
                success, result, _was_running = self.activate_model(filename)
            except PreconditionFailed as ex:
                logger.error(str(ex))
                return {"error": str(ex)}, 409
            if not success:
                logger.error(result)
                return {"error": result}, 500

            logger.info(f"Model {filename} uploaded and loaded successfully")
            return {
                "status": "success",
                "message": "Model uploaded and loaded successfully",
                "model": filename,
                "path": result,
            }, 200

    def select_model(self, model_name: str) -> Tuple[bool, str]:
        """
        Switch the service's state to *model_name* — without persisting it.

        Persistence is activate_model's job, after the runtime proved it can
        actually load the model; that keeps a broken selection from becoming
        the boot default.

        Args:
            model_name: Name of the model file

        Returns:
            Tuple of (success, model_path or error_message)
        """
        model_path, error = validate_model_filename(model_name, self.model_directory)
        if error:
            return False, error

        if not os.path.exists(model_path):
            return False, f"Model '{model_name}' not found"

        if os.path.splitext(model_path)[1].lower() in _FACES_EXTENSIONS:
            return False, (f"'{model_name}' is a face enrollment package; select the face "
                           "model built from it")

        if not RuntimeFactory.is_supported_model(model_path):
            return False, f"No supported runtime available for model '{model_name}'"

        classes_path = self.classes_file_for_model(model_path)
        areas_path = self.areas_file_for_model(model_path)
        settings_path = self.settings_file_for_model(model_path)

        with self.lock:
            self.current_model = model_name
            self.config.MODEL_PATH = model_path
            self.config.CLASSES_FILE_PATH = classes_path

        # Switch all per-model scoped state to this model's sibling files.
        # Areas and settings are switched before the detector is (re)initialized
        # by activate_model, so the new model boots with its own tuning.
        if self._area_service is not None:
            self._area_service.switch_storage(areas_path)
        if self._settings_service is not None:
            self._settings_service.switch_model(settings_path)

        logger.info(
            f"Model selected: {model_name} "
            f"(classes: {classes_path}, areas: {areas_path}, settings: {settings_path})"
        )
        return True, model_path

    class _Snapshot(NamedTuple):
        """The state activate_model restores when the new runtime fails."""
        current_model: str
        model_path: str
        classes_path: str

    def activate_model(self, model_name: str) -> Tuple[bool, str, bool]:
        """
        Full lifecycle to switch the active model in a live system —
        transactional: on any failure the previous model, its scoped stores,
        and its running state are restored, and nothing is persisted.

        Stops the detection loop (if running), selects the new model and its
        per-model classes file, re-initializes the detector, persists the
        selection only after that succeeded, and restarts the loop if it had
        been running.

        Args:
            model_name: Name of the model file to activate.

        Returns:
            A ``(success, model_path_or_error, was_running)`` tuple. On failure
            (``success=False``) ``model_path_or_error`` holds the error message
            and the previous model is active (and running again when it was
            before). On success ``model_path_or_error`` is the resolved model
            path and ``was_running`` says whether the loop was running
            beforehand (and has now been restarted).

        Raises:
            RuntimeError: if attach_detection_service() was not called.
            PreconditionFailed: the model's task is not the device's
                application task (nothing was stopped or switched).
        """
        if self._detection_service is None:
            raise RuntimeError(
                "ModelService.activate_model requires attach_detection_service() "
                "to be called during application wiring."
            )

        with self._op_lock:
            self._check_task(model_name)
            with self.lock:
                snapshot = self._Snapshot(
                    current_model=self.current_model,
                    model_path=self.config.MODEL_PATH,
                    classes_path=self.config.CLASSES_FILE_PATH,
                )

            was_running = self._detection_service.stop()

            success, result = self.select_model(model_name)
            if not success:
                # Nothing was switched; the old runtime is still loaded.
                if was_running:
                    self._detection_service.start()
                return False, result, was_running

            try:
                self._detection_service.initialize()
            except Exception as ex:  # noqa: BLE001 - surfaced to caller
                logger.error(
                    f"Failed to initialize detection after selecting "
                    f"'{model_name}': {ex}")
                self._rollback(snapshot, was_running)
                return False, f"Failed to load model: {ex}", was_running

            # The runtime proved the model loads — only now make it the boot
            # default.
            self._persist_current_model(model_name)

            if was_running:
                self._detection_service.start()

            return True, result, was_running

    def _rollback(self, snapshot: "_Snapshot", was_running: bool) -> None:
        """Restore the pre-activation model, stores, and running state."""
        assert self._detection_service is not None  # guarded by activate_model
        with self.lock:
            self.current_model = snapshot.current_model
            self.config.MODEL_PATH = snapshot.model_path
            self.config.CLASSES_FILE_PATH = snapshot.classes_path
        if self._area_service is not None:
            self._area_service.switch_storage(
                self.areas_file_for_model(snapshot.model_path))
        if self._settings_service is not None:
            self._settings_service.switch_model(
                self.settings_file_for_model(snapshot.model_path))
        try:
            self._detection_service.initialize()
            if was_running:
                self._detection_service.start()
            logger.info(
                f"Restored previous model '{snapshot.current_model}' after a "
                f"failed activation")
        except Exception:  # noqa: BLE001 - the failure is already surfaced
            logger.exception(
                f"Could not restore previous model '{snapshot.current_model}' "
                f"after a failed activation; detection is stopped")

    # ── Application type ──

    def set_task(self, task: str) -> bool:
        """Switch the device's application type. Returns False for a no-op.

        Serialized with every model lifecycle operation and with the GPU
        handover by ``_op_lock``, in this order:

        1. the current task again → nothing changes (the gateway still audits);
        2. refused while the GPU is handed over to training or a conversion
           runs (``PreconditionFailed``);
        3. detection stops — a failure aborts with the previous state intact;
        4. ``application.json`` is written — a failure restarts detection if
           it was running and aborts;
        5. an active model of another task is deselected (its files stay: they
           belong to the model, not to the application);
        6. the detection runtime, a labeling engine of another task and the
           last result are dropped, so nobody reads a stale result of the
           previous task;
        7. ``application_changed`` is published. Detection is not restarted.

        Steps 5–7 only drop in-memory state and a marker file whose absence
        is valid, so they cannot leave the persisted state inconsistent.
        """
        app = self._application_service
        if app is None:
            raise RuntimeError(
                "ModelService.set_task requires attach_application_service()")
        app.validate(task)
        with self._op_lock:
            if app.task == task:
                return False
            if self._runtime_released:
                raise PreconditionFailed(
                    "The GPU is handed over to model training; change the application "
                    "type after training finishes")
            if self._conversion_service is not None and self._conversion_service.get_active_jobs():
                raise PreconditionFailed(
                    "A model conversion is in progress; change the application type "
                    "after it finishes")
            previous = app.task
            detection = self._detection_service
            was_running = detection.stop() if detection is not None else False
            try:
                app.persist(task)
            except Exception:
                if was_running and detection is not None:
                    try:
                        detection.start()
                    except Exception:  # noqa: BLE001 - the write failure is surfaced
                        logger.exception("Could not restart detection after a failed "
                                         "application change")
                raise
            if self.current_model and self.task_of(self.current_model) != task:
                self._clear_selection()
            if detection is not None:
                detection.unload_runtime()
                detection.reset_results()
            labeling = self._labeling_service
            if labeling is not None and labeling.loaded_task() not in (None, task):
                labeling.unload()
            if previous == FACE_TASK:
                # The face embedder's private worker goes with the runtime.
                from ..postprocess import _face_assets
                from ..postprocess._face_embedder import close_worker
                close_worker(_face_assets.embed_worker_port())
            self._publish("application_changed", ["application", "models", "status"],
                          {"task": task})
        logger.info("Application type changed: %s → %s", previous or "(none)", task)
        return True

    def _clear_selection(self) -> None:
        """Deselect the active model: back to the default paths, no boot default."""
        default_model = (getattr(self.config, "DEFAULT_MODEL_PATH", None)
                         or os.path.join(self.model_directory, "weights.engine"))
        default_classes = (getattr(self.config, "DEFAULT_CLASSES_FILE_PATH", None)
                           or self.classes_file_for_model(default_model))
        with self.lock:
            previous = self.current_model
            self.current_model = ""
            self.config.MODEL_PATH = default_model
            self.config.CLASSES_FILE_PATH = default_classes
        self._forget_persisted_current_model()
        if self._area_service is not None:
            self._area_service.switch_storage(self.areas_file_for_model(default_model))
        if self._settings_service is not None:
            self._settings_service.switch_model(self.settings_file_for_model(default_model))
        logger.info(f"Model '{previous}' deselected: it belongs to another application")

    def _publish(self, event_type: str, keys: List[str], data: dict) -> None:
        if self._event_service is None:
            return
        try:
            self._event_service.publish(event_type, keys=keys, data=data)
        except Exception as ex:  # noqa: BLE001 - notification must not fail the change
            logger.warning(f"Could not publish {event_type}: {ex}")

    # ── GPU handover (training-service) ──

    def release_runtime(self) -> Tuple[bool, str]:
        """Stop detection and free every TensorRT worker for the training-service.

        Refused while a conversion runs (the engine build needs the worker).
        Holding ``_op_lock`` serializes the release against ``set_task`` and
        model activation: whichever runs first makes the other refuse or wait.
        """
        with self._op_lock:
            if self._conversion_service is not None and self._conversion_service.get_active_jobs():
                logger.warning("Runtime release refused: a model conversion is in progress")
                return False, "A model conversion is in progress; try again when it finishes"
            if self._detection_service is not None:
                self._detection_service.stop()
            # The labeling engine goes with the runtime: reset its state so a
            # stale "loaded" never survives the worker teardown below.
            if self._labeling_service is not None:
                self._labeling_service.unload()
            from ..runtime_management.worker_client import release_all_workers
            release_all_workers()
            self._runtime_released = True
            return True, "Runtime released"

    def resume_runtime(self, restart_detection: bool = True) -> Tuple[bool, str]:
        """End the GPU handover after training.

        With ``restart_detection`` (the default) the current model is
        re-activated and detection restarts. Without it only the handover ends:
        the runtime stays unloaded, so the model conversion that follows a
        training keeps the GPU, and detection waits for an explicit start. The
        application type can change again either way.
        """
        with self._op_lock:
            # Cleared first: a failed resume must not lock the application
            # type until the next boot.
            self._runtime_released = False
            if not restart_detection:
                return True, "GPU handover ended; detection left stopped"
            app = self._application_service
            if app is not None and app.task is None:
                return True, "Runtime resumed (no application type selected)"
            if not self.current_model:
                # Initializing would fall back to the config's default model,
                # which a device of another task refuses, and training mode
                # could then never be left.
                return True, "Runtime resumed (no model selected)"
            detection = self._detection_service
            if detection is not None and not detection.is_running:
                detection.initialize()
                detection.start()
            return True, "Runtime resumed"

    def start_detection(self) -> bool:
        """Start the detection loop; an explicit start ends a GPU handover.

        Returns True when the start took a released runtime back, so the caller
        can publish ``runtime_changed``. The flag is cleared only after the
        start succeeded, and both happen under ``_op_lock``: a training that
        releases the GPU afterwards stops detection and marks the handover
        again, so the two never interleave.
        """
        if self._detection_service is None:
            raise RuntimeError(
                "ModelService.start_detection requires attach_detection_service() "
                "to be called during application wiring."
            )
        with self._op_lock:
            if self._runtime_released and not self._detection_service.is_running:
                # The released workers are gone and a conversion may have
                # replaced the selected engine meanwhile: drop the old runtime
                # so start() initializes the model configured now, as a resume
                # does.
                self._detection_service.unload_runtime()
            self._detection_service.start()
            was_released, self._runtime_released = self._runtime_released, False
            return was_released

    def delete_model(self, model_name: str) -> Tuple[bool, str]:
        """
        Delete a model file.

        Args:
            model_name: Name of the model file

        Returns:
            Tuple of (success, error_message)
        """
        with self._op_lock:
            if model_name == self.current_model:
                return False, "Cannot delete the currently active model"

            # A file an active job is still reading (an enrollment package, a
            # checkpoint being converted) or about to write (its engine) must
            # stay: the job would fail, or finish and write its outputs back
            # after the delete.
            if model_name in self._job_owned_names():
                return False, (f"'{model_name}' is being built into a model; delete it "
                               "after that finishes")

            model_path, error = validate_model_filename(model_name, self.model_directory)
            if error:
                return False, error

            if not os.path.exists(model_path):
                return False, f"Model '{model_name}' not found"

            failures = []
            for path in self.model_artifacts(model_path):
                try:
                    os.remove(path)
                except FileNotFoundError:
                    continue  # a model need not have every sidecar
                except OSError as ex:
                    logger.error(f"Error deleting model artifact {path}: {ex}")
                    failures.append(f"{os.path.basename(path)}: {ex}")
            if failures:
                return False, "Some files could not be deleted: " + "; ".join(failures)
            logger.info(f"Model deleted with its sidecars: {model_name}")
            return True, ""

