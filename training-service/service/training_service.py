# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Training job orchestration.

One job at a time. The ultralytics run executes in a child process
(_yolo_trainer) that streams one JSON line per epoch; a reader thread folds
those into the job state and publishes ``training_progress`` events. On
success the resulting best.pt is uploaded through the api-gateway's existing
model-upload route, which renames it to the user-chosen model name and starts
the pt→onnx→engine conversion on the inference-service (classes sidecar and
SSE included) — the same path a manual upload takes.

Federated rounds (hub-orchestrated FedAvg) reuse the same job machinery but
start from a stashed checkpoint (``initial_weights_id``) and, instead of
uploading, stash the resulting last.pt back into the weights store
(``result_weights_id``) for the hub to collect and average.

A fine-tune starts from an existing device model instead (``base_model``, a
model-list name whose weights sidecar is the last best.pt it was built from):
the checkpoint is fetched through the gateway right before the split is built.

A ``face`` dataset is not trained at all: the job packages its labeled photos
as ``<model name>.faces`` (``DatasetService.build_face_package``) and uploads
the package through the same model-upload route with ``task=face``; the
inference-service builds the gallery as a conversion job, so the job ends with
a ``conversion_job_id`` exactly like a YOLO run. There is no trainer
subprocess, no base or federated weights and nothing stashed.
"""
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

import requests
from conecsa_common.tasks import FACE

from .config import Config, base_weights_for, img_size_for
from .dataset_registry import DatasetRegistry
from .dataset_service import (
    FACE_PACKAGE_SUFFIX,
    DatasetError,
    DatasetService,
    validate_model_name,
)
from .model_fetch import fetch_weights, list_models_with_weights, validate_model_ref
from .train_overrides import OverrideError, parse_overrides

logger = logging.getLogger(__name__)

_TERMINAL = {"done", "failed", "canceled"}


@dataclass
class TrainingJob:
    """State of a single training run (status, progress, epoch, metrics, result)."""

    job_id: str = ""
    status: str = "idle"      # idle/preparing/training/uploading/done/failed/canceled
    progress: int = 0         # 0-100
    epoch: int = 0
    total_epochs: int = 0
    message: str = ""
    error: str = ""
    model_name: str = ""
    conversion_job_id: str = ""
    metrics: Dict = field(default_factory=dict)
    started_at: float = 0.0
    dataset_id: str = ""
    patience: int = 0
    # Federated round (hub-orchestrated FedAvg): the result stays on-device as
    # a stashed checkpoint instead of going through the model-upload route.
    federated: bool = False
    result_weights_id: str = ""
    # Effective training geometry of the split ("frames" | "tiles:auto" |
    # "tiles:<px>"), declared on the model upload so the inference-service
    # can warn when TILING_MODE disagrees with it.
    geometry: str = ""
    # Existing device model (model-list name) the run fine-tunes from, if any.
    base_model: str = ""
    # The dataset's task: picks the base weights, the split layout and the
    # input size, and is declared on the model upload.
    task: str = ""


class TrainingService:
    """Orchestrates one ultralytics training job at a time.

    Runs the trainer in a child process (``_yolo_trainer``) that streams
    per-epoch JSON; a reader thread folds those into the :class:`TrainingJob`
    state and publishes ``training_progress`` events. On success it hands
    ``best.pt`` to the gateway's model-upload route (pt→onnx→engine).
    """

    def __init__(self, config: Config, registry: DatasetRegistry,
                 event_service=None, sam_service=None, weights_store=None):
        self._config = config
        self._registry = registry
        self._events = event_service
        self._sam = sam_service
        # Stash for federated rounds (initial weights in, last.pt out).
        self._weights = weights_store
        self._lock = threading.Lock()
        self._job = TrainingJob()
        # Dataset of the running job; frozen for the job's duration.
        self._job_dataset: Optional[DatasetService] = None
        self._process: Optional[subprocess.Popen] = None
        self._cancel_requested = False
        # Graceful early-stop (keep best.pt), distinct from a hard cancel.
        self._early_stop_requested = False
        # Monotonic timestamp of the trainer's last stdout/stderr line,
        # written by the reader threads and watched by the stall watchdog.
        self._last_output = 0.0

    # ── public API ────────────────────────────────────────────────────────────

    def get_job(self) -> TrainingJob:
        """Get job."""
        with self._lock:
            return TrainingJob(**vars(self._job))

    def is_active(self) -> bool:
        """Is active."""
        with self._lock:
            return self._job.status not in ("idle", *_TERMINAL)

    def start(self, dataset_id: str, model_name: str,
              epochs: int = 0, batch: int = 0, patience: int = 0,
              initial_weights_id: str = "", federated: bool = False,
              base_model: str = "") -> TrainingJob:
        """Start a job on ``dataset_id`` (a gallery build for a face dataset)."""
        if federated:
            # No model upload happens, so the name is only a display label.
            model_name = (model_name or "").strip() or "federated"
        else:
            model_name = validate_model_name(model_name)
        epochs = epochs or self._config.DEFAULT_EPOCHS
        batch = batch or self._config.TRAIN_BATCH
        patience = patience or self._config.DEFAULT_PATIENCE
        if epochs < 1 or epochs > 1000:
            raise DatasetError("Epochs must be between 1 and 1000")
        dataset = self._registry.get(dataset_id)
        task = dataset.task()
        base_model = (base_model or "").strip()
        if task == FACE:
            # A gallery build: no network is trained, so there are no weights
            # to start from and nothing a federated round could average.
            if federated:
                raise DatasetError(
                    "Face datasets cannot take part in federated training: a face "
                    "gallery stays on its device")
            if base_model:
                raise DatasetError(
                    "A face gallery is built from its photos; it cannot start from a "
                    "base model")
            if initial_weights_id:
                raise DatasetError(
                    "A face gallery is built from its photos; it takes no initial weights")
            weights_path = ""
            epochs = 0
        else:
            # A malformed TRAIN_OVERRIDES must fail the RPC, not a job that has
            # already frozen its dataset and released the inference runtime.
            try:
                parse_overrides(getattr(self._config, "TRAIN_OVERRIDES", ""))
            except OverrideError as exc:
                raise DatasetError(f"Invalid TRAIN_OVERRIDES: {exc}") from exc
            # Resolve the starting checkpoint up front so an unknown id fails the
            # RPC instead of the job. It must have been trained for the dataset's
            # task: YOLO(weights) infers the task from the checkpoint.
            weights_path = base_weights_for(self._config, task)
        if base_model and initial_weights_id:
            raise DatasetError("base_model and initial_weights_id are mutually exclusive")
        if initial_weights_id:
            if self._weights is None:
                raise DatasetError("Weights store is not available")
            weights_path = self._weights.path(initial_weights_id)
            stored_task = self._weights.task_of(initial_weights_id)
            if stored_task and stored_task != task:
                raise DatasetError(
                    f"The initial weights were trained for '{stored_task}'; this dataset "
                    f"is labeled for '{task}'")
        if base_model:
            # Existence is checked here against the device model list; the
            # download itself happens in the job (it is the first thing _run
            # does), so the RPC stays fast.
            base_model = validate_model_ref(base_model)
            try:
                available = list_models_with_weights(self._config.GATEWAY_ADDR)
            except requests.RequestException as exc:
                raise DatasetError(f"Could not list the device models: {exc}") from exc
            if base_model not in available:
                raise DatasetError(
                    f"Model '{base_model}' has no training checkpoint on the device")
            if available[base_model] != task:
                raise DatasetError(
                    f"Model '{base_model}' is a '{available[base_model]}' model; this "
                    f"dataset is labeled for '{task}'")

        with self._lock:
            if self._job.status not in ("idle", *_TERMINAL):
                raise DatasetError("A training job is already running")
            # The registry owns the frozen transition: freeze() claims the
            # dataset atomically against delete(), so it cannot vanish
            # between validation and the job taking it.
            dataset = self._registry.freeze(dataset_id)
            try:
                dataset.validate_for_training()
            except Exception:
                self._registry.release(dataset)
                raise
            job_id = str(uuid.uuid4())
            self._job = TrainingJob(
                job_id=job_id, status="preparing", progress=2,
                message=("Packaging the enrollment photos…" if task == FACE
                         else "Preparing dataset split…"),
                model_name=model_name, total_epochs=epochs,
                started_at=time.time(), dataset_id=dataset_id,
                patience=patience, federated=federated, base_model=base_model,
                task=task,
            )
            self._cancel_requested = False
            self._early_stop_requested = False
            self._job_dataset = dataset

        # SAM and training never share the GPU (8GB budget). The TensorRT
        # labeling engine lives in the inference-service, whose runtime the
        # gateway releases before every run.
        if self._sam is not None:
            self._sam.unload()

        threading.Thread(
            target=self._run, args=(job_id, epochs, batch, patience, weights_path),
            kwargs={"base_model": base_model},
            daemon=True, name=f"training-{job_id[:8]}",
        ).start()
        self._publish()
        return self.get_job()

    def cancel(self) -> bool:
        """Cancel."""
        with self._lock:
            if self._job.status not in ("preparing", "training"):
                return False
            self._cancel_requested = True
            proc = self._process
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError) as exc:
                logger.warning("Cancel signal failed: %s", exc)
        return True

    def finish_early(self) -> bool:
        """Gracefully stop the running job, keeping the best model so far.

        Unlike cancel (SIGTERM/kill → discarded), this nudges the trainer with
        SIGUSR1 so ultralytics finalizes the current epoch + validation, writes
        best.pt and exits 0 — the normal done/upload/conversion path then runs.
        """
        with self._lock:
            if self._job.status != "training":
                return False
            self._early_stop_requested = True
            proc = self._process
        if proc is None or proc.poll() is not None:
            return False
        try:
            # pid only (not the group) — a graceful signal, never a kill.
            os.kill(proc.pid, signal.SIGUSR1)
        except (ProcessLookupError, PermissionError) as exc:
            logger.warning("Finish-early signal failed: %s", exc)
            return False
        self._set(message="Finishing early — finalizing model…")
        return True

    # ── internals ─────────────────────────────────────────────────────────────

    def _set(self, **fields) -> None:
        """Set."""
        with self._lock:
            for k, v in fields.items():
                setattr(self._job, k, v)
        self._publish()

    def _publish(self) -> None:
        """Publish."""
        if self._events is None:
            return
        job = self.get_job()
        try:
            self._events.publish(
                "training_progress", keys=["training"],
                data={
                    "job_id": job.job_id, "status": job.status,
                    "progress": job.progress, "epoch": job.epoch,
                    "total_epochs": job.total_epochs, "message": job.message,
                    "error": job.error, "model_name": job.model_name,
                    "conversion_job_id": job.conversion_job_id,
                    "dataset_id": job.dataset_id,
                    "federated": job.federated,
                    "result_weights_id": job.result_weights_id,
                    "base_model": job.base_model,
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not publish training event: %s", exc)

    def _run(self, job_id: str, epochs: int, batch: int, patience: int,
             weights_path: str, base_model: str = "") -> None:
        """Run."""
        dataset = self._job_dataset
        assert dataset is not None
        try:
            if self.get_job().task == FACE:
                self._run_face(job_id, dataset)
                return
            if base_model:
                # The device model's last best.pt, through the gateway (this
                # service has no access to the inference-side model volume).
                self._set(message=f"Fetching base weights from {base_model}…")
                weights_path = fetch_weights(self._config.GATEWAY_ADDR, base_model,
                                             self._config.base_dir)
            task = self.get_job().task or "detect"
            # A classifier sees the whole frame: no tile crops.
            classify = task == "classify"
            self._set(message="Sorting the images into class folders…" if classify
                      else "Slicing the dataset into training tiles…")
            split = dataset.build_split(
                job_id,
                tile=None if classify else getattr(self._config, "TRAIN_TILE", "auto"),
                overlap=getattr(self._config, "TRAIN_TILE_OVERLAP", 0.2),
                min_visible=getattr(self._config, "TRAIN_TILE_MIN_VISIBLE", 0.25),
            )
            self._set(status="training", progress=5, geometry=split.geometry,
                      message=f"Training {epochs} epochs…")

            cmd = build_trainer_argv(
                self._config, split.yaml_path, weights_path,
                epochs=epochs, patience=patience, batch=batch, job_id=job_id,
                imgsz=img_size_for(self._config, task),
            )

            env = os.environ.copy()
            env["PYTHONPATH"] = "/app/training-service"

            logger.info("Starting trainer subprocess for job %s", job_id)
            with self._lock:
                self._process = subprocess.Popen(
                    cmd, env=env, start_new_session=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
                proc = self._process

            stderr_thread = threading.Thread(
                target=self._drain_stderr, args=(proc,), daemon=True,
                name=f"training-stderr-{job_id[:8]}",
            )
            stderr_thread.start()

            # Watchdog: stdout consumption below blocks until the trainer
            # exits, so hang detection has to come from the side. It fires on
            # output silence (stalled/hung trainer), not on total duration —
            # long runs are legitimate. `timed_out` carries the kill reason.
            self._last_output = time.monotonic()
            timed_out: list = []
            watchdog = threading.Thread(
                target=self._watch_trainer, args=(proc, timed_out),
                daemon=True, name=f"training-watchdog-{job_id[:8]}",
            )
            watchdog.start()
            best_path, last_path = self._consume_stdout(proc, epochs)
            proc.wait()

            if timed_out:
                raise RuntimeError(timed_out[0])
            if self._cancel_requested:
                self._set(status="canceled", message="Training canceled", progress=0)
                return
            if proc.returncode != 0 or best_path is None:
                raise RuntimeError(
                    self.get_job().error or
                    f"Trainer exited with code {proc.returncode}"
                )

            if self.get_job().federated:
                # Federated round: the hub collects last.pt for averaging, so
                # the result stays on-device instead of going through the
                # model-upload/conversion route.
                assert self._weights is not None
                weights_id = self._weights.stash_file(last_path or best_path, task=task)
                self._set(status="done", progress=100,
                          result_weights_id=weights_id,
                          message="Training complete; weights retained for aggregation")
                logger.info("Job %s done; weights stashed as %s", job_id, weights_id)
            else:
                conversion_job_id = self._upload_best(best_path)
                self._set(status="done", progress=100,
                          conversion_job_id=conversion_job_id,
                          message="Training complete; model uploaded for conversion")
                logger.info("Job %s done; conversion job %s", job_id, conversion_job_id)

        except Exception as exc:  # noqa: BLE001 - job state carries the failure
            logger.exception("Training job %s failed: %s", job_id, exc)
            if self._cancel_requested:
                self._set(status="canceled", message="Training canceled", progress=0)
            else:
                self._set(status="failed", error=str(exc),
                          message="Training failed", progress=0)
        finally:
            with self._lock:
                self._process = None
                self._job_dataset = None
            self._registry.release(dataset)
            # The split is job-scoped scratch: tile crops are real JPEGs (the
            # whole-frame path only symlinks) and nothing reads them once the
            # trainer has exited — best.pt lives in runs/<job>/weights.
            shutil.rmtree(os.path.join(self._config.runs_dir, job_id, "dataset"),
                          ignore_errors=True)

    def _run_face(self, job_id: str, dataset: DatasetService) -> None:
        """Gallery build: package the labeled photos and upload the package.

        No trainer runs. The ``.faces`` package goes through the model-upload
        route with ``task=face``; the inference-service embeds the photos as a
        conversion job and the job ends ``done`` with its
        ``conversion_job_id``, the same handoff a trained model takes. The
        local package is job scratch and is removed once uploaded (or on
        failure). Failures and a cancel surface through ``_run``'s handler.
        """
        package = ""
        try:
            package, count = dataset.build_face_package(job_id, self.get_job().model_name)
            self._set(progress=50, geometry="",
                      message=f"Packaged {count} enrollment photos")
            if self._cancel_requested:
                self._set(status="canceled", message="Training canceled", progress=0)
                return
            conversion_job_id = self._upload_best(package, suffix=FACE_PACKAGE_SUFFIX)
            self._set(status="done", progress=100,
                      conversion_job_id=conversion_job_id,
                      message="Photos uploaded; building the face gallery")
            logger.info("Job %s done; face gallery job %s", job_id, conversion_job_id)
        finally:
            if package and os.path.exists(package):
                os.remove(package)

    def _consume_stdout(self, proc: subprocess.Popen,
                        epochs: int) -> "tuple[Optional[str], Optional[str]]":
        """Fold the trainer's JSON lines into the job; return (best, last) paths."""
        best_path: Optional[str] = None
        last_path: Optional[str] = None
        assert proc.stdout is not None
        for line in proc.stdout:
            self._last_output = time.monotonic()
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                logger.debug("[trainer] %s", line)
                continue
            if payload.get("error"):
                self._set(error=str(payload["error"]))
            elif payload.get("stopping"):
                self._set(message="Finishing early — finalizing model…")
            elif payload.get("done"):
                best_path = str(payload.get("best", ""))
                last_path = str(payload.get("last", "")) or None
            elif payload.get("epoch"):
                epoch = int(payload["epoch"])
                progress = 5 + int(epoch / max(epochs, 1) * 90)
                self._set(epoch=epoch, progress=min(progress, 95),
                          metrics=payload.get("metrics") or {},
                          message=f"Epoch {epoch}/{epochs}")
        return best_path, last_path

    def _watch_trainer(self, proc: subprocess.Popen, timed_out: list) -> None:
        """Kill the trainer on output silence (hang) or the optional hard cap."""
        start = time.monotonic()
        while proc.poll() is None:
            time.sleep(15)
            now = time.monotonic()
            cap = self._config.TRAIN_TIMEOUT_SEC
            stall = self._config.TRAIN_STALL_TIMEOUT_SEC
            if cap and now - start > cap:
                reason = f"Training exceeded the {cap}s limit (TRAIN_TIMEOUT_SEC)"
            elif now - self._last_output > stall:
                reason = (
                    f"Training stalled: no trainer output for {stall}s "
                    f"(TRAIN_STALL_TIMEOUT_SEC)"
                )
            else:
                continue
            logger.error("%s; killing trainer", reason)
            timed_out.append(reason)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            return

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        # Ultralytics writes its progress bars and tracebacks to stderr; keep
        # them in our logs and keep the pipe from filling up and blocking it.
        """Drain stderr."""
        assert proc.stderr is not None
        for line in proc.stderr:
            self._last_output = time.monotonic()
            line = line.rstrip()
            if line:
                logger.info("[trainer] %s", line)

    def _upload_best(self, best_path: str, suffix: str = ".pt") -> str:
        """Upload a job result as ``{model_name}{suffix}`` through the gateway.

        The gateway relays to ModelControl.UploadModel on the inference-service,
        which saves it under /data/models and starts the conversion job
        (returned here so the frontend can track it): pt→onnx→engine for
        best.pt, where the split's effective geometry and the dataset's task
        ride along, so the model's settings sidecar records what it was
        trained on and the inference-service checks the task against the
        exported graph. A face package (``suffix=".faces"``) declares only
        ``task=face`` and starts the gallery build instead.
        """
        job = self.get_job()
        task = job.task or "detect"
        if task == FACE:
            message = "Uploading the photos for the face gallery build…"
            data = {"task": task}
        else:
            message = "Uploading model for conversion…"
            data = {"imgsz": str(img_size_for(self._config, task)),
                    "train_geometry": job.geometry,
                    "task": task}
        self._set(status="uploading", progress=97, message=message)
        url = f"{self._config.GATEWAY_ADDR}/api/v1/model"
        with open(best_path, "rb") as f:
            resp = requests.post(
                url,
                files={"file": (f"{job.model_name}{suffix}", f)},
                data=data,
                timeout=120,
            )
        if resp.status_code not in (200, 202):
            raise RuntimeError(
                f"Model upload failed (HTTP {resp.status_code}): {resp.text[:200]}"
            )
        try:
            return str(resp.json().get("job_id") or "")
        except ValueError:
            return ""


def build_trainer_argv(config, data_yaml: str, weights_path: str, *,
                       epochs: int, patience: int, batch: int, job_id: str,
                       imgsz: Optional[int] = None) -> list:
    """Argument vector for the ``service._yolo_trainer`` subprocess.

    Pure so tests can assert the contract without spawning anything.
    ``data_yaml`` is the split's data.yaml, or its root for a classifier. The
    model input size is ``imgsz`` (the task's size, ``img_size_for``) or
    ``config.IMG_SIZE`` (``TRAIN_IMG_SIZE``), and every allowlisted
    ``TRAIN_OVERRIDES`` pair is forwarded as its own ``--override key=value``
    (already validated in ``TrainingService.start``).
    """
    cmd = [
        sys.executable, "-m", "service._yolo_trainer",
        "--data", data_yaml,
        "--weights", weights_path,
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--batch", str(batch),
        "--imgsz", str(imgsz or config.IMG_SIZE),
        "--workers", str(config.TRAIN_WORKERS),
        "--project", config.runs_dir,
        "--name", job_id,
    ]
    if not config.TRAIN_AMP:
        cmd.append("--no-amp")
    for key, value in parse_overrides(getattr(config, "TRAIN_OVERRIDES", "")).items():
        cmd.extend(["--override", f"{key}={value}"])
    return cmd
