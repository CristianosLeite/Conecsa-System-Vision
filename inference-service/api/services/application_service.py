# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""The device's application type.

A device runs exactly one application — object detection, image
classification or instance segmentation. The choice lives in
``<models dir>/application.json`` beside ``.current_model``:

    {"task": "detect", "migrated": true}   migrated from older firmware
    {"task": "classify"}                   chosen by an administrator
    {"task": null}                         blank device, nothing chosen yet

This service only persists and validates that file; the switch itself (stop
detection, deselect a model of another task, notify clients) is orchestrated
by ``ModelService.set_task`` under the model lifecycle lock.
"""
import json
import logging
import os
from threading import Lock
from typing import Dict, List, Optional, Sequence

from conecsa_common.atomic import atomic_write_json
from conecsa_common.tasks import DETECT, TASKS, is_task, optional_task

from .errors import InvalidTask, PreconditionFailed

logger = logging.getLogger(__name__)

FILENAME = "application.json"

#: Files that prove a device was already in use before application types
#: existed. Any of them, even a dangling ``.current_model`` whose engine is
#: gone, makes an upgraded device an object-detection installation.
_ARTIFACT_EXTENSIONS = (".engine", ".plan", ".onnx", ".pt")
_STATE_MARKER = ".current_model"
_WEIGHTS_DIR = "weights"


def has_model_artifacts(model_directory: str) -> bool:
    """True when ``model_directory`` holds anything a pre-application device left."""
    try:
        names = os.listdir(model_directory)
    except OSError:
        return False
    for name in names:
        if name == _STATE_MARKER:
            return True
        if name == _WEIGHTS_DIR and os.path.isdir(os.path.join(model_directory, name)):
            return True
        if name.lower().endswith(_ARTIFACT_EXTENSIONS):
            return True
    return False


class ApplicationService:
    """Persisted application task plus the tasks this build can serve."""

    def __init__(self, model_directory: str, supported_tasks: Sequence[str]):
        self._model_directory = model_directory
        self._path = os.path.join(model_directory, FILENAME)
        self._supported: List[str] = list(supported_tasks)
        self._lock = Lock()
        self._task: Optional[str] = None
        self._migrated = False

    # ── state ────────────────────────────────────────────────────────────────

    @property
    def task(self) -> Optional[str]:
        """The application task, or ``None`` while none has been chosen."""
        with self._lock:
            return self._task

    @property
    def migrated(self) -> bool:
        """True when the task was set by the upgrade migration, not by a person."""
        with self._lock:
            return self._migrated

    @property
    def supported_tasks(self) -> List[str]:
        return list(self._supported)

    def info(self) -> Dict[str, object]:
        """``{task, supported_tasks, migrated}`` — the GetApplication payload."""
        with self._lock:
            return {"task": self._task, "supported_tasks": list(self._supported),
                    "migrated": self._migrated}

    # ── boot ─────────────────────────────────────────────────────────────────

    def load_or_migrate(self) -> None:
        """Read ``application.json``, writing it on the first boot of this build.

        A missing file on a device that already holds model artifacts is an
        upgrade: the device keeps doing what it always did, object detection
        (``migrated: true``). A device with no artifact at all is blank and
        records ``{"task": null}``, so the migration runs exactly once — a
        model copied in later never silently decides the application. A
        corrupt file is kept for diagnosis and read as "no application".
        """
        if os.path.exists(self._path):
            task, migrated = self._read()
            with self._lock:
                self._task, self._migrated = task, migrated
            logger.info("Application type: %s%s", task or "(none)",
                        " (migrated)" if migrated else "")
            return

        if has_model_artifacts(self._model_directory):
            payload: Dict[str, object] = {"task": DETECT, "migrated": True}
            logger.info("Existing installation without an application type: "
                        "recording object detection (migrated)")
        else:
            payload = {"task": None}
            logger.info("Blank device: no application type chosen yet")
        try:
            atomic_write_json(self._path, payload, mode=0o644, indent=2)
        except OSError as exc:
            # Keep the answer in memory; the next boot repeats the probe.
            logger.error("Could not write %s: %s", self._path, exc)
        with self._lock:
            self._task = optional_task(payload.get("task"))
            self._migrated = bool(payload.get("migrated", False))

    def _read(self) -> "tuple[Optional[str], bool]":
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            logger.error("Unreadable %s (%s): no application type until one is chosen",
                         self._path, exc)
            return None, False
        if not isinstance(data, dict):
            logger.error("%s is not a JSON object: no application type until one is chosen",
                         self._path)
            return None, False
        return optional_task(data.get("task")), bool(data.get("migrated", False))

    # ── validation and persistence ───────────────────────────────────────────

    def validate(self, task: object) -> str:
        """Return ``task`` if this build can run it, else raise.

        :class:`InvalidTask` for an id that is not a task at all,
        :class:`PreconditionFailed` for a known task this build does not
        support yet (its strategy ships in a later release).
        """
        if not is_task(task):
            raise InvalidTask(
                f"Unknown application type '{task}'; expected one of {', '.join(TASKS)}")
        assert isinstance(task, str)
        if task not in self._supported:
            raise PreconditionFailed(
                f"The '{task}' application is not available in this release "
                f"(supported: {', '.join(self._supported)})")
        return task

    def persist(self, task: str) -> None:
        """Durably record an administrator's choice; memory follows the disk.

        Raises ``OSError`` when the file cannot be written, leaving the
        in-memory task untouched.
        """
        atomic_write_json(self._path, {"task": task}, mode=0o644, indent=2)
        with self._lock:
            self._task = task
            self._migrated = False

    def resolve_upload_task(self, declared: object) -> str:
        """The task a model upload is recorded with.

        An empty declaration means "the device's application"; on a device
        with no application yet the uploader must declare it. A declared task
        must be one this build supports.
        """
        declared_task = optional_task(declared)
        if declared_task is None:
            current = self.task
            if current is None:
                raise InvalidTask(
                    "Declare the model's task: this device has no application type yet")
            return current
        return self.validate(declared_task)
