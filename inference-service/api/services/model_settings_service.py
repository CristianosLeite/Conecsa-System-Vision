# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Model settings service.

Persists per-model tuning to a sibling JSON file (weights.engine ->
weights.settings.json), like the per-model classes file and detection areas,
and applies it on selection and at startup. Keys, all preserved by ``save()``:

- confidence/overlay thresholds and camera configuration;
- ``"segment": {"max_instances": n}``, only once the operator set one (absent
  means the ``SEGMENT_MAX_MASKS`` defaults);
- ``"face": {"match_threshold", "min_size_px", "max_faces"}``, each only once
  the operator set it (absent means the ``FACE_*`` defaults);
- ``"imgsz"`` (640 or 1280), written by the ``.pt`` conversion; informational,
  since the engine dictates its own input size;
- ``"training": {"geometry": ...}`` (``"frames"``, ``"tiles:auto"`` or
  ``"tiles:<px>"``) when the uploader declared it; ``switch_model`` warns when it
  disagrees with the live ``TILING_MODE``/``TILING_TILE`` and never applies it;
- ``"task"`` ("detect", "classify", "segment", "face"), declared at upload, verified at
  activation and never rewritten; a file without it (older firmware) means
  ``"detect"``.
"""
import json
import logging
import os
from threading import Lock
from typing import Optional, Sequence, Union

from conecsa_common.atomic import atomic_write_json
from conecsa_common.tasks import task_or_default

from ..config import (
    Config,
    face_settings_from_env,
    valid_face_match_threshold,
    valid_face_max_faces,
    valid_face_min_size,
)

logger = logging.getLogger(__name__)

#: Per-model face settings: Config attribute → (key in the "face" block, validator).
_FACE_SETTINGS = (
    ("FACE_MATCH_THRESHOLD", "match_threshold", valid_face_match_threshold),
    ("FACE_MIN_SIZE_PX", "min_size_px", valid_face_min_size),
    ("FACE_MAX_FACES", "max_faces", valid_face_max_faces),
)

GEOMETRY_FRAMES = "frames"
GEOMETRY_TILES_AUTO = "tiles:auto"


def parse_train_geometry(value) -> Optional[str]:
    """A well-formed training geometry string, or ``None`` for anything else."""
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if text in (GEOMETRY_FRAMES, GEOMETRY_TILES_AUTO):
        return text
    pixels = text[len("tiles:"):] if text.startswith("tiles:") else ""
    if pixels.isdigit() and int(pixels) > 0:
        return f"tiles:{int(pixels)}"
    return None


def geometry_mismatch(train_geometry: Optional[str], tiling_mode: str,
                      tiling_tile: Optional[int]) -> Optional[str]:
    """Why ``train_geometry`` disagrees with the live tiling knobs, or ``None``.

    ``tiling_tile`` is ``None`` for the ``auto`` tile (the frame's short
    side), which is what ``tiles:auto`` models were trained with.
    """
    geometry = parse_train_geometry(train_geometry)
    if geometry is None:
        return None
    if geometry == GEOMETRY_FRAMES:
        if tiling_mode == "grid":
            return ("trained on whole frames but TILING_MODE=grid slices every frame "
                    "into tiles the model has not seen at that scale; set TILING_MODE=off "
                    "or retrain it with TRAIN_TILE=auto")
        return None
    trained = geometry.split(":", 1)[1]
    if tiling_mode != "grid":
        return (f"trained on {trained} tiles but TILING_MODE=off runs the whole "
                "letterboxed frame; set TILING_MODE=grid")
    live = "auto" if tiling_tile is None else str(tiling_tile)
    if live != trained:
        return (f"trained on {trained} tiles but TILING_TILE={live}; align TILING_TILE "
                "with the TRAIN_TILE the model was trained with")
    return None


class ModelSettingsService:
    """Per-model thresholds + camera config, persisted as a sibling JSON file."""

    def __init__(self, config: Config, video_service=None) -> None:
        self._config = config
        self._video_service = video_service
        # Resolved on switch_model(); None means "no model scoped yet".
        self._settings_path: Optional[str] = None
        self._lock = Lock()

    # ── Model switching ──

    def switch_model(self, settings_path: str) -> None:
        """Point at a model's settings file and apply it.

        If the file exists, its thresholds + camera config are loaded and
        applied. If it does not exist yet, the current in-memory settings are
        persisted to seed it, so the model immediately owns a snapshot that
        subsequent edits update.
        """
        with self._lock:
            self._settings_path = os.path.abspath(settings_path)
            path = self._settings_path

        if os.path.exists(path):
            data = self.load_and_apply()
            self._warn_on_geometry_mismatch(path, data)
            # A sidecar seeded by the conversion job holds only "imgsz" and
            # the training geometry; it still needs the threshold snapshot a
            # brand-new model would have got. A corrupt file (None) is left
            # alone, as before.
            if data is None or "thresholds" in data:
                return
        else:
            # The seeded snapshot inherits the thresholds, but the instance
            # limit and the face settings belong to one model: a new model
            # starts on the defaults.
            self._config.SEGMENT_MAX_INSTANCES = None
            self._apply_face({})
        self.save()

    def _apply_face(self, face: dict) -> None:
        """Apply a model's ``face`` block; a missing or invalid value is the default."""
        for (attr, key, valid), default in zip(_FACE_SETTINGS, face_settings_from_env(),
                                               strict=True):
            stored = face.get(key)
            value: float = default
            if (isinstance(stored, (int, float)) and not isinstance(stored, bool)
                    and valid(stored)):
                value = stored
            setattr(self._config, attr, float(value) if key == "match_threshold" else value)

    @staticmethod
    def _warn_on_geometry_mismatch(path: str, data: Optional[dict]) -> None:
        """Log when the model's recorded training geometry disagrees with TILING_*."""
        # Lazy: model_manager pulls in cv2 and the runtime registry, which the
        # settings sidecar does not otherwise need.
        from ..model_manager import tiling_mode_from_env, tiling_tile_from_env

        reason = geometry_mismatch(
            ModelSettingsService.training_geometry_of(data),
            tiling_mode_from_env(), tiling_tile_from_env())
        if reason:
            logger.warning("Model %s: %s", os.path.basename(path), reason)

    @staticmethod
    def training_geometry_of(data: Optional[dict]) -> Optional[str]:
        """The ``training.geometry`` value of a parsed settings payload, if well-formed."""
        training = (data or {}).get("training")
        if not isinstance(training, dict):
            return None
        return parse_train_geometry(training.get("geometry"))

    @classmethod
    def training_geometry(cls, settings_path: str) -> Optional[str]:
        """Read a model's recorded training geometry without activating it."""
        path = os.path.abspath(settings_path)
        if not os.path.exists(path):
            return None
        return cls.training_geometry_of(cls._read_payload(path))

    @classmethod
    def task_of(cls, settings_path: str) -> str:
        """Read a model's task without activating it (``"detect"`` when unrecorded)."""
        path = os.path.abspath(settings_path)
        if not os.path.exists(path):
            return task_or_default(None)
        return task_or_default((cls._read_payload(path) or {}).get("task"))

    # ── Load / apply ──

    def load_and_apply(self) -> Optional[dict]:
        """Read the active model's settings file and apply it to live state.

        Returns the parsed payload, or ``None`` when there is no file or it
        could not be read.
        """
        with self._lock:
            path = self._settings_path
        if not path or not os.path.exists(path):
            return None

        data = self._read_payload(path)
        if data is None:
            return None

        thresholds = data.get("thresholds", {}) or {}
        conf = thresholds.get("confidence")
        overlay = thresholds.get("overlay")
        if isinstance(conf, (int, float)) and 0.0 <= conf <= 1.0:
            self._config.CONFIDENCE_THRESHOLD = float(conf)
        if isinstance(overlay, (int, float)) and 0.0 <= overlay <= 1.0:
            self._config.OVERLAY_THRESHOLD = float(overlay)
        # A model without its own limit runs on the SEGMENT_MAX_MASKS defaults,
        # never on the limit of the model active before it.
        segment = data.get("segment")
        limit = segment.get("max_instances") if isinstance(segment, dict) else None
        valid = isinstance(limit, int) and not isinstance(limit, bool) and 1 <= limit <= 255
        self._config.SEGMENT_MAX_INSTANCES = limit if valid else None
        face = data.get("face")
        self._apply_face(face if isinstance(face, dict) else {})

        camera = data.get("camera")
        if camera and self._video_service is not None:
            # Best-effort: webcam server may not be reachable at boot; the
            # config is re-applied on the next selection/change regardless.
            self._video_service.apply_webcam_server_config(camera)

        stereo = data.get("stereo")
        if isinstance(stereo, dict) and self._video_service is not None:
            self._video_service.set_stereo_config(
                stereo.get("enabled"),
                stereo.get("alpha"),
                stereo.get("offset"),
                stereo.get("offset_y"),
            )

        logger.info("Applied per-model settings from %s", path)
        return data

    # ── Save ──

    #: Config attributes :meth:`save` can write on their own, and where they live.
    _FIELDS = {
        "OVERLAY_THRESHOLD": ("thresholds", "overlay", float),
        "SEGMENT_MAX_INSTANCES": ("segment", "max_instances", int),
        "FACE_MATCH_THRESHOLD": ("face", "match_threshold", float),
        "FACE_MIN_SIZE_PX": ("face", "min_size_px", int),
        "FACE_MAX_FACES": ("face", "max_faces", int),
    }

    def save(self, only: Union[str, Sequence[str], None] = None) -> bool:
        """Snapshot the live thresholds + camera config to the model's file.

        The informational ``imgsz`` and ``training`` and the model's ``task``
        already in the file (if any) are carried over so a threshold edit
        never erases them. Returns False only when the file could not be
        written; with no model scoped there is nothing to save.

        ``only`` names one ``Config`` attribute, or several (see ``_FIELDS``),
        to write into the file as it is on disk in one write, leaving every
        other value alone: changing the overlay threshold must not also
        persist a confidence threshold that ``SetThreshold`` deliberately
        kept in memory. A missing or unreadable file falls back to the full
        snapshot.
        """
        with self._lock:
            path = self._settings_path
        if not path:
            return True

        names = [only] if isinstance(only, str) else list(only or ())
        fields = [(name, self._FIELDS[name]) for name in names if name in self._FIELDS]
        if fields and len(fields) == len(names) and os.path.exists(path):
            existing = self._read_payload(path)
            values = [(getattr(self._config, name, None), field) for name, field in fields]
            if isinstance(existing, dict) and all(v is not None for v, _ in values):
                for value, (section, key, cast) in values:
                    if not isinstance(existing.get(section), dict):
                        existing[section] = {}
                    existing[section][key] = cast(value)
                return self._write_payload(path, existing)

        payload: dict = {
            "thresholds": {
                "confidence": self._config.CONFIDENCE_THRESHOLD,
                "overlay": self._config.OVERLAY_THRESHOLD,
            },
        }
        limit = getattr(self._config, "SEGMENT_MAX_INSTANCES", None)
        if limit is not None:
            payload["segment"] = {"max_instances": int(limit)}
        if self._video_service is not None:
            # Local tuning only: the capture source is device-level state.
            camera = self._video_service.get_model_camera_config()
            if camera:
                payload["camera"] = camera
            payload["stereo"] = self._video_service.get_stereo_config()

        existing = self._read_payload(path) if os.path.exists(path) else None
        imgsz = (existing or {}).get("imgsz")
        if isinstance(imgsz, int) and not isinstance(imgsz, bool):
            payload["imgsz"] = imgsz
        geometry = self.training_geometry_of(existing)
        if geometry is not None:
            payload["training"] = {"geometry": geometry}
        task = (existing or {}).get("task")
        if isinstance(task, str) and task:
            payload["task"] = task
        face = (existing or {}).get("face")
        if isinstance(face, dict) and face:
            # Only values the operator set are kept, as their live values.
            payload["face"] = {key: getattr(self._config, attr)
                               for attr, key, _ in _FACE_SETTINGS if key in face}

        return self._write_payload(path, payload)

    @classmethod
    def record_training(cls, settings_path: str, imgsz: Optional[int],
                        train_geometry: Optional[str] = None,
                        task: Optional[str] = None) -> None:
        """Store what a freshly uploaded or converted model was built as.

        Called before the model is ever activated — by the conversion job when
        the engine is written, or by the upload of a prebuilt engine: merges
        ``"imgsz"`` (when known), ``"training"`` (when the geometry is known
        and well-formed) and the model's ``"task"`` into an existing settings
        file or creates one holding just those keys (``switch_model``
        completes it with the threshold snapshot on first activation).
        Best-effort, like ``save``.
        """
        path = os.path.abspath(settings_path)
        payload = (cls._read_payload(path) if os.path.exists(path) else None) or {}
        if imgsz is not None:
            payload["imgsz"] = int(imgsz)
        geometry = parse_train_geometry(train_geometry)
        if geometry is not None:
            payload["training"] = {"geometry": geometry}
        if task:
            payload["task"] = task
        cls._write_payload(path, payload)

    @classmethod
    def record_imgsz(cls, settings_path: str, imgsz: int) -> None:
        """``record_training`` without a geometry (kept for callers that only know the size)."""
        cls.record_training(settings_path, imgsz)

    # ── File helpers ──

    @staticmethod
    def _read_payload(path: str) -> Optional[dict]:
        """Parse a settings file; ``None`` (logged) when unreadable or not a JSON object."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as exc:  # noqa: BLE001 - best-effort restore
            logger.error("Failed to read model settings %s: %s", path, exc)
            return None
        if not isinstance(data, dict):
            logger.error("Model settings %s is not a JSON object", path)
            return None
        return data

    @staticmethod
    def _write_payload(path: str, payload: dict) -> bool:
        """Durably write ``payload`` as JSON (``conecsa_common.atomic``); False (logged) on failure."""
        try:
            atomic_write_json(path, payload, indent=2)
        except Exception as exc:  # noqa: BLE001 - best-effort persist
            logger.error("Failed to persist model settings to %s: %s", path, exc)
            return False
        return True
