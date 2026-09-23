# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Configuration file for object detection.
Contains all configurations and environment variables.
"""
import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_FACE_MATCH_THRESHOLD = 0.363
DEFAULT_FACE_MIN_SIZE_PX = 40
DEFAULT_FACE_MAX_FACES = 5
#: Upper bounds of the face settings (the stage C budget caps faces per frame).
FACE_MAX_FACES_LIMIT = 20
FACE_MIN_SIZE_LIMIT = 1024


def valid_face_match_threshold(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= value <= 1.0


def valid_face_min_size(value) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)
            and 0 <= value <= FACE_MIN_SIZE_LIMIT)


def valid_face_max_faces(value) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)
            and 1 <= value <= FACE_MAX_FACES_LIMIT)


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except ValueError:
        logger.warning("%s is not a number; using %s", name, default)
        return default
    return value if valid_face_match_threshold(value) else default


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except ValueError:
        logger.warning("%s is not an integer; using %s", name, default)
        return default
    check = valid_face_max_faces if name == "FACE_MAX_FACES" else valid_face_min_size
    return value if check(value) else default


def face_settings_from_env() -> tuple:
    """``(match_threshold, min_size_px, max_faces)`` a model without its own values runs on."""
    return (_env_float("FACE_MATCH_THRESHOLD", DEFAULT_FACE_MATCH_THRESHOLD),
            _env_int("FACE_MIN_SIZE_PX", DEFAULT_FACE_MIN_SIZE_PX),
            _env_int("FACE_MAX_FACES", DEFAULT_FACE_MAX_FACES))


class Config:
    """Class to manage all system configurations."""

    def __init__(self):
        self.CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", default=0.75))
        self.OVERLAY_THRESHOLD = float(os.environ.get("OVERLAY_THRESHOLD", default=0.45))
        # Segmentation instance limit set by the operator for the active model
        # (1..255, both the per-frame and the per-tile cap); None means the
        # SEGMENT_MAX_MASKS / SEGMENT_MAX_MASKS_PER_TILE defaults.
        self.SEGMENT_MAX_INSTANCES = None
        # Face recognition, per model: the cosine similarity a face must beat
        # to take a person's name (SFace's published 0.363 by default), the
        # smallest face side recognized, and the faces embedded per frame
        # (largest first).
        (self.FACE_MATCH_THRESHOLD, self.FACE_MIN_SIZE_PX,
         self.FACE_MAX_FACES) = face_settings_from_env()

        # Capture settings - use appropriate device for platform
        if os.name == 'nt':  # Windows
            default_capture_device = "0"  # Webcam index
        else:  # Linux/Unix
            default_capture_device = "/dev/media0"

        self.CAPTURE_DEVICE = os.environ.get("CAPTURE_DEVICE", default=default_capture_device)
        self.CAPTURE_RESOLUTION_X = int(os.environ.get("CAPTURE_RESOLUTION_X", default=640))
        self.CAPTURE_RESOLUTION_Y = int(os.environ.get("CAPTURE_RESOLUTION_Y", default=640))
        self.CAPTURE_FRAMERATE = int(os.environ.get("CAPTURE_FRAMERATE", default=30))

        # Model settings — models live in the shared volume mounted from
        # the `os-base` container at /data/models.
        self.MODELS_DIR = os.environ.get("MODELS_DIR", default="/data/models")

        # Hard cap on a streamed model upload, counted server-side while the
        # chunks arrive (the 8 GB device cannot afford unbounded buffering).
        # Matches the gateway/nginx 600 MB body limit by default.
        self.MAX_MODEL_UPLOAD_BYTES = int(os.environ.get(
            "MAX_MODEL_UPLOAD_BYTES", default=600 * 1024 * 1024))
        self.MODEL_PATH = os.environ.get(
            "MODEL_PATH",
            default=os.path.join(self.MODELS_DIR, "weights.engine"),
        )

        # Label file path — sibling .txt of the active model file
        # (e.g. weights.engine -> weights.txt). Switched on model selection
        # by ModelService.select_model(); the env var overrides the default.
        model_base, _ = os.path.splitext(self.MODEL_PATH)
        self.CLASSES_FILE_PATH = os.environ.get(
            "CLASSES_FILE_PATH",
            default=f"{model_base}.txt",
        )
        # What the two paths above point at when no model is selected —
        # ModelService restores them when an application change deselects a
        # model of another task.
        self.DEFAULT_MODEL_PATH = self.MODEL_PATH
        self.DEFAULT_CLASSES_FILE_PATH = self.CLASSES_FILE_PATH
        
        self.DEFAULT_LABELS = ["CLASS1", "CLASS2", "CLASS3"]

        self.SHM_NAME = os.environ.get("SHM_NAME", "conecsa_frame_shm")

        # Offline detection buffer (store-and-forward while the hub is not
        # polling). Ring-buffer caps: whichever limit is hit first evicts the
        # oldest records. The threshold is how long without a hub snapshot
        # pull before the device considers the hub offline.
        self.DETECTION_BUFFER_MAX_RECORDS = int(
            os.environ.get("DETECTION_BUFFER_MAX_RECORDS", default=5000))
        self.DETECTION_BUFFER_MAX_BYTES = int(
            os.environ.get("DETECTION_BUFFER_MAX_BYTES", default=1_073_741_824))
        # Offline sampling interval; 1.0 matches the hub's snapshot poll.
        self.DETECTION_BUFFER_SAMPLE_SEC = float(
            os.environ.get("DETECTION_BUFFER_SAMPLE_SEC", default=1.0))
        self.HUB_OFFLINE_THRESHOLD_SEC = float(
            os.environ.get("HUB_OFFLINE_THRESHOLD_SEC", default=5.0))

    def set_overlay_threshold(self, threshold: float) -> bool:
        """Set the overlay threshold value."""
        if 0.0 <= threshold <= 1.0:
            self.OVERLAY_THRESHOLD = threshold
            return True
        return False

    def set_segment_max_instances(self, limit) -> bool:
        """Set the segmentation instance limit (an integer 1..255); False otherwise."""
        if isinstance(limit, int) and not isinstance(limit, bool) and 1 <= limit <= 255:
            self.SEGMENT_MAX_INSTANCES = limit
            return True
        return False

    def set_face_match_threshold(self, threshold) -> bool:
        """Set the face match similarity threshold (0..1); False otherwise."""
        if valid_face_match_threshold(threshold):
            self.FACE_MATCH_THRESHOLD = float(threshold)
            return True
        return False

    def set_face_min_size(self, size_px) -> bool:
        """Set the smallest recognized face side in pixels (0..1024); False otherwise."""
        if valid_face_min_size(size_px):
            self.FACE_MIN_SIZE_PX = size_px
            return True
        return False

    def set_face_max_faces(self, count) -> bool:
        """Set the faces recognized per frame (1..20); False otherwise."""
        if valid_face_max_faces(count):
            self.FACE_MAX_FACES = count
            return True
        return False

    def set_confidence_threshold(self, threshold: float) -> bool:
        """Set the detection confidence threshold (0..1); False when out of range."""
        if 0.0 <= threshold <= 1.0:
            self.CONFIDENCE_THRESHOLD = threshold
            return True
        return False
