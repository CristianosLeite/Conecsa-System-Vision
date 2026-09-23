# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Config service - reads and applies the app capture/inference configuration.

Holds the business logic so the gRPC servicer stays a thin adapter that
delegates here.
"""
import contextlib
import logging
from typing import Dict, Tuple

from .config_validation import (
    CAMERA_INT_BOUNDS,
    ConfigValidationError,
    parse_capture_device,
    validate_confidence,
    validate_int,
)

logger = logging.getLogger(__name__)

_WEBCAM_UNREACHABLE = ("Failed to reach webcam server. Ensure it is running and "
                       "shared memory is accessible.")


class ConfigService:
    """Owns get/update of the capture device + framerate + confidence threshold,
    proxying camera changes to the webcam-server and persisting per-model."""

    def __init__(self, config, video_service=None, settings_service=None,
                 lifecycle_lock=None):
        self._config = config
        self._video = video_service
        self._settings = settings_service
        # ModelService.op_lock: a concurrent model select or activation must not
        # swap the settings file between a change and its save.
        self._lifecycle_lock = lifecycle_lock or contextlib.nullcontext()

    def get_config(self) -> Dict:
        """Return the current configuration (device, resolution, thresholds, …)."""
        c = self._config
        return {
            "capture_device": c.CAPTURE_DEVICE,
            "capture_resolution": [c.CAPTURE_RESOLUTION_X, c.CAPTURE_RESOLUTION_Y],
            "capture_framerate": c.CAPTURE_FRAMERATE,
            "model_path": c.MODEL_PATH,
            "confidence_threshold": c.CONFIDENCE_THRESHOLD,
        }

    def update_config(self, data: Dict) -> Tuple[bool, str, int]:
        """Apply a partial config patch.

        Returns ``(ok, message, status)`` with the same status convention as
        ``VideoService.apply_camera_update`` (200 ok, 400 validation, 503
        webcam-server unreachable). The whole patch is validated through the
        shared camera bounds first, then pushed to the webcam-server, and only
        an acknowledged patch is written to the config and persisted — the
        stored settings never describe a state the camera did not accept.
        """
        if not data:
            return False, "No data provided", 400
        c = self._config
        try:
            camera_index = (parse_capture_device(data["capture_device"])
                            if "capture_device" in data else None)
            framerate = (validate_int("capture_framerate", data["capture_framerate"],
                                      CAMERA_INT_BOUNDS["framerate"])
                         if "capture_framerate" in data else None)
            confidence = (validate_confidence(data["confidence_threshold"])
                          if "confidence_threshold" in data else None)
        except ConfigValidationError as exc:
            return False, str(exc), 400

        webcam_patch: Dict = {}
        if camera_index is not None:
            webcam_patch["camera_index"] = camera_index
        if framerate is not None:
            webcam_patch["framerate"] = framerate
        try:
            if webcam_patch and self._video is not None:
                if not self._video.apply_webcam_server_config(webcam_patch):
                    return False, _WEBCAM_UNREACHABLE, 503
            with self._lifecycle_lock:
                if camera_index is not None:
                    c.CAPTURE_DEVICE = f"/dev/video{camera_index}"
                if framerate is not None:
                    c.CAPTURE_FRAMERATE = framerate
                if confidence is not None:
                    c.CONFIDENCE_THRESHOLD = confidence
                if self._settings is not None:
                    self._settings.save()
            return True, "Configuration updated", 200
        except Exception as ex:  # noqa: BLE001
            logger.error("Error updating config: %s", ex)
            return False, str(ex), 500
