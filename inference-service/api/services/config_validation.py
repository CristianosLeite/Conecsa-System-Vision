"""
One validation layer for every camera/config update (review M2).

Two API surfaces change the same state — the dedicated camera path
(``VideoService.apply_camera_update``, ``POST /api/v1/camera/config``) and the
generic config patch (``ConfigService.update_config``, ``PUT /api/v1/config``).
They used to validate differently: the generic one wrote framerate and
confidence straight into the config with no bounds and swallowed a device it
could not parse, and the camera one bounded framerate but not the resolution
or the camera index. Both now parse and bound their fields here, so a value
is either rejected identically on both paths (status 400) or applied
identically.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# Inclusive bounds for the integer fields relayed to the webcam-server.
CAMERA_INT_BOUNDS: Dict[str, Tuple[int, int]] = {
    "camera_index": (0, 63),
    "width": (16, 8192),
    "height": (16, 8192),
    "framerate": (1, 240),
    "exposure_time": (1, 300_000),
    "rgb_red": (0, 255),
    "rgb_green": (0, 255),
    "rgb_blue": (0, 255),
    "gamma": (1, 500),
    "gain": (0, 480),
}

# Inclusive bounds for the stereo-combine floats (inference-side only).
STEREO_FLOAT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "stereo_blend_alpha": (0.0, 1.0),
    "stereo_offset": (-0.5, 0.5),
    "stereo_offset_y": (-0.5, 0.5),
}

CONFIDENCE_BOUNDS: Tuple[float, float] = (0.0, 1.0)


class ConfigValidationError(ValueError):
    """A request value is missing, malformed or out of range (HTTP 400)."""


def coerce_bool(value: Any) -> bool:
    """Coerce a bool/str/number request value to a bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def validate_int(key: str, value: Any, bounds: Tuple[int, int]) -> int:
    """Parse ``value`` as an int inside ``bounds`` or raise ``ConfigValidationError``."""
    if isinstance(value, bool):
        raise ConfigValidationError(f"{key} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(f"{key} must be an integer") from exc
    lo, hi = bounds
    if not lo <= parsed <= hi:
        raise ConfigValidationError(f"{key} must be between {lo} and {hi}")
    return parsed


def validate_float(key: str, value: Any, bounds: Tuple[float, float]) -> float:
    """Parse ``value`` as a float inside ``bounds`` or raise ``ConfigValidationError``."""
    if isinstance(value, bool):
        raise ConfigValidationError(f"{key} must be a number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(f"{key} must be a number") from exc
    lo, hi = bounds
    if not lo <= parsed <= hi:
        raise ConfigValidationError(f"{key} must be between {lo} and {hi}")
    return parsed


def validate_confidence(value: Any) -> float:
    """The detection confidence threshold, bounded to 0..1."""
    return validate_float("confidence_threshold", value, CONFIDENCE_BOUNDS)


def parse_capture_device(value: Any) -> int:
    """Camera index from ``/dev/videoN``, ``"N"`` or ``N``; bounded like ``camera_index``."""
    text = str(value).strip()
    if text.startswith("/dev/video"):
        text = text[len("/dev/video"):]
    try:
        return validate_int("camera_index", text, CAMERA_INT_BOUNDS["camera_index"])
    except ConfigValidationError as exc:
        raise ConfigValidationError(
            f"capture_device must be /dev/videoN or a camera index ({exc})") from exc


@dataclass
class CameraPatch:
    """A validated camera update: the SHM fields plus the stereo settings."""
    webcam: Dict[str, Any]
    stereo_enabled: Optional[bool] = None
    stereo_alpha: Optional[float] = None
    stereo_offset: Optional[float] = None
    stereo_offset_y: Optional[float] = None

    @property
    def has_stereo(self) -> bool:
        return any(v is not None for v in (
            self.stereo_enabled, self.stereo_alpha, self.stereo_offset, self.stereo_offset_y))


def validate_camera_patch(data: Dict[str, Any]) -> CameraPatch:
    """Validate a camera-config request body; raise ``ConfigValidationError``.

    Every recognised field is checked before anything is applied, so a body
    with one bad value changes nothing.
    """
    if not isinstance(data, dict) or not data:
        raise ConfigValidationError("No data provided")
    webcam: Dict[str, Any] = {}
    for key, bounds in CAMERA_INT_BOUNDS.items():
        if key in data:
            webcam[key] = validate_int(key, data[key], bounds)
    if "auto_exposure" in data:
        webcam["auto_exposure"] = coerce_bool(data["auto_exposure"])

    stereo: Dict[str, Optional[float]] = {}
    for key, bounds in STEREO_FLOAT_BOUNDS.items():
        stereo[key] = validate_float(key, data[key], bounds) if key in data else None
    patch = CameraPatch(
        webcam=webcam,
        stereo_enabled=coerce_bool(data["stereo_enabled"]) if "stereo_enabled" in data else None,
        stereo_alpha=stereo["stereo_blend_alpha"],
        stereo_offset=stereo["stereo_offset"],
        stereo_offset_y=stereo["stereo_offset_y"],
    )
    if not patch.webcam and not patch.has_stereo:
        raise ConfigValidationError("No recognised camera fields provided")
    return patch
