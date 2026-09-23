# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
One validation layer for every camera/config update.

Two API surfaces change the same state — the dedicated camera path
(``VideoService.apply_camera_update``, ``POST /api/v1/camera/config``) and the
generic config patch (``ConfigService.update_config``, ``PUT /api/v1/config``).
Both parse and bound their fields here, so a value
is either rejected identically on both paths (status 400) or applied
identically.
"""
import ipaddress
import re
from dataclasses import dataclass, field
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

# The camera fields a model may carry in its settings file: the local-camera
# tuning. The capture source below is device-level state and never one of them.
LOCAL_CAMERA_KEYS: Tuple[str, ...] = (*CAMERA_INT_BOUNDS, "auto_exposure")

# Capture source: a local V4L2 camera or a remote camera's MJPEG stream.
SOURCE_LOCAL = "local"
SOURCE_NETWORK = "network"
SOURCE_KEYS: Tuple[str, ...] = ("source", "network_host", "network_port", "network_token")
NETWORK_PORT_BOUNDS: Tuple[int, int] = (1, 65_535)
# Crockford base32: digits and upper-case letters without I, L, O and U.
_NETWORK_TOKEN_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{8,32}$")


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


def validate_source(value: Any) -> str:
    """The capture source: ``local`` or ``network``."""
    if value not in (SOURCE_LOCAL, SOURCE_NETWORK):
        raise ConfigValidationError(f"source must be {SOURCE_LOCAL} or {SOURCE_NETWORK}")
    return value


def validate_network_host(value: Any) -> str:
    """The remote camera's address: an IPv4 literal a direct link can carry.

    Private and link-local ranges are what a hotspot or soft-AP hands out, so
    they are allowed. Loopback is refused everywhere, development included:
    the webcam-server runs in its own container, where loopback is itself.
    """
    try:
        address = ipaddress.IPv4Address(str(value).strip())
    except ValueError as exc:
        raise ConfigValidationError("network_host must be an IPv4 address") from exc
    if (address.is_unspecified or address.is_multicast or address.is_loopback
            or address == ipaddress.IPv4Address("255.255.255.255")):
        raise ConfigValidationError(
            "network_host must be a unicast address reachable by the device")
    return str(address)


def normalize_network_token(value: Any) -> str:
    """Normalize and check a stream token. The error never echoes the value."""
    token = value.replace("-", "").upper() if isinstance(value, str) else ""
    if not _NETWORK_TOKEN_RE.match(token):
        raise ConfigValidationError(
            "network_token must be 8 to 32 Crockford base32 characters")
    return token


def validate_source_patch(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the capture-source fields present in a request body.

    An omitted token keeps the stored one; an empty token is invalid rather
    than a way to clear it.
    """
    patch: Dict[str, Any] = {}
    if "source" in data:
        patch["source"] = validate_source(data["source"])
    if "network_host" in data:
        patch["network_host"] = validate_network_host(data["network_host"])
    if "network_port" in data:
        patch["network_port"] = validate_int(
            "network_port", data["network_port"], NETWORK_PORT_BOUNDS)
    if "network_token" in data:
        patch["network_token"] = normalize_network_token(data["network_token"])
    return patch


def validate_source_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a complete, merged capture-source state before it is written.

    A network source needs a full host/port/token. A local source may keep
    dormant network credentials — whole, partial or none — so switching back
    does not force the operator to type them again.
    """
    source = validate_source(state.get("source"))
    host = state.get("network_host") or ""
    port = state.get("network_port") or 0
    token = state.get("network_token") or ""
    if source == SOURCE_NETWORK and not (host and port and token):
        raise ConfigValidationError(
            "a network source needs network_host, network_port and network_token")
    return {
        "source": source,
        "network_host": validate_network_host(host) if host else "",
        "network_port": validate_int("network_port", port, NETWORK_PORT_BOUNDS) if port else 0,
        "network_token": normalize_network_token(token) if token else "",
    }


@dataclass
class CameraPatch:
    """A validated camera update: the SHM fields, the capture source and the
    stereo settings."""
    webcam: Dict[str, Any]
    source: Dict[str, Any] = field(default_factory=dict)
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
        source=validate_source_patch(data),
        stereo_enabled=coerce_bool(data["stereo_enabled"]) if "stereo_enabled" in data else None,
        stereo_alpha=stereo["stereo_blend_alpha"],
        stereo_offset=stereo["stereo_offset"],
        stereo_offset_y=stereo["stereo_offset_y"],
    )
    if not patch.webcam and not patch.source and not patch.has_stereo:
        raise ConfigValidationError("No recognised camera fields provided")
    return patch
