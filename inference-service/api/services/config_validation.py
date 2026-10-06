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
import secrets
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

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

# Capture source: a local V4L2 camera, a remote camera's MJPEG stream, or up to
# four RTSP IP cameras shown as a mosaic.
SOURCE_LOCAL = "local"
SOURCE_NETWORK = "network"
SOURCE_IP = "ip"
SOURCES: Tuple[str, ...] = (SOURCE_LOCAL, SOURCE_NETWORK, SOURCE_IP)
SOURCE_KEYS: Tuple[str, ...] = (
    "source", "network_host", "network_port", "network_token", "ip_cameras")

# IP cameras: the limits mirror the webcam-server's validator and keep four
# cameras well inside the SHM config region.
IP_CAMERAS_MAX = 4
IP_CAMERA_URL_MAX = 256
IP_CAMERA_CREDENTIAL_MAX = 64
IP_CAMERA_NAME_MAX = 40
_IP_CAMERA_ID_RE = re.compile(r"^cam-[0-9a-f]{8}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
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
    """The capture source: ``local``, ``network`` or ``ip``."""
    if value not in SOURCES:
        raise ConfigValidationError(f"source must be one of {', '.join(SOURCES)}")
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


def _text(key: str, value: Any, limit: int, allow_empty: bool = True) -> str:
    """A bounded single-line string field. The error never echoes the value."""
    if not isinstance(value, str):
        raise ConfigValidationError(f"{key} must be a string")
    text = value.strip()
    if not text and not allow_empty:
        raise ConfigValidationError(f"{key} is required")
    if len(text.encode()) > limit:
        raise ConfigValidationError(f"{key} must be at most {limit} bytes")
    if _CONTROL_RE.search(text):
        raise ConfigValidationError(f"{key} must not contain control characters")
    return text


def validate_rtsp_url(value: Any) -> str:
    """An IP camera's stream address: ``rtsp://host[:port]/path``, no credentials.

    The user name and password travel in their own fields (the password is
    write-only), so a URL carrying ``user:pass@`` is refused rather than stored
    where it would be echoed back. Loopback is refused as for the remote
    camera: the webcam-server runs in its own container.
    """
    url = _text("url", value, IP_CAMERA_URL_MAX, allow_empty=False)
    if any(ch.isspace() for ch in url):
        raise ConfigValidationError("url must not contain spaces")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise ConfigValidationError("url is not a valid RTSP address") from exc
    if parts.scheme != "rtsp":
        raise ConfigValidationError("url must start with rtsp://")
    if "@" in parts.netloc:
        raise ConfigValidationError(
            "url must not contain credentials; use the username and password fields")
    host = parts.hostname or ""
    if not host:
        raise ConfigValidationError("url must name a host")
    if port is not None and not 1 <= port <= 65_535:
        raise ConfigValidationError("url port must be between 1 and 65535")
    if host == "localhost":
        raise ConfigValidationError("url must name a host reachable by the device")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return url  # a host name
    if (address.is_unspecified or address.is_multicast or address.is_loopback
            or address == ipaddress.IPv4Address("255.255.255.255")):
        raise ConfigValidationError("url must name a host reachable by the device")
    return url


def new_ip_camera_id() -> str:
    """A fresh, stable identifier for a camera added to the list."""
    return f"cam-{secrets.token_hex(4)}"


def validate_ip_camera(entry: Any, position: int, with_id: bool) -> Dict[str, Any]:
    """One IP camera from a request (``with_id=False``) or the store.

    A request entry may omit ``id`` (a new camera) and ``password`` (keep the
    stored one); the returned dict then lacks those keys. A stored entry has
    every key.
    """
    if not isinstance(entry, dict):
        raise ConfigValidationError(f"ip_cameras[{position}] must be an object")
    out: Dict[str, Any] = {}
    if "id" in entry or with_id:
        camera_id = entry.get("id")
        if not isinstance(camera_id, str) or not _IP_CAMERA_ID_RE.match(camera_id):
            raise ConfigValidationError(f"ip_cameras[{position}].id is not a camera id")
        out["id"] = camera_id
    name = _text("name", entry.get("name", ""), IP_CAMERA_NAME_MAX)
    out["name"] = name or f"Camera {position + 1}"
    out["url"] = validate_rtsp_url(entry.get("url"))
    out["username"] = _text("username", entry.get("username", ""), IP_CAMERA_CREDENTIAL_MAX)
    if "password" in entry or with_id:
        password = entry.get("password", "")
        if not isinstance(password, str):
            raise ConfigValidationError("password must be a string")
        if len(password.encode()) > IP_CAMERA_CREDENTIAL_MAX:
            raise ConfigValidationError(
                f"password must be at most {IP_CAMERA_CREDENTIAL_MAX} bytes")
        if _CONTROL_RE.search(password):
            raise ConfigValidationError("password must not contain control characters")
        out["password"] = password
    return out


def _validate_ip_camera_list(value: Any, with_id: bool) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        raise ConfigValidationError("ip_cameras must be a list")
    if len(value) > IP_CAMERAS_MAX:
        raise ConfigValidationError(f"at most {IP_CAMERAS_MAX} IP cameras are supported")
    cameras = [validate_ip_camera(entry, i, with_id) for i, entry in enumerate(value)]
    ids = [c["id"] for c in cameras if "id" in c]
    if len(ids) != len(set(ids)):
        raise ConfigValidationError("ip_cameras holds the same camera twice")
    return cameras


def merge_ip_cameras(stored: List[Dict[str, Any]],
                     requested: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Resolve a validated request list against the stored one.

    The request is the complete, ordered list. An entry without ``id`` is a
    new camera; an entry with one must name a stored camera and, without
    ``password``, keeps that camera's stored password (it is write-only, so a
    client never has it to send back). ``password: ""`` clears it.
    """
    by_id = {c["id"]: c for c in stored}
    merged = []
    for position, entry in enumerate(requested):
        camera = dict(entry)
        if "id" in camera:
            previous = by_id.get(camera["id"])
            if previous is None:
                raise ConfigValidationError(f"ip_cameras[{position}].id is not a known camera")
            camera.setdefault("password", previous["password"])
        else:
            camera["id"] = new_ip_camera_id()
            camera.setdefault("password", "")
        merged.append(camera)
    return merged


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
    if "ip_cameras" in data:
        # Still request-shaped (ids/passwords may be missing): the service
        # resolves it against the stored list with merge_ip_cameras.
        patch["ip_cameras"] = _validate_ip_camera_list(data["ip_cameras"], with_id=False)
    return patch


def validate_source_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a complete, merged capture-source state before it is written.

    A network source needs a full host/port/token and an IP source at least
    one camera. Any other source may keep the dormant settings of the others
    — whole, partial or none — so switching back does not force the operator
    to type them again.
    """
    source = validate_source(state.get("source"))
    host = state.get("network_host") or ""
    port = state.get("network_port") or 0
    token = state.get("network_token") or ""
    if source == SOURCE_NETWORK and not (host and port and token):
        raise ConfigValidationError(
            "a network source needs network_host, network_port and network_token")
    cameras = _validate_ip_camera_list(state.get("ip_cameras") or [], with_id=True)
    if source == SOURCE_IP and not cameras:
        raise ConfigValidationError("an IP camera source needs at least one camera")
    return {
        "source": source,
        "network_host": validate_network_host(host) if host else "",
        "network_port": validate_int("network_port", port, NETWORK_PORT_BOUNDS) if port else 0,
        "network_token": normalize_network_token(token) if token else "",
        "ip_cameras": cameras,
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
    for key, float_bounds in STEREO_FLOAT_BOUNDS.items():
        stereo[key] = validate_float(key, data[key], float_bounds) if key in data else None
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
