# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Video service - owns the camera configuration exchanged with the webcam-server.

Frame transport (SHM read/fan-out) lives in ``ConsumerService`` and the
image operations (decode/encode/stereo/RGB) in ``FrameCodecService``; this
service is the camera-config facade used by the config/settings controllers:
it merges and writes ``CameraConfig`` into the SHM config region and reads the
producer health, and it delegates stereo settings to the codec so
callers keep a single entry point.

Two kinds of camera state meet here and are kept apart on purpose. The local
camera *tuning* (``_current_config``) is per-model: model settings load and save
it. The capture *source* (``_source``: local camera or a remote camera's stream, with
its address and secret token) is per-device: it lives in ``CameraSourceStore``,
survives a model switch, and never enters a model's settings file.
"""
import json
import logging
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

from .camera_source_store import CameraSourceStore
from .config_validation import (
    LOCAL_CAMERA_KEYS,
    SOURCE_KEYS,
    SOURCE_LOCAL,
    SOURCE_NETWORK,
    ConfigValidationError,
    validate_camera_patch,
    validate_source_state,
)

logger = logging.getLogger(__name__)

# Health values published by the webcam-server into the SHM header (see
# proto/shm.proto). Only "capturing" means frames are actually flowing.
CAMERA_STATUS_CAPTURING = "capturing"
CAMERA_STATUS_NO_CAMERA = "no_camera"

# Stable API text for ``CameraHealthDetail`` (proto/shm.proto): why a network
# source is not capturing. A value this build does not know reads as unspecified.
CAMERA_DETAIL_UNSPECIFIED = "unspecified"
_CAMERA_DETAILS = {
    0: CAMERA_DETAIL_UNSPECIFIED,
    1: "connecting",
    2: "unauthorized",
    3: "rate_limited",
    4: "unreachable",
    5: "stalled",
    6: "bad_stream",
}

_WEBCAM_UNREACHABLE = ("Failed to reach webcam server. Ensure it is running and "
                       "shared memory is accessible.")


class VideoService:
    """Camera configuration + producer health (via the shared SHM transport)."""

    def __init__(self, consumer_service, codec_service, model_directory: Optional[str] = None):
        """
        Args:
            consumer_service: ConsumerService owning the SHM transport.
            codec_service: FrameCodecService owning stereo/codec configuration.
            model_directory: where the device-level capture source is persisted;
                ``None`` keeps it in memory only.
        """
        self._consumer = consumer_service
        self._codec = codec_service
        # Serializes config writes: the gRPC threads and the SHM reader thread
        # (re-publish after a producer restart) both publish.
        self._lock = threading.RLock()

        # Last config successfully written to the webcam-server via SHM.
        # Used as the base when merging partial patches so we don't need to
        # round-trip config fields through HealthStatus.
        # Seeded from the same env vars compose sets on the webcam-server; the
        # code defaults below (640x640@30) match the webcam-server image ENV,
        # not the webcam-server binary defaults (2560x720@60).
        _framerate = int(os.environ.get("CAPTURE_FRAMERATE", 30))
        self._current_config: Dict = {
            "camera_index":  int(os.environ.get("CAMERA_INDEX",          0)),
            "width":         int(os.environ.get("CAPTURE_WIDTH",          640)),
            "height":        int(os.environ.get("CAPTURE_HEIGHT",         640)),
            "framerate":     _framerate,
            "auto_exposure": os.environ.get("CAPTURE_AUTO_EXPOSURE", "false").lower() == "true",
            "exposure_time": int(os.environ.get("CAPTURE_EXPOSURE_TIME",  10_000 // max(_framerate, 1))),
            "rgb_red":       int(os.environ.get("CAPTURE_RGB_RED",        128)),
            "rgb_green":     int(os.environ.get("CAPTURE_RGB_GREEN",      128)),
            "rgb_blue":      int(os.environ.get("CAPTURE_RGB_BLUE",       128)),
            "gamma":         int(os.environ.get("CAPTURE_GAMMA",          100)),
            "gain":          int(os.environ.get("CAPTURE_GAIN",           0)),
        }

        # Device-level capture source, loaded before any model settings apply.
        # Until an administrator saves one, messages carry no ``source`` at all,
        # which the webcam-server reads as "leave the source alone" — so its
        # CAMERA_SOURCE bootstrap env stays in charge.
        self._source_store = CameraSourceStore(model_directory) if model_directory else None
        stored = self._source_store.load() if self._source_store else None
        self._source_configured = stored is not None
        self._source: Dict[str, Any] = stored or {
            "source": SOURCE_LOCAL, "network_host": "", "network_port": 0, "network_token": "",
        }
        # Whether a config was ever meant to reach the webcam-server: what the
        # attach hook replays when the producer recreates its segment.
        self._publish_wanted = False

    # ── Webcam-server config (via SHM) ──

    def get_webcam_server_config(self) -> Optional[Dict]:
        """Return the webcam-server status plus the last-known config."""
        health = self._consumer.read_health()
        if health:
            return {"status": health.status, **self._current_config}
        return None

    # ── Camera health events ──

    def start_health_watch(self, event_service, interval: float = 1.0) -> None:
        """Publish ``camera_health_changed`` whenever the webcam-server's health
        (status, detail) changes, so the device screen and the hub follow the
        camera without polling the device list.

        The health lives in SHM and nobody is told when the webcam-server
        rewrites it, so a small daemon thread samples it every ``interval``
        seconds; only a change is published.
        """
        self._health_events = event_service
        self._health_seen: Optional[Tuple[str, str]] = None
        thread = threading.Thread(target=self._health_watch_loop, args=(interval,),
                                  name="camera-health", daemon=True)
        thread.start()

    def _health_watch_loop(self, interval: float) -> None:
        while True:
            try:
                self._health_tick()
            except Exception:  # noqa: BLE001 - a failed sample must not end the watch
                logger.exception("camera health sample failed")
            time.sleep(interval)

    def _health_tick(self) -> bool:
        """Sample the health once; publish and return True when it changed."""
        status, detail = self.camera_status(), self.camera_detail()
        if (status, detail) == self._health_seen:
            return False
        self._health_seen = (status, detail)
        with self._lock:
            source = (self._stored_source() or {}).get("source", SOURCE_LOCAL)
        self._health_events.publish(
            "camera_health_changed", keys=["camera_health"], source="camera",
            data={"status": status, "detail": detail, "source": source})
        return True

    # ── Camera liveness (SHM health) — the gate for starting detection ──

    def camera_status(self) -> str:
        """Return the webcam-server health status, or ``"no_camera"`` when the
        SHM segment carries no health yet (producer down / not started)."""
        health = self._consumer.read_health()
        return health.status if health else CAMERA_STATUS_NO_CAMERA

    def camera_connected(self) -> bool:
        """True only while the webcam-server is actually streaming a camera.

        The webcam-server publishes no frames at all without a camera, so any
        status other than ``"capturing"`` means detection would run blind.
        """
        return self.camera_status() == CAMERA_STATUS_CAPTURING

    def wait_for_camera(self, timeout: float = 15.0, interval: float = 0.5) -> bool:
        """Poll the SHM health until the camera streams, or ``timeout`` elapses.

        Used at startup: the webcam-server opens the device concurrently with
        the inference boot, so a plain check would race it.
        """
        deadline = time.monotonic() + timeout
        while True:
            if self.camera_connected():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(interval)

    def camera_detail(self) -> str:
        """Why the camera is not capturing, as stable lower-snake-case text."""
        health = self._consumer.read_health()
        return _CAMERA_DETAILS.get(getattr(health, "detail", 0), CAMERA_DETAIL_UNSPECIFIED)

    def get_current_camera_config(self) -> Dict:
        """Return a copy of the last-applied local camera tuning."""
        return dict(self._current_config)

    def get_model_camera_config(self) -> Dict:
        """The camera fields a model's settings file may hold: local tuning only.

        The capture source and its token are device-level and are not in here —
        by construction, not by a caller remembering to strip them.
        """
        return {key: self._current_config[key] for key in LOCAL_CAMERA_KEYS}

    def rgb_levels(self) -> Tuple[int, int, int]:
        """Return the current software RGB levels ``(r, g, b)`` (128 = neutral)."""
        c = self._current_config
        return (
            int(c.get("rgb_red", 128)),
            int(c.get("rgb_green", 128)),
            int(c.get("rgb_blue", 128)),
        )

    def apply_webcam_server_config(self, patch: Dict) -> bool:
        """Write camera config into the SHM config region.

        Merges the given patch over the last-known configuration stored in
        ``self._current_config`` so that partial updates don't reset other
        fields. Only local tuning keys are taken from ``patch``: a model's
        settings file cannot select a capture source. Returns True on success.
        """
        tuning = {key: patch[key] for key in LOCAL_CAMERA_KEYS if key in patch}
        with self._lock:
            try:
                # Merge patch over the last-known config.
                base = {**self._current_config, **tuning}
                self._publish(base, self._stored_source())
                self._current_config = base
                return True
            except Exception as ex:
                logger.error(f"Failed to write config to SHM: {ex}")
                return False

    def _stored_source(self) -> Optional[Dict[str, Any]]:
        """The source to put on the wire: ``None`` until one was ever saved."""
        return self._source if self._source_configured else None

    def _publish(self, config: Dict, source: Optional[Dict[str, Any]]) -> bool:
        """Serialize ``config`` plus ``source`` into SHM; True when it landed.

        ``source=None`` sends no source at all, which the webcam-server reads
        as "leave it alone". ``False`` means no segment is mapped (webcam-server
        down or not up yet); the attach hook publishes again once one is.
        """
        from ..proto import shm_pb2  # pyright: ignore[reportMissingImports]

        cfg = shm_pb2.CameraConfig(
            camera_index=config["camera_index"],
            width=config["width"],
            height=config["height"],
            framerate=config["framerate"],
            auto_exposure=config["auto_exposure"],
            exposure_time=config["exposure_time"],
            rgb_red=config["rgb_red"],
            rgb_green=config["rgb_green"],
            rgb_blue=config["rgb_blue"],
            gamma=config["gamma"],
            gain=config["gain"],
        )
        if source is not None:
            # The enum is set from a validated name, never from a raw number:
            # an out-of-range value would not fit the config region.
            cfg.source = (shm_pb2.CAMERA_SOURCE_NETWORK if source["source"] == SOURCE_NETWORK
                          else shm_pb2.CAMERA_SOURCE_LOCAL)
            cfg.network_host = source["network_host"]
            cfg.network_port = source["network_port"]
            cfg.network_token = source["network_token"]
        self._publish_wanted = True
        # ``None`` comes from transports predating the confirmed write.
        return self._consumer.write_config(cfg) is not False

    # ── Device-level capture source ──

    def publish_startup_source(self) -> None:
        """Send the stored capture source to the webcam-server at boot.

        Model settings only push the camera config when a model has some, so
        without this a device with no model selected would never restore its
        source. Nothing is sent when no source was ever saved.
        """
        with self._lock:
            if not self._source_configured:
                return
            try:
                if not self._publish(self._current_config, self._stored_source()):
                    logger.info("Camera source not published yet: no SHM segment; "
                                "it is sent when the webcam-server comes up")
            except Exception as ex:  # noqa: BLE001
                logger.error("Failed to publish the camera source: %s", ex)

    def republish(self) -> None:
        """Replay the desired config after the webcam-server recreated its segment.

        A fresh segment carries no config, so the webcam-server would fall back
        to its start-up defaults — for the source, silently a different camera.
        """
        with self._lock:
            if not self._publish_wanted:
                return
            try:
                self._publish(self._current_config, self._stored_source())
                logger.info("Camera config re-published to a new SHM segment")
            except Exception as ex:  # noqa: BLE001
                logger.error("Failed to re-publish the camera config: %s", ex)

    @staticmethod
    def is_source_only_update(data: Any) -> bool:
        """True when a request body touches nothing a model's settings hold."""
        return isinstance(data, dict) and bool(data) and set(data) <= set(SOURCE_KEYS)

    def _apply_source_update(self, source_patch: Dict[str, Any],
                             tuning: Dict[str, Any]) -> Tuple[bool, str, int]:
        """One transaction: validate the merged state, persist, publish, commit.

        Memory changes only after both the file and SHM took the candidate. If
        the publish fails, the previous file comes back (best effort).
        """
        candidate = validate_source_state({**self._source, **source_patch})
        config = {**self._current_config, **tuning}
        previous = dict(self._source) if self._source_configured else None

        if self._source_store is not None:
            try:
                self._source_store.save(candidate)
            except OSError as exc:
                logger.error("Failed to persist the camera source: %s", exc)
                return False, "Failed to persist the camera source", 500
        try:
            published = self._publish(config, candidate)
        except Exception as ex:  # noqa: BLE001
            logger.error("Failed to write the camera source to SHM: %s", ex)
            published = False
        if not published:
            if self._source_store is not None:
                self._source_store.restore(previous)
            return False, _WEBCAM_UNREACHABLE, 503

        self._source = candidate
        self._source_configured = True
        self._current_config = config
        return True, "Camera configuration applied", 200

    # ── Camera devices + update (business logic; controllers/gRPC are thin) ──

    def list_camera_devices(self) -> Dict:
        """Enumerate V4L2 devices and return them with the current camera +
        stereo configuration (the payload the camera-devices endpoint serves)."""
        devices = []
        try:
            if os.path.isdir("/dev"):
                for entry in sorted(os.listdir("/dev")):
                    if not entry.startswith("video"):
                        continue
                    full = os.path.join("/dev", entry)
                    try:
                        idx = int(entry.replace("video", ""))
                        name_path = f"/sys/class/video4linux/{entry}/name"
                        name = (open(name_path).read().strip()
                                if os.path.isfile(name_path) else entry)
                        devices.append({"path": full, "index": idx, "name": name})
                    except (ValueError, OSError):
                        devices.append({"path": full, "index": -1, "name": entry})
        except Exception as ex:  # noqa: BLE001
            logger.warning("Could not enumerate camera devices: %s", ex)

        supported_formats: Dict = {}
        try:
            fp = "/dev/shm/conecsa_camera_formats.json"
            if os.path.isfile(fp):
                with open(fp) as f:
                    supported_formats = json.load(f)
        except Exception as ex:  # noqa: BLE001
            logger.warning("Could not read camera formats: %s", ex)

        wc = self.get_webcam_server_config()
        g = lambda k, d: wc.get(k, d) if wc else d  # noqa: E731
        camera_status = g("status", CAMERA_STATUS_NO_CAMERA)
        stereo = self.get_stereo_config()
        cur_index = g("camera_index", 0)
        cur_path = f"/dev/video{cur_index}"
        if cur_path not in {d["path"] for d in devices} and cur_index >= 0:
            devices.insert(0, {"path": cur_path, "index": cur_index, "name": f"video{cur_index}"})
        for d in devices:
            d["supported_formats"] = supported_formats.get(d["path"], [])

        return {
            "devices": devices, "supported_formats": supported_formats,
            "camera_status": camera_status,
            "camera_detail": self.camera_detail(),
            "camera_connected": camera_status == CAMERA_STATUS_CAPTURING,
            # The token is write-only: only whether one is stored is reported.
            "current_source": self._source["source"],
            "current_network_host": self._source["network_host"],
            "current_network_port": self._source["network_port"],
            "network_token_set": bool(self._source["network_token"]),
            "current_device": cur_path, "current_index": cur_index,
            "current_width": g("width", 640), "current_height": g("height", 640),
            "current_framerate": g("framerate", 30),
            "current_auto_exposure": g("auto_exposure", False),
            "current_exposure_time": g("exposure_time", 333),
            "current_rgb_red": g("rgb_red", 128), "current_rgb_green": g("rgb_green", 128),
            "current_rgb_blue": g("rgb_blue", 128), "current_gamma": g("gamma", 100),
            "current_gain": g("gain", 0),
            "exposure_time_min": g("exposure_time_min", 1),
            "exposure_time_max": g("exposure_time_max", 300000),
            "current_stereo_enabled": bool(stereo.get("enabled", False)),
            "current_stereo_blend_alpha": float(stereo.get("alpha", 0.5)),
            "current_stereo_offset": float(stereo.get("offset", 0.0)),
            "current_stereo_offset_y": float(stereo.get("offset_y", 0.0)),
        }

    def apply_camera_update(self, data: Dict) -> Tuple[bool, str, int]:
        """Validate + apply a camera-config patch (webcam fields + stereo).

        Returns ``(ok, message, status)`` where status mirrors the HTTP codes
        (200 ok, 400 validation, 503 webcam unreachable) so both the REST
        controller and the gRPC servicer can stay thin. Validation lives in
        ``config_validation`` and is shared with the generic config patch. The
        webcam-server push comes before the stereo settings, so a body the
        camera cannot accept changes nothing on either side.
        """
        try:
            patch = validate_camera_patch(data)
        except ConfigValidationError as exc:
            return False, str(exc), 400
        try:
            if patch.source:
                # The source and any tuning in the same body go out as one message.
                with self._lock:
                    try:
                        ok, message, status = self._apply_source_update(
                            patch.source, patch.webcam)
                    except ConfigValidationError as exc:
                        return False, str(exc), 400
                if not ok:
                    return False, message, status
            elif patch.webcam and not self.apply_webcam_server_config(patch.webcam):
                return False, _WEBCAM_UNREACHABLE, 503
            if patch.has_stereo:
                self.set_stereo_config(patch.stereo_enabled, patch.stereo_alpha,
                                       patch.stereo_offset, patch.stereo_offset_y)
            return True, "Camera configuration applied", 200
        except Exception as ex:  # noqa: BLE001
            return False, str(ex), 500

    # ── Stereo configuration (delegated to the codec) ──

    def get_stereo_config(self) -> Dict:
        """Return the current stereo combine settings."""
        return self._codec.get_stereo_config()

    def set_stereo_config(
        self,
        enabled: Optional[bool] = None,
        alpha: Optional[float] = None,
        offset: Optional[float] = None,
        offset_y: Optional[float] = None,
    ) -> None:
        """Update stereo combine settings (partial; unset fields are kept)."""
        self._codec.set_stereo_config(enabled, alpha, offset, offset_y)
