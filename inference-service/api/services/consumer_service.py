# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Consumer service - owns the shared-memory transport with the webcam-server.

A single always-on background thread polls the POSIX SHM segment and publishes
the latest frame (cheap: just stores the JPEG bytes + a monotonically increasing
sequence number; no decode here). The processing pipeline reads from this
latest-frame slot, so the capture is consumed once regardless of how many
pipeline lanes are attached.

With the IP camera source the main segment carries no frames: its health says
how many cameras are configured, and every camera publishes into its own ring
(``conecsa_shm.mosaic``). The reader then keeps the latest frame of each camera
and hands the pipeline the whole set, which stage A composes into the mosaic —
composing here would cost a full-frame copy for every frame of every camera.
"""
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

# noinspection PyPackageRequirements
import numpy as np  # ships in conecsa-os-base:base
from conecsa_shm.camera_ring import CameraRingReader
from conecsa_shm.mosaic import IpCameraRings

# api/proto is created by Dockerfile.inference-service (and shimmed onto sys.path by
# tests/conftest.py); it does not exist in the checkout, so it is unresolvable statically.
from ..proto import shm_pb2  # pyright: ignore[reportMissingImports]
from .frame_codec import encode_frame

logger = logging.getLogger(__name__)

# How often the reader re-reads the main health for the IP camera count.
_COUNT_REFRESH_S = 0.25


@dataclass
class CapturedFrame:
    """One unit of capture: a single frame, or the IP cameras' latest set.

    ``jpg``/``npy`` carry a single-camera frame (as before). ``cameras`` is set
    instead for the IP camera source: the latest frame of each configured
    camera in mosaic order, ``None`` for one without a recent frame.
    """

    seq: int
    jpg: Optional[bytes] = None
    npy: Optional[np.ndarray] = None
    cameras: Optional[List[Optional[np.ndarray]]] = None


class ConsumerService:
    """Reads frames from the webcam-server SHM segment and fans them out."""

    def __init__(self, shm_name: str):
        # writable=True: the camera config/health channel writes back through the
        # same segment.
        self._shm = CameraRingReader(shm_name, writable=True)
        # The per-camera rings of the IP camera source (read only).
        self._ip = IpCameraRings(shm_name)
        self._ip_count = 0
        self._ip_count_read = 0.0

        # Shared latest frame — written by the background reader, read by the
        # processing pipeline. Protected by _cond.
        self._latest_jpg: Optional[bytes] = None
        self._latest_npy: Optional[np.ndarray] = None
        self._latest_cameras: Optional[List[Optional[np.ndarray]]] = None
        self._seq = 0  # monotonically increasing
        self._cond = threading.Condition(threading.Lock())

        # Called from the reader thread whenever the segment is (re)mapped.
        self._on_attach: Optional[Callable[[], None]] = None
        self._attach_seen = 0

        self._thread = threading.Thread(
            target=self._reader, daemon=True, name="webcam-reader"
        )
        self._thread.start()
        logger.info("[ConsumerService] Background camera reader started (/%s)", shm_name)

    # ── Background reader ──

    def _reader(self) -> None:
        """Poll the SHM segment for new frames and publish the latest."""
        last_cam_seq = 0
        while True:
            if not self._shm.is_available():
                time.sleep(0.5)
                self._shm.open()
                continue

            self._notify_attach()
            if self._poll_ip_cameras():
                continue
            result = self._shm.get_latest_frame(last_cam_seq)
            if result is not None:
                frame_data, last_cam_seq = result

                if isinstance(frame_data, np.ndarray):
                    # Raw-RGB producer → already a BGR ndarray. Encode a JPEG so
                    # the raw passthrough stream keeps working.
                    npy = frame_data
                    jpg = encode_frame(npy)
                else:
                    # JPEG producer (common path): keep only the bytes and defer
                    # decoding to the pipeline, which decodes at a reduced scale
                    # on its own worker thread. This keeps the reader cheap enough
                    # to track the full capture fps.
                    jpg = frame_data
                    npy = None

                with self._cond:
                    self._latest_jpg = jpg
                    self._latest_npy = npy
                    self._latest_cameras = None
                    self._seq += 1
                    self._cond.notify_all()
            else:
                time.sleep(0.001)  # 1 ms poll interval

    def _poll_ip_cameras(self) -> bool:
        """Serve the IP camera source; False when it is not the active one."""
        now = time.monotonic()
        if now - self._ip_count_read >= _COUNT_REFRESH_S:
            self._ip_count_read = now
            health = self.read_health()
            count = int(getattr(health, "ip_camera_count", 0) or 0)
            if count != self._ip_count:
                logger.info("[ConsumerService] IP cameras: %d", count)
                self._ip.clear()
                self._ip_count = count
        if self._ip_count <= 0:
            return False
        if self._ip.poll(self._ip_count):
            cameras = self._ip.frames(self._ip_count)
            with self._cond:
                self._latest_jpg = None
                self._latest_npy = None
                self._latest_cameras = cameras
                self._seq += 1
                self._cond.notify_all()
        else:
            time.sleep(0.002)
        return True

    @property
    def ip_camera_count(self) -> int:
        """Configured IP cameras as the webcam-server reports them (0 = other source)."""
        return self._ip_count

    def read_ip_health(self, index: int):
        """IP camera ``index``'s HealthStatus from its own ring, or ``None``."""
        data = self._ip.health_bytes(index)
        if not data:
            return None
        status = shm_pb2.HealthStatus()
        status.ParseFromString(data)
        return status

    def set_on_attach(self, callback: Callable[[], None]) -> None:
        """Register the hook run whenever the camera segment is (re)mapped.

        The webcam-server recreates its segment on every start, and the fresh
        one carries no config: the hook lets the config owner publish again.
        A mapping made before the hook was registered is reported too.
        """
        self._on_attach = callback

    def _notify_attach(self) -> None:
        generation = self._shm.attach_generation
        if self._on_attach is None or generation == self._attach_seen:
            return
        self._attach_seen = generation
        try:
            self._on_attach()
        except Exception:  # noqa: BLE001 - the reader thread must survive the hook
            logger.exception("[ConsumerService] camera segment attach hook failed")

    # ── Latest-frame access (fan-out) ──

    def wait_for(
        self, last_seq: int, timeout: float = 5.0
    ) -> Tuple[int, Optional[bytes], Optional[np.ndarray]]:
        """Block until a frame newer than ``last_seq`` arrives (or timeout).

        Returns ``(seq, jpg, npy)``; ``seq`` equals ``last_seq`` (with ``None``
        payloads) when the wait timed out without a new frame.
        """
        got = self.wait_for_frame(last_seq, timeout)
        return got.seq, got.jpg, got.npy

    def wait_for_frame(self, last_seq: int, timeout: float = 5.0) -> CapturedFrame:
        """Like :meth:`wait_for`, with the IP cameras' set when that is the source.

        On timeout the result carries ``last_seq`` and no payload.
        """
        with self._cond:
            self._cond.wait_for(lambda: self._seq > last_seq, timeout=timeout)
            if self._seq > last_seq:
                return CapturedFrame(self._seq, self._latest_jpg, self._latest_npy,
                                     self._latest_cameras)
            return CapturedFrame(last_seq)

    # ── Config / health passthrough (same SHM segment) ──

    def is_available(self) -> bool:
        """True once the camera SHM segment is mapped."""
        return self._shm.is_available()

    def write_config(self, config) -> bool:
        """Write a CameraConfig back into the SHM header for the webcam-server.

        Returns whether it reached the segment (``False`` while none is mapped).
        """
        return self._shm.write_config_bytes(config.SerializeToString())

    def read_health(self):
        """Return the webcam-server's HealthStatus from SHM, or ``None``."""
        data = self._shm.read_health_bytes()
        if not data:
            return None
        status = shm_pb2.HealthStatus()
        status.ParseFromString(data)
        return status
