# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: Apache-2.0

"""IP camera mosaic: the per-camera rings and the frame that shows them all.

With the IP camera source the webcam-server publishes each camera into its own
ring, ``<SHM_NAME>_ip<N>`` (``N`` = the camera's position in the configured
list), and the main ring carries no frames. Every consumer that wants one frame
(the inference pipeline, the raw video feed, the training preview) composes the
cameras into a mosaic with this module, so they all agree on the layout:

- the grid follows the number of *configured* cameras, not of the cameras
  online, so a camera that drops keeps its cell (dark placeholder) and the
  detection areas, normalized to the whole frame, keep meaning the same thing;
- 1 camera → 1×1, 2 → 2×1 (side by side), 3–4 → 2×2;
- every cell has the same size, and a camera is letterboxed into its cell.

Pure numpy/cv2 plus :class:`~conecsa_shm.camera_ring.CameraRingReader`; the
health payloads stay opaque bytes, like everywhere else in this package.
"""
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol, Sequence, Tuple, Union

import cv2
import numpy as np

from .camera_ring import CameraRingReader

#: Most IP cameras one device takes (the proto and the validators agree).
MAX_IP_CAMERAS = 4

#: Cell fill where a camera has no frame (offline or not started yet), BGR.
PLACEHOLDER_BGR = (32, 32, 32)

_DEFAULT_CELL = (1280, 720)
_DEFAULT_STALE_SECS = 2.0


@dataclass(frozen=True)
class Rect:
    """A pixel box in the mosaic; ``x0``/``y0`` inclusive, sizes positive."""

    x0: int
    y0: int
    width: int
    height: int

    @property
    def x1(self) -> int:
        return self.x0 + self.width

    @property
    def y1(self) -> int:
        return self.y0 + self.height

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1


def ip_ring_name(shm_name: str, index: int) -> str:
    """The segment name of IP camera ``index`` (0-based) next to ``shm_name``."""
    return f"{shm_name}_ip{index}"


def grid_for(count: int) -> Tuple[int, int]:
    """``(columns, rows)`` for ``count`` cameras: 1×1, 2×1, 2×2, …"""
    if count < 1:
        raise ValueError(f"a mosaic needs at least one camera, got {count}")
    cols = math.ceil(math.sqrt(count))
    return cols, math.ceil(count / cols)


def cell_size_from_env() -> Tuple[int, int]:
    """The cell size from ``MOSAIC_CELL_WIDTH``/``MOSAIC_CELL_HEIGHT`` (1280×720).

    Inference sees each camera at this size, so it bounds both the detail a
    model gets and the work per camera. Values under 64 or unparsable fall back
    to the default; odd values are rounded down to even (JPEG chroma blocks).
    """
    def read(name: str, default: int) -> int:
        try:
            value = int(os.environ.get(name, default))
        except ValueError:
            return default
        return value - value % 2 if value >= 64 else default

    return read("MOSAIC_CELL_WIDTH", _DEFAULT_CELL[0]), read("MOSAIC_CELL_HEIGHT", _DEFAULT_CELL[1])


def cell_rects(count: int, cell_w: int, cell_h: int) -> List[Rect]:
    """The cells of a ``count``-camera mosaic, in camera order (row-major)."""
    cols, _ = grid_for(count)
    return [Rect((i % cols) * cell_w, (i // cols) * cell_h, cell_w, cell_h) for i in range(count)]


def fit_rect(cell: Rect, width: int, height: int) -> Rect:
    """Where a ``width``×``height`` image lands, letterboxed and centered in ``cell``."""
    scale = min(cell.width / width, cell.height / height)
    w = max(1, min(cell.width, round(width * scale)))
    h = max(1, min(cell.height, round(height * scale)))
    return Rect(cell.x0 + (cell.width - w) // 2, cell.y0 + (cell.height - h) // 2, w, h)


def compose(frames: Sequence[Optional[np.ndarray]], cell_w: int,
            cell_h: int) -> Tuple[np.ndarray, List[Optional[Rect]]]:
    """Lay ``frames`` (BGR, ``None`` = no frame) out as one mosaic.

    Returns the mosaic and, per camera, the rectangle its picture occupies
    (``None`` for a placeholder cell). Inference runs on those rectangles, so a
    camera's letterbox bars never reach the model.
    """
    cells = cell_rects(len(frames), cell_w, cell_h)
    cols, rows = grid_for(len(frames))
    mosaic = np.empty((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    mosaic[:] = PLACEHOLDER_BGR
    content: List[Optional[Rect]] = []
    for frame, cell in zip(frames, cells, strict=True):
        if frame is None or frame.ndim != 3 or frame.shape[0] < 1 or frame.shape[1] < 1:
            content.append(None)
            continue
        h, w = frame.shape[:2]
        rect = fit_rect(cell, w, h)
        if (w, h) == (rect.width, rect.height):
            picture = frame
        else:
            shrink = rect.width < w
            picture = cv2.resize(frame, (rect.width, rect.height),
                                 interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
        mosaic[rect.y0:rect.y1, rect.x0:rect.x1] = picture
        content.append(rect)
    return mosaic, content


def camera_at(cells: Sequence[Rect], x: float, y: float) -> Optional[int]:
    """The index of the cell containing the point ``(x, y)``, or ``None``."""
    for index, cell in enumerate(cells):
        if cell.contains(x, y):
            return index
    return None


class IpCameraRings:
    """Latest frame of every IP camera ring, with staleness.

    Not thread-safe: one thread polls; it hands frames to others by reference
    (each one is a fresh array the ring reader allocated).
    """

    def __init__(self, shm_name: str, stale_secs: float = _DEFAULT_STALE_SECS):
        self._shm_name = shm_name
        self._stale_secs = stale_secs
        self._readers: List[Optional[CameraRingReader]] = [None] * MAX_IP_CAMERAS
        self._seqs = [0] * MAX_IP_CAMERAS
        self._frames: List[Optional[np.ndarray]] = [None] * MAX_IP_CAMERAS
        self._stamps = [0.0] * MAX_IP_CAMERAS
        # Each reader's attach generation when its frame was cached: a ring
        # the webcam-server recreated (another camera in that position, or a
        # new size) must not keep showing the previous camera's picture.
        self._generations = [0] * MAX_IP_CAMERAS
        # Which cameras had a frame when poll() last said "changed", and how
        # many there were: a camera going stale or the count changing alters
        # the mosaic as much as a new frame. None = nothing reported yet.
        self._shown: Optional[Tuple[bool, ...]] = None

    def reader(self, index: int) -> CameraRingReader:
        """The (lazily created) reader of camera ``index``'s ring."""
        reader = self._readers[index]
        if reader is None:
            reader = CameraRingReader(ip_ring_name(self._shm_name, index))
            self._readers[index] = reader
        return reader

    def poll(self, count: int) -> bool:
        """Pick up new frames from the first ``count`` rings; True when the set changed.

        Changed means a frame arrived, or which cameras have a frame changed
        (one went stale, or the count changed) — so a mosaic of offline
        cameras is shown once, and an expired last frame turns dark.
        """
        fresh = False
        now = time.monotonic()
        for index in range(min(count, MAX_IP_CAMERAS)):
            reader = self.reader(index)
            got = reader.get_latest_frame(self._seqs[index])
            if not reader.is_mapped or reader.attach_generation != self._generations[index]:
                # Ring gone or replaced: what we hold belongs to the old one.
                self._generations[index] = reader.attach_generation
                self._frames[index] = None
            if got is None:
                continue
            frame, seq = got
            self._seqs[index] = seq
            if isinstance(frame, bytes):
                frame = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
            self._frames[index] = frame
            self._stamps[index] = now
            fresh = True
        shown = tuple(frame is not None for frame in self.frames(count))
        if shown != self._shown:
            self._shown = shown
            fresh = True
        return fresh

    def frames(self, count: int) -> List[Optional[np.ndarray]]:
        """The latest frame per camera; ``None`` once one is older than the stale limit."""
        now = time.monotonic()
        return [
            self._frames[i] if self._frames[i] is not None
            and now - self._stamps[i] <= self._stale_secs else None
            for i in range(min(count, MAX_IP_CAMERAS))
        ]

    def health_bytes(self, index: int) -> Optional[bytes]:
        """Camera ``index``'s serialized health, or ``None`` while its ring is absent."""
        reader = self.reader(index)
        if not reader.is_available():
            reader.open()
        return reader.read_health_bytes()

    def clear(self) -> None:
        """Forget every frame (leaving the IP source, or a new camera list)."""
        self._frames = [None] * MAX_IP_CAMERAS
        self._stamps = [0.0] * MAX_IP_CAMERAS
        self._shown = None


# How often a frame source re-reads the main health for the camera count, and
# the fastest it recomposes the mosaic (previews, not the inference path).
_COUNT_REFRESH_S = 0.5
_MIN_COMPOSE_INTERVAL_S = 1 / 15


def ip_camera_count_of(health: Optional[bytes]) -> int:
    """``HealthStatus.ip_camera_count`` (field 3, a varint) from a serialized health.

    A minimal protobuf scan so every service can size the mosaic without
    importing its generated stubs here (this package stays protobuf-free);
    fields it does not need are skipped by wire type. 0 when absent.
    """
    if not health:
        return 0
    data, pos = memoryview(health), 0

    def varint() -> int:
        nonlocal pos
        value = shift = 0
        while True:
            byte = data[pos]
            pos += 1
            value |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return value
            shift += 7

    try:
        while pos < len(data):
            key = varint()
            field, wire = key >> 3, key & 7
            if wire == 0:
                value = varint()
                if field == 3:
                    return value
            elif wire == 2:
                length = varint()  # not `pos += varint()`: that reads pos first
                pos += length
            elif wire == 1:
                pos += 8
            elif wire == 5:
                pos += 4
            else:
                return 0
    except IndexError:
        return 0
    return 0


class FrameRing(Protocol):
    """What :class:`CameraFrameSource` needs of the main ring reader."""

    def read_health_bytes(self) -> Optional[bytes]: ...

    def get_latest_frame(
            self, last_seq: int) -> Optional[Tuple[Union[np.ndarray, bytes], int]]: ...


class CameraFrameSource:
    """The camera as one picture, whichever the source: for previews and captures.

    Wraps the main ring reader. While the main ring's health reports IP
    cameras (``ip_camera_count``), it serves the IP cameras' mosaic
    instead — composed at most 15 times a second and shared by every caller.
    ``count_of`` reads the count out of the serialized health
    (:func:`ip_camera_count_of` by default).
    Its sequence numbers are its own, monotonic across both modes.

    Thread-safe: the gateway serves many MJPEG clients from one instance.
    """

    def __init__(self, shm_name: str, main: FrameRing,
                 count_of: Callable[[Optional[bytes]], int] = ip_camera_count_of,
                 cell: Optional[Tuple[int, int]] = None):
        self._main = main
        self._count_of = count_of
        self._ip = IpCameraRings(shm_name)
        self._cell = cell or cell_size_from_env()
        self._lock = threading.Lock()
        self._count = 0
        self._count_read = 0.0
        self._main_seq = 0
        self._seq = 0
        self._frame: Optional[Union[np.ndarray, bytes]] = None
        self._composed_at = 0.0
        self._dirty = False
        self._jpeg: Optional[Tuple[bytes, int]] = None

    def ip_camera_count(self) -> int:
        """Configured IP cameras (0 for any other source)."""
        with self._lock:
            self._refresh_count()
            return self._count

    def _refresh_count(self) -> None:
        now = time.monotonic()
        if now - self._count_read < _COUNT_REFRESH_S:
            return
        self._count_read = now
        try:
            count = int(self._count_of(self._main.read_health_bytes()))
        except Exception:  # noqa: BLE001 - an unreadable health is "no IP cameras"
            count = 0
        count = max(0, min(count, MAX_IP_CAMERAS))
        if count != self._count:
            self._count = count
            self._ip.clear()
            self._frame = None

    def get_latest_frame(
            self, last_seq: int) -> Optional[Tuple[Union[np.ndarray, bytes], int]]:
        """``(frame, seq)`` newer than ``last_seq``, like ``CameraRingReader``."""
        with self._lock:
            self._refresh_count()
            if self._count == 0:
                got = self._main.get_latest_frame(self._main_seq)
                if got is not None:
                    self._frame, self._main_seq = got
                    self._seq += 1
            else:
                self._dirty |= self._ip.poll(self._count)
                now = time.monotonic()
                if self._dirty and now - self._composed_at >= _MIN_COMPOSE_INTERVAL_S:
                    self._frame, _ = compose(self._ip.frames(self._count), *self._cell)
                    self._composed_at = now
                    self._dirty = False
                    self._seq += 1
            if self._seq <= last_seq or self._frame is None:
                return None
            return self._frame, self._seq

    def get_latest(self, last_seq: int) -> Optional[Tuple[bytes, int]]:
        """``(jpeg, seq)`` newer than ``last_seq``; RAW frames and mosaics are encoded."""
        got = self.get_latest_frame(last_seq)
        if got is None:
            return None
        frame, seq = got
        if not isinstance(frame, np.ndarray):
            return frame, seq
        # Every MJPEG client asks for the same frame: encode it once.
        with self._lock:
            if self._jpeg is not None and self._jpeg[1] == seq:
                return self._jpeg
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return None
        encoded = (buf.tobytes(), seq)
        with self._lock:
            self._jpeg = encoded
        return encoded
