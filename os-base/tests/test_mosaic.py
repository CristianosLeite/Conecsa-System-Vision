# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for the IP camera mosaic layout and ring polling (conecsa_shm.mosaic)."""
import mmap
import os
import struct
import uuid

import numpy as np
import pytest
from conecsa_shm import mosaic
from conecsa_shm.camera_ring import (
    FORMAT_RAW_BGR,
    HEADER_SIZE,
    OFF_ACTIVE_SLOT,
    OFF_CHANNELS,
    OFF_FORMAT_FLAG,
    OFF_FRAME_SIZE,
    OFF_FRAME_WRITE_SEQ,
    OFF_MAGIC,
    OFF_MAX_FRAME_BYTES,
    OFF_WIDTH,
    SHM_MAGIC,
    SHM_VERSION,
)
from conecsa_shm.mosaic import (
    PLACEHOLDER_BGR,
    IpCameraRings,
    Rect,
    camera_at,
    cell_rects,
    compose,
    fit_rect,
    grid_for,
    ip_ring_name,
)


class StubCameraWriter:
    """A camera ring holding RAW frames, written like the webcam-server does."""

    def __init__(self, name, slot_size=64):
        self.path = f"/dev/shm/{name}"
        self.slot_size = slot_size
        total = HEADER_SIZE + 2 * slot_size
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_EXCL, 0o600)
        try:
            os.ftruncate(fd, total)
            self.mm = mmap.mmap(fd, total)
        finally:
            os.close(fd)
        struct.pack_into("<II", self.mm, OFF_MAGIC, SHM_MAGIC, SHM_VERSION)
        struct.pack_into("<I", self.mm, OFF_CHANNELS, 3)
        struct.pack_into("<I", self.mm, OFF_MAX_FRAME_BYTES, slot_size)
        self.seq = 0

    def publish(self, data, fmt, width, height):
        struct.pack_into("<Q", self.mm, OFF_FRAME_WRITE_SEQ, self.seq + 1)
        slot = 1 - struct.unpack_from("<I", self.mm, OFF_ACTIVE_SLOT)[0]
        base = HEADER_SIZE + slot * self.slot_size
        self.mm[base:base + len(data)] = data
        struct.pack_into("<IIII", self.mm, OFF_WIDTH, width, height, 3, self.slot_size)
        struct.pack_into("<I", self.mm, OFF_FORMAT_FLAG, fmt)
        struct.pack_into("<I", self.mm, OFF_FRAME_SIZE, len(data))
        struct.pack_into("<I", self.mm, OFF_ACTIVE_SLOT, slot)
        self.seq += 2
        struct.pack_into("<Q", self.mm, OFF_FRAME_WRITE_SEQ, self.seq)

    def close(self):
        self.mm.close()
        os.unlink(self.path)


class TestLayout:
    @pytest.mark.parametrize("count,grid", [(1, (1, 1)), (2, (2, 1)), (3, (2, 2)), (4, (2, 2))])
    def test_the_grid_follows_the_camera_count(self, count, grid):
        assert grid_for(count) == grid

    def test_no_camera_is_not_a_mosaic(self):
        with pytest.raises(ValueError):
            grid_for(0)

    def test_two_cameras_sit_side_by_side_each_taking_half(self):
        assert cell_rects(2, 1280, 720) == [Rect(0, 0, 1280, 720), Rect(1280, 0, 1280, 720)]

    def test_cells_are_row_major(self):
        cells = cell_rects(3, 100, 50)
        assert [(c.x0, c.y0) for c in cells] == [(0, 0), (100, 0), (0, 50)]

    def test_a_4_3_picture_is_pillarboxed_in_a_16_9_cell(self):
        assert fit_rect(Rect(1280, 0, 1280, 720), 640, 480) == Rect(1280 + 160, 0, 960, 720)

    def test_camera_at_maps_a_point_to_its_cell(self):
        cells = cell_rects(4, 100, 50)
        assert camera_at(cells, 150, 10) == 1
        assert camera_at(cells, 99.9, 49.9) == 0
        assert camera_at(cells, 10, 60) == 2
        assert camera_at(cells, 250, 10) is None

    def test_cell_size_comes_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("MOSAIC_CELL_WIDTH", "641")
        monkeypatch.setenv("MOSAIC_CELL_HEIGHT", "junk")
        assert mosaic.cell_size_from_env() == (640, 720)


class TestCompose:
    def test_one_camera_fills_the_frame(self):
        frame = np.full((720, 1280, 3), 200, np.uint8)
        out, content = compose([frame], 1280, 720)
        assert out.shape == (720, 1280, 3)
        assert content == [Rect(0, 0, 1280, 720)]
        assert (out == 200).all()

    def test_a_missing_camera_keeps_a_placeholder_cell(self):
        frame = np.full((360, 640, 3), 255, np.uint8)
        out, content = compose([frame, None, frame], 640, 360)
        assert out.shape == (720, 1280, 3)
        assert content[1] is None
        assert tuple(out[10, 700]) == PLACEHOLDER_BGR
        # The empty fourth cell of a 2×2 grid is a placeholder as well.
        assert tuple(out[400, 700]) == PLACEHOLDER_BGR
        assert (out[400:720, 0:640] == 255).all()

    def test_a_larger_camera_is_scaled_into_its_cell(self):
        frame = np.zeros((1080, 1920, 3), np.uint8)
        out, content = compose([frame, frame], 640, 360)
        assert out.shape == (360, 1280, 3)
        assert content == [Rect(0, 0, 640, 360), Rect(640, 0, 640, 360)]


@pytest.fixture
def base_name():
    return f"conecsa-test-mosaic-{uuid.uuid4().hex[:8]}"


class TestIpCameraRings:
    def test_frames_come_from_the_per_camera_rings(self, base_name):
        writer = StubCameraWriter(ip_ring_name(base_name, 1), slot_size=64)
        try:
            rings = IpCameraRings(base_name)
            assert rings.poll(2) is True, "the first set (both offline) is reported"
            assert rings.poll(2) is False
            pixels = bytes(range(24))
            writer.publish(pixels, fmt=FORMAT_RAW_BGR, width=4, height=2)
            assert rings.poll(2) is True
            frames = rings.frames(2)
            assert frames[0] is None
            assert frames[1] is not None
            assert frames[1].tobytes() == pixels
            assert rings.poll(2) is False, "the same frame is not new twice"
        finally:
            writer.close()

    def test_a_frame_goes_stale(self, base_name, monkeypatch):
        writer = StubCameraWriter(ip_ring_name(base_name, 0), slot_size=64)
        try:
            rings = IpCameraRings(base_name, stale_secs=2.0)
            writer.publish(bytes(24), fmt=FORMAT_RAW_BGR, width=4, height=2)
            clock = [100.0]
            monkeypatch.setattr(mosaic.time, "monotonic", lambda: clock[0])
            assert rings.poll(1)
            clock[0] += 1.9
            assert rings.poll(1) is False
            assert rings.frames(1)[0] is not None
            clock[0] += 0.2
            assert rings.poll(1) is True, "an expired frame changes the mosaic"
            assert rings.frames(1) == [None]
            assert rings.poll(1) is False
        finally:
            writer.close()

    def test_a_replaced_ring_drops_the_previous_cameras_frame(self, base_name):
        name = ip_ring_name(base_name, 0)
        writer = StubCameraWriter(name, slot_size=64)
        rings = IpCameraRings(base_name)
        writer.publish(bytes([255] * 24), fmt=FORMAT_RAW_BGR, width=4, height=2)
        assert rings.poll(1) and rings.frames(1)[0] is not None
        # The webcam-server unlinks the ring (camera removed or replaced)...
        writer.close()
        assert rings.poll(1) is True
        assert rings.frames(1) == [None], "no stale picture for whatever comes next"
        # ...and creates it again for the new camera, which has no frame yet.
        writer = StubCameraWriter(name, slot_size=64)
        try:
            rings.poll(1)
            assert rings.frames(1) == [None]
            writer.publish(bytes([7] * 24), fmt=FORMAT_RAW_BGR, width=4, height=2)
            assert rings.poll(1) is True
            frame = rings.frames(1)[0]
            assert frame is not None and int(frame[0, 0, 0]) == 7
        finally:
            writer.close()

    def test_a_new_camera_count_is_reported_even_without_frames(self, base_name):
        rings = IpCameraRings(base_name)
        assert rings.poll(1) is True
        assert rings.poll(1) is False
        assert rings.poll(3) is True
        assert rings.frames(3) == [None, None, None]

    def test_health_is_none_without_a_ring(self, base_name):
        assert IpCameraRings(base_name).health_bytes(3) is None


class _FakeMain:
    def __init__(self, health=b""):
        self.health = health
        self.frames = []

    def read_health_bytes(self):
        return self.health

    def get_latest_frame(self, last_seq):
        if self.frames and self.frames[0][1] > last_seq:
            return self.frames.pop(0)
        return None


class TestCameraFrameSource:
    def test_without_ip_cameras_it_serves_the_main_ring(self, base_name):
        main = _FakeMain()
        main.frames = [(b"jpeg-1", 7)]
        source = mosaic.CameraFrameSource(base_name, main, lambda b: 0, cell=(4, 2))
        assert source.get_latest(0) == (b"jpeg-1", 1)
        assert source.get_latest(1) is None

    def test_with_ip_cameras_it_serves_the_mosaic(self, base_name):
        writer = StubCameraWriter(ip_ring_name(base_name, 1), slot_size=64)
        try:
            main = _FakeMain(health=b"2")
            source = mosaic.CameraFrameSource(
                base_name, main, lambda b: int(b or 0), cell=(4, 2))
            assert source.ip_camera_count() == 2
            writer.publish(bytes([255] * 24), fmt=FORMAT_RAW_BGR, width=4, height=2)
            got = source.get_latest_frame(0)
            assert got is not None
            frame, seq = got
            assert isinstance(frame, np.ndarray)
            assert seq == 1 and frame.shape == (2, 8, 3)
            assert tuple(frame[0, 0]) == PLACEHOLDER_BGR and tuple(frame[0, 5]) == (255,) * 3
            encoded = source.get_latest(0)
            assert encoded is not None
            jpeg, again = encoded
            assert again == 1 and jpeg[:2] == b"\xff\xd8"
        finally:
            writer.close()

    def test_a_health_parse_error_reads_as_no_ip_cameras(self, base_name):
        def broken(_):
            raise ValueError("bad payload")
        source = mosaic.CameraFrameSource(base_name, _FakeMain(), broken, cell=(4, 2))
        assert source.ip_camera_count() == 0
