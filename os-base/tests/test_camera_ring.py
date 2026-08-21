"""Tests for the camera-ring seqlock reader (conecsa_shm.camera_ring, v2).

A stub writer emits the exact byte protocol of webcam-server's ShmProducer
(the shared spec comments in both files are the contract); the tests cover
odd-seq rejection, torn-copy retry, the RAW resolution race, version-1
rejection, and a checksummed stress run.
"""
import hashlib
import mmap
import os
import struct
import threading
import uuid

import numpy as np
import pytest
from conecsa_shm import camera_ring
from conecsa_shm.camera_ring import (
    FORMAT_JPEG,
    FORMAT_RAW_RGB,
    HEADER_SIZE,
    OFF_ACTIVE_SLOT,
    OFF_CHANNELS,
    OFF_FORMAT_FLAG,
    OFF_FRAME_SIZE,
    OFF_FRAME_WRITE_SEQ,
    OFF_HEIGHT,
    OFF_MAGIC,
    OFF_MAX_FRAME_BYTES,
    OFF_WIDTH,
    SHM_MAGIC,
    SHM_VERSION,
    CameraRingReader,
)


class StubCameraWriter:
    """Emits the v2 camera-ring publication protocol like the Rust producer."""

    def __init__(self, name, width=4, height=2, slot_size=256, version=SHM_VERSION):
        self.path = f"/dev/shm/{name}"
        self.slot_size = slot_size
        total = HEADER_SIZE + 2 * slot_size
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_EXCL, 0o600)
        try:
            os.ftruncate(fd, total)
            self.mm = mmap.mmap(fd, total)
        finally:
            os.close(fd)
        struct.pack_into("<II", self.mm, OFF_MAGIC, SHM_MAGIC, version)
        struct.pack_into("<I", self.mm, OFF_WIDTH, width)
        struct.pack_into("<I", self.mm, OFF_HEIGHT, height)
        struct.pack_into("<I", self.mm, OFF_CHANNELS, 3)
        struct.pack_into("<I", self.mm, OFF_MAX_FRAME_BYTES, slot_size)
        self.seq = 0

    def begin(self):
        struct.pack_into("<Q", self.mm, OFF_FRAME_WRITE_SEQ, self.seq + 1)

    def finish(self):
        self.seq += 2
        struct.pack_into("<Q", self.mm, OFF_FRAME_WRITE_SEQ, self.seq)

    def publish(self, data, fmt=FORMAT_JPEG, width=None, height=None):
        self.begin()
        active = struct.unpack_from("<I", self.mm, OFF_ACTIVE_SLOT)[0]
        new_slot = 1 - active
        base = HEADER_SIZE + new_slot * self.slot_size
        self.mm[base:base + len(data)] = data
        if width is not None:
            struct.pack_into("<I", self.mm, OFF_WIDTH, width)
        if height is not None:
            struct.pack_into("<I", self.mm, OFF_HEIGHT, height)
        struct.pack_into("<I", self.mm, OFF_FORMAT_FLAG, fmt)
        struct.pack_into("<I", self.mm, OFF_FRAME_SIZE, len(data))
        struct.pack_into("<I", self.mm, OFF_ACTIVE_SLOT, new_slot)
        self.finish()

    def close(self):
        self.mm.close()
        os.unlink(self.path)


@pytest.fixture
def ring():
    name = f"conecsa-test-cam-{uuid.uuid4().hex[:8]}"
    writer = StubCameraWriter(name)
    reader = CameraRingReader(name)
    yield writer, reader
    reader.close()
    writer.close()


class _TornStruct:
    """struct proxy that fires a hook right before the validation seq read,
    simulating the writer lapping the reader mid-copy."""

    def __init__(self, hook):
        self._hook = hook
        self._seq_reads = 0

    def unpack_from(self, fmt, buf, offset=0):
        if offset == OFF_FRAME_WRITE_SEQ:
            self._seq_reads += 1
            if self._seq_reads == 2:
                self._hook()
        return struct.unpack_from(fmt, buf, offset)

    def __getattr__(self, name):
        return getattr(struct, name)


class TestSeqlockReader:
    def test_a_stable_publication_is_returned(self, ring):
        writer, reader = ring
        writer.publish(b"\xffjpeg-one")
        got = reader.get_latest_frame(0)
        assert got is not None
        frame, seq = got
        assert frame == b"\xffjpeg-one"
        assert seq == writer.seq
        # No newer frame: same seq yields None.
        assert reader.get_latest_frame(seq) is None

    def test_an_open_publication_window_is_never_read(self, ring):
        writer, reader = ring
        writer.publish(b"stable")
        writer.begin()  # odd seq: a write is in flight
        assert reader.get_latest_frame(0) is None
        writer.finish()
        assert reader.get_latest_frame(0) is not None

    def test_a_torn_copy_is_retried_never_mixed(self, ring, monkeypatch):
        writer, reader = ring
        writer.publish(b"AAAAAAAA")
        # Lap the reader between its copy and its validation read.
        monkeypatch.setattr(
            camera_ring, "struct",
            _TornStruct(lambda: writer.publish(b"BBBBBBBB")))
        got = reader.get_latest_frame(0)
        assert got is not None
        frame, seq = got
        assert frame == b"BBBBBBBB", "a torn read must retry, not mix frames"
        assert seq == writer.seq

    def test_a_resolution_change_never_reshapes_stale_bytes(self, ring,
                                                            monkeypatch):
        writer, reader = ring
        writer.publish(b"\x01" * (4 * 2 * 3), fmt=FORMAT_RAW_RGB,
                       width=4, height=2)
        # Mid-read, the camera renegotiates to 2x2 — dimensions and payload
        # change together inside the writer's window.
        monkeypatch.setattr(
            camera_ring, "struct",
            _TornStruct(lambda: writer.publish(
                b"\x02" * (2 * 2 * 3), fmt=FORMAT_RAW_RGB, width=2, height=2)))
        got = reader.get_latest_frame(0)
        assert got is not None
        frame, _ = got
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (2, 2, 3)

    def test_a_version_1_producer_is_rejected_loudly(self, caplog):
        name = f"conecsa-test-cam-{uuid.uuid4().hex[:8]}"
        writer = StubCameraWriter(name, version=1)
        try:
            with caplog.at_level("WARNING"):
                reader = CameraRingReader(name)
            assert not reader.is_available()
            assert any("magic/version" in r.message for r in caplog.records)
        finally:
            writer.close()

    def test_checksummed_stress_run_yields_no_corrupt_frame(self, ring):
        writer, reader = ring
        payload_len = 64
        stop = threading.Event()

        def make_frame(n):
            body = hashlib.sha256(str(n).encode()).digest()
            return (body * 3)[:payload_len]

        def produce():
            n = 0
            while not stop.is_set():
                writer.publish(make_frame(n))
                n += 1

        producer = threading.Thread(target=produce, daemon=True)
        producer.start()
        try:
            accepted = 0
            last = 0
            while accepted < 500:
                got = reader.get_latest_frame(last)
                if got is None:
                    continue
                frame, seq = got
                last = seq
                n = seq // 2 - 1  # the writer publishes frame n at seq 2(n+1)
                assert frame == make_frame(n), f"corrupt frame at seq {seq}"
                accepted += 1
        finally:
            stop.set()
            producer.join(5)


class TestProducerRestartRecovery:
    def test_the_reader_survives_a_producer_restart(self):
        # webcam-server unlinks + recreates its segment on start; a consumer
        # holding the old mapping must detect the new inode and keep going,
        # with seqs still monotonic (REFACTORING.md H4).
        name = f"conecsa-test-cam-{uuid.uuid4().hex[:8]}"
        writer = StubCameraWriter(name)
        reader = CameraRingReader(name)
        try:
            writer.publish(b"frame-before-restart")
            got = reader.get_latest_frame(0)
            assert got is not None
            frame, last = got
            assert frame == b"frame-before-restart"

            writer.close()  # unlink, like ShmProducer::drop
            assert not reader.is_available(), \
                "a stale mapping must not report available"

            writer = StubCameraWriter(name)  # new inode, same name
            writer.publish(b"frame-after-restart")
            got = reader.get_latest_frame(last)
            assert got is not None, "the reader must remap the new segment"
            frame, seq = got
            assert frame == b"frame-after-restart"
            assert seq > last, "seqs must stay monotonic across the restart"
            assert reader.is_available()
        finally:
            reader.close()
            try:
                writer.close()
            except OSError:
                pass

    def test_unlink_without_recreate_reports_unavailable(self):
        name = f"conecsa-test-cam-{uuid.uuid4().hex[:8]}"
        writer = StubCameraWriter(name)
        reader = CameraRingReader(name)
        try:
            writer.publish(b"x")
            assert reader.is_available()
            writer.close()
            assert not reader.is_available()
            assert reader.get_latest_frame(0) is None
        finally:
            reader.close()
