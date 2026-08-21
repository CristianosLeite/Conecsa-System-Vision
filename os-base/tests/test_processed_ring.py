"""Tests for the processed-frame seqlock ring (conecsa_shm.processed_ring, v2)."""
import hashlib
import struct
import threading
import uuid

import pytest
from conecsa_shm import processed_ring
from conecsa_shm.processed_ring import (
    _OFF_WRITE_SEQ,
    ProcessedFrameReader,
    ProcessedFrameWriter,
)


@pytest.fixture
def ring():
    name = f"conecsa-test-proc-{uuid.uuid4().hex[:8]}"
    writer = ProcessedFrameWriter(shm_name=name, slot_bytes=4096)
    reader = ProcessedFrameReader(shm_name=name)
    yield writer, reader
    reader.close()
    writer.close()
    import os
    os.unlink(f"/dev/shm/{name}")


class _TornStruct:
    """struct proxy firing a hook right before the validation seq read."""

    def __init__(self, hook):
        self._hook = hook
        self._seq_reads = 0

    def unpack_from(self, fmt, buf, offset=0):
        if offset == _OFF_WRITE_SEQ:
            self._seq_reads += 1
            if self._seq_reads == 2:
                self._hook()
        return struct.unpack_from(fmt, buf, offset)

    def __getattr__(self, name):
        return getattr(struct, name)

    def pack_into(self, *args, **kwargs):
        return struct.pack_into(*args, **kwargs)


class TestProcessedRing:
    def test_round_trip(self, ring):
        writer, reader = ring
        writer.publish(b"\xff\xd8jpeg-bytes")
        got = reader.get_latest(0)
        assert got is not None
        jpg, seq = got
        assert jpg == b"\xff\xd8jpeg-bytes"
        assert seq % 2 == 0
        assert reader.get_latest(seq) is None

    def test_an_open_publication_window_is_never_read(self, ring):
        writer, reader = ring
        writer.publish(b"stable")
        # Manually open a window (odd seq) as a crashed/mid-write producer.
        struct.pack_into("<Q", writer._mm, _OFF_WRITE_SEQ, writer._seq + 1)
        assert reader.get_latest(0) is None
        struct.pack_into("<Q", writer._mm, _OFF_WRITE_SEQ, writer._seq)
        assert reader.get_latest(0) is not None

    def test_a_torn_copy_is_retried_never_mixed(self, ring, monkeypatch):
        writer, reader = ring
        writer.publish(b"A" * 32)
        monkeypatch.setattr(
            processed_ring, "struct",
            _TornStruct(lambda: writer.publish(b"B" * 32)))
        got = reader.get_latest(0)
        assert got is not None
        jpg, _ = got
        assert jpg == b"B" * 32, "a torn read must retry, not mix frames"

    def test_checksummed_stress_run_yields_no_corrupt_frame(self, ring):
        writer, reader = ring
        stop = threading.Event()

        def make_frame(n):
            return (hashlib.sha256(str(n).encode()).digest() * 2)[:48]

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
                got = reader.get_latest(last)
                if got is None:
                    continue
                jpg, seq = got
                last = seq
                n = seq // 2 - 1
                assert jpg == make_frame(n), f"corrupt frame at seq {seq}"
                accepted += 1
        finally:
            stop.set()
            producer.join(5)
