"""Processed-frame publication is serialized with its ordering (review M5).

``_publish`` used to advance the last-published sequence under the lock and
then write to the ring outside it, so a frame that paused between the two
could be overtaken and then overwrite the newer one — and two lanes could
enter the single-producer ring at the same time.
"""
import threading
import time
from types import SimpleNamespace

import pytest
from api.services.processing_pipeline import ProcessingPipelineService


class BlockingRing:
    """Records publishes in order; the first one parks until released."""

    def __init__(self):
        self.published = []
        self.first_entered = threading.Event()
        self.release = threading.Event()
        self.concurrent = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()

    def publish(self, jpg):
        with self._lock:
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        if not self.first_entered.is_set():
            self.first_entered.set()
            self.release.wait(5.0)
        self.published.append(jpg)
        with self._lock:
            self.concurrent -= 1


@pytest.fixture
def pipeline(monkeypatch):
    import conecsa_shm.processed_ring as ring_mod
    ring = BlockingRing()
    monkeypatch.setattr(ring_mod, "ProcessedFrameWriter", lambda: ring)
    fake = SimpleNamespace()
    service = ProcessingPipelineService(
        fake, fake, SimpleNamespace(is_running=False), fake, fake, fake, fake,
        autostart=False)
    return service, ring


class TestPublishOrdering:
    def test_a_newer_frame_waits_for_the_older_publish_to_finish(self, pipeline):
        service, ring = pipeline
        older = threading.Thread(target=service._publish, args=(b"one", 1))
        older.start()
        assert ring.first_entered.wait(3.0)
        newer = threading.Thread(target=service._publish, args=(b"two", 2))
        newer.start()
        time.sleep(0.2)
        # The newer frame must not have entered the ring while the older one
        # is still writing.
        assert ring.published == []
        ring.release.set()
        older.join(5.0)
        newer.join(5.0)
        assert ring.published == [b"one", b"two"]
        assert ring.max_concurrent == 1

    def test_an_older_frame_arriving_late_is_dropped(self, pipeline):
        service, ring = pipeline
        ring.release.set()
        service._publish(b"two", 2)
        service._publish(b"one", 1)
        assert ring.published == [b"two"]
