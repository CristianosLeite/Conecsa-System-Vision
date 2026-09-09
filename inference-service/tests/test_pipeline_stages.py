"""Every pipeline stage survives a per-item failure (review M6).

The stage threads are started once and never restarted, so an unguarded
exception in the infer hand-off, the finish bookkeeping or the encode stage
used to end that thread for the life of the process while the gRPC port
stayed healthy. These tests inject one failure per stage and check that the
next frame still reaches the ring, and that the health snapshot records it.
"""
import time
from types import SimpleNamespace

import pytest
from api.services.processing_pipeline import ProcessingPipelineService
from test_pipeline_quiesce import (
    FakeCodec,
    FakeConsumer,
    FakeDetection,
    FakeGpio,
    FakeOverlay,
    FakeRing,
    FakeStats,
    FakeVideo,
    _wait_until,
)


class FailingOnceCodec(FakeCodec):
    def __init__(self):
        self.calls = 0

    def encode_frame(self, frame):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("encoder blew up")
        return b"jpg"


class FailingOnceDetection(FakeDetection):
    def __init__(self):
        super().__init__()
        self.release.set()
        self.finish_calls = 0

    def finish(self, outputs, frame, metas, inference_time=0.0, generation=None):
        self.finish_calls += 1
        if self.finish_calls == 1:
            raise RuntimeError("postprocess blew up")
        return super().finish(outputs, frame, metas, inference_time, generation)


class ExplodingStats(FakeStats):
    def __init__(self):
        self.calls = 0

    def update(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("stats blew up")


@pytest.fixture
def build(monkeypatch):
    monkeypatch.setenv("TENSORRT_CONTEXTS", "1")
    import conecsa_shm.processed_ring as ring_mod
    ring = FakeRing()
    monkeypatch.setattr(ring_mod, "ProcessedFrameWriter", lambda: ring)
    built = []

    def _build(codec=None, detection=None, stats=None):
        consumer = FakeConsumer()
        detection = detection or FailingOnceDetection()
        pipeline = ProcessingPipelineService(
            consumer, codec or FakeCodec(), detection, stats or FakeStats(),
            FakeGpio(), FakeOverlay(), FakeVideo())
        built.append(pipeline)
        return SimpleNamespace(pipeline=pipeline, consumer=consumer, ring=ring,
                               detection=detection)

    yield _build
    for pipeline in built:
        pipeline.close()


class TestStageBoundaries:
    def test_a_finish_failure_publishes_the_raw_frame_and_keeps_going(self, build):
        rig = build()
        rig.consumer.push(1)
        rig.consumer.push(2)
        assert _wait_until(lambda: len(rig.ring.published) == 2)
        assert rig.pipeline.stage_errors["finish"] == 1
        assert rig.pipeline.inflight == 0

    def test_a_stats_failure_drops_only_that_frame(self, build):
        detection = FakeDetection()
        detection.release.set()
        rig = build(detection=detection, stats=ExplodingStats())
        rig.consumer.push(1)
        rig.consumer.push(2)
        assert _wait_until(lambda: len(rig.ring.published) == 1)
        assert rig.pipeline.stage_errors["finish"] == 1
        assert rig.pipeline.inflight == 0

    def test_an_encode_failure_does_not_end_the_encoder_thread(self, build):
        detection = FakeDetection()
        detection.release.set()
        rig = build(codec=FailingOnceCodec(), detection=detection)
        rig.consumer.push(1)
        rig.consumer.push(2)
        assert _wait_until(lambda: len(rig.ring.published) == 1)
        assert rig.pipeline.stage_errors["encode"] == 1
        assert all(t.is_alive() for t in rig.pipeline._threads)

    def test_a_malformed_item_in_the_encode_queue_is_dropped(self, build):
        detection = FakeDetection()
        detection.release.set()
        rig = build(detection=detection)
        rig.pipeline._q_encode.put("not a (seq, frame) tuple")  # the unpack raises
        assert _wait_until(lambda: rig.pipeline.stage_errors["encode"] == 1)
        rig.consumer.push(1)
        assert _wait_until(lambda: rig.ring.published == [b"jpg"])


class TestHealth:
    def test_snapshot_reports_errors_progress_and_gate_state(self, build):
        rig = build()
        rig.consumer.push(1)
        assert _wait_until(lambda: rig.ring.published)
        time.sleep(0.05)
        health = rig.pipeline.health()
        assert set(health) == {"stage_errors", "progress_age_s", "inflight",
                               "stale_drops", "draining"}
        assert health["stage_errors"]["finish"] == 1
        assert health["stage_errors"]["encode"] == 0
        assert health["progress_age_s"]["encode"] >= 0.05
        assert health["inflight"] == 0
        assert health["draining"] is False

    def test_fps_uses_the_monotonic_clock(self, build, monkeypatch):
        rig = build()
        calls = []
        monkeypatch.setattr(time, "monotonic", lambda: calls.append(1) or 100.0)
        rig.pipeline._tick_fps()
        assert calls, "wall-clock time.time() would be stepped by the hub"
