# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""The pipeline quiesces before a runtime swap and rejects frames prepared for
an older runtime.

Runs the real ``ProcessingPipelineService`` threads against fakes: a consumer
that hands out scripted frames, a detection service whose ``infer`` blocks
until the test releases it, and a recording stand-in for the processed SHM
ring. No TensorRT, no shared memory.
"""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from api.services.detection_service import DetectionService, StaleGeneration
from api.services.processing_pipeline import ProcessingPipelineService

_FRAME = np.zeros((8, 8, 3), dtype=np.uint8)


class FakeConsumer:
    """Hands out queued ``(seq, jpg, npy)`` frames, then times out like the real one."""

    def __init__(self):
        self.frames = []
        self._lock = threading.Lock()

    def push(self, seq):
        with self._lock:
            self.frames.append((seq, None, _FRAME.copy()))

    def wait_for(self, last_seq, timeout=1.0):
        with self._lock:
            for i, (seq, jpg, npy) in enumerate(self.frames):
                if seq > last_seq:
                    del self.frames[i]
                    return seq, jpg, npy
        time.sleep(0.01)
        return last_seq, None, None


class FakeCodec:
    def decode_frame_scaled(self, jpg):
        return _FRAME.copy()

    def apply_rgb_levels(self, frame, r, g, b):
        return frame

    def combine_stereo(self, frame):
        return frame

    def encode_frame(self, frame):
        return b"jpg"


class FakeDetection:
    """``infer`` parks on ``release``; ``generation`` is bumped by the test."""

    def __init__(self):
        self.is_running = True
        self.generation = 1
        self.infer_started = threading.Event()
        self.release = threading.Event()
        self.pipeline = None
        self.finished = 0

    def attach_pipeline(self, pipeline):
        self.pipeline = pipeline

    def get_trigger_status(self):
        return True

    def is_model_loaded(self):
        return True

    def prepare(self, frame):
        return self.generation, [frame], [SimpleNamespace()]

    def infer(self, inputs, generation=None):
        self.infer_started.set()
        self.release.wait(5.0)
        if generation != self.generation:
            raise StaleGeneration("stale")
        return [np.zeros(1)], 0.01

    def finish(self, outputs, frame, metas, inference_time=0.0, generation=None):
        if generation != self.generation:
            raise StaleGeneration("stale")
        self.finished += 1
        return SimpleNamespace(processed_image=frame, num_detections=0)

    def increment_detection_count(self, n):
        pass


class FakeRing:
    def __init__(self):
        self.published = []

    def publish(self, jpg):
        self.published.append(jpg)


class FakeStats:
    def __init__(self):
        self.resets = 0
        self.updates = 0

    def update(self, **kwargs):
        self.updates += 1

    def record_timings(self, **kwargs):
        pass

    def reset(self):
        self.resets += 1


class FakeGpio:
    def should_process_frame(self):
        return True


class FakeOverlay:
    def draw_detection_off_overlay(self, frame):
        return frame


class FakeVideo:
    def rgb_levels(self):
        return 128, 128, 128


@pytest.fixture
def rig(monkeypatch):
    monkeypatch.setenv("TENSORRT_CONTEXTS", "1")
    import conecsa_shm.processed_ring as ring_mod
    ring = FakeRing()
    monkeypatch.setattr(ring_mod, "ProcessedFrameWriter", lambda: ring)
    consumer, detection = FakeConsumer(), FakeDetection()
    pipeline = ProcessingPipelineService(
        consumer, FakeCodec(), detection, FakeStats(), FakeGpio(), FakeOverlay(), FakeVideo())
    yield SimpleNamespace(pipeline=pipeline, consumer=consumer, detection=detection, ring=ring)
    detection.release.set()
    pipeline.close()


def _wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class TestDrain:
    def test_registers_itself_with_the_detection_service(self, rig):
        assert rig.detection.pipeline is rig.pipeline

    def test_waits_for_the_frame_a_worker_is_busy_with(self, rig):
        rig.consumer.push(1)
        assert rig.detection.infer_started.wait(3.0)
        assert rig.pipeline.inflight == 1

        drained = []
        t = threading.Thread(target=lambda: drained.append(rig.pipeline.drain(5.0)))
        t.start()
        time.sleep(0.2)
        # The worker is still mid-inference: drain must not have returned.
        assert drained == []
        rig.detection.release.set()
        t.join(5.0)
        assert drained == [True]
        assert rig.pipeline.inflight == 0
        # The frame finished against the old runtime and was published.
        assert _wait_until(lambda: rig.ring.published == [b"jpg"])

    def test_no_new_frame_enters_the_detection_path_while_draining(self, rig):
        rig.detection.release.set()
        assert rig.pipeline.drain(1.0)
        rig.consumer.push(1)
        # The frame is consumed, but as a "Detection Off" frame: nothing inferred.
        assert _wait_until(lambda: rig.ring.published == [b"jpg"])
        assert not rig.detection.infer_started.is_set()
        assert rig.pipeline.inflight == 0

        rig.pipeline.resume()
        rig.consumer.push(2)
        assert rig.detection.infer_started.wait(3.0)

    def test_drain_zeroes_the_stopped_run_stats(self, rig):
        # Regression: /stats kept the previous run's counters after a stop.
        rig.pipeline._frame_times.extend([1.0, 2.0])
        rig.detection.release.set()
        assert rig.pipeline.drain(1.0)
        assert rig.pipeline._stats.resets == 1
        assert rig.pipeline._frame_times == []

    def test_resume_starts_a_fresh_fps_window(self, rig):
        rig.detection.release.set()
        assert rig.pipeline.drain(1.0)
        rig.pipeline._frame_times.append(1.0)
        rig.pipeline.resume()
        assert rig.pipeline._frame_times == []

    def test_a_frame_finishing_after_a_drain_timeout_publishes_no_stats(self, rig):
        # Regression: a worker outliving drain() refilled the zeroed stats.
        rig.consumer.push(1)
        assert rig.detection.infer_started.wait(3.0)
        assert rig.pipeline.drain(0.2) is False
        rig.detection.release.set()
        assert _wait_until(lambda: rig.ring.published == [b"jpg"])
        assert rig.pipeline._stats.resets == 1
        assert rig.pipeline._stats.updates == 0

    def test_times_out_and_reports_when_a_worker_is_wedged(self, rig):
        rig.consumer.push(1)
        assert rig.detection.infer_started.wait(3.0)
        assert rig.pipeline.drain(0.2) is False
        assert rig.pipeline.inflight == 1


class TestStaleGeneration:
    def test_a_frame_prepared_for_the_old_runtime_is_dropped(self, rig):
        rig.consumer.push(1)
        assert rig.detection.infer_started.wait(3.0)
        # The runtime is swapped while the frame sits in inference.
        rig.detection.generation += 1
        rig.detection.release.set()
        assert _wait_until(lambda: rig.pipeline.stale_drops == 1)
        assert rig.pipeline.inflight == 0
        assert rig.detection.finished == 0
        assert rig.ring.published == []


class TestDetectionServiceTransitions:
    """The real DetectionService drives drain/resume around stop/initialize/start."""

    class _Pipeline:
        def __init__(self):
            self.calls = []

        def drain(self, timeout):
            self.calls.append(("drain", timeout))
            return True

        def resume(self):
            self.calls.append(("resume",))

    def test_stop_drains_even_when_not_running(self):
        from api.config import Config
        service = DetectionService(Config())
        pipeline = self._Pipeline()
        service.attach_pipeline(pipeline)
        assert service.stop() is False
        assert [c[0] for c in pipeline.calls] == ["drain"]

    def test_initialize_advances_the_generation_and_resumes(self, tmp_path, monkeypatch):
        from api.config import Config
        from api.services import detection_service as mod
        model = tmp_path / "weights.engine"
        model.write_bytes(b"x")
        config = Config()
        config.MODEL_PATH = str(model)
        monkeypatch.setattr(mod, "ModelManager", lambda cfg, **kw: SimpleNamespace(
            tiling_active=False, output_details=[{"shape": [1, 300, 6]}],
            input_details=[{"shape": [1, 3, 640, 640]}]))
        monkeypatch.setattr(mod, "load_class_labels", lambda cfg: ["a"])
        monkeypatch.setattr(mod.postprocess, "create",
                            lambda task, labels, cfg: SimpleNamespace())
        service = DetectionService(config)
        pipeline = self._Pipeline()
        service.attach_pipeline(pipeline)

        before = service.generation
        service.initialize()
        assert service.generation == before + 1
        assert pipeline.calls == [("resume",)]
        # An item stamped with the old generation is now refused by both stages.
        with pytest.raises(StaleGeneration):
            service.infer([np.zeros(1)], before)
        with pytest.raises(StaleGeneration):
            service.finish([np.zeros(1)], _FRAME, [SimpleNamespace()], 0.0, before)
