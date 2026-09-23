# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Processing pipeline - single shared producer that turns raw camera frames into
the processed (detection-overlaid) MJPEG stream.

Why this exists
---------------
A serial ``decode → preprocess → infer → draw → encode`` loop leaves the device
mostly idle: the CPU waits while the GPU infers, and the GPU waits while the
CPU decodes/encodes. The TensorRT inference runs in a separate process (its IPC
wait releases the GIL) and OpenCV decode/encode release the GIL too, so the
stages parallelize across the idle cores.

This service runs the stages on separate threads connected by bounded blocking
hand-offs (``queue.Queue``):

    A "prepare"  decode (reduced scale) + RGB + stereo + preprocess
    B "infer"    TensorRT inference (the GPU stage)
    C "finish"   postprocess (NMS/draw) + stats
    D "encode"   JPEG encode + publish

With ``TENSORRT_CONTEXTS = N > 1`` the decode (A) and inference (B) stages each
get N threads — N inference contexts run in parallel (measured ~1.8x GPU
scaling) and N decode threads keep them fed — while the postprocess (C) and
encode (D) stages stay single-threaded but pipelined behind them. N=1 collapses
to a single-lane decode∥infer∥encode pipeline.

Stage A always grabs the *freshest* camera frame and drops stale ones, so the
pipeline stays realtime; the blocking hand-offs give backpressure so the whole
thing settles at the slowest stage. The published frame is mirrored into a
shared-memory ring from which the api-gateway fans out to all HTTP clients, so
N viewers cost the same as one.

The trigger/GPIO-gate/freeze/"Detection Off" logic lives here too (Stage A/C),
computed once and shared.

Runtime transitions
-------------------
Frames on the detection path (A → B → C) hold a reference to the live runtime:
a model swap or a GPU release while they are in flight would submit an input
prepared for one engine to another, decode outputs with the wrong labels, or
tear a worker down mid-call. So the pipeline counts the frames it has claimed
for detection and ``DetectionService.stop()`` calls :meth:`drain` before the
runtime changes: the detection path closes, the stages finish what they hold,
and only then does the caller swap/release. Every item also carries the runtime
generation it was prepared against, and the detection service rejects a
mismatch (``StaleGeneration``), which covers the timeout path of ``drain``.
"""
import logging
import os
import queue
import threading
import time
from typing import Optional

# noinspection PyPackageRequirements
import cv2  # ships in conecsa-os-base:base

# noinspection PyPackageRequirements
import numpy as np  # ships in conecsa-os-base:base

from .detection_service import StaleGeneration

logger = logging.getLogger(__name__)

_STAGES = ("prepare", "infer", "finish", "encode")
# After the first failure of a stage, log every Nth so a broken camera or
# model cannot flood the journal at frame rate.
_ERROR_LOG_EVERY = 100

# Accepted PROCESSED_OUTPUT_SCALE factors (1 = publish the drawn frame as is).
_OUTPUT_SCALE_FACTORS = (1, 2, 4)
DEFAULT_OUTPUT_SCALE = 1


def output_scale_from_env() -> int:
    """Downscale factor for the published processed stream (``PROCESSED_OUTPUT_SCALE``).

    Stage D is single-threaded and, with ``PROCESSING_DECODE_SCALE=1`` (the
    default, required by tiled inference), JPEG-encodes every frame at camera
    resolution. Publishing at half size halves that work while the overlay is
    still drawn at full resolution upstream. Accepts 1, 2 or 4; anything else
    (including a non-integer) falls back to 1 with a warning so a typo cannot
    silently change the stream.
    """
    raw = os.environ.get("PROCESSED_OUTPUT_SCALE", str(DEFAULT_OUTPUT_SCALE)).strip()
    try:
        factor = int(raw)
    except ValueError:
        logger.warning("PROCESSED_OUTPUT_SCALE=%r is not an int; using %d", raw, DEFAULT_OUTPUT_SCALE)
        return DEFAULT_OUTPUT_SCALE
    if factor not in _OUTPUT_SCALE_FACTORS:
        logger.warning("PROCESSED_OUTPUT_SCALE=%d not in %s; using %d",
                       factor, _OUTPUT_SCALE_FACTORS, DEFAULT_OUTPUT_SCALE)
        return DEFAULT_OUTPUT_SCALE
    return factor


def downscale_for_publish(frame: np.ndarray, factor: int) -> np.ndarray:
    """Shrink an already-drawn frame by ``factor`` (INTER_AREA) before JPEG encode.

    Returns the frame untouched for ``factor <= 1`` or when it is already too
    small to shrink; overlays are drawn upstream at full size and simply get
    scaled with the pixels.
    """
    if factor <= 1:
        return frame
    h, w = frame.shape[:2]
    new_w, new_h = w // factor, h // factor
    if new_w < 1 or new_h < 1:
        return frame
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


class ProcessingPipelineService:
    """Owns the decode→infer→encode pipeline and publishes the latest frame."""

    def __init__(self, consumer_service, codec_service, detection_service,
                 stats_service, gpio_service, overlay_renderer, video_service,
                 autostart: bool = True):
        self._consumer = consumer_service
        self._codec = codec_service
        self._detection = detection_service
        self._stats = stats_service
        self._gpio = gpio_service
        self._overlay = overlay_renderer
        self._video = video_service

        # Number of parallel lanes = number of inference contexts. With N>1 the
        # decode and inference stages each get N threads so neither becomes the
        # bottleneck (decode ~30 ms and inference ~30 ms are co-limiting at N=1).
        # N=1 keeps exactly the single-lane decode∥infer∥encode pipeline.
        try:
            self._n = max(1, int(os.environ.get("TENSORRT_CONTEXTS", "1")))
        except ValueError:
            self._n = 1

        # Stage hand-offs: bounded blocking queues sized to the lane count so all
        # lanes stay fed without unbounded latency build-up.
        self._q_infer: "queue.Queue" = queue.Queue(maxsize=self._n)
        self._q_finish: "queue.Queue" = queue.Queue(maxsize=self._n)
        self._q_encode: "queue.Queue" = queue.Queue(maxsize=self._n)

        # Shared frame-claim cursor so multiple decode threads each grab a
        # distinct (freshest-available) camera frame instead of duplicating work.
        self._grab_lock = threading.Lock()
        self._last_grabbed = 0

        # In-order publish across lanes (a slower lane finishing after a newer
        # frame already went out must not push the stream backwards).
        self._last_published_seq = 0
        self._publish_lock = threading.Lock()

        # The published frame goes into a shared-memory ring from which the
        # API-gateway container fans out the processed feed — the frames never
        # cross a gRPC boundary.
        self._proc_shm = None
        try:
            from conecsa_shm.processed_ring import ProcessedFrameWriter
            self._proc_shm = ProcessedFrameWriter()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Pipeline] processed-frame SHM unavailable: %s", exc)

        # Published-frame downscale factor, read once (see output_scale_from_env).
        self._output_scale = output_scale_from_env()

        # Last successfully encoded frame — re-emitted while output is frozen
        # (GPIO trigger pin low, or detection trigger disabled). Written by the
        # encode thread and every prepare lane, read by the prepare lanes: a
        # single reference assignment, so readers see either the previous or
        # the new complete frame, never a torn one — no lock needed.
        self._frozen: Optional[bytes] = None

        # Run gate: the pipeline only does work while detection is running.
        self._gate = threading.Condition(threading.Lock())

        # Runtime-transition gate (see the module docstring): while draining,
        # stage A claims no new frame for detection; ``_inflight`` counts the
        # frames currently owned by the detection path (A → B → C).
        self._draining = False
        self._inflight = 0
        self._inflight_cv = threading.Condition(threading.Lock())
        # Frames rejected because the runtime changed under them (StaleGeneration).
        self.stale_drops = 0

        # Per-stage failure counters and last-progress clock (see health()).
        # A stage failure drops that one item and nothing else: the threads are
        # started once and never restarted, so an unguarded raise would silently
        # end the processed stream while the gRPC port stayed healthy.
        self._health_lock = threading.Lock()
        self.stage_errors: dict = {s: 0 for s in _STAGES}
        self._last_progress: dict = {s: time.monotonic() for s in _STAGES}

        # FPS over the last N processed frames.
        self._frame_times: list = []
        self._fps_lock = threading.Lock()

        # Set by close(): every stage loop exits at its next iteration.
        self._stop = threading.Event()
        self._threads: list = []

        # Let the detection service quiesce us before it swaps/releases the
        # runtime (DetectionService.stop → drain, initialize/start → resume).
        attach = getattr(detection_service, "attach_pipeline", None)
        if callable(attach):
            attach(self)

        if autostart:
            self.start()

    # ── Lifecycle ──

    def start(self) -> None:
        """Start the stage threads (once). ``autostart=False`` defers this for tests."""
        if self._threads:
            return
        targets = []
        for i in range(self._n):
            targets.append((self._stage_prepare, f"pipeline-prepare-{i}"))
            targets.append((self._stage_infer, f"pipeline-infer-{i}"))
        targets.append((self._stage_finish, "pipeline-finish"))
        targets.append((self._stage_encode, "pipeline-encode"))
        for target, name in targets:
            thread = threading.Thread(target=target, daemon=True, name=name)
            self._threads.append(thread)
            thread.start()
        logger.info("[Pipeline] decode∥infer∥encode pipeline started (%d lane(s))", self._n)

    def close(self, timeout: float = 5.0) -> None:
        """Stop the stage threads (tests); the stages exit at their next poll."""
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout)

    # ── Runtime-transition gate ──

    def drain(self, timeout: float) -> bool:
        """Close the detection path and wait until no frame is in flight on it.

        Called by ``DetectionService.stop()`` before a model swap or a GPU
        release. Stage A stops claiming frames for detection at once; the items
        already held by B and C complete against the *old* runtime, which is
        still valid until the caller swaps it. Returns ``False`` if the stages
        did not empty within ``timeout`` (the generation check then rejects
        whatever was left).

        The run's counters are zeroed once the path is closed, so ``/stats``
        does not keep reporting a stopped run until the next frame.
        """
        self._draining = True
        with self._inflight_cv:
            quiesced = self._inflight_cv.wait_for(
                lambda: self._inflight == 0, timeout=timeout)
        if not quiesced:
            logger.error("[Pipeline] drain timed out with %d frame(s) in flight",
                         self._inflight)
        self._reset_run_stats()
        return quiesced

    def resume(self) -> None:
        """Reopen the detection path after the runtime transition completed."""
        self._clear_fps_window()
        self._draining = False

    def _reset_run_stats(self) -> None:
        """Zero the published stats and the rolling FPS window."""
        self._clear_fps_window()
        if self._stats is not None:
            self._stats.reset()

    def _clear_fps_window(self) -> None:
        """Forget frame timestamps, so the first FPS after a restart skips the gap."""
        with self._fps_lock:
            self._frame_times.clear()

    # ── Health ──

    def _stage_error(self, stage: str, ex: BaseException) -> None:
        """Count a per-item stage failure; log the first and every Nth after."""
        with self._health_lock:
            self.stage_errors[stage] += 1
            count = self.stage_errors[stage]
        if count == 1 or count % _ERROR_LOG_EVERY == 0:
            logger.error("[Pipeline] %s stage error (#%d): %s", stage, count, ex)

    def _progress(self, stage: str) -> None:
        """Record that ``stage`` completed an item (monotonic clock)."""
        with self._health_lock:
            self._last_progress[stage] = time.monotonic()

    def health(self) -> dict:
        """Snapshot for diagnostics: per-stage error counts, seconds since each
        stage last completed an item, in-flight count, stale drops, draining."""
        now = time.monotonic()
        with self._health_lock:
            errors = dict(self.stage_errors)
            ages = {s: round(now - t, 3) for s, t in self._last_progress.items()}
        return {
            "stage_errors": errors,
            "progress_age_s": ages,
            "inflight": self.inflight,
            "stale_drops": self.stale_drops,
            "draining": self._draining,
        }

    @property
    def inflight(self) -> int:
        """Frames currently owned by the detection path (for tests/diagnostics)."""
        with self._inflight_cv:
            return self._inflight

    def _begin_inflight(self) -> None:
        with self._inflight_cv:
            self._inflight += 1

    def _end_inflight(self) -> None:
        with self._inflight_cv:
            self._inflight -= 1
            if self._inflight <= 0:
                self._inflight = 0
                self._inflight_cv.notify_all()

    # ── Run gate ──

    def _wait_active(self) -> None:
        """Park the prepare stage while nothing needs processing."""
        with self._gate:
            # Short timeout so we re-check is_running, which flips via
            # DetectionService.start()/stop() without touching this condition.
            while not self._detection.is_running and not self._stop.is_set():
                self._gate.wait(timeout=0.5)

    # ── Publish ──

    def _publish(self, jpg: bytes, seq: int) -> None:
        """Publish a processed frame, keeping display in camera-frame order.

        ``seq`` is the source camera-frame sequence; frames older than the last
        published one (a slower lane finishing after a newer frame already went
        out) are dropped so the stream never goes backwards.

        The ring write stays under the ordering lock: N prepare lanes (frozen
        and "Detection Off" frames) and the encode thread all publish, and the
        processed ring is a single-producer seqlock — two writers entering it
        at once would pick the same slot and tear it, and releasing the lock
        between the sequence update and the write let a newer frame land first
        and be overwritten by the older one. The write is a memcpy of one JPEG
        plus a few header stores, so holding the lock costs nothing visible.
        """
        with self._publish_lock:
            if seq <= self._last_published_seq:
                return
            self._last_published_seq = seq
            if self._proc_shm is not None:
                self._proc_shm.publish(jpg)

    # ── Stage A — decode / color / stereo / preprocess (+ gate & freeze) ──

    def _grab_next(self):
        """Claim the freshest not-yet-claimed camera frame (shared across lanes).

        Holding the lock across the wait makes decode threads naturally alternate
        frames: each waits for the next frame, claims it, releases. Returns
        ``(seq, jpg, npy)`` or ``None`` on timeout.
        """
        with self._grab_lock:
            seq, jpg, npy = self._consumer.wait_for(self._last_grabbed, timeout=1.0)
            if seq <= self._last_grabbed:
                return None
            self._last_grabbed = seq
            return seq, jpg, npy

    def _stage_prepare(self) -> None:
        """Stage A loop: grab the latest frame, apply the trigger/freeze gates,
        decode + color + stereo-combine it, and hand the prepared input to the
        inference stage (or emit a freeze/"Detection Off" frame)."""
        while not self._stop.is_set():
            try:
                self._wait_active()
                grabbed = self._grab_next()
                if grabbed is None:
                    continue  # timed out — re-check the run gate
                seq, jpg, npy = grabbed
                # Frame age runs from here: the ring carries no capture
                # timestamp (and the wall clock is stepped by the hub).
                t_grab = time.monotonic()

                # GPIO trigger gate: pin low → freeze output (re-emit last frame).
                if not self._gpio.should_process_frame():
                    if self._frozen is not None:
                        self._publish(self._frozen, seq)
                    continue

                # Decode (reduced scale) + software RGB + stereo combine.
                frame = self._decode(jpg, npy)
                if frame is None:
                    continue
                frame = self._codec.combine_stereo(frame)

                # Detection trigger disabled → freeze (re-emit, or make one).
                if not self._detection.get_trigger_status():
                    if self._frozen is not None:
                        self._publish(self._frozen, seq)
                    else:
                        self._emit_off(frame, seq)
                    continue

                # Active detection path → hand off to the inference stage. Not
                # while draining: the runtime is about to change under us.
                if (self._detection.is_running and self._detection.is_model_loaded()
                        and not self._draining):
                    self._dispatch_detection(frame, seq, t_grab)
                else:
                    # Detection off → "Detection Off" overlay.
                    self._emit_off(frame, seq)
                self._progress("prepare")
            except Exception as ex:  # noqa: BLE001 - never let the stage thread die
                self._stage_error("prepare", ex)

    def _dispatch_detection(self, frame: np.ndarray, seq: int,
                            t_grab: Optional[float] = None) -> None:
        """Preprocess ``frame`` and hand it to the inference stage.

        The frame counts as in flight from before ``prepare`` (which touches
        the runtime) until the hand-off succeeds, at which point stage B owns
        the count; every other exit releases it. ``t_grab`` (monotonic) rides
        along so stage D can report the frame's age at publication.
        """
        self._begin_inflight()
        handed = False
        try:
            prepared = self._detection.prepare(frame)
            if prepared is None:
                self._emit_off(frame, seq)
                return
            generation, input_data, meta = prepared
            handed = self._handoff(
                self._q_infer, (seq, generation, frame, input_data, meta, t_grab))
        finally:
            if not handed:
                self._end_inflight()

    def _decode(self, jpg, npy) -> Optional[np.ndarray]:
        """Return a BGR frame from a raw-RGB array or a JPEG (decoded + RGB levels)."""
        if npy is not None:
            # Raw-RGB producer: color already applied in the webcam-server.
            return npy
        if jpg is not None:
            frame = self._codec.decode_frame_scaled(jpg)
            r, g, b = self._video.rgb_levels()
            return self._codec.apply_rgb_levels(frame, r, g, b)
        return None

    # ── Stage B — inference (one thread per context from the model-manager pool) ──

    def _stage_infer(self) -> None:
        """Stage B loop: run inference on prepared inputs (one thread per context,
        drawing a free context from the pool) and hand results to the finish stage.

        The item owns one in-flight count from the moment it is taken; it moves
        to stage C with a successful hand-off and is released on every other
        exit so ``drain`` can never wait on a frame that was dropped.
        """
        while not self._stop.is_set():
            item = self._take(self._q_infer)
            if item is None:
                continue
            handed = False
            try:
                seq, generation, frame, input_data, meta, t_grab = item
                output_data, inference_time = self._detection.infer(input_data, generation)
                handed = self._handoff(
                    self._q_finish,
                    (seq, generation, frame, output_data, meta, inference_time, t_grab))
                self._progress("infer")
            except StaleGeneration as ex:
                self.stale_drops += 1
                logger.info("[Pipeline] infer: dropped stale frame (%s)", ex)
            except Exception as ex:  # noqa: BLE001 - drop the item, keep the thread
                self._stage_error("infer", ex)
            finally:
                if not handed:
                    self._end_inflight()

    # ── Stage C — postprocess / draw / stats (single thread) ──

    def _stage_finish(self) -> None:
        """Stage C loop: postprocess/draw, update stats, drop
        out-of-order frames a faster lane already superseded, then hand the drawn
        frame to the encode stage. Releases the in-flight count on every exit:
        the encode stage does not touch the runtime."""
        while not self._stop.is_set():
            item = self._take(self._q_finish)
            if item is None:
                continue
            try:
                seq, generation, frame, output_data, meta, inference_time, t_grab = item

                # Drop frames a faster infer lane has already superseded (out-of-order
                # completion): no stats/GPIO churn, no wasted postprocess/encode.
                with self._publish_lock:
                    if seq <= self._last_published_seq:
                        continue
                t_finish = time.monotonic()

                try:
                    result = self._detection.finish(
                        output_data, frame, meta, inference_time, generation)
                except StaleGeneration as ex:
                    self.stale_drops += 1
                    logger.info("[Pipeline] finish: dropped stale frame (%s)", ex)
                    continue
                except Exception as ex:  # noqa: BLE001 - publish the raw frame instead
                    self._stage_error("finish", ex)
                    result = None

                if result is not None:
                    out = result.processed_image
                    num = result.num_detections
                    # Detection counts objects; classification counts class
                    # transitions (count_increment).
                    increment = getattr(result, "count_increment", None)
                    if increment is None:
                        increment = num
                    if increment > 0:
                        self._detection.increment_detection_count(increment)
                    # A frame that completes while the path is draining belongs
                    # to the run being stopped: drain() zeroes the stats, and a
                    # late completion (after a drain timeout) must not refill them.
                    if not self._draining:
                        self._stats.update(
                            fps=self._tick_fps(),
                            inference_time=inference_time * 1000,
                            detections=num,
                            increment_frames_with_detections=(num > 0),
                        )
                        # Stage C service time: postprocess + counters, not the
                        # wait for the encode stage below.
                        self._stats.record_timings(
                            finish_ms=(time.monotonic() - t_finish) * 1000.0)
                else:
                    out = frame

                self._handoff(self._q_encode, (seq, out, t_grab))
                self._progress("finish")
            except Exception as ex:  # noqa: BLE001 - drop the item, keep the thread
                self._stage_error("finish", ex)
            finally:
                self._end_inflight()

    # ── Stage D — JPEG encode + publish (single thread; cv2 releases the GIL) ──

    def _stage_encode(self) -> None:
        """Stage D loop: JPEG-encode the drawn frame, cache it as the frozen frame
        and publish it to the processed SHM ring."""
        while not self._stop.is_set():
            item = self._take(self._q_encode)
            if item is None:
                continue
            try:
                seq, out, t_grab = item
                t_encode = time.monotonic()
                # Shrink before encode: with PROCESSING_DECODE_SCALE=1 (the default,
                # required by tiled inference) this single-threaded stage encodes the
                # frame at camera resolution; PROCESSED_OUTPUT_SCALE=2 halves that work.
                # No-op at the default factor 1.
                encoded = self._codec.encode_frame(
                    downscale_for_publish(out, self._output_scale))
                if encoded:
                    self._frozen = encoded
                    self._publish(encoded, seq)
                    if not self._draining:  # see Stage C: no stats for a stopping run
                        now = time.monotonic()
                        self._stats.record_timings(
                            encode_ms=(now - t_encode) * 1000.0,
                            age_ms=(now - t_grab) * 1000.0 if t_grab is not None else None)
                self._progress("encode")
            except Exception as ex:  # noqa: BLE001 - this is the only encoder thread
                self._stage_error("encode", ex)

    # ── Helpers ──

    def _emit_off(self, frame: np.ndarray, seq: int) -> None:
        """Draw the 'Detection Off' overlay, encode, publish and cache as frozen."""
        off = self._overlay.draw_detection_off_overlay(frame)
        # Same downscale as the live path so the stream keeps one resolution.
        encoded = self._codec.encode_frame(downscale_for_publish(off, self._output_scale))
        if encoded:
            self._frozen = encoded
            self._publish(encoded, seq)

    @staticmethod
    def _handoff(q: "queue.Queue", item) -> bool:
        """Hand an item to the next stage, blocking until it is free.

        The bounded queue makes this a rendezvous: it throttles the upstream
        stage to the downstream rate (so stage A only decodes as fast as
        inference consumes — no wasted decodes), while still overlapping work
        (A decodes the next frame while B infers the current one). Realtime
        freshness is kept because stage A re-grabs the *latest* camera frame
        after each hand-off, skipping any frames captured while it was blocked.
        A long stall (>2 s) drops the frame rather than wedging the thread;
        returns whether the item was handed over.
        """
        try:
            q.put(item, timeout=2.0)
            return True
        except queue.Full:
            logger.warning("[Pipeline] downstream stalled — dropping frame")
            return False

    @staticmethod
    def _take(q: "queue.Queue"):
        """Pop the next item from a stage queue, or ``None`` after a 1s timeout."""
        try:
            return q.get(timeout=1.0)
        except queue.Empty:
            return None

    def _tick_fps(self) -> float:
        """Record a frame timestamp and return the rolling FPS over the last ~30."""
        with self._fps_lock:
            now = time.monotonic()
            self._frame_times.append(now)
            if len(self._frame_times) > 30:
                self._frame_times.pop(0)
            if len(self._frame_times) > 1:
                span = self._frame_times[-1] - self._frame_times[0]
                if span > 0:
                    return (len(self._frame_times) - 1) / span
            return 0.0
