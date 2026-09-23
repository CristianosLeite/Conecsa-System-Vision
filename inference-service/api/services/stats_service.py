# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Statistics service - Manages performance statistics.
"""
import threading
from typing import Optional

from ..models.detection_models import SystemStats
from .stage_timer import StageTimer


class StatsService:
    """Service for managing system statistics."""

    def __init__(self):
        """Initialize the stats service."""
        self.stats = SystemStats()
        # Service times of the single-threaded pipeline stages and the age of
        # a published frame (benchmark protocol).
        self._finish = StageTimer()
        self._encode = StageTimer()
        self._age = StageTimer()
        # Protects update()/reset() against concurrent readers so nothing
        # observes a partially-updated snapshot (or a stats object swapped out
        # by reset() mid-read).
        self._lock = threading.Lock()
        # Optional sink (EventService.publish_stats) so stats flow down the
        # unified app-event stream, letting clients use a single connection
        # instead of a dedicated stats stream. Invoked under ``self._lock``
        # with the full stats dict on every update/reset.
        self._update_listener = None

    def set_update_listener(self, listener) -> None:
        """Register a callback fired with the stats dict on every update."""
        self._update_listener = listener

    def update(self, fps: Optional[float] = None, inference_time: Optional[float] = None,
               detections: Optional[int] = None, increment_frames_with_detections: bool = False):
        """
        Update statistics.

        Args:
            fps: Frames per second
            inference_time: Inference time in milliseconds
            detections: Number of detections
            increment_frames_with_detections: Whether to increment frames with detections counter
        """
        with self._lock:
            if fps is not None:
                self.stats.fps = fps

            if inference_time is not None:
                self.stats.inference_time = inference_time

            if detections is not None:
                self.stats.detections = detections

            if increment_frames_with_detections:
                self.stats.frames_with_detections += 1

            self._fanout_locked()

    def record_timings(self, finish_ms: Optional[float] = None,
                       encode_ms: Optional[float] = None,
                       age_ms: Optional[float] = None) -> None:
        """Add one frame's stage service times (ms); no fan-out of its own.

        Called once per frame by stage C (``finish_ms``) and stage D
        (``encode_ms`` and the frame's age at publication); the next
        ``update`` carries the refreshed summary to listeners.
        """
        with self._lock:
            if finish_ms is not None:
                self._finish.add(finish_ms)
            if encode_ms is not None:
                self._encode.add(encode_ms)
            if age_ms is not None:
                self._age.add(age_ms)

    def get_stats(self) -> SystemStats:
        """Get a consistent snapshot of the current statistics."""
        with self._lock:
            return self._snapshot_locked()

    def reset(self):
        """Reset all statistics to zero."""
        with self._lock:
            self.stats = SystemStats()
            self._finish.clear()
            self._encode.clear()
            self._age.clear()
            self._fanout_locked()

    def _snapshot_locked(self) -> SystemStats:
        """A copy of the counters plus the timing summaries (caller holds the lock)."""
        finish_mean, finish_p95, finish_p99 = self._finish.summary()
        encode_mean, encode_p95, _ = self._encode.summary()
        _, age_p95, _ = self._age.summary()
        return SystemStats(
            fps=self.stats.fps,
            inference_time=self.stats.inference_time,
            detections=self.stats.detections,
            frames_with_detections=self.stats.frames_with_detections,
            finish_mean_ms=finish_mean,
            finish_p95_ms=finish_p95,
            finish_p99_ms=finish_p99,
            encode_mean_ms=encode_mean,
            encode_p95_ms=encode_p95,
            frame_age_p95_ms=age_p95,
        )

    def _fanout_locked(self) -> None:
        """Mirror the latest stats to the registered listener (if any).

        Caller must hold ``self._lock``. The listener (EventService) takes its
        own lock; nothing acquires ``self._lock`` while holding the event lock,
        so there is no inversion.
        """
        if self._update_listener is not None:
            try:
                self._update_listener(self._to_dict_locked())
            except Exception:  # noqa: BLE001
                pass

    def _to_dict_locked(self) -> dict:
        """Build the stats dict. Caller must hold ``self._lock``."""
        s = self._snapshot_locked()
        return {
            "fps": s.fps,
            "inference_time": s.inference_time,
            "detections": s.detections,
            "frames_with_detections": s.frames_with_detections,
            "finish_mean_ms": s.finish_mean_ms,
            "finish_p95_ms": s.finish_p95_ms,
            "finish_p99_ms": s.finish_p99_ms,
            "encode_mean_ms": s.encode_mean_ms,
            "encode_p95_ms": s.encode_p95_ms,
            "frame_age_p95_ms": s.frame_age_p95_ms,
        }
