# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Rolling service-time window for one pipeline stage (benchmark protocol).

Stages C (postprocess) and D (encode + publish) are single-threaded, so
their service time bounds the frame rate whatever ``TENSORRT_CONTEXTS`` is.
Each keeps the last few hundred samples; the mean/p95/p99 summary is cached
and recomputed at most once per ``refresh_s`` so reading it on every stats
fan-out stays O(1). Not thread-safe on its own: ``StatsService`` holds its
lock around every call.
"""
import math
import time
from collections import deque
from typing import Callable, Deque, Optional, Tuple

Summary = Tuple[float, float, float]  # (mean, p95, p99) in milliseconds

_EMPTY: Summary = (0.0, 0.0, 0.0)


def _nearest_rank(ordered, percent: float) -> float:
    """The nearest-rank percentile of an ascending, non-empty sequence."""
    rank = max(1, math.ceil(percent / 100.0 * len(ordered)))
    return ordered[rank - 1]


def summarize(samples) -> Summary:
    """``(mean, p95, p99)`` of ``samples``; zeros when there are none."""
    if not samples:
        return _EMPTY
    ordered = sorted(samples)
    return (sum(ordered) / len(ordered), _nearest_rank(ordered, 95.0),
            _nearest_rank(ordered, 99.0))


class StageTimer:
    """The last ``maxlen`` samples of one stage, in milliseconds."""

    def __init__(self, maxlen: int = 512, refresh_s: float = 1.0,
                 clock: Callable[[], float] = time.monotonic):
        self._samples: Deque[float] = deque(maxlen=maxlen)
        self._refresh_s = refresh_s
        self._clock = clock
        self._cached: Summary = _EMPTY
        self._computed_at: Optional[float] = None
        self._dirty = False

    def add(self, ms: float) -> None:
        self._samples.append(float(ms))
        self._dirty = True

    def clear(self) -> None:
        self._samples.clear()
        self._cached = _EMPTY
        self._computed_at = None
        self._dirty = False

    def summary(self) -> Summary:
        """``(mean, p95, p99)``, at most ``refresh_s`` old."""
        now = self._clock()
        if self._dirty and (self._computed_at is None
                            or now - self._computed_at >= self._refresh_s):
            self._cached = summarize(self._samples)
            self._computed_at = now
            self._dirty = False
        return self._cached
