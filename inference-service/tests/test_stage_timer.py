# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Pipeline stage timings (benchmark protocol): StageTimer and StatsService."""
import pytest
from api.services.stage_timer import StageTimer, summarize
from api.services.stats_service import StatsService


class TestSummarize:
    def test_empty_is_zero(self):
        assert summarize([]) == (0.0, 0.0, 0.0)

    def test_nearest_rank_percentiles(self):
        mean, p95, p99 = summarize(range(1, 101))
        assert mean == pytest.approx(50.5)
        assert (p95, p99) == (95, 99)

    def test_one_sample_is_every_percentile(self):
        assert summarize([7.0]) == (7.0, 7.0, 7.0)


class TestStageTimer:
    def test_keeps_only_the_window(self):
        timer = StageTimer(maxlen=4, refresh_s=0.0)
        for ms in (100, 100, 1, 1, 1, 1):
            timer.add(ms)
        assert timer.summary() == (1.0, 1.0, 1.0)

    def test_the_summary_is_recomputed_at_most_once_per_refresh(self):
        now = [0.0]
        timer = StageTimer(refresh_s=1.0, clock=lambda: now[0])
        timer.add(10)
        assert timer.summary()[0] == 10.0
        timer.add(30)
        assert timer.summary()[0] == 10.0, "cached within the refresh window"
        now[0] = 1.0
        assert timer.summary()[0] == 20.0

    def test_clear(self):
        timer = StageTimer(refresh_s=0.0)
        timer.add(5)
        timer.clear()
        assert timer.summary() == (0.0, 0.0, 0.0)


class TestStatsServiceTimings:
    def test_timings_reach_the_snapshot_and_the_listener(self):
        stats = StatsService()
        seen = []
        stats.set_update_listener(seen.append)
        stats.record_timings(finish_ms=12.0, encode_ms=8.0, age_ms=90.0)
        snap = stats.get_stats()
        assert (snap.finish_mean_ms, snap.finish_p95_ms, snap.finish_p99_ms) == (12.0,) * 3
        assert (snap.encode_mean_ms, snap.encode_p95_ms) == (8.0, 8.0)
        assert snap.frame_age_p95_ms == 90.0
        assert seen == [], "recording timings does not fan out on its own"
        stats.update(fps=25.0)
        assert seen[-1]["finish_p95_ms"] == 12.0
        assert seen[-1]["frame_age_p95_ms"] == 90.0

    def test_reset_clears_the_timings(self):
        stats = StatsService()
        stats.record_timings(finish_ms=12.0)
        stats.reset()
        assert stats.get_stats().finish_mean_ms == 0.0

    def test_partial_records_leave_the_other_stages_alone(self):
        stats = StatsService()
        stats.record_timings(encode_ms=4.0)
        snap = stats.get_stats()
        assert snap.finish_mean_ms == 0.0 and snap.encode_mean_ms == 4.0
