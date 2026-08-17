"""Unit tests for the hub-driven system clock (agent/time_agent.py)."""
from datetime import datetime, timezone

import pytest

from agent.time_agent import TimeAgent
from agent import time_agent


def _millis(text: str) -> int:
    """UTC 'YYYY-MM-DD HH:MM:SS' → epoch milliseconds."""
    moment = datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return int(moment.timestamp() * 1000)


@pytest.fixture
def floor(tmp_path, monkeypatch):
    """Point the floor file at a temp dir and stub out the actual clock write."""
    path = tmp_path / "state" / "fake-hwclock"
    monkeypatch.setattr(time_agent, "FLOOR_PATH", str(path))
    applied = []
    monkeypatch.setattr(TimeAgent, "_apply", staticmethod(applied.append))
    monkeypatch.setattr(TimeAgent, "_write_rtc", staticmethod(lambda: None))
    return path, applied


class TestFloorFile:
    def test_absent_floor_reads_as_none(self, floor):
        assert TimeAgent.read_floor() is None

    def test_round_trips_through_the_shared_format(self, floor):
        path, _ = floor
        moment = datetime(2026, 8, 3, 10, 30, 0, tzinfo=timezone.utc)
        TimeAgent.write_floor(moment)
        # conecsa-fake-hwclock.sh sorts these stamps lexicographically, so the
        # fixed-width UTC layout is part of the contract.
        assert path.read_text() == "2026-08-03 10:30:00\n"
        assert TimeAgent.read_floor() == moment

    def test_garbage_floor_is_ignored_not_fatal(self, floor):
        path, _ = floor
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("last tuesday\n")
        assert TimeAgent.read_floor() is None


class TestSetSystemTime:
    def test_accepts_a_time_and_records_the_new_floor(self, floor):
        _, applied = floor
        result = TimeAgent.set_system_time(_millis("2026-08-03 10:00:00"), "pairing")
        assert result["success"] is True
        assert applied == [datetime(2026, 8, 3, 10, 0, tzinfo=timezone.utc)]
        assert TimeAgent.read_floor() == datetime(2026, 8, 3, 10, 0, tzinfo=timezone.utc)

    def test_refuses_to_move_below_the_persisted_floor(self, floor):
        _, applied = floor
        TimeAgent.write_floor(datetime(2026, 8, 3, 10, 0, tzinfo=timezone.utc))
        # A hub that lost its own clock (the kiosk runs on a Jetson too) must
        # not be able to drag the device backwards.
        result = TimeAgent.set_system_time(_millis("2020-01-02 00:00:00"), "hub-poll")
        assert result["success"] is False
        assert "floor" in result["message"]
        assert applied == []

    def test_moving_forward_from_a_floor_is_allowed(self, floor):
        _, applied = floor
        TimeAgent.write_floor(datetime(2026, 8, 3, 10, 0, tzinfo=timezone.utc))
        assert TimeAgent.set_system_time(_millis("2026-08-03 11:00:00"))["success"] is True
        assert applied == [datetime(2026, 8, 3, 11, 0, tzinfo=timezone.utc)]

    @pytest.mark.parametrize("value", [0, -1])
    def test_rejects_a_non_positive_timestamp(self, floor, value):
        _, applied = floor
        assert TimeAgent.set_system_time(value, "hub-poll")["success"] is False
        assert applied == []

    def test_reports_a_failed_clock_write(self, floor, monkeypatch):
        monkeypatch.setattr(TimeAgent, "_apply", staticmethod(
            lambda _moment: (_ for _ in ()).throw(OSError(1, "Operation not permitted"))))
        result = TimeAgent.set_system_time(_millis("2026-08-03 10:00:00"), "hub-poll")
        assert result["success"] is False
        assert "clock_settime" in result["message"]
        # A failed write must not advance the floor.
        assert TimeAgent.read_floor() is None
