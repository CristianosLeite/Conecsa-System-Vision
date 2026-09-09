"""Unit tests for the shared conecsa_common.events.EventBus."""
import threading

from conecsa_common.events import EventBus


def _bus(limit=200):
    return EventBus(source="svc", snapshot_keys=["a", "b"], history_limit=limit)


class TestPublishAndSnapshot:
    def test_versions_and_defaults(self):
        bus = _bus()
        e1 = bus.publish("x_changed", keys=["x"])
        e2 = bus.publish("y_changed")
        assert (e1["version"], e2["version"]) == (1, 2)
        assert e1["source"] == "svc"
        assert e2["keys"] == [] and e2["data"] == {}
        assert bus.publish("z", source="other")["source"] == "other"

    def test_snapshot_carries_the_configured_keys(self):
        bus = _bus()
        bus.publish("a")
        version, snap = bus.snapshot()
        assert version == 1
        assert snap["type"] == "state_snapshot"
        assert snap["keys"] == ["a", "b"]
        assert snap["source"] == "svc"


class TestWaitForChanges:
    def test_replays_events_since_version(self):
        bus = _bus()
        bus.publish("a")
        bus.publish("b")
        version, events, _sv, _stats, changed = bus.wait_for_changes(1, None, 0.1)
        assert (version, changed) == (2, True)
        assert [e["type"] for e in events] == ["b"]

    def test_timeout_returns_not_changed(self):
        bus = _bus()
        version, events, _sv, _stats, changed = bus.wait_for_changes(0, None, 0.02)
        assert (version, events, changed) == (0, [], False)

    def test_a_subscriber_behind_the_buffer_gets_one_snapshot(self):
        bus = _bus(limit=2)
        for name in ("a", "b", "c", "d"):
            bus.publish(name)
        # Versions 3 and 4 remain; a subscriber at 1 cannot be replayed.
        version, events, _sv, _stats, changed = bus.wait_for_changes(1, None, 0.1)
        assert (version, changed) == (4, True)
        assert [e["type"] for e in events] == ["state_snapshot"]
        assert events[0]["version"] == 4

    def test_stats_channel_wakes_a_tracking_waiter_only(self):
        bus = _bus()
        threading.Timer(0.02, bus.publish_stats, args=({"fps": 3},)).start()
        version, events, stats_version, stats, changed = bus.wait_for_changes(0, 0, 1.0)
        assert changed is True
        assert (version, events) == (0, [])
        assert (stats_version, stats) == (1, {"fps": 3})
        # Opted out: the same stats bump is invisible.
        _v, _e, sv, st, changed = bus.wait_for_changes(0, None, 0.02)
        assert (sv, st, changed) == (None, None, False)
