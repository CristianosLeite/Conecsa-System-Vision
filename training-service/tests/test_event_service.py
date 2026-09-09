"""Unit tests for the training-service event bus."""
from service.event_service import EventService


class TestPublishSnapshot:
    def test_publish_increments_version(self):
        svc = EventService()
        e1 = svc.publish("training_changed", keys=["training"])
        e2 = svc.publish("dataset_changed", keys=["dataset"])
        assert e1["version"] == 1
        assert e2["version"] == 2
        assert e1["source"] == "training"

    def test_publish_defaults(self):
        e = EventService().publish("x")
        assert e["keys"] == []
        assert e["data"] == {}
        assert e["source"] == "training"

    def test_snapshot(self):
        svc = EventService()
        svc.publish("a")
        version, snap = svc.snapshot()
        assert version == 1
        assert snap["type"] == "state_snapshot"
        assert snap["keys"] == ["training", "dataset", "sam"]


class TestWaitForChanges:
    def test_returns_new_events(self):
        svc = EventService()
        svc.publish("a")
        svc.publish("b")
        version, events, changed = svc.wait_for_changes(last_version=1, timeout=0.1)
        assert changed is True
        assert version == 2
        assert [e["type"] for e in events] == ["b"]

    def test_timeout_when_no_change(self):
        svc = EventService()
        version, events, changed = svc.wait_for_changes(last_version=0, timeout=0.05)
        assert changed is False
        assert events == []
        assert version == 0

    def test_a_subscriber_behind_the_replay_buffer_gets_a_snapshot(self):
        svc = EventService(history_limit=2)
        for name in ("a", "b", "c", "d"):
            svc.publish(name)
        version, events, changed = svc.wait_for_changes(last_version=1, timeout=0.1)
        assert (version, changed) == (4, True)
        # One snapshot instead of an empty list with changed=True, which used
        # to leave the gateway relay believing nothing was missed.
        assert [e["type"] for e in events] == ["state_snapshot"]
        assert events[0]["keys"] == ["training", "dataset", "sam"]
