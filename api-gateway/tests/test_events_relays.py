"""start_relays() starts its three threads once per process (review L4)."""
import threading

from gateway import events


class RecordingThread:
    """threading.Thread stand-in: records the target, never runs it."""

    started = []

    def __init__(self, target=None, daemon=None, name=None):
        self.target, self.name = target, name

    def start(self):
        RecordingThread.started.append(self.name)


def test_a_second_call_starts_nothing(monkeypatch):
    RecordingThread.started.clear()
    monkeypatch.setattr(events.threading, "Thread", RecordingThread)
    monkeypatch.setattr(events, "_relays_started", False)
    monkeypatch.setattr(events, "_relays_lock", threading.Lock())

    events.start_relays()
    events.start_relays()
    assert RecordingThread.started == ["event-relay", "stats-relay", "training-event-relay"]
