# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""SSE stream controllers: what each generator waits on between messages."""
from flask import Flask
from gateway.controllers import streams


class FakeEvents:
    """Records the versions each wait was given; replays scripted replies."""

    _version = 5

    def __init__(self, replies):
        self.replies = list(replies)
        self.waits = []

    def stats_snapshot(self):
        return 1, {"fps": 0.0}

    def wait_for_changes(self, last_version, last_stats_version, timeout):
        self.waits.append((last_version, last_stats_version))
        return self.replies.pop(0)


def test_the_stats_stream_follows_the_event_version(monkeypatch):
    # Regression: the stream kept the event version it subscribed at, so after
    # one invalidation every wait returned at once and keepalives spun (about
    # 500 per second on a device, one waitress core while idle).
    events = FakeEvents([
        (6, [{"type": "thresholds_changed"}], 1, None, True),  # invalidation only
        (6, [], 1, None, False),                               # quiet: timeout
    ])
    monkeypatch.setattr(streams, "event_service", events)
    with Flask(__name__).test_request_context():
        body = iter(streams.stream_stats().response)
        assert next(body) == 'data: {"fps": 0.0}\n\n'
        assert next(body) == ": keepalive\n\n"
        assert next(body) == ": keepalive\n\n"
    assert events.waits == [(5, 1), (6, 1)]
