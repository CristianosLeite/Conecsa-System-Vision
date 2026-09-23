# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: Apache-2.0

"""In-process event bus shared by the inference and training services.

Publishes lightweight invalidation events that a gRPC server stream relays to
the api-gateway, which republishes them over SSE so every client surface (web
UI, Node-RED, curl-driven flows) can reconcile with the backend state. The
transport-neutral logic lives here; each service configures its event source,
its snapshot keys and whether it uses the stats channel.

This module is dependency-free (stdlib only).
"""
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple


class EventBus:
    """Thread-safe event bus with a small replay buffer and a stats channel.

    Invalidation events get a monotonically increasing version and are kept
    in a bounded replay deque so a subscriber can catch up after a short gap.
    A subscriber whose ``last_version`` is older than the oldest replayed
    event receives one ``state_snapshot`` event instead, forcing a full
    reconciliation rather than silently dropping invalidations.

    High-rate performance stats are a "latest value" channel with their own
    version, never appended to the replay deque, so the per-frame update rate
    cannot evict invalidation events; both channels share one condition so a
    single waiter wakes on either.
    """

    def __init__(self, source: str, snapshot_keys: Sequence[str],
                 history_limit: int = 200) -> None:
        self._cond = threading.Condition()
        self._version = 0
        self._events: Deque[Dict] = deque(maxlen=history_limit)
        self._source = source
        self._snapshot_keys = list(snapshot_keys)
        self._stats_version = 0
        self._stats: Dict = {}

    @property
    def source(self) -> str:
        """The default ``source`` stamped on events and snapshots."""
        return self._source

    def publish(
        self,
        event_type: str,
        keys: Optional[List[str]] = None,
        source: Optional[str] = None,
        data: Optional[Dict] = None,
    ) -> Dict:
        """Publish one invalidation event and wake every waiter."""
        with self._cond:
            self._version += 1
            event = {
                "version": self._version,
                "type": event_type,
                "timestamp": time.time(),
                "source": source or self._source,
                "keys": keys or [],
                "data": data or {},
            }
            self._events.append(event)
            self._cond.notify_all()
            return event

    def publish_stats(self, stats: Dict) -> None:
        """Update the latest-value stats channel and wake every waiter."""
        with self._cond:
            self._stats_version += 1
            self._stats = stats or {}
            self._cond.notify_all()

    def snapshot(self) -> Tuple[int, Dict]:
        """Return ``(version, snapshot_event)`` for a new subscriber."""
        with self._cond:
            return self._version, self._snapshot_locked()

    def wait_for_changes(
        self, last_version: int, last_stats_version: Optional[int], timeout: float
    ) -> Tuple[int, List[Dict], Optional[int], Optional[Dict], bool]:
        """Wait for events newer than ``last_version`` or a stats update.

        Pass ``last_stats_version=None`` to ignore the stats channel. Returns
        ``(version, events, stats_version, stats, changed)``: ``events`` is
        the replay since ``last_version`` (or a single snapshot when the
        subscriber fell behind the buffer), ``stats`` the latest stats dict
        when it advanced past ``last_stats_version`` (else ``None``), and
        ``changed`` is ``False`` on timeout (the caller emits a keepalive).
        """
        track_stats = last_stats_version is not None
        with self._cond:
            changed = self._cond.wait_for(
                lambda: self._version != last_version
                or (track_stats and self._stats_version != last_stats_version),
                timeout=timeout,
            )
            stats_version = self._stats_version if track_stats else None
            if not changed:
                return self._version, [], stats_version, None, False

            events: List[Dict] = []
            if self._version != last_version:
                events = [e for e in self._events if e["version"] > last_version]
                oldest = self._events[0]["version"] if self._events else self._version
                if not events or oldest > last_version + 1:
                    # The subscriber fell behind the replay buffer (its next
                    # event was evicted) or is ahead of it (a counter reset).
                    # Force a full reconciliation instead of silently dropping
                    # invalidations.
                    events = [self._snapshot_locked()]

            stats = (
                dict(self._stats)
                if track_stats and self._stats_version != last_stats_version
                else None
            )
            return self._version, events, stats_version, stats, True

    def _snapshot_locked(self) -> Dict:
        """Build a snapshot event. Caller must hold ``self._cond``."""
        return {
            "version": self._version,
            "type": "state_snapshot",
            "timestamp": time.time(),
            "source": self._source,
            "keys": list(self._snapshot_keys),
            "data": {},
        }
