"""Training-service event bus.

A facade over ``conecsa_common.events.EventBus`` (shared with the
inference-service) with the training source name, the training snapshot keys
and no stats channel. The gateway tails ``TrainingControl.StreamEvents`` and
republishes onto its unified SSE bus, so web clients keep their single event
stream.
"""
from typing import Dict, List, Optional, Tuple

from conecsa_common.events import EventBus

SNAPSHOT_KEYS = ("training", "dataset", "sam")


class EventService:
    """Thread-safe in-process event bus with a small replay buffer."""

    def __init__(self, history_limit: int = 200):
        self._bus = EventBus(source="training", snapshot_keys=SNAPSHOT_KEYS,
                             history_limit=history_limit)

    def publish(
        self,
        event_type: str,
        keys: Optional[List[str]] = None,
        source: str = "training",
        data: Optional[Dict] = None,
    ) -> Dict:
        """Publish one event and wake all stream subscribers."""
        return self._bus.publish(event_type, keys=keys, source=source, data=data)

    def snapshot(self) -> Tuple[int, Dict]:
        """Return an initial snapshot event for new subscribers."""
        return self._bus.snapshot()

    def wait_for_changes(
        self, last_version: int, timeout: float
    ) -> Tuple[int, List[Dict], bool]:
        """Wait for events newer than ``last_version``.

        Returns ``(version, events, changed)``; ``changed`` is ``False`` on
        timeout (caller emits a keepalive). A subscriber that fell behind the
        replay buffer receives a single snapshot event, like on the inference
        side, instead of an empty list that hid the gap.
        """
        version, events, _stats_version, _stats, changed = self._bus.wait_for_changes(
            last_version, None, timeout)
        return version, events, changed
