# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Application event service.

Publishes lightweight invalidation events over SSE so every client surface
(web UI, Node-RED, curl-driven flows) can reconcile with the backend state.
The bus itself is ``conecsa_common.events.EventBus`` (shared with the
training-service); this facade only fixes the inference source name and the
snapshot keys, and keeps the stats channel the pipeline multiplexes onto the
same stream.
"""
from conecsa_common.events import EventBus

# What a fresh subscriber must reconcile after a snapshot event.
SNAPSHOT_KEYS = (
    "status",
    "models",
    "classes",
    "thresholds",
    "camera",
    "network",
    "gpio",
    "trigger",
    "areas",
    "application",
)


class EventService(EventBus):
    """Thread-safe in-process event bus with a small replay buffer."""

    def __init__(self, history_limit: int = 200):
        super().__init__(source="api", snapshot_keys=SNAPSHOT_KEYS,
                         history_limit=history_limit)
