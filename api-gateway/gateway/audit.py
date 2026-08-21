"""Audit buffer — the device's own record of what users did to it.

Every mutating request that reaches the gateway is appended to a small SQLite
ring buffer on disk; the hub drains it over mTLS (``/api/v1/audit/backlog``)
and only after it confirms persisting a page are the local rows deleted
(``/api/v1/audit/backlog/ack``). Rows are written unconditionally, unlike the
detection buffer's store-and-forward: an audit trail that only records what
happened while the hub was watching would be worthless.

The device authenticates nobody, so the actor is whatever identity the hub
stamped on the request (see ``audit_events.actor``); the trail is explicit
about its absence rather than inventing one.

The gateway must never fail because of this buffer: every public entry point
swallows database errors (one reopen/recreate attempt, then the buffer cools
down and retries later). The ring/drain/recovery machinery is the shared
``conecsa_common.BoundedSqliteQueue``; this module adds the audit schema and
the request hook.
"""
import logging
import os
import sqlite3
import threading
import time
from typing import Optional

from conecsa_common import BoundedSqliteQueue

from . import audit_events
from .config import settings
from .helpers import _hub_verified, _is_trusted_proxy

logger = logging.getLogger(__name__)

# Page size for a backlog request that does not specify one, and the hard cap.
DEFAULT_PAGE_LIMIT = 25
MAX_PAGE_LIMIT = 200


class AuditBuffer(BoundedSqliteQueue):
    """Persistent ring buffer of audit records awaiting collection by the hub."""

    TABLE = "audit_events"
    LOG_NAME = "audit buffer"
    SELECT_COLUMNS = ("id", "captured_at", "username", "role", "event",
                      "detail", "source_ip", "outcome")
    DEFAULT_PAGE_LIMIT = DEFAULT_PAGE_LIMIT
    MAX_PAGE_LIMIT = MAX_PAGE_LIMIT

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS audit_events ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " captured_at REAL NOT NULL,"
            " username TEXT NOT NULL DEFAULT '',"
            " role TEXT NOT NULL DEFAULT '',"
            " event TEXT NOT NULL,"
            " detail TEXT NOT NULL DEFAULT '',"
            " source_ip TEXT NOT NULL DEFAULT '',"
            " outcome TEXT NOT NULL DEFAULT 'ok',"
            " size_bytes INTEGER NOT NULL)"
        )

    def _row_to_record(self, row) -> dict:
        return {"id": row[0], "captured_at": row[1], "username": row[2],
                "role": row[3], "event": row[4], "detail": row[5],
                "source_ip": row[6], "outcome": row[7]}

    # ── recording (request threads) ──────────────────────────────────────────

    def record(self, event: str, username: str = "", role: str = "",
               detail: str = "", source_ip: str = "", outcome: str = "ok",
               captured_at: Optional[float] = None) -> None:
        """Append one action. Never raises."""
        if self._disabled:
            return
        captured_at = time.time() if captured_at is None else captured_at
        # Encoded length, not character count: names and dataset titles are
        # written in Portuguese and Spanish, where a character is often two
        # bytes, and the ring cap is a promise about disk.
        size = sum(len(part.encode("utf-8")) for part in
                   (event, username, role, detail, source_ip, outcome))
        with self._lock:
            if self._disabled:
                return
            try:
                self._db().execute(
                    "INSERT INTO audit_events"
                    " (captured_at, username, role, event, detail, source_ip,"
                    "  outcome, size_bytes)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (captured_at, username, role, event, detail, source_ip,
                     outcome, size),
                )
                self._db().commit()
                self._count += 1
                self._bytes += size
                self._evict_over_caps()
            except sqlite3.Error:
                self._handle_db_error("record")

# ── module singleton + request hook ──────────────────────────────────────────

_buffer: Optional[AuditBuffer] = None
_buffer_lock = threading.Lock()


def buffer() -> AuditBuffer:
    """The process-wide audit buffer, opened on first use.

    Lazy rather than built at import: importing the gateway must not create
    files on disk, so tests and tooling can load the app without provisioning
    the audit volume.
    """
    global _buffer
    if _buffer is None:
        with _buffer_lock:
            if _buffer is None:
                _buffer = AuditBuffer(
                    db_path=os.path.join(settings.AUDIT_DIR, "audit.db"),
                    max_records=settings.AUDIT_MAX_RECORDS,
                    max_bytes=settings.AUDIT_MAX_BYTES,
                )
    return _buffer


def record_request(request, response) -> None:
    """Append the finished request to the trail, when it was a user action.

    Called from an ``after_request`` hook, so it runs for every route including
    the ``/api/*`` aliases, and it must never raise: a failure here would turn
    a successful action into a 500.
    """
    try:
        if not audit_events.should_audit(request):
            return
        # The identity headers are only meaningful when the request actually
        # came through the mTLS terminator; anything else on the compose
        # network could set them, so an unverified caller gets no actor.
        from_hub = _hub_verified()
        username, role = audit_events.actor(request) if from_hub else ("", "")
        # A hub-verified request with no operator on it is the hub acting on its
        # own — applying a recipe, deleting a dataset, pairing. The hub records
        # those itself, against the session it authenticated, so recording them
        # here as well would put an anonymous duplicate beside every named row.
        # Requests relayed from the device UI always carry an operator, and
        # anything not hub-verified is a local client worth recording.
        if from_hub and not username:
            return
        # Forwarded addresses are only worth anything from nginx: it sets
        # X-Forwarded-For itself and clears the hub's origin header on the
        # plaintext listener. A caller reaching the gateway directly could
        # otherwise name any address it liked in the trail.
        from_proxy = _is_trusted_proxy(request.remote_addr or "")
        buffer().record(
            event=audit_events.event_for(request),
            username=username,
            role=role,
            detail=audit_events.detail_for(request),
            source_ip=audit_events.source_ip(request, from_proxy),
            outcome=audit_events.outcome_for(response),
        )
    except Exception:  # noqa: BLE001 - the trail must never break a request
        logger.exception("failed to record an audit event")
