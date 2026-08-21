"""Bounded SQLite ring queue — the shared core of the device's durable buffers.

Two services keep a small store-and-forward ring on disk that the hub drains
over mTLS: the inference-service's detection buffer and the api-gateway's
audit trail. They used to be ~200-line near-copies of the same machinery;
this base class owns that machinery once:

- one connection (WAL, ``busy_timeout``) serialized by one lock;
- record/byte caps enforced by oldest-first eviction (ring semantics);
- the drain protocol: ``list_backlog`` pages oldest-first (optionally
  byte-capped) and ``ack`` deletes confirmed rows idempotently;
- failure policy: the host service must never fail because of its buffer.
  Every operation swallows database errors behind one reopen attempt; if
  that also fails the queue *cools down* (``DISABLE_COOLDOWN_S``) and retries
  after, instead of disabling itself for the rest of the process's life.
  A corrupt database is quarantined as ``<name>.corrupt-<ts>`` — never
  deleted, the audit trail is the device's tamper evidence — before a fresh
  one is created. ``health()`` reports the degraded state so status
  endpoints can surface it.

Subclasses provide the schema (``TABLE``, ``_create_schema``), the drain row
shape (``SELECT_COLUMNS``, ``_row_to_record``) and their own write paths.
"""
import logging
import os
import sqlite3
import threading
import time
from typing import Optional, Sequence

logger = logging.getLogger(__name__)


class BoundedSqliteQueue:
    """Persistent bounded ring of records awaiting collection by the hub."""

    # ── subclass contract ─────────────────────────────────────────────────────
    #: Main ring table. Must have `id INTEGER PRIMARY KEY AUTOINCREMENT` and a
    #: `size_bytes INTEGER NOT NULL` column (the byte cap and page cap key on it).
    TABLE = ""
    #: Human name used in log lines ("audit buffer", "detection buffer").
    LOG_NAME = "queue"
    #: Columns list_backlog selects; `id` must come first.
    SELECT_COLUMNS: Sequence[str] = ()
    #: Optional per-page cap on the summed raw `size_bytes`; when set,
    #: SIZE_INDEX names the position of `size_bytes` in SELECT_COLUMNS.
    PAGE_SOFT_BYTES: Optional[int] = None
    SIZE_INDEX: int = -1

    DEFAULT_PAGE_LIMIT = 25
    MAX_PAGE_LIMIT = 200
    EVICT_BATCH = 100
    #: How long to stay degraded after a failed reopen before trying again.
    DISABLE_COOLDOWN_S = 60.0
    _DISCARD_LOG_INTERVAL_S = 60.0

    def __init__(self, db_path: str, max_records: int, max_bytes: int,
                 clock=time.monotonic):
        self._db_path = db_path
        self._max_records = max_records
        self._max_bytes = max_bytes
        self._clock = clock
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._disabled_until: Optional[float] = None
        self._count = 0
        self._bytes = 0
        self._discarded_since_log = 0
        self._last_discard_log = 0.0
        with self._lock:
            try:
                self._open(recreate_on_error=True)
            except (sqlite3.Error, OSError):
                logger.exception("%s unavailable; will retry in %.0fs",
                                 self.LOG_NAME, self.DISABLE_COOLDOWN_S)
                self._disabled = True

    # ── hooks ─────────────────────────────────────────────────────────────────

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        """Create the ring table(s). Called on every connect."""
        raise NotImplementedError

    def _row_to_record(self, row) -> dict:
        """Serialize one SELECT_COLUMNS row for the drain page.

        Runs outside the lock, so expensive formatting (base64, JSON) never
        stalls the write path.
        """
        raise NotImplementedError

    def _after_connect(self) -> None:
        """Extra per-subclass state to load after a (re)connect."""

    # ── degraded state ────────────────────────────────────────────────────────

    @property
    def _disabled(self) -> bool:
        """Whether the queue is currently cooling down after a failure."""
        return (self._disabled_until is not None
                and self._clock() < self._disabled_until)

    @_disabled.setter
    def _disabled(self, value: bool) -> None:
        if value:
            self._disabled_until = self._clock() + self.DISABLE_COOLDOWN_S
        else:
            self._disabled_until = None

    def health(self) -> dict:
        """Durability status for health/status endpoints."""
        degraded = self._disabled
        return {
            "enabled": not degraded,
            "pending": 0 if degraded else self._count,
            "degraded_reason": (
                "database unavailable; retrying" if degraded else None),
        }

    # ── shared public surface ─────────────────────────────────────────────────

    def pending_count(self) -> int:
        """Rows currently buffered and awaiting the hub."""
        return 0 if self._disabled else self._count

    def list_backlog(self, limit: int = 0) -> dict:
        """One page of buffered records, oldest first.

        ``device_now`` and each ``captured_at`` come from the same wall clock,
        so the hub can rebase them onto its own timeline as relative offsets
        even when this device's absolute clock is wrong — which it is on every
        boot, since the hardware has no RTC battery.
        """
        limit = max(1, min(int(limit) or self.DEFAULT_PAGE_LIMIT,
                           self.MAX_PAGE_LIMIT))
        empty: dict = {"records": [], "device_now": time.time(), "pending": 0}
        if self._disabled:
            return empty
        with self._lock:
            try:
                cursor = self._db().execute(
                    f"SELECT {', '.join(self.SELECT_COLUMNS)}"
                    f" FROM {self.TABLE} ORDER BY id ASC LIMIT ?",
                    (limit,),
                )
                # Stream instead of fetchall(): stop stepping the cursor once
                # a byte cap is reached so SQLite never materializes BLOBs of
                # rows this page will not include.
                rows = []
                page_bytes = 0
                for row in cursor:
                    if self.PAGE_SOFT_BYTES is not None:
                        size = row[self.SIZE_INDEX]
                        if rows and page_bytes + size > self.PAGE_SOFT_BYTES:
                            break
                        page_bytes += size
                    rows.append(row)
                pending = self._count
            except sqlite3.Error:
                self._handle_db_error("list_backlog")
                return empty
        # Formatting happens outside the lock so drain-page serialization
        # never stalls the producer thread's writes.
        records = [self._row_to_record(row) for row in rows]
        return {"records": records, "device_now": time.time(),
                "pending": pending}

    def ack(self, ids: list) -> int:
        """Delete records the hub confirmed persisting. Idempotent."""
        ids = [int(i) for i in ids]
        if self._disabled or not ids:
            return 0
        deleted = 0
        with self._lock:
            try:
                for start in range(0, len(ids), 500):
                    chunk = ids[start:start + 500]
                    marks = ",".join("?" * len(chunk))
                    freed_bytes, freed_count = self._db().execute(
                        "SELECT COALESCE(SUM(size_bytes), 0), COUNT(*)"
                        f" FROM {self.TABLE} WHERE id IN ({marks})",
                        chunk,
                    ).fetchone()
                    self._db().execute(
                        f"DELETE FROM {self.TABLE} WHERE id IN ({marks})",
                        chunk,
                    )
                    self._count -= freed_count
                    self._bytes -= freed_bytes
                    deleted += freed_count
                self._db().commit()
            except sqlite3.Error:
                self._handle_db_error("ack")
        return deleted

    # ── internals (all called with self._lock held) ───────────────────────────

    def _open(self, recreate_on_error: bool) -> None:
        """Open (or create) the database; quarantine + recreate on corruption."""
        directory = os.path.dirname(self._db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        try:
            self._connect()
        except sqlite3.DatabaseError:
            if not recreate_on_error:
                raise
            logger.exception(
                "%s at %s is unusable; quarantining and recreating",
                self.LOG_NAME, self._db_path)
            self._close()
            self._quarantine()
            self._connect()
        self._disabled = False

    def _quarantine(self) -> None:
        """Preserve a corrupt database for diagnosis instead of deleting it.

        The audit trail is the device's tamper evidence — destroying it on
        the first read error would be indistinguishable from tampering.
        """
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for suffix in ("", "-wal", "-shm"):
            src = self._db_path + suffix
            if not os.path.exists(src):
                continue
            try:
                os.replace(src, f"{self._db_path}.corrupt-{stamp}{suffix}")
            except OSError as exc:
                # Last resort: an unmovable sidecar must not block recovery.
                logger.warning("%s could not quarantine %s (%s); removing",
                               self.LOG_NAME, src, exc)
                try:
                    os.remove(src)
                except OSError:
                    logger.debug("%s cleanup skipped for %s",
                                 self.LOG_NAME, src, exc_info=True)

    def _connect(self) -> None:
        # One connection shared by every calling thread; self._lock serializes.
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._db().execute("PRAGMA journal_mode=WAL")
        self._db().execute("PRAGMA synchronous=NORMAL")
        self._db().execute("PRAGMA busy_timeout=5000")
        self._create_schema(self._db())
        self._db().commit()
        self._count, self._bytes = self._db().execute(
            "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0)"
            f" FROM {self.TABLE}"
        ).fetchone()
        self._after_connect()

    def _db(self) -> sqlite3.Connection:
        # Reopens lazily after a cooldown elapsed (the connection was closed
        # when the queue degraded). Raises sqlite3.Error on failure, which
        # every caller routes through _handle_db_error.
        if self._conn is None:
            self._open(recreate_on_error=True)
        assert self._conn is not None
        return self._conn

    def _close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                logger.debug("ignoring sqlite error while closing %s",
                             self.LOG_NAME, exc_info=True)
            self._conn = None

    def _handle_db_error(self, op: str) -> None:
        """One reopen/recreate attempt; if that also fails, cool down."""
        logger.exception("%s %s failed; reopening database", self.LOG_NAME, op)
        self._close()
        try:
            self._open(recreate_on_error=True)
        except (sqlite3.Error, OSError):
            logger.exception("%s reopen failed; retrying in %.0fs",
                             self.LOG_NAME, self.DISABLE_COOLDOWN_S)
            self._close()
            self._disabled = True

    def _evict_over_caps(self) -> None:
        """Drop oldest rows while over the record/byte caps (ring semantics)."""
        discarded = 0
        while self._count > 1 and (
            self._count > self._max_records or self._bytes > self._max_bytes
        ):
            rows = self._db().execute(
                f"SELECT id, size_bytes FROM {self.TABLE}"
                " ORDER BY id ASC LIMIT ?",
                (self.EVICT_BATCH,),
            ).fetchall()
            over = max(self._count - self._max_records, 0)
            for rec_id, size in rows:
                if not (self._count > 1 and (
                        over > 0 or self._bytes > self._max_bytes)):
                    break
                self._db().execute(
                    f"DELETE FROM {self.TABLE} WHERE id = ?", (rec_id,))
                self._count -= 1
                self._bytes -= size
                over -= 1
                discarded += 1
        if discarded:
            self._db().commit()
            self._discarded_since_log += discarded
            now = self._clock()
            if now - self._last_discard_log >= self._DISCARD_LOG_INTERVAL_S:
                logger.warning("%s full: discarded %d oldest record(s)",
                               self.LOG_NAME, self._discarded_since_log)
                self._discarded_since_log = 0
                self._last_discard_log = now
