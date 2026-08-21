"""
Detection buffer service - store-and-forward for hub outages.

While the hub is polling (`/detections/snapshot` ~1x/s over the gateway), this
service only tracks the last-pull time and the current detection signature.
When no snapshot pull arrives within the offline threshold, on-change
detection results (same change semantics as the hub collector: class@area
counts + total + model) are persisted to a small SQLite ring buffer on disk,
so they survive a device reboot. The hub drains them on reconnect via
``ListBacklog`` and only after it confirms persistence (``AckBacklog``) are
the local rows deleted.

Buffering never starts before the first hub contact of the device's lifetime
(persisted ``hub_seen`` flag), so a standalone device never fills its eMMC.
The pipeline must never fail because of this buffer: every public entry point
swallows database errors (one reopen/recreate attempt, then the buffer cools
down and retries later). The ring/drain/recovery machinery is the shared
``conecsa_common.BoundedSqliteQueue``; this module adds the detection schema,
the offline gating, and the frame encoding.
"""
import base64
import json
import logging
import sqlite3
import time
from typing import Optional

from conecsa_common import BoundedSqliteQueue

logger = logging.getLogger(__name__)

# Page size for ListBacklog when the request does not specify one, and the
# hard upper bound (each record may carry a base64 JPEG frame).
DEFAULT_PAGE_LIMIT = 25
MAX_PAGE_LIMIT = 100
# Byte-aware page trim: after including the first record, stop adding more once
# the raw stored bytes would exceed this. This is a best-effort cap to keep
# each page's JSON size (≈4/3 after base64 frames) below typical transport
# limits; remaining rows are picked up by the next page request.
PAGE_SOFT_BYTES = 3 * 1024 * 1024

_HUB_SEEN_KEY = "hub_seen"


def signature(detections: list, total: int, model: str) -> str:
    """Change signature of a detection set.

    Mirrors the hub collector's dedup signature
    (hub-vision/src/detections/mod.rs::signature): ordered counts of
    ``class_name@area_label`` (``none`` when the detection has no area),
    joined with the total and model. Confidence/bbox/color deliberately do
    not participate, so jitter on a static scene does not look like change.
    """
    counts: dict = {}
    for d in detections:
        area = d.get("area")
        label = (area or {}).get("label") if isinstance(area, dict) else None
        key = f"{d.get('class_name', '')}@{label or 'none'}"
        counts[key] = counts.get(key, 0) + 1
    # Rust Debug formatting of a BTreeMap<String, u32>: {"a": 1, "b": 2}
    counts_repr = "{" + ", ".join(
        f'"{k}": {counts[k]}' for k in sorted(counts)
    ) + "}"
    return f"{counts_repr}|{total}|{model}"


def _encode_jpeg(image) -> Optional[bytes]:
    """JPEG-encode a BGR frame (quality 80, matching the snapshot encoder)."""
    # noinspection PyPackageRequirements
    import cv2  # Package is included on os build; lazy so tests can stub this.
    ok, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return None
    return buf.tobytes()


class DetectionBufferService(BoundedSqliteQueue):
    """Persistent ring buffer of detection records for offline-hub periods."""

    TABLE = "buffered_detections"
    LOG_NAME = "detection buffer"
    SELECT_COLUMNS = ("id", "captured_at", "payload", "frame", "frame_is_raw",
                      "size_bytes")
    PAGE_SOFT_BYTES = PAGE_SOFT_BYTES
    SIZE_INDEX = 5
    DEFAULT_PAGE_LIMIT = DEFAULT_PAGE_LIMIT
    MAX_PAGE_LIMIT = MAX_PAGE_LIMIT
    EVICT_BATCH = 50

    def __init__(
        self,
        db_path: str,
        max_records: int,
        max_bytes: int,
        offline_threshold_s: float,
        clock=time.monotonic,
    ):
        self._offline_threshold_s = offline_threshold_s
        self._hub_seen = False
        self._last_sig: Optional[str] = None
        # No pull yet = offline: a device rebooting mid-outage must buffer the
        # scene it wakes up to. Booting next to a live hub is fine too — the
        # first snapshot poll (~1s) lands before the pipeline produces frames,
        # and the hub's drain dedup absorbs any overlap.
        self._last_pull: Optional[float] = None
        super().__init__(db_path, max_records, max_bytes, clock=clock)

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS buffered_detections ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " captured_at REAL NOT NULL,"
            " payload TEXT NOT NULL,"
            " frame BLOB,"
            " frame_is_raw INTEGER NOT NULL DEFAULT 1,"
            " size_bytes INTEGER NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS buffer_meta"
            " (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )

    def _after_connect(self) -> None:
        self._hub_seen = self._db().execute(
            "SELECT 1 FROM buffer_meta WHERE key = ?", (_HUB_SEEN_KEY,)
        ).fetchone() is not None

    def _row_to_record(self, row) -> dict:
        # base64/JSON happen outside the lock (BoundedSqliteQueue contract),
        # so drain-page formatting never stalls the pipeline-finish thread.
        rec_id, captured_at, payload, frame, frame_is_raw, _size = row
        record = {"id": rec_id, "captured_at": captured_at,
                  "frame": None, "raw_frame": None}
        record.update(json.loads(payload))
        if frame is not None:
            b64 = base64.b64encode(frame).decode('ascii')
            record["raw_frame" if frame_is_raw else "frame"] = b64
        return record

    # ── hub-contact tracking (gRPC pool threads) ─────────────────────────────

    def note_snapshot_pull(self) -> None:
        """Record a hub snapshot pull (the hub-is-online heartbeat)."""
        with self._lock:
            self._last_pull = self._clock()
            if self._hub_seen or self._disabled:
                return
            try:
                self._db().execute(
                    "INSERT OR REPLACE INTO buffer_meta (key, value) VALUES (?, '1')",
                    (_HUB_SEEN_KEY,),
                )
                self._db().commit()
                self._hub_seen = True
                logger.info("detection buffer armed: first hub contact recorded")
            except sqlite3.Error:
                self._handle_db_error("note_snapshot_pull")

    # ── producer (single pipeline-finish thread) ─────────────────────────────

    def observe(self, detections: list, total: int, model: str,
                raw_image, processed_image) -> None:
        """Consider one finished frame for buffering.

        Always tracks the change signature (so the first offline frame that
        equals the last state the hub saw is not re-recorded); writes a row
        only when the hub is offline, the set changed, is non-empty, and the
        hub has been seen at least once in this device's lifetime.
        """
        if self._disabled:
            return
        sig = signature(detections, total, model)
        with self._lock:
            changed = sig != self._last_sig
            self._last_sig = sig
            if not (self._hub_seen and changed and total > 0):
                return
            if not self._offline():
                return
        # Encode outside the lock: JPEG compression is the expensive step, and
        # holding the lock through it would stall the drain RPCs
        # (list_backlog/ack) behind frame processing.
        frame_is_raw = raw_image is not None
        image = raw_image if frame_is_raw else processed_image
        frame_bytes = _encode_jpeg(image) if image is not None else None
        payload = json.dumps(
            {"detections": detections, "total": total, "model": model}
        )
        size = len(payload) + len(frame_bytes or b"")
        with self._lock:
            # Re-check: a hub pull may have landed while encoding (the live
            # snapshot now covers this state), and a concurrent DB error may
            # have disabled the buffer.
            if self._disabled or not self._offline():
                return
            try:
                self._db().execute(
                    "INSERT INTO buffered_detections"
                    " (captured_at, payload, frame, frame_is_raw, size_bytes)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (time.time(), payload, frame_bytes, int(frame_is_raw), size),
                )
                self._db().commit()
                self._count += 1
                self._bytes += size
                self._evict_over_caps()
            except sqlite3.Error:
                self._handle_db_error("observe")

    # ── internals ─────────────────────────────────────────────────────────────

    def _offline(self) -> bool:
        """Whether the hub has gone silent past the offline threshold."""
        return self._last_pull is None or (
            self._clock() - self._last_pull) > self._offline_threshold_s
