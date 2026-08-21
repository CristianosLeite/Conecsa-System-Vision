"""Conformance tests for the shared bounded SQLite queue.

Both subclasses (the gateway's AuditBuffer, the inference-service's
DetectionBufferService) inherit this machinery; a minimal subclass here pins
the shared contract: ring eviction, drain/ack, corruption quarantine, and the
cooldown-retry degraded mode.
"""
import os
import sqlite3

import pytest
from conecsa_common import BoundedSqliteQueue


class MiniQueue(BoundedSqliteQueue):
    TABLE = "items"
    LOG_NAME = "mini queue"
    SELECT_COLUMNS = ("id", "body", "size_bytes")
    DISABLE_COOLDOWN_S = 60.0

    def _create_schema(self, conn):
        conn.execute(
            "CREATE TABLE IF NOT EXISTS items ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " body TEXT NOT NULL,"
            " size_bytes INTEGER NOT NULL)"
        )

    def _row_to_record(self, row):
        return {"id": row[0], "body": row[1]}

    def push(self, body: str):
        if self._disabled:
            return
        size = len(body.encode("utf-8"))
        with self._lock:
            if self._disabled:
                return
            try:
                self._db().execute(
                    "INSERT INTO items (body, size_bytes) VALUES (?, ?)",
                    (body, size))
                self._db().commit()
                self._count += 1
                self._bytes += size
                self._evict_over_caps()
            except sqlite3.Error:
                self._handle_db_error("push")


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return FakeClock()


def make_queue(tmp_path, clock, max_records=100, max_bytes=1 << 20):
    return MiniQueue(str(tmp_path / "q.db"), max_records, max_bytes,
                     clock=clock)


class TestRingContract:
    def test_round_trip_and_persistence(self, tmp_path, clock):
        q = make_queue(tmp_path, clock)
        q.push("one")
        q.push("two")
        assert q.pending_count() == 2
        q._close()
        again = make_queue(tmp_path, clock)
        assert again.pending_count() == 2
        page = again.list_backlog()
        assert [r["body"] for r in page["records"]] == ["one", "two"]

    def test_record_cap_evicts_oldest(self, tmp_path, clock):
        q = make_queue(tmp_path, clock, max_records=3)
        for i in range(6):
            q.push(f"item-{i}")
        bodies = [r["body"] for r in q.list_backlog()["records"]]
        assert bodies == ["item-3", "item-4", "item-5"]

    def test_byte_cap_evicts_oldest_but_keeps_the_newest(self, tmp_path, clock):
        q = make_queue(tmp_path, clock, max_bytes=10)
        q.push("aaaaaaaa")   # 8 bytes
        q.push("bbbbbbbb")   # over: evicts the oldest
        bodies = [r["body"] for r in q.list_backlog()["records"]]
        assert bodies == ["bbbbbbbb"]

    def test_ack_is_idempotent_and_batched(self, tmp_path, clock):
        q = make_queue(tmp_path, clock, max_records=1000)
        for i in range(700):  # crosses the 500-id chunking
            q.push(f"i{i}")
        ids = []
        while True:
            page = q.list_backlog(limit=200)
            if not page["records"]:
                break
            page_ids = [r["id"] for r in page["records"]]
            ids.extend(page_ids)
            q.ack(page_ids)
        assert len(set(ids)) == 700
        assert q.pending_count() == 0
        assert q.ack(ids) == 0


class TestRecovery:
    def test_corruption_is_quarantined_never_deleted(self, tmp_path, clock):
        path = tmp_path / "q.db"
        path.write_bytes(b"this is not a sqlite database at all")
        q = make_queue(tmp_path, clock)
        q.push("fresh")
        assert q.pending_count() == 1
        quarantined = [f for f in os.listdir(tmp_path)
                       if f.startswith("q.db.corrupt-")]
        assert quarantined, "the corrupt trail must survive for diagnosis"
        assert (tmp_path / quarantined[0]).read_bytes().startswith(b"this is")

    def test_a_failed_reopen_cools_down_then_recovers(self, tmp_path, clock,
                                                      monkeypatch):
        q = make_queue(tmp_path, clock)
        q.push("before")

        # Break both the live connection and the reopen path.
        def explode(*_args, **_kwargs):
            raise sqlite3.OperationalError("disk I/O error")

        real_open = q._open
        monkeypatch.setattr(q, "_open", explode)
        assert q._conn is not None
        q._conn.close()
        q._conn = None
        q.push("lost")
        assert q._disabled is True
        assert q.pending_count() == 0
        assert q.health()["enabled"] is False
        assert q.health()["degraded_reason"]

        # Still degraded within the cooldown, even after the fault clears.
        monkeypatch.setattr(q, "_open", real_open)
        clock.now += 30
        assert q._disabled is True

        # Past the cooldown the next operation reopens and the queue works.
        clock.now += 31
        q.push("after")
        assert q.health()["enabled"] is True
        bodies = [r["body"] for r in q.list_backlog()["records"]]
        assert "after" in bodies
        assert "before" in bodies, "the pre-fault data was never deleted"
