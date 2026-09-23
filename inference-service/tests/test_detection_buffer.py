# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for the offline detection buffer (store-and-forward)."""
import base64
import json
import sqlite3

import api.services.detection_buffer as buffer_mod
import pytest

THRESHOLD = 5.0

RAW = object()        # stand-in for the clean BGR frame
ANNOTATED = object()  # stand-in for the overlay frame


class FakeClock:
    """Deterministic monotonic clock."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture(autouse=True)
def stub_jpeg(monkeypatch):
    """Encode 'frames' without cv2: raw and annotated get distinct bytes."""
    monkeypatch.setattr(
        buffer_mod, "_encode_jpeg",
        lambda image: b"RAWJPEG" if image is RAW else b"ANNJPEG",
    )


@pytest.fixture
def clock():
    return FakeClock()


def make_buffer(tmp_path, clock, max_records: int = 100,
                max_bytes: int = 10_000_000,
                sample_interval_s: float = 0.0) -> buffer_mod.DetectionBufferService:
    # Every frame is a sample unless a test exercises the offline sampling.
    return buffer_mod.DetectionBufferService(
        str(tmp_path / "buffer.db"), max_records=max_records,
        max_bytes=max_bytes, offline_threshold_s=THRESHOLD, clock=clock,
        sample_interval_s=sample_interval_s)


def go_offline(buf: buffer_mod.DetectionBufferService, clock: FakeClock) -> None:
    """Record a hub pull (arming hub_seen), then let the threshold lapse."""
    buf.note_snapshot_pull()
    clock.advance(THRESHOLD + 1)


def dets(*classes, area=None):
    return [
        {"class_name": c, "confidence": 0.9, "area": area,
         "bbox": [0.1, 0.2, 0.3, 0.4], "color": "#ff0000"}
        for c in classes
    ]


class TestSignature:
    """Must mirror hub-vision/src/detections/mod.rs::signature exactly
    (cases copied from hub-vision/src/detections/tests.rs)."""

    def test_encodes_class_area_counts_total_and_model(self):
        assert buffer_mod.signature(dets("cap"), 1, "yolo") == '{"cap@none": 1}|1|yolo'

    def test_groups_by_class_and_area(self):
        detections = dets("cap", "cap", area={"label": "zone-1"}) + dets("cap")
        assert buffer_mod.signature(detections, 3, "m") == \
            '{"cap@none": 1, "cap@zone-1": 2}|3|m'

    def test_ignores_confidence_bbox_and_color(self):
        a = dets("cap")
        b = [dict(a[0], confidence=0.1, bbox=[0.5, 0.5, 0.9, 0.9], color=None)]
        assert buffer_mod.signature(a, 1, "m") == buffer_mod.signature(b, 1, "m")

    def test_ignores_segmentation_polygons(self):
        # Polygon outlines jitter frame to frame; the same instances must not
        # count as a change.
        a = [dict(d, polygons=[[[0.1, 0.2], [0.3, 0.2], [0.2, 0.4]]]) for d in dets("cap")]
        b = [dict(a[0], polygons=[[[0.11, 0.21], [0.3, 0.19], [0.25, 0.4], [0.2, 0.41]]])]
        c = dets("cap")
        assert buffer_mod.signature(a, 1, "m") == buffer_mod.signature(b, 1, "m")
        assert buffer_mod.signature(a, 1, "m") == buffer_mod.signature(c, 1, "m")

    def test_distinguishes_class_total_and_model(self):
        base = buffer_mod.signature(dets("cap"), 1, "m")
        assert buffer_mod.signature(dets("bottle"), 1, "m") != base
        assert buffer_mod.signature(dets("cap"), 2, "m") != base
        assert buffer_mod.signature(dets("cap"), 1, "other") != base


class TestGating:
    def test_never_buffers_before_first_hub_contact(self, tmp_path, clock):
        buf = make_buffer(tmp_path, clock)
        clock.advance(THRESHOLD * 10)  # "offline", but no hub was ever seen
        buf.observe(dets("cap"), 1, "m", RAW, ANNOTATED)
        assert buf.pending_count() == 0

    def test_does_not_buffer_while_the_hub_is_polling(self, tmp_path, clock):
        buf = make_buffer(tmp_path, clock)
        buf.note_snapshot_pull()
        clock.advance(THRESHOLD - 1)
        buf.observe(dets("cap"), 1, "m", RAW, ANNOTATED)
        assert buf.pending_count() == 0

    def test_buffers_a_change_after_the_offline_threshold(self, tmp_path, clock):
        buf = make_buffer(tmp_path, clock)
        go_offline(buf, clock)
        buf.observe(dets("cap"), 1, "m", RAW, ANNOTATED)
        assert buf.pending_count() == 1

    def test_the_record_is_built_only_when_one_is_written(self, tmp_path, clock):
        # The pipeline passes light items for the signature and a provider for
        # the stored items (segmentation rings are extracted only then).
        buf = make_buffer(tmp_path, clock)
        calls = []

        def full():
            calls.append(1)
            return [dict(dets("cap")[0], polygons=[[[0.1, 0.2], [0.3, 0.2], [0.2, 0.4]]])]

        buf.note_snapshot_pull()
        buf.observe(dets("cap"), 1, "m", RAW, ANNOTATED, record_detections=full)  # online
        assert calls == []
        clock.advance(THRESHOLD + 1)
        buf.observe(dets("bottle"), 1, "m", RAW, ANNOTATED, record_detections=full)
        assert calls == [1] and buf.pending_count() == 1
        with sqlite3.connect(str(tmp_path / "buffer.db")) as db:
            (payload,) = db.execute("SELECT payload FROM buffered_detections").fetchone()
        assert json.loads(payload)["detections"][0]["polygons"] == [[[0.1, 0.2], [0.3, 0.2],
                                                                     [0.2, 0.4]]]

    def test_a_reboot_mid_outage_buffers_the_scene_it_wakes_up_to(
            self, tmp_path, clock):
        # hub_seen persisted by an earlier life; device rebooted with the hub
        # still down: a static scene present at boot must be recorded once.
        make_buffer(tmp_path, clock).note_snapshot_pull()
        buf = make_buffer(tmp_path, clock)
        buf.observe(dets("cap"), 1, "m", RAW, ANNOTATED)
        assert buf.pending_count() == 1
        buf.observe(dets("cap"), 1, "m", RAW, ANNOTATED)  # unchanged scene
        assert buf.pending_count() == 1

    def test_ignores_empty_sets_and_repeated_signatures(self, tmp_path, clock):
        buf = make_buffer(tmp_path, clock)
        go_offline(buf, clock)
        buf.observe([], 0, "m", RAW, ANNOTATED)
        assert buf.pending_count() == 0
        buf.observe(dets("cap"), 1, "m", RAW, ANNOTATED)
        buf.observe(dets("cap"), 1, "m", RAW, ANNOTATED)  # same signature
        assert buf.pending_count() == 1
        buf.observe(dets("cap", "cap"), 2, "m", RAW, ANNOTATED)  # changed
        assert buf.pending_count() == 2

    def test_a_pull_landing_during_encoding_skips_the_write(
            self, tmp_path, clock, monkeypatch):
        # The JPEG encode runs outside the lock; if the hub reconnects in that
        # window the live snapshot covers this state, so observe must re-check
        # and drop the row instead of buffering a duplicate.
        buf = make_buffer(tmp_path, clock)
        go_offline(buf, clock)

        def encode_and_reconnect(image):
            buf.note_snapshot_pull()  # hub pulls mid-encode (no deadlock)
            return b"JPG"

        monkeypatch.setattr(buffer_mod, "_encode_jpeg", encode_and_reconnect)
        buf.observe(dets("cap"), 1, "m", RAW, ANNOTATED)
        assert buf.pending_count() == 0

    def test_state_seen_online_is_not_rerecorded_when_going_offline(
            self, tmp_path, clock):
        # The hub already snapshot the "cap" state while online; the first
        # offline frame with the same signature must not be buffered again.
        buf = make_buffer(tmp_path, clock)
        buf.note_snapshot_pull()
        buf.observe(dets("cap"), 1, "m", RAW, ANNOTATED)  # online: sig tracked
        clock.advance(THRESHOLD + 1)
        buf.observe(dets("cap"), 1, "m", RAW, ANNOTATED)  # unchanged
        assert buf.pending_count() == 0
        buf.observe(dets("bottle"), 1, "m", RAW, ANNOTATED)
        assert buf.pending_count() == 1


class TestOfflineSampling:
    """Offline, frames are compared at the hub's cadence, not every frame."""

    def test_a_flickering_set_writes_at_most_one_row_per_interval(self, tmp_path, clock):
        # Regression: A/B flips at 30 fps wrote one row per flip (499 in ~2 min).
        buf = make_buffer(tmp_path, clock, sample_interval_s=1.0)
        go_offline(buf, clock)
        for i in range(60):  # two seconds at 30 fps
            buf.observe(dets("cap") if i % 2 else dets("bottle"), 1, "m", RAW, ANNOTATED)
            clock.advance(1 / 30)
        assert 1 <= buf._count <= 3

    def test_a_change_inside_the_interval_is_taken_at_the_next_sample(self, tmp_path, clock):
        buf = make_buffer(tmp_path, clock, sample_interval_s=1.0)
        go_offline(buf, clock)
        buf.observe(dets("cap"), 1, "m", RAW, ANNOTATED)
        clock.advance(0.5)
        buf.observe(dets("bottle"), 1, "m", RAW, ANNOTATED)
        assert buf._count == 1
        clock.advance(0.6)
        buf.observe(dets("bottle"), 1, "m", RAW, ANNOTATED)
        assert buf._count == 2


class TestBacklogProtocol:
    def _filled(self, tmp_path, clock, n=3):
        buf = make_buffer(tmp_path, clock)
        go_offline(buf, clock)
        for i in range(n):
            buf.observe(dets(f"class-{i}"), 1, "m", RAW, ANNOTATED)
        return buf

    def test_roundtrip_list_then_ack(self, tmp_path, clock):
        buf = self._filled(tmp_path, clock, n=3)
        page = buf.list_backlog()
        assert page["pending"] == 3
        assert isinstance(page["device_now"], float)
        assert [r["class_name"] for rec in page["records"]
                for r in rec["detections"]] == ["class-0", "class-1", "class-2"]
        rec = page["records"][0]
        assert rec["total"] == 1
        assert rec["model"] == "m"
        assert isinstance(rec["captured_at"], float)
        assert rec["raw_frame"] == base64.b64encode(b"RAWJPEG").decode()
        assert rec["frame"] is None

        ids = [r["id"] for r in page["records"]]
        assert buf.ack(ids) == 3
        assert buf.pending_count() == 0
        assert buf.list_backlog()["records"] == []

    def test_pagination_is_oldest_first_with_stable_ids(self, tmp_path, clock):
        buf = self._filled(tmp_path, clock, n=5)
        first = buf.list_backlog(limit=2)
        assert len(first["records"]) == 2
        assert first["pending"] == 5
        ids = [r["id"] for r in first["records"]]
        assert ids == sorted(ids)
        buf.ack(ids)
        second = buf.list_backlog(limit=2)
        assert all(r["id"] > ids[-1] for r in second["records"])

    def test_pages_are_trimmed_by_bytes(self, tmp_path, clock, monkeypatch):
        # Each record stores payload + 7-byte fake JPEG; force a byte cap smaller than
        # a single record so trimming returns exactly one record per page.
        buf = self._filled(tmp_path, clock, n=3)
        one_record = buf.list_backlog(limit=1)["records"][0]
        monkeypatch.setattr(buffer_mod.DetectionBufferService,
                            "PAGE_SOFT_BYTES", 1)
        page = buf.list_backlog(limit=25)
        # The first record always ships (a page must make progress), the rest
        # wait for the drain's next page request.
        assert len(page["records"]) == 1
        assert page["records"][0]["id"] == one_record["id"]
        assert page["pending"] == 3

    def test_ack_is_idempotent(self, tmp_path, clock):
        buf = self._filled(tmp_path, clock, n=1)
        ids = [r["id"] for r in buf.list_backlog()["records"]]
        assert buf.ack(ids) == 1
        assert buf.ack(ids) == 0          # already gone
        assert buf.ack([987654]) == 0     # never existed
        assert buf.ack([]) == 0

    def test_falls_back_to_the_annotated_frame(self, tmp_path, clock):
        buf = make_buffer(tmp_path, clock)
        go_offline(buf, clock)
        buf.observe(dets("cap"), 1, "m", None, ANNOTATED)
        rec = buf.list_backlog()["records"][0]
        assert rec["raw_frame"] is None
        assert rec["frame"] == base64.b64encode(b"ANNJPEG").decode()

    def test_each_record_keeps_the_task_it_was_captured_under(self, tmp_path, clock):
        # Records outlive an application switch: a classification record
        # buffered before a switch to detection still drains as one.
        buf = make_buffer(tmp_path, clock)
        go_offline(buf, clock)
        busy = [{"class_name": "busy", "confidence": 0.93, "area": None,
                 "color": "#3b82f6"}]
        buf.observe(busy, 1, "states.engine", RAW, ANNOTATED, task="classify")
        buf.observe(dets("cap"), 1, "boxes.engine", RAW, ANNOTATED, task="detect")
        records = buf.list_backlog()["records"]
        assert [r["task"] for r in records] == ["classify", "detect"]
        assert "bbox" not in records[0]["detections"][0]

    def test_a_row_from_older_firmware_has_no_task(self, tmp_path, clock):
        make_buffer(tmp_path, clock)  # creates the schema
        payload = '{"detections": [], "total": 1, "model": "m"}'
        conn = sqlite3.connect(tmp_path / "buffer.db")
        conn.execute(
            "INSERT INTO buffered_detections"
            " (captured_at, payload, frame, frame_is_raw, size_bytes)"
            " VALUES (1.0, ?, NULL, 1, ?)", (payload, len(payload)))
        conn.commit()
        conn.close()
        record = make_buffer(tmp_path, clock).list_backlog()["records"][0]
        assert "task" not in record

    def test_survives_a_reopen(self, tmp_path, clock):
        self._filled(tmp_path, clock, n=2)
        reopened = make_buffer(tmp_path, clock)
        assert reopened.pending_count() == 2
        assert len(reopened.list_backlog()["records"]) == 2


class TestRingCaps:
    def test_evicts_oldest_over_the_record_cap(self, tmp_path, clock):
        buf = make_buffer(tmp_path, clock, max_records=3)
        go_offline(buf, clock)
        for i in range(5):
            buf.observe(dets(f"class-{i}"), 1, "m", RAW, ANNOTATED)
        assert buf.pending_count() == 3
        kept = [r["detections"][0]["class_name"]
                for r in buf.list_backlog()["records"]]
        assert kept == ["class-2", "class-3", "class-4"]

    def test_evicts_oldest_over_the_byte_cap(self, tmp_path, clock):
        # Each record is ~7 bytes of frame + ~120 of payload; cap at ~2 records.
        buf = make_buffer(tmp_path, clock, max_bytes=300)
        go_offline(buf, clock)
        for i in range(4):
            buf.observe(dets(f"class-{i}"), 1, "m", RAW, ANNOTATED)
        page = buf.list_backlog()
        assert 1 <= len(page["records"]) < 4
        newest = page["records"][-1]["detections"][0]["class_name"]
        assert newest == "class-3"  # newest always survives


class TestResilience:
    def test_recreates_a_corrupted_database(self, tmp_path, clock):
        path = tmp_path / "buffer.db"
        path.write_bytes(b"this is not a sqlite database at all")
        buf = buffer_mod.DetectionBufferService(
            str(path), max_records=10, max_bytes=1000,
            offline_threshold_s=THRESHOLD, clock=clock)
        go_offline(buf, clock)
        buf.observe(dets("cap"), 1, "m", RAW, ANNOTATED)
        assert buf.pending_count() == 1

    def test_hub_seen_is_persisted(self, tmp_path, clock):
        make_buffer(tmp_path, clock).note_snapshot_pull()
        row = sqlite3.connect(str(tmp_path / "buffer.db")).execute(
            "SELECT value FROM buffer_meta WHERE key = 'hub_seen'").fetchone()
        assert row == ("1",)
