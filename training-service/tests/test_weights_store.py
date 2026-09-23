# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for the federated weights stash (WeightsStore)."""
import os
import time

import pytest
from service.config import Config
from service.dataset_service import DatasetError
from service.weights_store import WeightsStore


@pytest.fixture
def cfg(tmp_path):
    cfg = Config()
    cfg.DATA_DIR = str(tmp_path)
    cfg.BASE_WEIGHTS = str(tmp_path / "missing-base.pt")  # no symlink in tests
    return cfg


@pytest.fixture
def store(cfg):
    return WeightsStore(cfg)


class TestSaveStream:
    def test_roundtrip(self, store):
        weights_id, size = store.save_stream([b"abc", b"def"])
        assert size == 6
        with open(store.path(weights_id), "rb") as f:
            assert f.read() == b"abcdef"

    def test_ids_are_hex_only(self, store):
        weights_id, _ = store.save_stream([b"x"])
        assert all(c in "0123456789abcdef" for c in weights_id)

    def test_empty_upload_rejected(self, store, cfg):
        with pytest.raises(DatasetError):
            store.save_stream([])
        # A failed save leaves no temp file behind.
        assert [e for e in os.listdir(cfg.weights_dir)] == []

    def test_budget_enforced(self, store, cfg):
        cfg.MAX_WEIGHTS_UPLOAD_MB = 1
        with pytest.raises(DatasetError):
            store.save_stream([b"x" * (1 << 20), b"y"])
        assert [e for e in os.listdir(cfg.weights_dir)] == []


class TestPathValidation:
    def test_rejects_traversal(self, store):
        for bad in ("", "../etc/passwd", "abc/def", "UPPER", "a" * 32 + "/.."):
            with pytest.raises(DatasetError):
                store.path(bad)

    def test_unknown_id_raises(self, store):
        with pytest.raises(DatasetError):
            store.path("0" * 32)


class TestStashAndDelete:
    def test_stash_file_copies(self, store, tmp_path):
        src = tmp_path / "last.pt"
        src.write_bytes(b"checkpoint")
        weights_id = store.stash_file(str(src))
        with open(store.path(weights_id), "rb") as f:
            assert f.read() == b"checkpoint"
        assert src.exists()  # copy, not move

    def test_stash_missing_file_raises(self, store, tmp_path):
        with pytest.raises(DatasetError):
            store.stash_file(str(tmp_path / "nope.pt"))

    def test_delete_is_idempotent(self, store):
        weights_id, _ = store.save_stream([b"x"])
        store.delete(weights_id)
        store.delete(weights_id)  # missing id: no-op
        with pytest.raises(DatasetError):
            store.path(weights_id)


class TestPrune:
    def test_prunes_only_stale_store_entries(self, store, cfg):
        stale_id, _ = store.save_stream([b"old"])
        fresh_id, _ = store.save_stream([b"new"])
        foreign = os.path.join(cfg.weights_dir, "yolo26s.pt")
        with open(foreign, "wb") as f:
            f.write(b"base")
        old = time.time() - cfg.WEIGHTS_TTL_SEC - 10
        os.utime(store.path(stale_id), (old, old))
        os.utime(foreign, (old, old))

        store.prune()

        with pytest.raises(DatasetError):
            store.path(stale_id)
        store.path(fresh_id)  # untouched
        assert os.path.exists(foreign)  # non-store files are never pruned

    def test_a_stale_task_sidecar_goes_with_its_blob(self, store, cfg):
        stale_id, _ = store.save_stream([b"old"], task="classify")
        sidecar = os.path.join(cfg.weights_dir, f"{stale_id}.task")
        old = time.time() - cfg.WEIGHTS_TTL_SEC - 10
        os.utime(store.path(stale_id), (old, old))
        os.utime(sidecar, (old, old))
        store.prune()
        assert not os.path.exists(sidecar)


class TestTask:
    """A checkpoint remembers the task it was trained for."""

    def test_an_upload_records_its_task(self, store):
        weights_id, _ = store.save_stream([b"x"], task="classify")
        assert store.task_of(weights_id) == "classify"

    def test_an_unknown_task_reads_as_empty(self, store):
        weights_id, _ = store.save_stream([b"x"])
        assert store.task_of(weights_id) == ""
        assert store.task_of("0" * 32) == ""

    def test_a_stash_records_its_task(self, store, tmp_path):
        src = tmp_path / "last.pt"
        src.write_bytes(b"checkpoint")
        assert store.task_of(store.stash_file(str(src), task="detect")) == "detect"

    def test_delete_removes_the_sidecar(self, store, cfg):
        weights_id, _ = store.save_stream([b"x"], task="classify")
        store.delete(weights_id)
        assert os.listdir(cfg.weights_dir) == []

    def test_a_failed_upload_leaves_no_sidecar(self, store, cfg):
        with pytest.raises(DatasetError):
            store.save_stream([], task="classify")
        assert os.listdir(cfg.weights_dir) == []

    def test_both_base_weights_are_linked_into_the_store(self, cfg, tmp_path):
        for name in ("yolo26s.pt", "yolo26s-cls.pt"):
            (tmp_path / name).write_bytes(b"base")
        cfg.BASE_WEIGHTS = str(tmp_path / "yolo26s.pt")
        cfg.BASE_WEIGHTS_CLS = str(tmp_path / "yolo26s-cls.pt")
        WeightsStore(cfg)
        linked = sorted(e for e in os.listdir(cfg.weights_dir) if e.startswith("yolo26s"))
        assert linked == ["yolo26s-cls.pt", "yolo26s.pt"]
