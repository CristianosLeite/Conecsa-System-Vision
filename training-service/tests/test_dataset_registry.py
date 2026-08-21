"""Unit tests for DatasetRegistry lifecycle (create/list/get/delete)."""
import pytest
from service.config import Config
from service.dataset_registry import DatasetRegistry
from service.dataset_service import DatasetError


@pytest.fixture
def registry(tmp_path):
    cfg = Config()
    cfg.DATA_DIR = str(tmp_path)
    return DatasetRegistry(cfg, event_service=None)


class TestCheckId:
    def test_rejects_invalid_id(self):
        for bad in ("", "UPPER", "has space"):
            with pytest.raises(DatasetError):
                DatasetRegistry._check_id(bad)


class TestCreateListGet:
    def test_create_then_list(self, registry):
        meta = registry.create("My Dataset")
        assert meta["name"] == "My Dataset"
        listed = registry.list()
        assert len(listed) == 1
        assert listed[0]["dataset_id"] == meta["dataset_id"]

    def test_get_returns_service(self, registry):
        meta = registry.create("D1")
        ds = registry.get(meta["dataset_id"])
        assert ds.dataset_id == meta["dataset_id"]

    def test_get_unknown_raises(self, registry):
        with pytest.raises(DatasetError):
            registry.get("11111111-1111-1111-1111-111111111111")

    def test_create_rejects_invalid_name(self, registry):
        with pytest.raises(DatasetError):
            registry.create("bad/name")


class TestRenameDelete:
    def test_rename(self, registry):
        meta = registry.create("Old")
        renamed = registry.rename(meta["dataset_id"], "New")
        assert renamed["name"] == "New"

    def test_delete(self, registry):
        meta = registry.create("Doomed")
        registry.delete(meta["dataset_id"])
        assert registry.list() == []

    def test_delete_unknown_raises(self, registry):
        with pytest.raises(DatasetError):
            registry.delete("11111111-1111-1111-1111-111111111111")


class TestReloadFromDisk:
    def test_datasets_rescanned_by_new_registry(self, tmp_path):
        cfg = Config()
        cfg.DATA_DIR = str(tmp_path)
        r1 = DatasetRegistry(cfg, event_service=None)
        r1.create("Persisted")
        # A fresh registry over the same data dir rescans the dataset.
        r2 = DatasetRegistry(cfg, event_service=None)
        assert len(r2.list()) == 1
        assert r2.list()[0]["name"] == "Persisted"


class TestFreezeDeleteRace:
    """One lock owns the frozen transition (REFACTORING.md M4): a delete can
    never interleave between a job validating a dataset and freezing it."""

    def test_a_frozen_dataset_cannot_be_deleted(self, registry):
        meta = registry.create("D1")
        registry.freeze(meta["dataset_id"])
        with pytest.raises(DatasetError, match="locked"):
            registry.delete(meta["dataset_id"])
        # Still listed and intact.
        assert registry.get(meta["dataset_id"]) is not None

    def test_release_reopens_deletion(self, registry):
        meta = registry.create("D1")
        ds = registry.freeze(meta["dataset_id"])
        registry.release(ds)
        registry.delete(meta["dataset_id"])
        with pytest.raises(DatasetError):
            registry.get(meta["dataset_id"])

    def test_a_deleted_dataset_cannot_be_frozen(self, registry):
        meta = registry.create("D1")
        registry.delete(meta["dataset_id"])
        with pytest.raises(DatasetError, match="not found"):
            registry.freeze(meta["dataset_id"])

    def test_double_freeze_is_refused(self, registry):
        meta = registry.create("D1")
        registry.freeze(meta["dataset_id"])
        with pytest.raises(DatasetError, match="locked"):
            registry.freeze(meta["dataset_id"])

    def test_concurrent_freeze_and_delete_never_both_succeed(self, registry):
        import threading

        for _ in range(20):
            meta = registry.create("D1")
            dataset_id = meta["dataset_id"]
            barrier = threading.Barrier(2)
            outcomes = {}

            def freeze(dataset_id=dataset_id, barrier=barrier, outcomes=outcomes):
                barrier.wait()
                try:
                    registry.freeze(dataset_id)
                    outcomes["freeze"] = True
                except DatasetError:
                    outcomes["freeze"] = False

            def delete(dataset_id=dataset_id, barrier=barrier, outcomes=outcomes):
                barrier.wait()
                try:
                    registry.delete(dataset_id)
                    outcomes["delete"] = True
                except DatasetError:
                    outcomes["delete"] = False

            threads = [threading.Thread(target=freeze),
                       threading.Thread(target=delete)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(10)

            assert outcomes["freeze"] != outcomes["delete"], \
                "exactly one of freeze/delete may win"
            if outcomes["freeze"]:
                # The dataset survived; its directory must still exist for
                # the training job that claimed it.
                ds = registry.get(dataset_id)
                import os
                assert os.path.isdir(ds.root)
                registry.release(ds)
                registry.delete(dataset_id)

    def test_a_failed_rmtree_is_reported(self, registry, monkeypatch):
        import shutil as _shutil
        meta = registry.create("D1")

        def boom(path):
            raise OSError("file is busy")

        monkeypatch.setattr(_shutil, "rmtree", boom)
        with pytest.raises(DatasetError, match="could not be deleted"):
            registry.delete(meta["dataset_id"])
