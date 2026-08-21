"""Tests for whole-artifact model deletion (REFACTORING.md L3).

Sidecars are found by basename, so a leftover .txt/.areas.json/.settings.json
from a deleted model used to be silently inherited by the next model uploaded
under the same name — wrong class labels on live detections.
"""

import pytest
from api.config import Config
from api.services.model_service import ModelService


@pytest.fixture
def service(tmp_path):
    return ModelService(Config(), str(tmp_path)), tmp_path


def _plant_model(tmp_path, stem):
    files = [f"{stem}.engine", f"{stem}.txt", f"{stem}.areas.json",
             f"{stem}.settings.json"]
    for name in files:
        (tmp_path / name).write_text("x")
    return files


class TestArtifactDeletion:
    def test_delete_removes_the_binary_and_every_sidecar(self, service):
        svc, tmp_path = service
        files = _plant_model(tmp_path, "old")
        ok, error = svc.delete_model("old.engine")
        assert ok, error
        for name in files:
            assert not (tmp_path / name).exists(), f"{name} survived"

    def test_a_model_without_sidecars_still_deletes(self, service):
        svc, tmp_path = service
        (tmp_path / "bare.engine").write_text("x")
        ok, error = svc.delete_model("bare.engine")
        assert ok, error

    def test_a_reupload_does_not_inherit_stale_labels(self, service):
        svc, tmp_path = service
        _plant_model(tmp_path, "m")
        (tmp_path / "m.txt").write_text("cat\ndog\n")
        assert svc.delete_model("m.engine")[0]
        # The next model with the same basename starts clean.
        assert not (tmp_path / "m.txt").exists()

    def test_the_active_model_is_still_protected(self, service):
        svc, tmp_path = service
        _plant_model(tmp_path, "active")
        svc.current_model = "active.engine"
        ok, error = svc.delete_model("active.engine")
        assert not ok and "active" in error
        assert (tmp_path / "active.engine").exists()
        assert (tmp_path / "active.txt").exists()
