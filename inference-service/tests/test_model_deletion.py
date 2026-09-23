# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for whole-artifact model deletion.

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


class TestWeightsSidecar:
    def test_weights_file_lives_under_the_weights_subdir(self):
        assert (
            ModelService.weights_file_for_model("/data/models/Teste.engine")
            == "/data/models/weights/Teste.pt"
        )

    def test_list_models_reports_the_sidecar_and_hides_it(self, service):
        svc, tmp_path = service
        (tmp_path / "a.engine").write_text("x")
        (tmp_path / "b.engine").write_text("x")
        (tmp_path / "weights").mkdir()
        (tmp_path / "weights" / "a.pt").write_text("pt")
        listed = {m.name: m.has_weights for m in svc.list_models()}
        assert listed == {"a.engine": True, "b.engine": False}

    def test_weights_file_path_resolves_only_existing_sidecars(self, service):
        svc, tmp_path = service
        (tmp_path / "weights").mkdir()
        (tmp_path / "weights" / "a.pt").write_text("pt")
        assert svc.weights_file_path("a.engine") == str(tmp_path / "weights" / "a.pt")
        assert svc.weights_file_path("b.engine") == ""
        assert svc.weights_file_path("../a.engine") == ""

    def test_delete_removes_the_weights_sidecar(self, service):
        svc, tmp_path = service
        _plant_model(tmp_path, "old")
        (tmp_path / "weights").mkdir()
        (tmp_path / "weights" / "old.pt").write_text("pt")
        ok, error = svc.delete_model("old.engine")
        assert ok, error
        assert not (tmp_path / "weights" / "old.pt").exists()

    def test_overwriting_with_an_engine_drops_the_stale_sidecar(self, service):
        svc, tmp_path = service
        _plant_model(tmp_path, "old")
        (tmp_path / "weights").mkdir()
        (tmp_path / "weights" / "old.pt").write_text("pt")

        class Upload:
            def save(self, path):
                open(path, "w").write("new engine")

        ok, _, error = svc.save_model("old.engine", Upload())
        assert ok, error
        assert not (tmp_path / "weights" / "old.pt").exists()
        assert not [m for m in svc.list_models() if m.has_weights]

    def test_a_pt_upload_keeps_the_sidecar_for_its_conversion(self, service):
        svc, tmp_path = service
        _plant_model(tmp_path, "old")
        (tmp_path / "weights").mkdir()
        (tmp_path / "weights" / "old.pt").write_text("pt")

        class Upload:
            def save(self, path):
                open(path, "w").write("checkpoint")

        ok, _, error = svc.save_model("old.pt", Upload())
        assert ok, error
        assert (tmp_path / "weights" / "old.pt").exists()
