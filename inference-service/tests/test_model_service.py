# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for ModelService path helpers and validation."""
from api.config import Config
from api.services.model_service import ModelService


class TestSiblingPathHelpers:
    def test_classes_file_for_model(self):
        assert (
            ModelService.classes_file_for_model("/data/models/weights.engine")
            == "/data/models/weights.txt"
        )

    def test_areas_file_for_model(self):
        assert (
            ModelService.areas_file_for_model("/data/models/weights.engine")
            == "/data/models/weights.areas.json"
        )

    def test_settings_file_for_model(self):
        assert (
            ModelService.settings_file_for_model("/data/models/weights.engine")
            == "/data/models/weights.settings.json"
        )

    def test_handles_other_extensions(self):
        assert ModelService.classes_file_for_model("/m/model.onnx") == "/m/model.txt"


class TestModelFilePath:
    def _svc(self, tmp_path):
        return ModelService(Config(), str(tmp_path))

    def test_valid_existing_model(self, tmp_path):
        svc = self._svc(tmp_path)
        (tmp_path / "weights.engine").write_bytes(b"x")
        assert svc.model_file_path("weights.engine") == str(tmp_path / "weights.engine")

    def test_missing_file_returns_empty(self, tmp_path):
        svc = self._svc(tmp_path)
        assert svc.model_file_path("absent.engine") == ""

    def test_path_traversal_rejected(self, tmp_path):
        svc = self._svc(tmp_path)
        assert svc.model_file_path("../weights.engine") == ""
        assert svc.model_file_path("/etc/passwd") == ""

    def test_disallowed_extension_rejected(self, tmp_path):
        svc = self._svc(tmp_path)
        (tmp_path / "notes.txt").write_text("x")
        assert svc.model_file_path("notes.txt") == ""

    def test_control_characters_rejected(self, tmp_path):
        svc = self._svc(tmp_path)
        assert svc.model_file_path("weights\n.engine") == ""


class TestListModels:
    def test_extensions_match_case_insensitively(self, tmp_path):
        # validate_model_filename accepts "yard.ENGINE"; the list must show it.
        (tmp_path / "yard.ENGINE").write_bytes(b"x")
        (tmp_path / "notes.TXT").write_bytes(b"x")
        names = [m.name for m in ModelService(Config(), str(tmp_path)).list_models()]
        assert names == ["yard.ENGINE"]


class TestDeleteModel:
    def test_no_model_is_current_until_one_is_activated(self, tmp_path):
        # Regression: the default was "weights.engine", so a blank device's
        # status named a model it never loaded.
        assert ModelService(Config(), str(tmp_path)).current_model == ""

    def test_cannot_delete_active_model(self, tmp_path):
        svc = ModelService(Config(), str(tmp_path))
        svc.current_model = "weights.engine"
        ok, msg = svc.delete_model("weights.engine")
        assert ok is False
        assert "active" in msg

    def test_delete_missing_model(self, tmp_path):
        svc = ModelService(Config(), str(tmp_path))
        ok, msg = svc.delete_model("other.engine")
        assert ok is False
        assert "not found" in msg

    def test_delete_existing_model(self, tmp_path):
        svc = ModelService(Config(), str(tmp_path))
        (tmp_path / "other.engine").write_bytes(b"x")
        ok, msg = svc.delete_model("other.engine")
        assert ok is True
        assert not (tmp_path / "other.engine").exists()


class TestSaveModel:
    class _FakeUpload:
        def save(self, path):
            with open(path, "wb") as f:
                f.write(b"model-bytes")

    def test_rejects_bad_extension(self, tmp_path):
        svc = ModelService(Config(), str(tmp_path))
        ok, path, err = svc.save_model("bad.txt", self._FakeUpload())
        assert ok is False
        assert path == ""
        assert "Invalid file type" in err

    def test_saves_valid_model(self, tmp_path):
        svc = ModelService(Config(), str(tmp_path))
        ok, path, err = svc.save_model("m.onnx", self._FakeUpload())
        assert ok is True
        assert err == ""
        assert (tmp_path / "m.onnx").read_bytes() == b"model-bytes"


class TestUpdateSetting:
    """Per-model settings change under the lifecycle lock and roll back when unsaved."""

    class _Settings:
        def __init__(self, saved):
            self.saved, self.calls = saved, 0

        def save(self, only=None):
            self.calls += 1
            return self.saved

    def _svc(self, tmp_path, saved=True):
        config = Config()
        config.SEGMENT_MAX_INSTANCES = 32
        svc = ModelService(config, str(tmp_path))
        settings = self._Settings(saved)
        svc.attach_settings_service(settings)
        return svc, config, settings

    def test_a_valid_value_is_applied_and_saved(self, tmp_path):
        svc, config, settings = self._svc(tmp_path)
        outcome = svc.update_setting(
            "SEGMENT_MAX_INSTANCES", lambda: config.set_segment_max_instances(12))
        assert outcome == ModelService.SETTING_SAVED
        assert config.SEGMENT_MAX_INSTANCES == 12 and settings.calls == 1

    def test_a_refused_value_is_not_saved(self, tmp_path):
        svc, config, settings = self._svc(tmp_path)
        outcome = svc.update_setting(
            "SEGMENT_MAX_INSTANCES", lambda: config.set_segment_max_instances(0))
        assert outcome == ModelService.SETTING_INVALID
        assert config.SEGMENT_MAX_INSTANCES == 32 and settings.calls == 0

    def test_an_unsaved_value_is_rolled_back(self, tmp_path):
        svc, config, _ = self._svc(tmp_path, saved=False)
        outcome = svc.update_setting(
            "SEGMENT_MAX_INSTANCES", lambda: config.set_segment_max_instances(12))
        assert outcome == ModelService.SETTING_UNSAVED
        assert config.SEGMENT_MAX_INSTANCES == 32

    def _face(self, tmp_path, saved=True):
        svc, config, settings = self._svc(tmp_path, saved)
        config.FACE_MIN_SIZE_PX, config.FACE_MAX_FACES = 40, 5
        return svc, config, settings

    def test_several_fields_are_applied_and_saved_in_one_write(self, tmp_path):
        svc, config, settings = self._face(tmp_path)
        outcome = svc.update_settings([
            ("FACE_MIN_SIZE_PX", lambda: config.set_face_min_size(80)),
            ("FACE_MAX_FACES", lambda: config.set_face_max_faces(3)),
        ])
        assert outcome == (ModelService.SETTING_SAVED, None)
        assert (config.FACE_MIN_SIZE_PX, config.FACE_MAX_FACES) == (80, 3)
        assert settings.calls == 1

    def test_one_refused_field_leaves_every_field_as_it_was(self, tmp_path):
        svc, config, settings = self._face(tmp_path)
        outcome = svc.update_settings([
            ("FACE_MIN_SIZE_PX", lambda: config.set_face_min_size(80)),
            ("FACE_MAX_FACES", lambda: config.set_face_max_faces(0)),
        ])
        assert outcome == (ModelService.SETTING_INVALID, "FACE_MAX_FACES")
        assert (config.FACE_MIN_SIZE_PX, config.FACE_MAX_FACES) == (40, 5)
        assert settings.calls == 0

    def test_an_unsaved_write_restores_every_field(self, tmp_path):
        svc, config, _ = self._face(tmp_path, saved=False)
        outcome = svc.update_settings([
            ("FACE_MIN_SIZE_PX", lambda: config.set_face_min_size(80)),
            ("FACE_MAX_FACES", lambda: config.set_face_max_faces(3)),
        ])
        assert outcome == (ModelService.SETTING_UNSAVED, None)
        assert (config.FACE_MIN_SIZE_PX, config.FACE_MAX_FACES) == (40, 5)

    def test_the_change_holds_the_lifecycle_lock(self, tmp_path):
        svc, config, _ = self._svc(tmp_path)
        held = []

        def set_value():
            # _op_lock is reentrant: another thread would block, this one sees it owned.
            held.append(svc._op_lock._is_owned())  # type: ignore[attr-defined]
            return config.set_segment_max_instances(12)

        svc.update_setting("SEGMENT_MAX_INSTANCES", set_value)
        assert held == [True]
