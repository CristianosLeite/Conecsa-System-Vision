# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""ApplicationService: boot migration, validation, persistence."""
import json

import pytest
from api import postprocess
from api.services import application_service as app_mod
from api.services.application_service import ApplicationService, has_model_artifacts
from api.services.errors import InvalidTask, PreconditionFailed


def _file(tmp_path):
    return json.loads((tmp_path / "application.json").read_text())


def _service(tmp_path, supported=("detect",)):
    svc = ApplicationService(str(tmp_path), list(supported))
    svc.load_or_migrate()
    return svc


class TestBootMigration:
    def test_blank_device_records_no_application_once(self, tmp_path):
        svc = _service(tmp_path)
        assert svc.task is None and svc.migrated is False
        assert _file(tmp_path) == {"task": None}
        # A model copied in later never decides the application on reboot.
        (tmp_path / "late.engine").write_bytes(b"x")
        assert _service(tmp_path).task is None

    @pytest.mark.parametrize("artifact", [
        "weights.engine", "model.plan", "model.onnx", "best.pt",
    ])
    def test_a_device_with_models_is_an_object_detection_installation(self, tmp_path,
                                                                        artifact):
        (tmp_path / artifact).write_bytes(b"x")
        svc = _service(tmp_path)
        assert svc.task == "detect" and svc.migrated is True
        assert _file(tmp_path) == {"task": "detect", "migrated": True}

    def test_a_dangling_selection_marker_still_counts(self, tmp_path):
        # Damaged install: .current_model names an engine that is gone.
        (tmp_path / ".current_model").write_text("gone.engine")
        assert _service(tmp_path).task == "detect"

    def test_a_weights_directory_still_counts(self, tmp_path):
        (tmp_path / "weights").mkdir()
        assert _service(tmp_path).task == "detect"

    def test_settings_sidecars_alone_do_not(self, tmp_path):
        (tmp_path / "weights.settings.json").write_text("{}")
        assert has_model_artifacts(str(tmp_path)) is False

    def test_an_existing_file_is_read_not_migrated(self, tmp_path):
        (tmp_path / "application.json").write_text(json.dumps({"task": "classify"}))
        (tmp_path / "weights.engine").write_bytes(b"x")
        svc = _service(tmp_path, ("detect", "classify"))
        assert svc.task == "classify" and svc.migrated is False

    def test_an_unknown_task_from_a_newer_build_is_kept_verbatim(self, tmp_path):
        (tmp_path / "application.json").write_text(json.dumps({"task": "pose"}))
        assert _service(tmp_path).task == "pose"

    def test_a_corrupt_file_reads_as_unset_and_is_kept(self, tmp_path):
        (tmp_path / "application.json").write_text("{not json")
        svc = _service(tmp_path)
        assert svc.task is None
        assert (tmp_path / "application.json").read_text() == "{not json"

    def test_an_unwritable_directory_keeps_the_answer_in_memory(self, tmp_path, monkeypatch):
        (tmp_path / "weights.engine").write_bytes(b"x")

        def refuse(*args, **kwargs):
            raise OSError("read-only file system")

        monkeypatch.setattr(app_mod, "atomic_write_json", refuse)
        assert _service(tmp_path).task == "detect"


class TestValidation:
    def test_unknown_ids_are_invalid(self, tmp_path):
        svc = _service(tmp_path)
        for bad in ("pose", "", None, 3):
            with pytest.raises(InvalidTask):
                svc.validate(bad)

    def test_known_but_unsupported_tasks_are_a_precondition(self, tmp_path):
        svc = _service(tmp_path)
        for task in ("classify", "segment"):
            with pytest.raises(PreconditionFailed, match="not available in this release"):
                svc.validate(task)
        assert svc.validate("detect") == "detect"

    def test_the_supported_list_is_the_registered_strategies(self, tmp_path):
        svc = ApplicationService(str(tmp_path), postprocess.supported_tasks())
        assert svc.supported_tasks == ["detect", "classify", "segment"]
        assert svc.info() == {"task": None, "supported_tasks": ["detect", "classify", "segment"],
                              "migrated": False}


class TestPersist:
    def test_writes_then_updates_memory(self, tmp_path):
        (tmp_path / "weights.engine").write_bytes(b"x")
        svc = _service(tmp_path, ("detect", "classify"))
        svc.persist("classify")
        assert svc.task == "classify" and svc.migrated is False
        assert _file(tmp_path) == {"task": "classify"}

    def test_a_failed_write_leaves_memory_untouched(self, tmp_path, monkeypatch):
        svc = _service(tmp_path)

        def refuse(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(app_mod, "atomic_write_json", refuse)
        with pytest.raises(OSError):
            svc.persist("detect")
        assert svc.task is None


class TestUploadTask:
    def test_empty_declaration_means_the_device_task(self, tmp_path):
        (tmp_path / "weights.engine").write_bytes(b"x")
        assert _service(tmp_path).resolve_upload_task("") == "detect"

    def test_empty_declaration_on_a_blank_device_is_refused(self, tmp_path):
        with pytest.raises(InvalidTask, match="Declare the model's task"):
            _service(tmp_path).resolve_upload_task("")

    def test_a_declaration_must_be_supported(self, tmp_path):
        svc = _service(tmp_path)
        assert svc.resolve_upload_task("detect") == "detect"
        with pytest.raises(PreconditionFailed):
            svc.resolve_upload_task("classify")
        with pytest.raises(InvalidTask):
            svc.resolve_upload_task("pose")
