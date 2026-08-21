"""Tests for the centralized model filename validation (api/model_paths.py).

The traversal matrix here is the regression net for REFACTORING.md C2:
before centralization only the download path checked for traversal, and
saving '../escaped.engine' wrote outside the model directory.
"""
import os

import pytest
from api.config import Config
from api.model_paths import ALLOWED_MODEL_EXTENSIONS, validate_model_filename
from api.services.model_service import ModelService


@pytest.fixture
def model_root(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    return str(root)


class TestValidateModelFilename:
    def test_a_plain_name_resolves_under_the_root(self, model_root):
        path, error = validate_model_filename("weights.engine", model_root)
        assert error == ""
        assert path == os.path.join(model_root, "weights.engine")

    @pytest.mark.parametrize("ext", ALLOWED_MODEL_EXTENSIONS)
    def test_every_allowed_extension_is_accepted(self, model_root, ext):
        path, error = validate_model_filename(f"m{ext}", model_root)
        assert error == ""

    def test_extensions_are_case_normalized(self, model_root):
        path, error = validate_model_filename("m.ENGINE", model_root)
        assert error == ""

    @pytest.mark.parametrize("name", [
        "../escaped.engine",
        "../../escaped.engine",
        "/etc/escaped.engine",
        "sub/dir.engine",
        "..\\escaped.engine",
        "a\\b.engine",
    ])
    def test_traversal_and_separators_are_rejected(self, model_root, name):
        path, error = validate_model_filename(name, model_root)
        assert path == "" and error

    @pytest.mark.parametrize("name", [
        "",
        " padded.engine",
        "padded.engine ",
        "a\r\nb.engine",
        'a"b.engine',
        "a\x00b.engine",
    ])
    def test_empty_whitespace_and_control_characters_are_rejected(
            self, model_root, name):
        path, error = validate_model_filename(name, model_root)
        assert path == "" and error

    @pytest.mark.parametrize("name", [
        ".current_model",
        ".hidden.engine",
        "..",
    ])
    def test_hidden_and_state_names_are_reserved(self, model_root, name):
        path, error = validate_model_filename(name, model_root)
        assert path == "" and error

    @pytest.mark.parametrize("name", [
        "labels.txt",
        "areas.areas.json",
        "settings.settings.json",
        "model.exe",
        "model",
    ])
    def test_disallowed_extensions_are_rejected(self, model_root, name):
        path, error = validate_model_filename(name, model_root)
        assert path == "" and error

    def test_encoded_traversal_arrives_literal_and_is_rejected(self, model_root):
        # The gateway does not decode %2e%2e; if it ever did, the separator
        # check still catches the decoded form.
        path, error = validate_model_filename("%2e%2e%2fescaped.engine", model_root)
        # Literal percent characters make a weird but *contained* name; the
        # decoded form must be rejected.
        decoded_path, decoded_error = validate_model_filename(
            "../escaped.engine", model_root)
        assert decoded_path == "" and decoded_error
        if not error:
            assert path.startswith(model_root + os.sep)

    def test_a_symlinked_root_still_confines(self, tmp_path):
        real = tmp_path / "real-models"
        real.mkdir()
        link = tmp_path / "link-models"
        link.symlink_to(real)
        path, error = validate_model_filename("m.engine", str(link))
        assert error == ""
        assert path == str(real / "m.engine")


class FakeUpload:
    def __init__(self):
        self.saved_to = None

    def save(self, path):
        self.saved_to = path
        with open(path, "wb") as fh:
            fh.write(b"weights")


class TestModelServiceUsesTheValidator:
    def make_service(self, model_root):
        return ModelService(Config(), model_root)

    def test_save_model_rejects_traversal(self, model_root, tmp_path):
        service = self.make_service(model_root)
        upload = FakeUpload()
        ok, path, error = service.save_model("../escaped.engine", upload)
        assert not ok and error
        assert upload.saved_to is None
        assert not (tmp_path / "escaped.engine").exists()

    def test_save_model_accepts_a_plain_name(self, model_root):
        service = self.make_service(model_root)
        ok, path, error = service.save_model("m.engine", FakeUpload())
        assert ok, error
        assert os.path.isfile(os.path.join(model_root, "m.engine"))

    def test_select_model_rejects_traversal(self, model_root, tmp_path):
        # Even with a real file sitting outside the root, selection by a
        # traversal name must fail on the name, not on existence.
        outside = tmp_path / "outside.engine"
        outside.write_bytes(b"x")
        service = self.make_service(model_root)
        ok, msg = service.select_model("../outside.engine")
        assert not ok
        assert "not allowed" in msg or "Invalid" in msg

    def test_delete_model_rejects_traversal(self, model_root, tmp_path):
        outside = tmp_path / "outside.engine"
        outside.write_bytes(b"x")
        service = self.make_service(model_root)
        ok, msg = service.delete_model("../outside.engine")
        assert not ok
        assert outside.exists(), "the file outside the root must survive"
