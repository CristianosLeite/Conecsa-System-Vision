# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""A face model's lifecycle in ModelService: the gallery sidecar is one of the
model's files, a ``.faces`` package builds a gallery instead of converting, a
face model is never uploaded directly, a package is not selectable, and leaving
the application frees the embedder's worker. Plus the per-model face settings
in the sidecar."""
import json
import os
from types import SimpleNamespace

import pytest
from api.config import Config, face_settings_from_env
from api.services import model_service as ms
from api.services.model_service import ModelService
from api.services.model_settings_service import ModelSettingsService


class FakeConversions:
    def __init__(self):
        self.face_calls = []
        self.jobs = []

    def start_face_gallery(self, faces_path, original_filename, model_directory):
        self.face_calls.append((faces_path, original_filename, model_directory))
        return SimpleNamespace(job_id="job1")

    def get_active_jobs(self):
        return self.jobs


class FileData:
    def __init__(self, payload=b"data"):
        self.payload = payload

    def save(self, path):
        with open(path, "wb") as f:
            f.write(self.payload)


@pytest.fixture
def service(tmp_path):
    config = Config()
    config.MODEL_PATH = str(tmp_path / "weights.engine")
    service = ModelService(config, str(tmp_path))
    service.attach_conversion_service(FakeConversions())
    return service


def test_the_gallery_is_one_of_a_models_files(tmp_path):
    path = str(tmp_path / "staff.engine")
    assert ModelService.gallery_file_for_model(path) == str(tmp_path / "staff.gallery.npz")
    assert ModelService.gallery_file_for_model(path) in ModelService.model_artifacts(path)


def test_deleting_a_face_model_removes_its_gallery(service, tmp_path):
    for name in ("staff.engine", "staff.txt", "staff.gallery.npz", "staff.settings.json"):
        (tmp_path / name).write_bytes(b"x")
    assert service.delete_model("staff.engine") == (True, "")
    assert not (tmp_path / "staff.gallery.npz").exists()


def test_a_package_starts_a_gallery_build(service, tmp_path):
    body, status = service.process_upload("staff.faces", FileData(), task="face")
    assert status == 202 and body["job_id"] == "job1"
    assert "face gallery" in body["message"]
    assert service._conversion_service.face_calls == [
        (str(tmp_path / "staff.faces"), "staff.faces", str(tmp_path))]


def test_a_face_model_is_never_uploaded_directly(service, tmp_path):
    body, status = service.process_upload("staff.engine", FileData(), task="face")
    assert status == 400 and "built from an enrollment dataset" in body["error"]
    assert not (tmp_path / "staff.engine").exists()


def test_a_package_is_refused_on_a_device_of_another_task(service, tmp_path):
    body, status = service.process_upload("staff.faces", FileData(), task="detect")
    assert status == 400 and "enrollment package" in body["error"]
    assert not (tmp_path / "staff.faces").exists()


def test_a_package_is_not_selectable(service, tmp_path):
    (tmp_path / "staff.faces").write_bytes(b"package")
    ok, message = service.select_model("staff.faces")
    assert not ok and "enrollment package" in message


def test_a_package_is_listed_with_its_job_task_while_it_builds(service, tmp_path):
    (tmp_path / "staff.faces").write_bytes(b"package")
    service._conversion_service.jobs = [SimpleNamespace(
        task="face", pt_path="", onnx_path="", faces_path=str(tmp_path / "staff.faces"))]
    assert [(m.name, m.task) for m in service.list_models()] == [("staff.faces", "face")]


def test_leaving_face_frees_the_embedder_worker(service, monkeypatch):
    closed = []
    monkeypatch.setattr("api.postprocess._face_embedder.close_worker", closed.append)
    app = SimpleNamespace(task="face", validate=lambda t: None,
                          persist=lambda t: setattr(app, "task", t))
    service.attach_application_service(app)
    service.attach_detection_service(SimpleNamespace(
        stop=lambda: False, unload_runtime=lambda: None, reset_results=lambda: None))

    assert service.set_task("detect") is True
    from api.postprocess import _face_assets
    assert closed == [_face_assets.embed_worker_port()]


def test_switching_between_other_tasks_leaves_the_worker_alone(service, monkeypatch):
    closed = []
    monkeypatch.setattr("api.postprocess._face_embedder.close_worker", closed.append)
    app = SimpleNamespace(task="detect", validate=lambda t: None,
                          persist=lambda t: setattr(app, "task", t))
    service.attach_application_service(app)
    service.attach_detection_service(SimpleNamespace(
        stop=lambda: False, unload_runtime=lambda: None, reset_results=lambda: None))

    assert service.set_task("classify") is True
    assert closed == []


# ── per-model face settings ──

def _settings(tmp_path, payload=None):
    path = tmp_path / "staff.settings.json"
    if payload is not None:
        path.write_text(json.dumps(payload))
    config = Config()
    service = ModelSettingsService(config)
    service.switch_model(str(path))
    return config, service, path


def test_a_model_without_face_settings_runs_on_the_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("FACE_MAX_FACES", "7")
    config, _, _ = _settings(tmp_path, {"thresholds": {"confidence": 0.5, "overlay": 0.4}})
    assert (config.FACE_MATCH_THRESHOLD, config.FACE_MIN_SIZE_PX,
            config.FACE_MAX_FACES) == face_settings_from_env()
    assert config.FACE_MAX_FACES == 7


def test_a_models_own_face_settings_are_applied(tmp_path):
    config, _, _ = _settings(tmp_path, {
        "thresholds": {"confidence": 0.5, "overlay": 0.4}, "task": "face",
        "face": {"match_threshold": 0.5, "min_size_px": 80, "max_faces": 2}})
    assert (config.FACE_MATCH_THRESHOLD, config.FACE_MIN_SIZE_PX,
            config.FACE_MAX_FACES) == (0.5, 80, 2)


@pytest.mark.parametrize("face", [
    {"match_threshold": 2.0}, {"min_size_px": -5}, {"max_faces": 0},
    {"max_faces": True}, {"max_faces": "3"},
])
def test_invalid_stored_values_fall_back_to_the_defaults(tmp_path, face):
    config, _, _ = _settings(tmp_path, {"thresholds": {}, "face": face})
    assert (config.FACE_MATCH_THRESHOLD, config.FACE_MIN_SIZE_PX,
            config.FACE_MAX_FACES) == face_settings_from_env()


def test_saving_one_face_setting_leaves_the_others_alone(tmp_path):
    config, service, path = _settings(tmp_path, {
        "thresholds": {"confidence": 0.5, "overlay": 0.4}, "task": "face",
        "face": {"max_faces": 2}})
    config.FACE_MAX_FACES = 9
    config.CONFIDENCE_THRESHOLD = 0.99
    assert service.save(only="FACE_MAX_FACES") is True
    stored = json.loads(path.read_text())
    assert stored["face"] == {"max_faces": 9}
    assert stored["thresholds"]["confidence"] == 0.5     # not persisted by this save
    assert stored["task"] == "face"


def test_a_full_save_keeps_the_task_and_the_settings_the_operator_set(tmp_path):
    config, service, path = _settings(tmp_path, {
        "thresholds": {"confidence": 0.5, "overlay": 0.4}, "task": "face",
        "face": {"match_threshold": 0.5}})
    config.FACE_MATCH_THRESHOLD = 0.6
    assert service.save() is True
    stored = json.loads(path.read_text())
    assert stored["face"] == {"match_threshold": pytest.approx(0.6)}
    assert stored["task"] == "face"


def test_settings_of_a_previous_model_never_leak_into_the_next(tmp_path):
    config, service, _ = _settings(tmp_path, {
        "thresholds": {}, "face": {"max_faces": 2}})
    assert config.FACE_MAX_FACES == 2
    other = tmp_path / "other.settings.json"
    other.write_text(json.dumps({"thresholds": {"confidence": 0.5, "overlay": 0.4}}))
    service.switch_model(str(other))
    assert config.FACE_MAX_FACES == face_settings_from_env()[2]


def test_a_brand_new_model_starts_on_the_defaults(tmp_path):
    config = Config()
    config.FACE_MAX_FACES = 11
    service = ModelSettingsService(config)
    service.switch_model(str(tmp_path / "fresh.settings.json"))
    assert config.FACE_MAX_FACES == face_settings_from_env()[2]
    assert "face" not in json.loads((tmp_path / "fresh.settings.json").read_text())


def test_the_faces_extension_is_a_known_model_file():
    assert ".faces" in ms.ALLOWED_MODEL_EXTENSIONS
    assert os.path.splitext("staff.faces")[1] in ms._FACES_EXTENSIONS


# ── publishing over the active model ──

class FakeDetection:
    """Records the lifecycle calls a reload makes."""

    def __init__(self, running=True, broken=False):
        self.calls = []
        self._running = running
        self._broken = broken

    def stop(self):
        self.calls.append("stop")
        return self._running

    def initialize(self):
        self.calls.append("initialize")
        if self._broken:
            raise RuntimeError("the gallery is out of step")

    def start(self):
        self.calls.append("start")


@pytest.fixture
def active(service, tmp_path, monkeypatch):
    """``staff.engine`` is the active, running model."""
    monkeypatch.setattr(ms.RuntimeFactory, "is_supported_model", staticmethod(lambda path: True))
    (tmp_path / "staff.engine").write_bytes(b"engine")
    service.current_model = "staff.engine"
    service.attach_detection_service(FakeDetection())
    return service


def test_publishing_over_the_active_model_reloads_it(active, tmp_path):
    """The running strategy would otherwise keep the previous gallery and names."""
    detection = active._detection_service
    with active.publication("staff.engine"):
        assert detection.calls == []          # the files are replaced first
    assert detection.calls == ["stop", "initialize", "start"]
    assert active.current_model == "staff.engine"


def test_a_stopped_device_stays_stopped_after_the_reload(active):
    active.attach_detection_service(FakeDetection(running=False))
    with active.publication("staff.engine"):
        pass
    assert active._detection_service.calls == ["stop", "initialize"]


def test_publishing_another_model_leaves_the_running_one_alone(active, tmp_path):
    with active.publication("visitors.engine"):
        pass
    assert active._detection_service.calls == []


def test_a_publication_that_fails_reloads_nothing(active):
    with pytest.raises(OSError):
        with active.publication("staff.engine"):
            raise OSError("disk full")
    assert active._detection_service.calls == []


def test_a_reload_that_fails_is_the_publications_failure(active):
    active.attach_detection_service(FakeDetection(broken=True))
    with pytest.raises(RuntimeError, match="rebuilt but could not be reloaded.*out of step"):
        with active.publication("staff.engine"):
            pass


def test_a_publication_holds_the_lifecycle_lock(active):
    """A rename (``SetClasses`` runs under ``op_lock``) waits for the reload."""
    import threading
    inside, release, renamed = threading.Event(), threading.Event(), threading.Event()

    def publish():
        with active.publication("staff.engine"):
            inside.set()
            release.wait(5)

    def rename():
        with active.op_lock:
            renamed.set()

    publisher = threading.Thread(target=publish)
    publisher.start()
    assert inside.wait(5)
    renamer = threading.Thread(target=rename)
    renamer.start()
    assert not renamed.wait(0.2)
    release.set()
    publisher.join(5)
    renamer.join(5)
    assert renamed.is_set()
    assert active._detection_service.calls == ["stop", "initialize", "start"]
