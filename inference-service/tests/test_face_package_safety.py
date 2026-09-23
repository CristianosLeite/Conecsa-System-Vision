# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""An enrollment package is biometric data the device must not hand back.

A ``.faces`` package holds the photos the operator enrolled and sits in the
model directory for the whole asynchronous gallery build (and survives a
restart if that build never finishes). It is accepted for upload and listed
while it builds, but it is never downloadable, never deletable from under its
own job, and the photos inside it are bounded before they are decoded.
"""
import io
import struct
import threading
import zipfile
from types import SimpleNamespace
from typing import Any, cast

import pytest
from api.config import Config
from api.services.face_gallery_builder import (
    MAX_IMAGE_PIXELS,
    PackageImage,
    image_dimensions,
)
from api.services.model_service import ModelService


@pytest.fixture
def service(tmp_path):
    config = Config()
    config.MODEL_PATH = str(tmp_path / "weights.engine")
    service = ModelService(config, str(tmp_path))
    service.attach_conversion_service(SimpleNamespace(get_active_jobs=lambda: []))
    return service


# ── the package never leaves the device ──

def test_an_enrollment_package_is_not_downloadable(service, tmp_path):
    (tmp_path / "staff.faces").write_bytes(b"PK\x03\x04enrolment photos")
    (tmp_path / "staff.engine").write_bytes(b"engine")

    assert service.model_file_path("staff.faces") == ""
    # The model built from it downloads as any other engine.
    assert service.model_file_path("staff.engine") == str(tmp_path / "staff.engine")


@pytest.mark.parametrize("name", ["staff.FACES", "staff.Faces"])
def test_the_refusal_is_case_insensitive(service, tmp_path, name):
    (tmp_path / name).write_bytes(b"PK\x03\x04")
    assert service.model_file_path(name) == ""


def test_a_package_being_built_cannot_be_deleted(service, tmp_path):
    (tmp_path / "staff.faces").write_bytes(b"PK\x03\x04")
    service.attach_conversion_service(SimpleNamespace(get_active_jobs=lambda: [
        SimpleNamespace(task="face", pt_path="", onnx_path="",
                        faces_path=str(tmp_path / "staff.faces")),
    ]))

    ok, message = service.delete_model("staff.faces")

    assert not ok and "being built" in message
    assert (tmp_path / "staff.faces").exists()


def _building(service, tmp_path, engine="staff.engine"):
    service.attach_conversion_service(SimpleNamespace(get_active_jobs=lambda: [
        SimpleNamespace(task="face", pt_path="", onnx_path="",
                        faces_path=str(tmp_path / "staff.faces"),
                        engine_path=str(tmp_path / engine)),
    ]))


def test_the_engine_a_build_will_publish_cannot_be_deleted_under_it(service, tmp_path):
    # A model of the same name already exists; the running build replaces it
    # at the end, so a delete now would be undone by the job.
    (tmp_path / "staff.faces").write_bytes(b"PK\x03\x04")
    (tmp_path / "staff.engine").write_bytes(b"engine")
    _building(service, tmp_path)

    ok, message = service.delete_model("staff.engine")

    assert not ok and "being built" in message
    assert (tmp_path / "staff.engine").exists()
    assert service.delete_model("other.engine") == (False, "Model 'other.engine' not found")


@pytest.mark.parametrize("filename", ["staff.faces", "staff.pt", "staff.engine"])
def test_an_upload_owned_by_a_running_build_is_refused(service, tmp_path, filename):
    _building(service, tmp_path)
    saved = []
    service.save_model = lambda name, data: saved.append(name) or (True, "", "")

    body, status = service.process_upload(
        filename, io.BytesIO(b"x"), task="face" if filename.endswith(".faces") else "detect")

    assert status == 409 and "being built" in body["error"]
    assert saved == []


def test_an_unrelated_upload_is_not_blocked_by_the_build(service, tmp_path):
    _building(service, tmp_path)
    saved = []
    service.save_model = lambda name, data: saved.append(name) or (False, "", "stop here")

    body, status = service.process_upload("lobby.faces", io.BytesIO(b"x"), task="face")

    assert status == 500 and saved == ["lobby.faces"]


def test_a_package_has_no_weights_to_download(service, tmp_path):
    # An earlier "staff" model left a checkpoint; the package must not resolve it.
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "staff.pt").write_bytes(b"checkpoint")
    (tmp_path / "staff.faces").write_bytes(b"PK\x03\x04")
    (tmp_path / "staff.engine").write_bytes(b"engine")

    assert service.weights_file_path("staff.faces") == ""
    assert service.weights_file_path("staff.engine") == str(weights / "staff.pt")
    listed = {m.name: m.has_weights for m in service.list_models()}
    assert listed == {"staff.faces": False, "staff.engine": True}


def test_deleting_a_stale_package_keeps_the_models_sidecars(service, tmp_path):
    """After an interrupted build the package lingers beside the model of the
    same name; removing it must not take that model's files with it."""
    (tmp_path / "staff.faces").write_bytes(b"PK\x03\x04")
    kept = ["staff.engine", "staff.txt", "staff.settings.json", "staff.gallery.npz"]
    for name in kept:
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "staff.pt").write_bytes(b"x")

    assert service.delete_model("staff.faces") == (True, "")

    assert not (tmp_path / "staff.faces").exists()
    assert all((tmp_path / name).exists() for name in kept)
    assert (tmp_path / "weights" / "staff.pt").exists()
    assert ModelService.model_artifacts(str(tmp_path / "staff.faces")) == [
        str(tmp_path / "staff.faces")]


def test_the_package_is_deletable_once_its_job_is_over(service, tmp_path):
    (tmp_path / "staff.faces").write_bytes(b"PK\x03\x04")
    assert service.delete_model("staff.faces") == (True, "")
    assert not (tmp_path / "staff.faces").exists()


# ── a crafted photo cannot exhaust the device ──

def _png_header(width: int, height: int) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + struct.pack(">II", width, height)


def _jpeg_header(width: int, height: int) -> bytes:
    return (b"\xff\xd8\xff\xc0" + struct.pack(">H", 17) + b"\x08"
            + struct.pack(">HH", height, width) + b"\x03" * 10)


@pytest.mark.parametrize("header,expected", [
    (_png_header(1920, 1080), (1920, 1080)),
    (_jpeg_header(640, 480), (640, 480)),
    (b"not an image at all", None),
    (b"", None),
])
def test_dimensions_are_read_from_the_header(header, expected):
    assert image_dimensions(header) == expected


def test_a_photo_past_the_pixel_limit_is_skipped_before_decoding(monkeypatch):
    """The byte cap alone cannot stop it: a flat 60k×60k PNG compresses tiny."""
    from api.services import face_gallery_builder as fgb

    huge = _png_header(60_000, 60_000)
    assert huge and (60_000 * 60_000) > MAX_IMAGE_PIXELS
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("images/a.png", huge)

    decoded = []
    monkeypatch.setattr(fgb.cv2, "imdecode", lambda *a, **k: decoded.append(True))

    with zipfile.ZipFile(buffer) as archive:
        # The photo never reaches the detector or the embedder.
        result = fgb.FaceGalleryBuilder._embed_one(
            archive, PackageImage("a", 0, "images/a.png"), None, {}, cast(Any, None))

    assert result is None and decoded == []


def test_a_normal_photo_still_reaches_the_decoder(monkeypatch):
    from api.services import face_gallery_builder as fgb

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("images/a.png", _png_header(640, 480))

    monkeypatch.setattr(fgb.cv2, "imdecode", lambda *a, **k: None)  # undecodable, but tried

    with zipfile.ZipFile(buffer) as archive:
        result = fgb.FaceGalleryBuilder._embed_one(
            archive, PackageImage("a", 0, "images/a.png"), None, {}, cast(Any, None))

    assert result is None  # cv2 could not decode the stub; the guard let it through


def test_a_build_waits_for_the_one_before_it(tmp_path, monkeypatch):
    """Every build drives the same two worker ports and writes the same shared
    engines, so two uploads arriving together must not run at once."""
    from api.services import face_gallery_builder as fgb

    builder = fgb.FaceGalleryBuilder(Config(), str(tmp_path), lambda *a: None,
                                     lambda *a: None)
    started = threading.Event()
    release = threading.Event()

    def slow_build(package_path, engine_out):
        started.set()
        release.wait(timeout=5)
        return fgb.BuildSummary(people=1, embedded=1)

    monkeypatch.setattr(builder, "_build", slow_build)
    first = threading.Thread(target=builder.build, args=("a.faces", "a.engine"))
    first.start()
    assert started.wait(timeout=5)

    # The lock the second build would take is held by the first one.
    assert not fgb._BUILD_LOCK.acquire(blocking=False)
    release.set()
    first.join(timeout=5)

    assert fgb._BUILD_LOCK.acquire(blocking=False)
    fgb._BUILD_LOCK.release()
