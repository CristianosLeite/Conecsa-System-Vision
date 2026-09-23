# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Building a face model from an enrollment package: manifest validation, the
shared engines built once, photos without a face skipped and counted, the
model files written, and the conversion job that drives it (status, events,
cleanup)."""
import json
import os
import zipfile
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from api.postprocess import _face_assets, _face_gallery
from api.postprocess.face import gallery_file_for_model
from api.services import conversion_service as cs
from api.services import face_gallery_builder as fgb
from api.services.conversion_service import ConversionJob, ConversionService, ConversionStatus
from api.services.face_gallery_builder import FaceGalleryBuilder, PackageError, read_manifest
from face_fixtures import unit, yunet_details, yunet_outputs

JPEG = cv2.imencode(".jpg", np.full((480, 640, 3), 60, np.uint8))[1].tobytes()


def _package(path, classes=("Ana", "Bruno"), images=None, manifest=None):
    """A ``.faces`` package; ``images`` is a list of ``(image_id, class_id)``."""
    images = [("a1", 0), ("a2", 0), ("b1", 1)] if images is None else images
    payload = manifest if manifest is not None else {
        "format": 1,
        "classes": list(classes),
        "images": [{"image_id": i, "class_id": c, "file": f"images/{i}.jpg"}
                   for i, c in images],
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(fgb.MANIFEST, json.dumps(payload))
        for image_id, _ in images:
            archive.writestr(f"images/{image_id}.jpg", JPEG)
    return path


class FakeDetector:
    """A ModelManager stand-in: one face per photo, none for ``blind`` calls,
    and two faces of similar size for ``ambiguous`` ones."""

    def __init__(self, blind=(), ambiguous=()):
        self.input_details = [{"shape": [1, 3, 640, 640], "index": 0}]
        self.output_details = yunet_details()
        self._blind = set(blind)
        self._ambiguous = set(ambiguous)
        # The second person of an ambiguous photo (115 × 115 beside 120 × 120).
        self.second_box = (300.0, 180.0, 415.0, 295.0)
        self.calls = 0

    def preprocess_tiles(self, frame):
        from api.model_manager import TileMeta
        return [np.zeros((1, 3, 640, 640), np.float32)], [TileMeta(1.0, 80, 640, 0, 0, 640, 480)]

    def run_inference(self, tensor):
        index, self.calls = self.calls, self.calls + 1
        # One face fires on several neighbouring anchors, as YuNet really
        # does; the builder must suppress them before judging the photo.
        faces = [((100.0, 180.0, 220.0, 300.0), 0.9), ((132.0, 212.0, 252.0, 332.0), 0.8)]
        if index in self._blind:
            faces = []
        elif index in self._ambiguous:
            faces.append((self.second_box, 0.9))
        return yunet_outputs(faces), 0.001


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """Bundled assets, a fake detector/embedder and an engine builder that touches files."""
    assets = tmp_path / "assets"
    assets.mkdir()
    for name in (_face_assets.DETECTOR_ONNX, _face_assets.EMBEDDER_ONNX):
        (assets / name).write_bytes(b"onnx")
    monkeypatch.setenv("FACE_MODELS_DIR", str(assets))
    monkeypatch.setattr(_face_assets, "embedder_sha256", lambda: "sha-test")

    detector = FakeDetector()
    embedder = SimpleNamespace(
        embed=lambda crops: np.stack([unit(1, 0)] * len(crops)),
        close=lambda: None)
    monkeypatch.setattr("api.model_manager.ModelManager", lambda *a, **k: detector)
    monkeypatch.setattr(fgb, "FaceEmbedder", lambda *a, **k: embedder)
    monkeypatch.setattr(fgb, "close_worker", lambda port: closed.append(port))
    monkeypatch.setattr(fgb, "check", lambda *a, **k: None)
    closed: list = []

    built: list = []

    def build_engine(onnx, engine):
        built.append((onnx, engine))
        with open(engine, "wb") as f:
            f.write(b"engine")

    models = tmp_path / "models"
    models.mkdir()
    return SimpleNamespace(models=models, detector=detector, build_engine=build_engine,
                           built=built, closed=closed, assets=assets)


def _builder(wired, progress=None):
    return FaceGalleryBuilder(SimpleNamespace(MODEL_PATH=""), str(wired.models),
                              wired.build_engine, progress or (lambda p, m: None))


# ── manifest ──

def test_a_manifest_is_parsed(tmp_path):
    with zipfile.ZipFile(_package(tmp_path / "p.faces")) as archive:
        classes, images = read_manifest(archive)
    assert classes == ["Ana", "Bruno"]
    assert [(i.image_id, i.class_id) for i in images] == [("a1", 0), ("a2", 0), ("b1", 1)]


@pytest.mark.parametrize("manifest,message", [
    ({"format": 2, "classes": ["Ana"], "images": [{"image_id": "a", "class_id": 0,
                                                   "file": "images/a.jpg"}]}, "format"),
    ({"format": 1, "classes": [], "images": []}, "has no people"),
    ({"format": 1, "classes": ["Ana\nBruno"], "images": [{"image_id": "a", "class_id": 0,
                                                          "file": "images/a.jpg"}]},
     "characters"),
    ({"format": 1, "classes": ["x" * 65], "images": [{"image_id": "a", "class_id": 0,
                                                      "file": "images/a.jpg"}]},
     "too long"),
    ({"format": 1, "classes": ["Ana", "Ana #ff0000"], "images": [{"image_id": "a", "class_id": 0,
                                                                  "file": "images/a.jpg"}]},
     "share a name"),
    ({"format": 1, "classes": ["#ff0000"], "images": [{"image_id": "a", "class_id": 0,
                                                       "file": "images/a.jpg"}]},
     "colour alone"),
    ({"format": 1, "classes": ["Ana", "Bruno"],
      "images": [{"image_id": "a", "class_id": 0, "file": "images/a.jpg"},
                 {"image_id": "b", "class_id": 1, "file": "images/a.jpg"}]},
     "more than once"),
    ({"format": 1, "classes": ["Ana", "Unknown"], "images": [{"image_id": "a", "class_id": 0,
                                                              "file": "images/a.jpg"}]},
     "reserved"),
    ({"format": 1, "classes": ["Ana", "ana"], "images": [{"image_id": "a", "class_id": 0,
                                                          "file": "images/a.jpg"}]},
     "share a name"),
    ({"format": 1, "classes": ["Ana"], "images": []}, "no photos"),
    ({"format": 1, "classes": ["Ana"], "images": [{"image_id": "a", "class_id": 3,
                                                   "file": "images/a.jpg"}]}, "invalid class"),
    ({"format": 1, "classes": ["Ana"], "images": [{"image_id": "a", "class_id": 0,
                                                   "file": "images/gone.jpg"}]}, "missing"),
])
def test_a_malformed_manifest_is_refused(tmp_path, manifest, message):
    path = _package(tmp_path / "p.faces", images=[("a", 0)], manifest=manifest)
    with zipfile.ZipFile(path) as archive, pytest.raises(PackageError, match=message):
        read_manifest(archive)


def test_a_package_without_a_manifest_is_refused(tmp_path):
    path = tmp_path / "empty.faces"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("images/a.jpg", JPEG)
    with zipfile.ZipFile(path) as archive, pytest.raises(PackageError, match="no manifest"):
        read_manifest(archive)


# ── build ──

def test_a_build_writes_the_model_its_classes_and_its_gallery(tmp_path, wired):
    engine = str(wired.models / "staff.engine")
    summary = _builder(wired).build(str(_package(tmp_path / "staff.faces")), engine)

    assert (summary.people, summary.embedded, summary.skipped) == (2, 3, {})
    gallery = _face_gallery.load(gallery_file_for_model(engine))
    assert gallery.names == ["Ana", "Bruno"] and gallery.labels.tolist() == [0, 0, 1]
    assert gallery.image_ids == ["a1", "a2", "b1"]
    assert gallery.embedder_sha256 == "sha-test"
    assert np.allclose(np.linalg.norm(gallery.embeddings, axis=1), 1.0)
    with open(str(wired.models / "staff.txt")) as f:
        assert f.read().split() == ["Ana", "Bruno"]
    assert open(engine, "rb").read() == b"engine"
    # The gallery is stamped with the names it was published with, and the
    # settings sidecar records the task with the same publication.
    assert gallery.labels_sha256 == _face_gallery.labels_stamp(["Ana", "Bruno"])
    from api.services.model_settings_service import ModelSettingsService
    assert ModelSettingsService.task_of(str(wired.models / "staff.settings.json")) == "face"


def test_a_rebuild_keeps_the_operators_settings_of_the_previous_model(tmp_path, wired):
    engine = str(wired.models / "staff.engine")
    settings = wired.models / "staff.settings.json"
    settings.write_text('{"task": "face", "face": {"match_threshold": 0.5}}')

    _builder(wired).build(str(_package(tmp_path / "staff.faces")), engine)

    import json
    assert json.loads(settings.read_text()) == {"task": "face", "face": {"match_threshold": 0.5}}


def test_a_newer_bundled_graph_never_reuses_the_old_engine(tmp_path, wired):
    builder = _builder(wired)
    builder.build(str(_package(tmp_path / "a.faces")), str(wired.models / "a.engine"))
    first = [engine for _, engine in wired.built]
    wired.built.clear()
    (wired.assets / _face_assets.EMBEDDER_ONNX).write_bytes(b"onnx v2")

    builder.build(str(_package(tmp_path / "b.faces")), str(wired.models / "b.engine"))

    rebuilt = [engine for _, engine in wired.built]
    assert rebuilt == [_face_assets.engine_path(str(wired.models), _face_assets.EMBEDDER_ONNX)]
    assert rebuilt[0] != first[1] and os.path.isfile(first[1])


def test_a_publish_that_fails_part_way_restores_the_previous_model(tmp_path, wired,
                                                                   monkeypatch):
    engine = str(wired.models / "staff.engine")
    previous = {engine: b"old-engine", str(wired.models / "staff.txt"): b"Old\n",
                gallery_file_for_model(engine): b"old-gallery"}
    for path, content in previous.items():
        open(path, "wb").write(content)
    real_replace = os.replace

    def replace_but_not_the_engine(src, dst):
        if dst == engine:
            raise OSError("disk full")
        real_replace(src, dst)

    monkeypatch.setattr(fgb.os, "replace", replace_but_not_the_engine)
    with pytest.raises(OSError, match="disk full"):
        _builder(wired).build(str(_package(tmp_path / "staff.faces")), engine)

    # The gallery and the names had already been replaced: both are back.
    for path, content in previous.items():
        assert open(path, "rb").read() == content
    leftovers = [p.name for p in wired.models.iterdir()
                 if p.name.endswith((fgb.STAGING_SUFFIX, fgb.BACKUP_SUFFIX))]
    assert leftovers == []


def test_a_build_that_fails_while_publishing_leaves_no_partial_model(tmp_path, wired,
                                                                     monkeypatch):
    """The gallery, the names and the engine are published together: an
    existing model of the same name keeps all three files, and no staged
    file is left behind."""
    engine = str(wired.models / "staff.engine")
    for path, content in ((engine, b"old-engine"), (str(wired.models / "staff.txt"), b"Old\n"),
                          (gallery_file_for_model(engine), b"old-gallery")):
        open(path, "wb").write(content)

    def fail_copy(src, dst, *a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(fgb.shutil, "copyfile", fail_copy)
    with pytest.raises(OSError, match="disk full"):
        _builder(wired).build(str(_package(tmp_path / "staff.faces")), engine)

    assert open(engine, "rb").read() == b"old-engine"
    assert open(str(wired.models / "staff.txt"), "rb").read() == b"Old\n"
    assert open(gallery_file_for_model(engine), "rb").read() == b"old-gallery"
    assert not [p for p in wired.models.iterdir() if p.name.endswith(fgb.STAGING_SUFFIX)]


def test_a_failed_job_discards_what_it_staged(tmp_path, monkeypatch):
    service = ConversionService()
    job = _job(tmp_path, service)
    staged = [f"{p}{fgb.STAGING_SUFFIX}" for p in fgb.model_outputs(job.engine_path)]

    def stage_then_explode(*args, **kwargs):
        for path in staged:
            open(path, "wb").write(b"partial")
        raise RuntimeError("worker died")

    monkeypatch.setattr("api.services.face_gallery_builder.FaceGalleryBuilder",
                        lambda *a, **k: SimpleNamespace(build=stage_then_explode))

    service._run_face_job(job.job_id)

    assert job.status is ConversionStatus.FAILED
    assert not any(os.path.exists(p) for p in staged)


def test_the_shared_engines_are_built_once_and_reused(tmp_path, wired):
    builder = _builder(wired)
    builder.build(str(_package(tmp_path / "a.faces")), str(wired.models / "a.engine"))
    assert [engine for _, engine in wired.built] == [
        _face_assets.engine_path(str(wired.models), _face_assets.DETECTOR_ONNX),
        _face_assets.engine_path(str(wired.models), _face_assets.EMBEDDER_ONNX)]
    wired.built.clear()
    builder.build(str(_package(tmp_path / "b.faces")), str(wired.models / "b.engine"))
    assert wired.built == []


def test_photos_without_a_face_are_skipped_and_counted(tmp_path, wired, monkeypatch):
    monkeypatch.setattr(fgb, "FaceEmbedder", lambda *a, **k: SimpleNamespace(
        embed=lambda crops: np.stack([unit(1, 0)] * len(crops)), close=lambda: None))
    wired.detector._blind = {1}
    summary = _builder(wired).build(str(_package(tmp_path / "s.faces")),
                                    str(wired.models / "s.engine"))
    assert summary.embedded == 2 and summary.skipped == {"Ana": 1}
    assert "skipped" in summary.message()


def test_one_face_on_several_anchors_still_enrolls(tmp_path, wired):
    """Neighbouring anchors are the same face, not an ambiguous photo.

    Judging overlap before NMS made every photo look like it held two faces
    of the same size, and the whole build failed with "no photo shows a clear
    face".
    """
    summary = _builder(wired).build(str(_package(tmp_path / "s.faces")),
                                    str(wired.models / "s.engine"))
    assert summary.embedded == 3 and summary.skipped == {}


def test_a_photo_with_two_faces_of_similar_size_is_skipped(tmp_path, wired):
    """One photo carries one name; the builder must not guess between faces."""
    wired.detector._ambiguous = {1}
    summary = _builder(wired).build(str(_package(tmp_path / "s.faces")),
                                    str(wired.models / "s.engine"))
    assert summary.embedded == 2 and summary.skipped == {"Ana": 1}


@pytest.mark.parametrize("height, skipped", [(78.0, {"Ana": 1}), (77.0, {})])
def test_a_second_face_that_reaches_the_ratio_is_ambiguous(tmp_path, wired, height, skipped):
    """The boundary is inclusive: 120 × 78 is exactly 65 % of the 120 × 120 face."""
    assert fgb.AMBIGUOUS_AREA_RATIO == 0.65
    wired.detector._ambiguous = {1}
    wired.detector.second_box = (300.0, 180.0, 420.0, 180.0 + height)
    summary = _builder(wired).build(str(_package(tmp_path / "s.faces")),
                                    str(wired.models / "s.engine"))
    assert summary.skipped == skipped


def test_a_build_where_no_photo_shows_a_face_fails(tmp_path, wired):
    wired.detector._blind = {0, 1, 2}
    with pytest.raises(RuntimeError, match="no photo shows a clear face"):
        _builder(wired).build(str(_package(tmp_path / "s.faces")),
                              str(wired.models / "s.engine"))


def test_both_build_workers_are_always_closed(tmp_path, wired):
    wired.detector._blind = {0, 1, 2}
    with pytest.raises(RuntimeError):
        _builder(wired).build(str(_package(tmp_path / "s.faces")),
                              str(wired.models / "s.engine"))
    assert wired.closed == list(_face_assets.build_worker_ports())


def test_only_the_publication_runs_inside_the_guard(tmp_path, wired):
    """Enrolling takes minutes; the lifecycle lock is held for the file moves only."""
    from contextlib import contextmanager
    seen = []

    @contextmanager
    def guard(engine_filename):
        seen.append(("enter", engine_filename, wired.detector.calls,
                     (wired.models / "s.engine").exists()))
        yield
        seen.append(("exit", (wired.models / "s.engine").exists()))

    FaceGalleryBuilder(SimpleNamespace(MODEL_PATH=""), str(wired.models), wired.build_engine,
                       lambda p, m: None, publish_guard=guard).build(
        str(_package(tmp_path / "s.faces")), str(wired.models / "s.engine"))

    assert seen == [("enter", "s.engine", 3, False), ("exit", True)]


def test_progress_is_reported_while_enrolling(tmp_path, wired):
    seen = []
    _builder(wired, progress=lambda p, m: seen.append((p, m))).build(
        str(_package(tmp_path / "s.faces")), str(wired.models / "s.engine"))
    assert any("face engines" in m for _, m in seen)
    assert any("Enrolling photo 1 of 3" in m for _, m in seen)
    assert all(0 <= p <= 100 for p, _ in seen)


# ── the conversion job ──

def _job(tmp_path, service):
    job = ConversionJob(job_id="job1", original_filename="staff.faces", pt_path="",
                        onnx_path="", engine_path=str(tmp_path / "staff.engine"),
                        task="face", faces_path=str(tmp_path / "staff.faces"))
    open(job.faces_path, "wb").write(b"package")
    service._jobs[job.job_id] = job
    return job


def test_the_job_reports_done_records_the_task_and_removes_the_package(tmp_path, monkeypatch):
    events = []
    service = ConversionService(SimpleNamespace(
        publish=lambda event, keys, **kw: events.append((event, kw.get("data")))))
    job = _job(tmp_path, service)
    summary = fgb.BuildSummary(people=2, embedded=5)
    monkeypatch.setattr(cs, "_build_engine_from_onnx", lambda *a: None)
    monkeypatch.setattr(fgb, "FaceGalleryBuilder",
                        lambda *a, **k: SimpleNamespace(build=lambda p, e: summary))
    monkeypatch.setattr("api.services.face_gallery_builder.FaceGalleryBuilder",
                        lambda *a, **k: SimpleNamespace(build=lambda p, e: summary))

    service._run_face_job(job.job_id)

    assert job.status is ConversionStatus.DONE and job.progress == 100
    assert job.engine_filename == "staff.engine" and "2 people" in job.message
    assert not (tmp_path / "staff.faces").exists()
    assert [e for e, _ in events][-2:] == ["classes_changed", "models_changed"]


def test_the_job_publishes_inside_the_guard_and_reports_done_after_it(tmp_path, monkeypatch):
    """"Done" means live: the guard reloads a rebuilt active model before it exits."""
    from contextlib import contextmanager
    order = []
    service = ConversionService(SimpleNamespace(
        publish=lambda event, keys, **kw: order.append(event)))
    job = _job(tmp_path, service)

    @contextmanager
    def guard(engine_filename):
        order.append(f"publish {engine_filename}")
        yield
        order.append("reloaded")

    def builder(*args, publish_guard, **kwargs):
        def build(package, engine):
            with publish_guard(os.path.basename(engine)):
                assert job.status is not ConversionStatus.DONE
            return fgb.BuildSummary(people=1, embedded=1)
        return SimpleNamespace(build=build)

    service.attach_publication_guard(guard)
    monkeypatch.setattr("api.services.face_gallery_builder.FaceGalleryBuilder", builder)

    service._run_face_job(job.job_id)

    assert job.status is ConversionStatus.DONE
    assert order.index("publish staff.engine") < order.index("reloaded")
    assert order.index("reloaded") < order.index("classes_changed")
    assert order[-2:] == ["classes_changed", "models_changed"]


def test_a_reload_that_fails_fails_the_job(tmp_path, monkeypatch):
    from contextlib import contextmanager
    service = ConversionService()
    job = _job(tmp_path, service)

    @contextmanager
    def guard(engine_filename):
        yield
        raise RuntimeError("'staff.engine' was rebuilt but could not be reloaded")

    def builder(*args, publish_guard, **kwargs):
        def build(package, engine):
            with publish_guard(os.path.basename(engine)):
                pass
        return SimpleNamespace(build=build)

    service.attach_publication_guard(guard)
    monkeypatch.setattr("api.services.face_gallery_builder.FaceGalleryBuilder", builder)

    service._run_face_job(job.job_id)

    assert job.status is ConversionStatus.FAILED
    assert "could not be reloaded" in (job.error or "")


def test_a_failed_job_is_reported_and_removes_the_package(tmp_path, monkeypatch):
    service = ConversionService()
    job = _job(tmp_path, service)

    def explode(*args, **kwargs):
        raise RuntimeError("no photo shows a clear face")

    monkeypatch.setattr("api.services.face_gallery_builder.FaceGalleryBuilder",
                        lambda *a, **k: SimpleNamespace(build=explode))

    service._run_face_job(job.job_id)

    assert job.status is ConversionStatus.FAILED
    assert job.error == "no photo shows a clear face"
    assert not (tmp_path / "staff.faces").exists()


def test_the_package_is_listed_with_its_task_while_it_builds(tmp_path):
    service = ConversionService()
    job = _job(tmp_path, service)
    assert service.get_active_jobs() == [job]
    assert service.to_dict(job)["task"] == "face"
