# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Renaming a model's classes applies to the running strategy.

``SetClasses`` rewrites the model's sibling ``.txt`` while the stream runs.
The strategies cache the names — and the colors parsed out of them — when they
are built, so before this the burned-in overlay and the snapshot kept the old
names until the next model load. It shows worst on ``face``, where the name is
the whole result: a person renamed in the UI stayed under the old name on
screen.
"""
from types import SimpleNamespace

import api.inference_grpc as ig
import grpc
import inference_pb2 as pb
import numpy as np
import pytest
from api.config import Config
from api.postprocess import _face_assets
from api.postprocess.classify import ClassifyPostprocessor
from api.postprocess.detect import DetectPostprocessor
from api.postprocess.face import FacePostprocessor, gallery_file_for_model
from api.postprocess.segment import SegmentPostprocessor
from face_fixtures import FakeEmbedder, unit, write_gallery


def _config(model_path: str = "") -> Config:
    cfg = Config()
    if model_path:
        cfg.MODEL_PATH = model_path
    cfg.CONFIDENCE_THRESHOLD = 0.5
    return cfg


@pytest.mark.parametrize("build", [DetectPostprocessor, SegmentPostprocessor])
def test_the_box_tasks_adopt_renamed_labels(build):
    pp = build(["cap", "bolt"], _config())
    pp.set_class_labels(["boné", "parafuso #00ff00"])
    assert pp.class_labels == ["boné", "parafuso"]


def test_classification_reports_the_new_name_on_the_next_frame():
    pp = ClassifyPostprocessor(["cat", "dog"], _config())
    frame = np.zeros((48, 64, 3), np.uint8)
    outputs = [[np.array([[0.9, 0.1]], np.float32)]]
    assert pp.process(outputs, frame, [None], False).items[0].class_name == "cat"

    pp.set_class_labels(["gato", "cachorro"])

    result = pp.process(outputs, frame, [None], False)
    assert result.items[0].class_name == "gato"
    assert [c["class_name"] for c in (result.candidates or [])] == ["gato", "cachorro"]


def test_a_renamed_person_is_adopted_without_rebuilding_the_gallery(tmp_path, monkeypatch):
    monkeypatch.setattr(_face_assets, "embedder_sha256", lambda: "sha-test")
    model = str(tmp_path / "staff.engine")
    write_gallery(gallery_file_for_model(model), ["Pessoa 1"], [unit(1, 0)], [0])
    pp = FacePostprocessor(["Pessoa 1"], _config(model), embedder=FakeEmbedder())
    assert pp.class_labels == ["Pessoa 1"]

    pp.set_class_labels(["Cristiano"])

    assert pp.class_labels == ["Cristiano"]


def _servicer(tmp_path, fail: bool = False):
    """The management servicer over a fake app that records the live apply."""
    applied: list = []

    def refresh():
        if fail:
            raise RuntimeError("runtime is unloaded")
        applied.append(True)

    app = SimpleNamespace(
        config=SimpleNamespace(CLASSES_FILE_PATH=str(tmp_path / "model.txt")),
        detection_service=SimpleNamespace(refresh_class_labels=refresh),
    )
    return ig.ManagementControlServicer(app), app, applied


def test_saving_classes_applies_them_to_the_running_model(tmp_path):
    servicer, app, applied = _servicer(tmp_path)

    result = servicer.SetClasses(pb.ClassList(classes=["Cristiano"]), None)

    assert result.success and applied == [True]
    with open(app.config.CLASSES_FILE_PATH) as f:
        assert f.read().split() == ["Cristiano"]


def test_clearing_classes_applies_too(tmp_path):
    servicer, _, applied = _servicer(tmp_path)
    servicer.SetClasses(pb.ClassList(classes=["Cristiano"]), None)
    applied.clear()

    assert servicer.ClearClasses(pb.Empty(), None).success
    assert applied == [True]


def test_a_failed_apply_still_saves_the_classes(tmp_path):
    """The names are on disk; the next model load picks them up regardless."""
    servicer, app, _ = _servicer(tmp_path, fail=True)

    assert servicer.SetClasses(pb.ClassList(classes=["Cristiano"]), None).success
    with open(app.config.CLASSES_FILE_PATH) as f:
        assert f.read().split() == ["Cristiano"]


def test_a_face_sidecar_shorter_than_the_gallery_is_padded_from_it(tmp_path, monkeypatch):
    monkeypatch.setattr(_face_assets, "embedder_sha256", lambda: "sha-test")
    model = str(tmp_path / "staff.engine")
    write_gallery(gallery_file_for_model(model), ["Ana", "Bruno"], [unit(1, 0), unit(0, 1)], [0, 1])
    pp = FacePostprocessor(["Ana"], _config(model), embedder=FakeEmbedder())

    pp.set_class_labels([])

    assert pp.class_labels == ["Ana", "Bruno"]


def test_the_face_names_and_colors_are_swapped_as_one(tmp_path, monkeypatch):
    """The frame thread reads ``_people`` once per frame: whatever list it
    picks up is complete and its colors match its names, so a live rename
    can never index a half-built list."""
    monkeypatch.setattr(_face_assets, "embedder_sha256", lambda: "sha-test")
    model = str(tmp_path / "staff.engine")
    write_gallery(gallery_file_for_model(model), ["Ana", "Bruno"], [unit(1, 0), unit(0, 1)], [0, 1])
    pp = FacePostprocessor(["Ana"], _config(model), embedder=FakeEmbedder())
    before = pp._people

    pp.set_class_labels(["Ana Souza #ff0000"])

    after = pp._people
    assert before is not after and before == (["Ana", "Bruno"], before[1])
    assert after[0] == ["Ana Souza", "Bruno"] and len(after[1]) == 2
    assert after[1][0] == (0, 0, 255)  # BGR of #ff0000
    assert not hasattr(pp, "_names") and not hasattr(pp, "_colors")


class FakeContext:
    def __init__(self):
        self.code, self.details = None, ""

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


def _face_servicer(tmp_path, monkeypatch, people=("Ana", "Bruno")):
    """The management servicer over a face model with ``people`` in its gallery."""
    from api.services.model_settings_service import ModelSettingsService

    monkeypatch.setattr(_face_assets, "embedder_sha256", lambda: "sha-test")
    model = str(tmp_path / "staff.engine")
    ModelSettingsService.record_training(str(tmp_path / "staff.settings.json"), None,
                                         task="face")
    write_gallery(gallery_file_for_model(model), list(people),
                  [unit(1, 0)] * len(people), list(range(len(people))))
    applied: list = []
    app = SimpleNamespace(
        config=SimpleNamespace(CLASSES_FILE_PATH=str(tmp_path / "staff.txt"),
                               MODEL_PATH=model),
        detection_service=SimpleNamespace(refresh_class_labels=lambda: applied.append(True)),
    )
    return ig.ManagementControlServicer(app), app, applied


@pytest.mark.parametrize("classes,reason", [
    (["Ana"], "exactly its 2 enrolled people"),
    (["Ana", "Bruno", "Carla"], "exactly its 2 enrolled people"),
    (["Ana", "unknown"], "reserved"),
    (["Ana", "Unknown #ff0000"], "reserved"),
    (["Ana", "ana"], "share a name"),
    (["Ana", "ANA #ff0000"], "share a name"),
    (["Ana", ""], "must not be empty"),
    (["Ana", "#ff0000"], "colour alone"),
    (["Ana", "Bru/no"], "characters"),
    (["Bruno", "Ana"], "rename people in place"),
    (["Carla", "Ana"], "rename people in place"),
])
def test_a_face_list_that_does_not_fit_the_gallery_is_refused(tmp_path, monkeypatch,
                                                              classes, reason):
    servicer, app, applied = _face_servicer(tmp_path, monkeypatch)
    ctx = FakeContext()

    result = servicer.SetClasses(pb.ClassList(classes=classes), ctx)

    assert not result.success and reason in result.message
    assert ctx.code == grpc.StatusCode.INVALID_ARGUMENT and reason in ctx.details
    assert applied == [] and not (tmp_path / "staff.txt").exists()


def test_a_face_rename_in_place_is_saved_and_applied(tmp_path, monkeypatch):
    servicer, app, applied = _face_servicer(tmp_path, monkeypatch)

    result = servicer.SetClasses(pb.ClassList(classes=["Ana Souza #ff0000", "Bruno"]),
                                 FakeContext())

    assert result.success and applied == [True]
    with open(app.config.CLASSES_FILE_PATH) as f:
        assert f.read().splitlines() == ["Ana Souza #ff0000", "Bruno"]
    # The gallery now goes with the renamed list: the next activation accepts it.
    from api.postprocess import _face_gallery
    gallery = _face_gallery.load(gallery_file_for_model(app.config.MODEL_PATH))
    assert gallery.labels_sha256 == _face_gallery.labels_stamp(["Ana Souza #ff0000", "Bruno"])
    FacePostprocessor(["Ana Souza #ff0000", "Bruno"], _config(app.config.MODEL_PATH),
                      embedder=FakeEmbedder())


def test_a_gallery_out_of_step_with_its_names_is_refused(tmp_path, monkeypatch):
    """A build interrupted between the gallery and the names would pair these
    embeddings with another model's names; the strategy refuses it."""
    from api.postprocess import _face_gallery
    from api.postprocess.contract import ContractError

    monkeypatch.setattr(_face_assets, "embedder_sha256", lambda: "sha-test")
    model = str(tmp_path / "staff.engine")
    write_gallery(gallery_file_for_model(model), ["Ana", "Bruno"], [unit(1, 0), unit(0, 1)], [0, 1])
    gallery = _face_gallery.load(gallery_file_for_model(model))
    gallery.labels_sha256 = _face_gallery.labels_stamp(["Ana", "Bruno"])
    _face_gallery.save(gallery_file_for_model(model), gallery)

    FacePostprocessor(["Ana", "Bruno"], _config(model), embedder=FakeEmbedder())
    with pytest.raises(ContractError, match="out of step"):
        FacePostprocessor(["Old", "Names"], _config(model), embedder=FakeEmbedder())


def test_a_face_rename_holds_the_model_lifecycle_lock(tmp_path, monkeypatch):
    from threading import RLock

    servicer, app, _ = _face_servicer(tmp_path, monkeypatch)
    lock = RLock()
    app.model_service = SimpleNamespace(op_lock=lock)
    held = []
    app.detection_service = SimpleNamespace(
        refresh_class_labels=lambda: held.append(lock._is_owned()))  # type: ignore[attr-defined]

    assert servicer.SetClasses(pb.ClassList(classes=["Ana Souza", "Bruno"]), FakeContext()).success
    assert held == [True]


def test_a_failed_restamp_changes_nothing(tmp_path, monkeypatch):
    from api.postprocess import _face_gallery

    servicer, app, applied = _face_servicer(tmp_path, monkeypatch)
    servicer.SetClasses(pb.ClassList(classes=["Ana", "Bruno"]), FakeContext())
    applied.clear()
    monkeypatch.setattr(_face_gallery, "save", lambda *a, **k: (_ for _ in ()).throw(OSError("ro")))
    ctx = FakeContext()

    result = servicer.SetClasses(pb.ClassList(classes=["Ana Souza", "Bruno"]), ctx)

    assert not result.success and ctx.code == grpc.StatusCode.INTERNAL
    with open(app.config.CLASSES_FILE_PATH) as f:
        assert f.read().splitlines() == ["Ana", "Bruno"]
    assert applied == []


def test_a_failed_sidecar_write_moves_the_stamp_back(tmp_path, monkeypatch):
    from api.postprocess import _face_gallery
    from api.repositories.class_labels_repository import ClassLabelsRepository

    servicer, app, applied = _face_servicer(tmp_path, monkeypatch)
    servicer.SetClasses(pb.ClassList(classes=["Ana", "Bruno"]), FakeContext())
    applied.clear()
    monkeypatch.setattr(ClassLabelsRepository, "save_labels", lambda self, labels: False)

    result = servicer.SetClasses(pb.ClassList(classes=["Ana Souza", "Bruno"]), FakeContext())

    assert not result.success and applied == []
    gallery = _face_gallery.load(gallery_file_for_model(app.config.MODEL_PATH))
    assert gallery.labels_sha256 == _face_gallery.labels_stamp(["Ana", "Bruno"])


def test_clearing_a_face_list_restamps_the_gallery(tmp_path, monkeypatch):
    from api.postprocess import _face_gallery

    servicer, app, _ = _face_servicer(tmp_path, monkeypatch)
    assert servicer.ClearClasses(pb.Empty(), FakeContext()).success
    gallery = _face_gallery.load(gallery_file_for_model(app.config.MODEL_PATH))
    assert gallery.labels_sha256 == _face_gallery.labels_stamp([])


def test_clearing_a_face_list_falls_back_to_the_enrolled_names(tmp_path, monkeypatch):
    servicer, _, applied = _face_servicer(tmp_path, monkeypatch)
    assert servicer.ClearClasses(pb.Empty(), FakeContext()).success and applied == [True]


def test_other_tasks_keep_any_list(tmp_path, monkeypatch):
    from api.services.model_settings_service import ModelSettingsService

    ModelSettingsService.record_training(str(tmp_path / "m.settings.json"), None,
                                         task="detect")
    app = SimpleNamespace(
        config=SimpleNamespace(CLASSES_FILE_PATH=str(tmp_path / "m.txt"),
                               MODEL_PATH=str(tmp_path / "m.engine")),
        detection_service=SimpleNamespace(refresh_class_labels=lambda: None),
    )
    result = ig.ManagementControlServicer(app).SetClasses(
        pb.ClassList(classes=["unknown", "cap", "cap"]), FakeContext())
    assert result.success
