# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""TrainingControl RPCs for classification datasets: the hub-record →
dataset path with an image class, the label RPCs, the ZIP import's task and the
task of federated weights."""
from types import SimpleNamespace

import cv2
import grpc
import numpy as np
import pytest
import training_pb2 as pb
from service.dataset_service import DatasetService
from service.training_grpc import TrainingControlServicer


class FakeContext:
    def __init__(self):
        self.code = None
        self.details = None

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


def _jpeg():
    ok, buf = cv2.imencode(".jpg", np.full((48, 64, 3), 70, np.uint8))
    assert ok
    return buf.tobytes()


@pytest.fixture
def rig(tmp_path):
    dataset = DatasetService("d1", str(tmp_path / "d1"),
                             SimpleNamespace(MIN_IMAGES=1, runs_dir=str(tmp_path / "runs")))
    dataset.write_meta("pets", geometry={"native": True}, task="classify")
    events = []
    app = SimpleNamespace(
        dataset_registry=SimpleNamespace(get=lambda dataset_id: dataset),
        config=SimpleNamespace(DATASET_IMG_SIZE=0),
        event_service=SimpleNamespace(publish=lambda *a, **k: events.append(a[0])),
    )
    return SimpleNamespace(servicer=TrainingControlServicer(app), dataset=dataset,
                           events=events)


def _ingest(rig, image_class="", boxes=()):
    ctx = FakeContext()
    info = rig.servicer.AddDatasetImage(pb.LabeledImageUpload(
        dataset_id="d1", jpeg=_jpeg(), image_class=image_class, boxes=list(boxes)), ctx)
    return info, ctx


class TestRecordToDataset:
    def test_the_class_name_is_resolved_or_created(self, rig):
        infos = [_ingest(rig, name)[0] for name in ("cat", "dog", "cat")]
        assert [i.image_class for i in infos] == [0, 1, 0]
        # Class 0 is a real class: presence, not the value, marks it labeled.
        assert infos[0].HasField("image_class") and infos[0].labeled
        assert rig.dataset.get_classes() == ["cat", "dog"]
        assert rig.events.count("dataset_changed") == 3

    def test_an_unlabeled_record_is_stored_without_a_class(self, rig):
        info, ctx = _ingest(rig)
        assert ctx.code is None
        assert info.labeled is False and not info.HasField("image_class")

    def test_boxes_on_a_classification_dataset_are_invalid(self, rig):
        _, ctx = _ingest(rig, boxes=[pb.NamedBox(class_name="cat", x1=0.1, y1=0.1,
                                                 x2=0.5, y2=0.5)])
        assert ctx.code == grpc.StatusCode.INVALID_ARGUMENT
        assert rig.dataset.list_images() == []

    def test_list_images_reports_each_class(self, rig):
        _ingest(rig, "cat")
        _ingest(rig)
        listed = rig.servicer.ListImages(pb.DatasetId(dataset_id="d1"), FakeContext())
        assert sorted(i.HasField("image_class") for i in listed.images) == [False, True]


class TestLabelRpcs:
    def test_get_set_and_clear_the_class(self, rig):
        info, _ = _ingest(rig, "cat")
        _ingest(rig, "dog")
        ref = pb.ImageId(dataset_id="d1", image_id=info.image_id)
        assert rig.servicer.GetLabels(ref, FakeContext()).image_class == 0
        assert rig.servicer.SetLabels(pb.Labels(dataset_id="d1", image_id=info.image_id,
                                                image_class=1), FakeContext()).success
        assert rig.servicer.GetLabels(ref, FakeContext()).image_class == 1
        assert rig.servicer.SetLabels(pb.Labels(dataset_id="d1", image_id=info.image_id),
                                      FakeContext()).success
        assert not rig.servicer.GetLabels(ref, FakeContext()).HasField("image_class")

    def test_boxes_are_refused(self, rig):
        info, _ = _ingest(rig, "cat")
        r = rig.servicer.SetLabels(pb.Labels(
            dataset_id="d1", image_id=info.image_id,
            boxes=[pb.Box(class_id=0, cx=0.5, cy=0.5, w=0.1, h=0.1)]), FakeContext())
        assert not r.success and "an image class" in r.message


def _upload_servicer(tmp_path, imported):
    def import_zip(name, zip_path, task="detect"):
        imported.append((name, task))
        return {"dataset_id": "d9", "name": name, "created_at": 0.0, "cover_image_id": "",
                "image_count": 2, "labeled_count": 2, "class_count": 2, "task": task}

    return TrainingControlServicer(SimpleNamespace(
        dataset_registry=SimpleNamespace(import_zip=import_zip),
        config=SimpleNamespace(MAX_DATASET_UPLOAD_MB=1, datasets_dir=str(tmp_path))))


def _dataset_stream(**meta):
    return iter([pb.DatasetUploadChunk(meta=pb.DatasetUploadMeta(name="pets", **meta)),
                 pb.DatasetUploadChunk(chunk=b"PK")])


class TestUploadDatasetTask:
    def test_the_task_reaches_the_import(self, tmp_path):
        imported = []
        r = _upload_servicer(tmp_path, imported).UploadDataset(
            _dataset_stream(task="classify"), FakeContext())
        assert r.success and r.dataset.task == "classify"
        assert imported == [("pets", "classify")]

    def test_no_task_imports_detection(self, tmp_path):
        imported = []
        assert _upload_servicer(tmp_path, imported).UploadDataset(
            _dataset_stream(), FakeContext()).success
        assert imported == [("pets", "detect")]

    def test_an_unknown_task_is_refused(self, tmp_path):
        imported = []
        r = _upload_servicer(tmp_path, imported).UploadDataset(
            _dataset_stream(task="pose"), FakeContext())
        assert not r.success and imported == []


class FakeWeights:
    def __init__(self, tasks=None):
        self.saved = []
        self.tasks = dict(tasks or {})

    def save_stream(self, chunks, task=""):
        data = b"".join(chunks)
        self.saved.append((data, task))
        return "a" * 32, len(data)

    def path(self, weights_id):
        return f"/w/{weights_id}.pt"

    def task_of(self, weights_id):
        return self.tasks.get(weights_id, "")


def _weights_stream(task):
    return iter([pb.WeightsChunk(meta=pb.WeightsUploadMeta(name="round-1", task=task)),
                 pb.WeightsChunk(chunk=b"pt")])


class TestWeightsTask:
    def test_the_upload_records_the_task(self):
        store = FakeWeights()
        servicer = TrainingControlServicer(SimpleNamespace(weights_store=store))
        assert servicer.UploadWeights(_weights_stream("classify"), FakeContext()).success
        assert store.saved == [(b"pt", "classify")]

    def test_an_unknown_task_is_refused(self):
        store = FakeWeights()
        servicer = TrainingControlServicer(SimpleNamespace(weights_store=store))
        assert not servicer.UploadWeights(_weights_stream("pose"), FakeContext()).success
        assert store.saved == []

    def test_checkpoints_of_different_tasks_are_not_averaged(self, tmp_path):
        a, b = "a" * 32, "b" * 32
        store = FakeWeights({a: "detect", b: "classify"})
        servicer = TrainingControlServicer(SimpleNamespace(
            weights_store=store, config=SimpleNamespace(weights_dir=str(tmp_path))))
        r = servicer.AverageWeights(SimpleNamespace(weights_ids=[a, b]), FakeContext())
        assert not r.success and "different tasks" in r.message
