# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Face recognition datasets: one person (image class) per photo, the gallery
gate, the ``.faces`` enrollment package, the folder-per-person import and the
federated / SAM refusals."""
import json
import zipfile
from types import SimpleNamespace

import cv2
import grpc
import numpy as np
import pytest
import training_pb2 as pb
from service.dataset_import import DatasetImportError, import_dataset_zip
from service.dataset_service import (
    DatasetError,
    DatasetService,
    LabelKindError,
    NamedBox,
    check_label_kinds,
    uses_image_class,
)
from service.training_grpc import TrainingControlServicer


def _jpeg(value=90):
    ok, buf = cv2.imencode(".jpg", np.full((40, 60, 3), value, np.uint8))
    assert ok
    return buf.tobytes()


def _service(tmp_path, dataset_id, task, min_images=20):
    config = SimpleNamespace(MIN_IMAGES=min_images, runs_dir=str(tmp_path / "runs"))
    service = DatasetService(dataset_id, str(tmp_path / dataset_id), config)
    service.write_meta(dataset_id, geometry={"native": True}, task=task)
    return service


@pytest.fixture
def ds(tmp_path):
    return _service(tmp_path, "staff", "face")


def _add(ds, name):
    return ds.add_labeled_image(_jpeg(), [], image_class=name)


class TestLabelKinds:
    def test_image_class_tasks(self):
        assert uses_image_class("classify") and uses_image_class("face")
        assert not uses_image_class("detect") and not uses_image_class("segment")

    def test_a_face_dataset_takes_an_image_class_only(self):
        check_label_kinds("face", image_class="Alice")
        with pytest.raises(LabelKindError, match="'face' dataset takes an image class"):
            check_label_kinds("face", boxes=1)
        with pytest.raises(LabelKindError):
            check_label_kinds("face", polygons=1)

    def test_photos_are_labeled_with_a_person(self, ds, tmp_path):
        alice, bob = _add(ds, "Alice"), _add(ds, "Bob")
        assert (alice.image_class, bob.image_class) == (0, 1)
        assert ds.get_classes() == ["Alice", "Bob"]
        assert (tmp_path / "staff" / "labels" / f"{bob.image_id}.txt").read_text() == "1\n"
        listed = {e.image_id: e for e in ds.list_images()}
        assert listed[bob.image_id].labeled and listed[bob.image_id].image_class == 1
        assert ds.get_image_class(alice.image_id) == 0

    def test_boxes_are_refused(self, ds):
        with pytest.raises(LabelKindError):
            ds.add_labeled_image(_jpeg(), [NamedBox("Alice", 0.5, 0.5, 0.2, 0.2)])
        assert ds.list_images() == []

    def test_set_labels_remove_class_and_replicate(self, ds):
        alice, bob = _add(ds, "Alice"), _add(ds, "Bob")
        ds.set_labels(alice.image_id, [], image_class=1)
        assert ds.get_image_class(alice.image_id) == 1
        ds.set_labels(alice.image_id, [], image_class=None)
        assert ds.get_image_class(alice.image_id) is None
        assert ds.replicate_image(bob.image_id, 2) == 2
        assert sorted(e.image_class for e in ds.list_images() if e.labeled) == [1, 1, 1]
        ds.remove_class(0)
        assert ds.get_classes() == ["Bob"]
        assert ds.get_image_class(bob.image_id) == 0


class TestGalleryGate:
    def test_one_person_with_one_photo_is_enough(self, ds):
        _add(ds, "Alice")
        ds.validate_for_training()
        assert ds.info()["min_images"] == 1

    def test_a_person_and_a_labeled_photo_are_required(self, ds):
        with pytest.raises(DatasetError, match="at least one person"):
            ds.validate_for_training()
        ds.add_class("Alice")
        ds.add_labeled_image(_jpeg(), [])
        with pytest.raises(DatasetError, match="at least one photo"):
            ds.validate_for_training()

    def test_a_classifier_still_needs_its_minimums(self, tmp_path):
        classify = _service(tmp_path, "pets", "classify", min_images=1)
        classify.add_labeled_image(_jpeg(), [], image_class="cat")
        with pytest.raises(DatasetError, match="at least 2 classes"):
            classify.validate_for_training()
        assert classify.info()["min_images"] == 1
        big = _service(tmp_path, "big", "classify", min_images=20)
        assert big.info()["min_images"] == 20

    def test_a_face_dataset_has_no_training_split(self, ds):
        _add(ds, "Alice")
        with pytest.raises(DatasetError, match="gallery"):
            ds.build_split("job-1")


class TestFacePackage:
    def test_the_manifest_names_every_labeled_photo(self, ds, tmp_path):
        alice, bob = _add(ds, "Alice"), _add(ds, "Bob")
        ds.add_class("Carol")  # a person without photos keeps their class id
        unlabeled = ds.add_labeled_image(_jpeg(), [])
        path, count = ds.build_face_package("job-1", "front-door")
        assert path == str(tmp_path / "runs" / "job-1" / "front-door.faces")
        assert count == 2
        with zipfile.ZipFile(path) as zf:
            manifest = json.loads(zf.read("manifest.json"))
            names = set(zf.namelist())
            assert zf.read(f"images/{alice.image_id}.jpg") == ds.get_image_bytes(alice.image_id)
        assert manifest["format"] == 1
        assert manifest["classes"] == ["Alice", "Bob", "Carol"]
        assert sorted(manifest["images"], key=lambda e: e["class_id"]) == [
            {"image_id": alice.image_id, "class_id": 0, "file": f"images/{alice.image_id}.jpg"},
            {"image_id": bob.image_id, "class_id": 1, "file": f"images/{bob.image_id}.jpg"},
        ]
        assert names == {"manifest.json", f"images/{alice.image_id}.jpg",
                         f"images/{bob.image_id}.jpg"}
        assert f"images/{unlabeled.image_id}.jpg" not in names

    def test_the_package_matches_the_gallery_builder_contract(self, ds):
        # Mirrors inference-service face_gallery_builder.read_manifest, which
        # cannot be imported here (it needs the TensorRT stack).
        _add(ds, "Alice")
        path, _ = ds.build_face_package("job-2", "lobby")
        with zipfile.ZipFile(path) as zf:
            manifest = json.loads(zf.read("manifest.json"))
            members = set(zf.namelist())
        classes = manifest["classes"]
        assert classes and all(isinstance(c, str) and c.strip() for c in classes)
        for entry in manifest["images"]:
            assert isinstance(entry["class_id"], int) and 0 <= entry["class_id"] < len(classes)
            assert entry["file"] in members

    def test_nothing_to_package_is_refused(self, ds):
        ds.add_class("Alice")
        ds.add_labeled_image(_jpeg(), [])
        with pytest.raises(DatasetError, match="at least one photo"):
            ds.build_face_package("job-3", "lobby")

    def test_only_face_datasets_and_valid_names(self, ds, tmp_path):
        classify = _service(tmp_path, "pets", "classify")
        classify.add_labeled_image(_jpeg(), [], image_class="cat")
        with pytest.raises(DatasetError, match="Only a face dataset"):
            classify.build_face_package("job-4", "pets")
        _add(ds, "Alice")
        with pytest.raises(DatasetError, match="Model name"):
            ds.build_face_package("job-4", "../escape")

    def test_a_face_dataset_is_never_exported(self, ds, tmp_path):
        # The enrolment photos are biometric data: no ZIP, whole or sharded.
        _add(ds, "Alice")
        zip_path = tmp_path / "export.zip"
        with pytest.raises(DatasetError, match="stay on the device"):
            ds.export_zip(str(zip_path))
        with pytest.raises(DatasetError, match="stay on the device"):
            ds.export_zip(str(zip_path), num_shards=2, shard_index=0, seed="s")
        assert not zip_path.exists()


class TestReservedPersonName:
    @pytest.mark.parametrize("name", ["unknown", "Unknown", "UNKNOWN", " unknown ",
                                      "unknown #ff0000"])
    def test_unknown_is_not_a_person(self, ds, name):
        with pytest.raises(DatasetError, match="reserved"):
            ds.add_class(name)
        ds.add_class("Alice")
        with pytest.raises(DatasetError, match="reserved"):
            ds.rename_class(0, name)
        assert ds.get_classes() == ["Alice"]

    def test_other_tasks_keep_the_name(self, tmp_path):
        pets = _service(tmp_path, "pets", "classify")
        assert pets.add_class("unknown") == ["unknown"]

    @pytest.mark.parametrize("name", ["unknown", "Unknown #ff0000", "#ff0000"])
    def test_a_labeled_image_cannot_create_such_a_person(self, ds, name):
        # AddDatasetImage (camera capture, the hub's add-to-dataset) names the
        # person itself and must follow the same rules as the people panel.
        with pytest.raises(DatasetError):
            ds.add_labeled_image(_jpeg(), [], image_class=name)
        assert ds.get_classes() == [] and ds.list_images() == []

    @pytest.mark.parametrize("name", ["#ff0000", " #ABCDEF "])
    def test_a_colour_alone_is_not_a_person(self, ds, tmp_path, name):
        with pytest.raises(DatasetError, match="colour alone"):
            ds.add_class(name)
        ds.add_class("Alice")
        with pytest.raises(DatasetError, match="colour alone"):
            ds.rename_class(0, name)
        with pytest.raises(DatasetImportError, match="colour alone"):
            import_dataset_zip(_zip(tmp_path, {"#ff0000/1.jpg": _jpeg()}),
                               str(tmp_path / "out"), img_size=0, task="face")
        pets = _service(tmp_path, "pets", "classify")
        assert pets.add_class("#ff0000") == ["#ff0000"]

    def test_two_people_cannot_differ_only_in_case(self, ds, tmp_path):
        ds.add_class("Alice")
        ds.add_class("Bob")
        for name in ("alice", "ALICE #ff0000", " Alice "):
            with pytest.raises(DatasetError, match="already exists"):
                ds.add_class(name)
            with pytest.raises(DatasetError, match="already exists"):
                ds.rename_class(1, name)
        # Renaming a person to another spelling of their own name is fine.
        assert ds.rename_class(0, "ALICE") == ["ALICE", "Bob"]
        pets = _service(tmp_path, "pets", "classify")
        pets.add_class("cat")
        assert pets.add_class("Cat") == ["cat", "Cat"]

    @pytest.mark.parametrize("other", ["alice", "ALICE #ff0000"])
    def test_an_imported_package_with_two_spellings_is_refused(self, tmp_path, other):
        with pytest.raises(DatasetImportError, match="Duplicate"):
            import_dataset_zip(_zip(tmp_path, {
                "Alice/1.jpg": _jpeg(), f"{other}/1.jpg": _jpeg(),
            }), str(tmp_path / "out"), img_size=0, task="face")

    def test_an_imported_package_is_refused_too(self, tmp_path):
        with pytest.raises(DatasetImportError, match="reserved"):
            import_dataset_zip(_zip(tmp_path, {
                "Alice/1.jpg": _jpeg(), "Unknown/1.jpg": _jpeg(),
            }), str(tmp_path / "out"), img_size=0, task="face")


def _zip(tmp_path, entries, name="faces.zip"):
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as z:
        for arcname, data in entries.items():
            z.writestr(arcname, data)
    return str(path)


class TestFaceImport:
    def test_one_folder_per_person(self, tmp_path):
        dest = tmp_path / "out"
        classes, count = import_dataset_zip(_zip(tmp_path, {
            "Alice/1.jpg": _jpeg(), "Alice/2.jpg": _jpeg(), "Bob/1.jpg": _jpeg(),
        }), str(dest), img_size=0, task="face")
        assert classes == ["Alice", "Bob"] and count == 3
        labels = sorted(p.read_text() for p in (dest / "labels").glob("*.txt"))
        assert labels == ["0\n", "0\n", "1\n"]

    def test_a_single_person_archive(self, tmp_path):
        classes, count = import_dataset_zip(_zip(tmp_path, {"Alice/1.jpg": _jpeg()}),
                                            str(tmp_path / "out"), img_size=0, task="face")
        assert (classes, count) == (["Alice"], 1)

    def test_a_detection_archive_is_refused(self, tmp_path):
        with pytest.raises(DatasetImportError, match="face recognition.*per person"):
            import_dataset_zip(_zip(tmp_path, {
                "data.yaml": "names: ['a']\n", "images/1.jpg": _jpeg(),
                "labels/1.txt": "0 0.5 0.5 0.1 0.1\n",
            }), str(tmp_path / "out"), img_size=0, task="face")


class FakeContext:
    def __init__(self):
        self.code = None
        self.details = None

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


class FakeWeights:
    def __init__(self, tasks=None):
        self.saved = []
        self.tasks = dict(tasks or {})

    def save_stream(self, chunks, task=""):
        self.saved.append(task)
        return "a" * 32, 2

    def path(self, weights_id):
        return f"/w/{weights_id}.pt"

    def task_of(self, weights_id):
        return self.tasks.get(weights_id, "")


class TestRpcRefusals:
    @pytest.fixture
    def servicer(self, ds, tmp_path):
        _add(ds, "Alice")
        app = SimpleNamespace(
            dataset_registry=SimpleNamespace(get=lambda dataset_id: ds),
            config=SimpleNamespace(datasets_dir=str(tmp_path), weights_dir=str(tmp_path)),
            training_service=SimpleNamespace(is_active=lambda: False),
            sam_service=SimpleNamespace(segment=lambda *a, **k: pytest.fail("SAM called")),
            weights_store=FakeWeights({"a" * 32: "face", "b" * 32: "face"}),
        )
        return TrainingControlServicer(app)

    def test_a_face_dataset_is_not_exported(self, servicer, tmp_path):
        ctx = FakeContext()
        chunks = list(servicer.ExportDataset(pb.DatasetId(dataset_id="d1"), ctx))
        assert chunks == [] and ctx.code == grpc.StatusCode.FAILED_PRECONDITION
        assert "biometric" in str(ctx.details)
        assert not list(tmp_path.glob(".export-*.zip"))

    def test_a_face_dataset_is_not_sharded(self, servicer):
        ctx = FakeContext()
        chunks = list(servicer.ExportDatasetShard(pb.ShardExportRequest(
            dataset_id="d1", num_shards=2, shard_index=0, seed="s"), ctx))
        assert chunks == [] and ctx.code == grpc.StatusCode.FAILED_PRECONDITION
        assert "federated" in str(ctx.details)

    def test_sam_does_not_label_faces(self, servicer, ds):
        image_id = ds.list_images()[0].image_id
        r = servicer.SamSegment(SimpleNamespace(
            dataset_id="d1", image_id=image_id, points=[], text_prompt="face",
            threshold=0.5), FakeContext())
        assert not r.success and "face datasets" in r.message

    def test_face_weights_are_not_uploaded_or_averaged(self, servicer):
        stream = iter([pb.WeightsChunk(meta=pb.WeightsUploadMeta(name="r1", task="face")),
                       pb.WeightsChunk(chunk=b"pt")])
        assert not servicer.UploadWeights(stream, FakeContext()).success
        r = servicer.AverageWeights(SimpleNamespace(weights_ids=["a" * 32, "b" * 32]),
                                    FakeContext())
        assert not r.success and "Face" in r.message
