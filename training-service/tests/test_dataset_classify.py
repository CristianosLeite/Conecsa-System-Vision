# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Classification datasets: one image class per image stored as a one-line
label file, the training gate, the class-folder split and the folder-per-class
export."""
import os
import shutil
import zipfile
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from service.dataset_import import import_dataset_zip
from service.dataset_service import Box, DatasetError, DatasetService, LabelKindError, NamedBox


def _jpeg():
    ok, buf = cv2.imencode(".jpg", np.full((40, 60, 3), 90, np.uint8))
    assert ok
    return buf.tobytes()


def _service(tmp_path, dataset_id, task):
    config = SimpleNamespace(MIN_IMAGES=1, runs_dir=str(tmp_path / "runs"))
    service = DatasetService(dataset_id, str(tmp_path / dataset_id), config)
    service.write_meta(dataset_id, geometry={"native": True}, task=task)
    return service


@pytest.fixture
def ds(tmp_path):
    return _service(tmp_path, "pets", "classify")


def _add(ds, name):
    return ds.add_labeled_image(_jpeg(), [], image_class=name)


class TestLabels:
    def test_the_class_is_resolved_by_name_and_created_when_missing(self, ds):
        cat, dog, again = _add(ds, "cat"), _add(ds, "dog"), _add(ds, " cat ")
        assert (cat.image_class, dog.image_class, again.image_class) == (0, 1, 0)
        assert ds.get_classes() == ["cat", "dog"]
        assert cat.labeled and cat.box_count == 0

    def test_the_label_file_is_one_line_with_the_class_id(self, ds, tmp_path):
        _add(ds, "cat")
        dog = _add(ds, "dog")
        assert (tmp_path / "pets" / "labels" / f"{dog.image_id}.txt").read_text() == "1\n"

    @pytest.mark.parametrize("empty", [None, "", "  "])
    def test_an_unlabeled_image_is_allowed(self, ds, empty):
        entry = ds.add_labeled_image(_jpeg(), [], image_class=empty)
        assert entry.labeled is False and entry.image_class is None
        assert ds.get_classes() == []

    def test_boxes_are_refused_before_any_write(self, ds):
        with pytest.raises(LabelKindError, match="an image class"):
            ds.add_labeled_image(_jpeg(), [NamedBox("cat", 0.5, 0.5, 0.1, 0.1)])
        assert ds.list_images() == [] and ds.get_classes() == []

    def test_list_images_reports_the_class(self, ds):
        _add(ds, "cat")
        ds.add_labeled_image(_jpeg(), [])
        states = sorted((e.labeled, e.image_class if e.image_class is not None else -1)
                        for e in ds.list_images())
        assert states == [(False, -1), (True, 0)]
        assert ds.info()["labeled_count"] == 1

    def test_set_labels_sets_and_clears_the_class(self, ds):
        entry = _add(ds, "cat")
        _add(ds, "dog")
        ds.set_labels(entry.image_id, [], image_class=1)
        assert ds.get_image_class(entry.image_id) == 1
        ds.set_labels(entry.image_id, [])
        assert ds.get_image_class(entry.image_id) is None
        assert [e.labeled for e in ds.list_images() if e.image_id == entry.image_id] == [False]

    def test_set_labels_refuses_an_unknown_class_and_boxes(self, ds):
        entry = _add(ds, "cat")
        with pytest.raises(DatasetError, match="Unknown class id"):
            ds.set_labels(entry.image_id, [], image_class=5)
        with pytest.raises(LabelKindError):
            ds.set_labels(entry.image_id, [Box(0, 0.5, 0.5, 0.1, 0.1)])
        assert ds.get_image_class(entry.image_id) == 0

    def test_a_detection_dataset_refuses_an_image_class(self, tmp_path):
        det = _service(tmp_path, "parts", "detect")
        with pytest.raises(LabelKindError, match="not an image class"):
            det.add_labeled_image(_jpeg(), [], image_class="cap")
        entry = det.add_image(_jpeg())
        with pytest.raises(LabelKindError):
            det.set_labels(entry.image_id, [], image_class=0)
        assert det.get_image_class(entry.image_id) is None

    def test_removing_a_class_unlabels_its_images_and_shifts_the_rest(self, ds):
        cat, dog, bird = _add(ds, "cat"), _add(ds, "dog"), _add(ds, "bird")
        assert ds.remove_class(1) == ["cat", "bird"]
        assert [ds.get_image_class(e.image_id) for e in (cat, dog, bird)] == [0, None, 1]

    def test_replicas_keep_the_class(self, ds):
        _add(ds, "cat")
        dog = _add(ds, "dog")
        assert ds.replicate_image(dog.image_id, 2) == 2
        assert sorted(e.image_class or 0 for e in ds.list_images()) == [0, 1, 1, 1]

    def test_an_unlabeled_image_cannot_be_replicated(self, ds):
        entry = ds.add_labeled_image(_jpeg(), [])
        with pytest.raises(DatasetError, match="Only labeled images"):
            ds.replicate_image(entry.image_id, 1)

    @pytest.mark.parametrize("name", [".", ".."])
    def test_dots_only_class_names_are_refused(self, ds, name):
        # A classification split turns class names into folder names.
        with pytest.raises(DatasetError, match="dots"):
            ds.add_class(name)


class TestTrainingGate:
    def test_a_classifier_needs_two_labeled_classes(self, ds):
        _add(ds, "cat")
        _add(ds, "cat")
        with pytest.raises(DatasetError, match="at least 2 classes"):
            ds.validate_for_training()
        _add(ds, "dog")
        ds.validate_for_training()


class TestSplit:
    def test_class_folders_in_both_splits(self, ds, tmp_path):
        for name in ["cat"] * 3 + ["dog"] * 3:
            _add(ds, name)
        ds.add_class("bird")                 # no images: still a folder in both splits
        ds.add_labeled_image(_jpeg(), [])    # unlabeled: left out
        split = ds.build_split("job-1", tile="auto")   # tiling does not apply
        root = tmp_path / "runs" / "job-1" / "dataset"
        assert split.yaml_path == str(root)
        assert split.geometry == "frames"
        assert (split.train_count, split.valid_count) == (4, 2)
        for name in ("train", "val"):
            assert sorted(os.listdir(root / name)) == ["bird", "cat", "dog"]
        assert [len(os.listdir(root / "val" / c)) for c in ("bird", "cat", "dog")] == [0, 1, 1]
        assert not (root / "data.yaml").exists()
        # Symlinks onto the dataset's own images (same volume).
        linked = root / "train" / "cat" / os.listdir(root / "train" / "cat")[0]
        assert os.path.islink(linked) and os.path.exists(linked)

    def test_the_split_is_deterministic_per_job(self, ds, tmp_path):
        for name in ["cat"] * 5 + ["dog"] * 5:
            _add(ds, name)
        val = tmp_path / "runs" / "job-a" / "dataset" / "val" / "cat"
        ds.build_split("job-a")
        first = sorted(os.listdir(val))
        shutil.rmtree(tmp_path / "runs" / "job-a")
        ds.build_split("job-a")
        assert sorted(os.listdir(val)) == first

    def test_two_classes_must_reach_the_validation_split(self, ds):
        # cat splits 1/1; dog has a single image, which can only train.
        _add(ds, "cat")
        _add(ds, "cat")
        _add(ds, "dog")
        with pytest.raises(DatasetError, match="at least 2 classes with 2 or more"):
            ds.build_split("job-1")


class TestExport:
    def test_the_export_reimports_as_the_same_classification_dataset(self, ds, tmp_path):
        _add(ds, "cat")
        _add(ds, "dog")
        _add(ds, "dog")
        ds.add_class("bird")
        ds.add_labeled_image(_jpeg(), [])
        zip_path = tmp_path / "pets.zip"
        assert ds.export_zip(str(zip_path)) == 3
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
        assert "classes.txt" in names
        assert sorted(n.rsplit("/", 1)[0] for n in names if n.endswith(".jpg")) == [
            "train/cat", "train/dog", "train/dog"]
        classes, count = import_dataset_zip(str(zip_path), str(tmp_path / "copy"),
                                            img_size=0, task="classify")
        assert (classes, count) == (["cat", "dog", "bird"], 3)
        labels = sorted(p.read_text() for p in (tmp_path / "copy" / "labels").glob("*.txt"))
        assert labels == ["0\n", "1\n", "1\n"]

    def test_a_leading_dot_class_survives_the_round_trip(self, ds, tmp_path):
        _add(ds, ".defective")
        _add(ds, "good")
        zip_path = tmp_path / "parts.zip"
        assert ds.export_zip(str(zip_path)) == 2
        classes, count = import_dataset_zip(str(zip_path), str(tmp_path / "copy"),
                                            img_size=0, task="classify")
        assert (classes, count) == ([".defective", "good"], 2)
        labels = sorted(p.read_text() for p in (tmp_path / "copy" / "labels").glob("*.txt"))
        assert labels == ["0\n", "1\n"]

    def test_every_shard_carries_every_class(self, ds, tmp_path):
        for name in ["cat", "dog", "cat", "dog"]:
            _add(ds, name)
        for index in range(2):
            path = tmp_path / f"shard{index}.zip"
            assert ds.export_zip(str(path), num_shards=2, shard_index=index, seed="s") == 2
            with zipfile.ZipFile(path) as z:
                assert z.read("classes.txt").decode() == "cat\ndog\n"
