# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""A dataset's task: fixed at creation, legacy = detect, and the one label
kind each task accepts."""
import json

import pytest
from service.config import Config
from service.dataset_registry import DatasetRegistry
from service.dataset_service import LabelKindError, NamedBox, check_label_kinds


@pytest.fixture
def registry(tmp_path):
    cfg = Config()
    cfg.DATA_DIR = str(tmp_path)
    return DatasetRegistry(cfg, event_service=None)


def _meta_on_disk(ds):
    with open(ds._meta_file) as fh:
        return json.load(fh)


class TestCreation:
    def test_the_default_task_is_detect(self, registry):
        meta = registry.create("Parts")
        ds = registry.get(meta["dataset_id"])
        assert meta["task"] == "detect"
        assert ds.task() == "detect"
        assert ds.info()["task"] == "detect"
        assert _meta_on_disk(ds)["task"] == "detect"

    def test_the_device_task_is_recorded(self, registry):
        meta = registry.create("Kinds", "classify")
        assert meta["task"] == "classify"
        assert _meta_on_disk(registry.get(meta["dataset_id"]))["task"] == "classify"

    def test_the_task_is_fixed_at_creation(self, registry):
        meta = registry.create("Kinds", "classify")
        ds = registry.get(meta["dataset_id"])
        ds.write_meta("Renamed", task="segment")
        ds.rename("Renamed again")
        assert ds.task() == "classify"
        assert registry.list()[0]["task"] == "classify"


class TestLegacy:
    def test_a_meta_without_task_is_detect_and_is_backfilled(self, registry):
        meta = registry.create("Old")
        ds = registry.get(meta["dataset_id"])
        data = _meta_on_disk(ds)
        del data["task"]
        with open(ds._meta_file, "w") as fh:
            json.dump(data, fh)
        assert ds.task() == "detect"
        assert ds.meta()["task"] == "detect"
        ds.rename("Older")
        assert _meta_on_disk(ds)["task"] == "detect"


class TestLabelKinds:
    @pytest.mark.parametrize("task,kw", [
        ("detect", {"boxes": 2}),
        ("segment", {"polygons": 1}),
        ("classify", {"image_class": "cap"}),
        ("detect", {}), ("segment", {}), ("classify", {}),  # unlabeled is always fine
    ])
    def test_accepted(self, task, kw):
        check_label_kinds(task, **kw)

    @pytest.mark.parametrize("task,kw", [
        ("detect", {"polygons": 1}),
        ("detect", {"image_class": "cap"}),
        ("segment", {"boxes": 1}),
        ("segment", {"image_class": "cap"}),
        ("classify", {"boxes": 1}),
        ("classify", {"polygons": 1}),
        ("detect", {"boxes": 1, "polygons": 1}),
    ])
    def test_refused(self, task, kw):
        with pytest.raises(LabelKindError, match=f"'{task}' dataset"):
            check_label_kinds(task, **kw)

    def test_boxes_on_a_classification_dataset_are_refused_before_any_write(self, registry):
        meta = registry.create("Kinds", "classify")
        ds = registry.get(meta["dataset_id"])
        with pytest.raises(LabelKindError):
            ds.add_labeled_image(b"jpeg", [NamedBox("cap", 0.5, 0.5, 0.2, 0.2)])
        assert ds.list_images() == []
        assert ds.get_classes() == []
        # An unlabeled image is always welcome.
        ds.add_labeled_image(b"jpeg", [])
        assert len(ds.list_images()) == 1

    def test_detection_datasets_keep_taking_boxes(self, registry):
        meta = registry.create("Parts")
        ds = registry.get(meta["dataset_id"])
        entry = ds.add_labeled_image(b"jpeg", [NamedBox("cap", 0.5, 0.5, 0.2, 0.2)])
        assert entry.box_count == 1
