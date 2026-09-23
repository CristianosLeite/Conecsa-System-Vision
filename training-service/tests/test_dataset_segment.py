# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Segmentation datasets: polygon label rows, the label-kind exclusion,
normalization, class removal, replication, the tiled split and the export →
import round trip."""
import os
import zipfile
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from service.dataset_import import DatasetImportError, import_dataset_zip
from service.dataset_service import (
    Box,
    DatasetError,
    DatasetService,
    LabelKindError,
    NamedPolygon,
    Polygon,
    normalize_polygons,
    points_to_pairs,
)

W, H = 1280, 720
SQUARE = [[0.1, 0.1], [0.3, 0.1], [0.3, 0.4], [0.1, 0.4]]
U_SHAPE = [[0.5, 0.2], [0.9, 0.2], [0.9, 0.8], [0.8, 0.8], [0.8, 0.4], [0.6, 0.4],
           [0.6, 0.8], [0.5, 0.8]]


def _jpeg(width=W, height=H):
    ok, buf = cv2.imencode(".jpg", np.full((height, width, 3), 90, np.uint8))
    assert ok
    return buf.tobytes()


def _service(tmp_path, dataset_id="parts", geometry=None, task="segment"):
    config = SimpleNamespace(MIN_IMAGES=1, runs_dir=str(tmp_path / "runs"))
    service = DatasetService(dataset_id, str(tmp_path / dataset_id), config)
    service.write_meta(dataset_id, geometry=geometry or {"native": True}, task=task)
    return service


@pytest.fixture
def ds(tmp_path):
    return _service(tmp_path)


def _add(ds, *rings, name="bolt"):
    return ds.add_labeled_image(_jpeg(), [], polygons=[NamedPolygon(name, r) for r in rings])


def _rows(path):
    with open(path) as f:
        return [line.split() for line in f.read().splitlines()]


class TestLabels:
    def test_rings_are_resolved_by_name_and_written_one_row_each(self, ds):
        entry = _add(ds, SQUARE, U_SHAPE)
        assert ds.get_classes() == ["bolt"]
        assert entry.labeled and entry.box_count == 2
        # Largest ring first; valid rings are kept as given (normalization's fixed point).
        assert [len(r) for r in _rows(ds._label_path(entry.image_id))] == [1 + 2 * 8, 1 + 2 * 4]
        stored = ds.get_polygons(entry.image_id)[0].points
        assert [v for p in stored for v in p] == pytest.approx([v for p in U_SHAPE for v in p])

    def test_reading_back_gives_one_instance_per_row(self, ds):
        entry = _add(ds, SQUARE, U_SHAPE)
        assert [p.instance for p in ds.get_polygons(entry.image_id)] == [0, 1]

    def test_set_labels_normalizes_a_self_intersecting_ring(self, ds):
        entry = _add(ds, SQUARE)
        bowtie = [[0.1, 0.1], [0.4, 0.4], [0.4, 0.1], [0.1, 0.4]]
        ds.set_labels(entry.image_id, [], polygons=[Polygon(0, bowtie)])
        rings = ds.get_polygons(entry.image_id)
        assert rings and all(len(p.points) >= 3 and p.points != bowtie for p in rings)

    def test_clearing_the_polygons_unlabels_the_image(self, ds):
        entry = _add(ds, SQUARE)
        ds.set_labels(entry.image_id, [], polygons=[])
        assert not ds.list_images()[0].labeled
        assert not os.path.exists(ds._label_path(entry.image_id))

    def test_boxes_and_image_classes_are_refused(self, ds):
        entry = _add(ds, SQUARE)
        with pytest.raises(LabelKindError):
            ds.set_labels(entry.image_id, [Box(0, 0.5, 0.5, 0.2, 0.2)])
        with pytest.raises(LabelKindError):
            ds.set_labels(entry.image_id, [], image_class=0)

    def test_polygons_are_refused_on_a_detection_dataset(self, tmp_path):
        detect = _service(tmp_path, "d", task="detect")
        with pytest.raises(LabelKindError):
            detect.add_labeled_image(_jpeg(), [], polygons=[NamedPolygon("bolt", SQUARE)])

    @pytest.mark.parametrize("points", [
        [[0.1, 0.1], [0.2, 0.2]],
        [[0.1, 0.1], [1.2, 0.1], [0.5, 0.5]],
    ])
    def test_invalid_rings_are_refused(self, ds, points):
        entry = _add(ds, SQUARE)
        with pytest.raises(DatasetError):
            ds.set_labels(entry.image_id, [], polygons=[Polygon(0, points)])

    def test_an_unknown_class_id_is_refused(self, ds):
        entry = _add(ds, SQUARE)
        with pytest.raises(DatasetError, match="Unknown class"):
            ds.set_labels(entry.image_id, [], polygons=[Polygon(3, SQUARE)])

    def test_an_unlabeled_image_is_allowed(self, ds):
        entry = ds.add_labeled_image(_jpeg(), [])
        assert not entry.labeled and entry.box_count == 0

    def test_a_letterboxed_dataset_normalizes_at_its_square(self, tmp_path):
        square = _service(tmp_path, "sq", geometry={"letterbox": 64})
        entry = square.add_labeled_image(_jpeg(64, 64), [],
                                         polygons=[NamedPolygon("bolt", SQUARE)])
        assert entry.box_count == 1


class TestHelpers:
    def test_points_to_pairs(self):
        assert points_to_pairs([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) == [
            [pytest.approx(0.1), pytest.approx(0.2)], [pytest.approx(0.3), pytest.approx(0.4)],
            [pytest.approx(0.5), pytest.approx(0.6)]]

    @pytest.mark.parametrize("flat", [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 0.4]])
    def test_odd_or_short_point_lists_are_refused(self, flat):
        with pytest.raises(DatasetError):
            points_to_pairs(flat)

    def test_rings_of_one_instance_merge(self):
        a = [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]]
        b = [[0.3, 0.3], [0.6, 0.3], [0.6, 0.6], [0.3, 0.6]]
        (out,) = normalize_polygons([Polygon(1, a, 7), Polygon(1, b, 7)], 640, 480)
        assert (out.class_id, out.instance) == (1, 7)

    def test_named_rings_keep_their_name(self):
        (out,) = normalize_polygons([NamedPolygon("nut", SQUARE, 2)], 640, 480)
        assert isinstance(out, NamedPolygon) and (out.class_name, out.instance) == ("nut", 2)


class TestDatasetOps:
    def test_listing_counts_rings(self, ds):
        _add(ds, SQUARE, U_SHAPE)
        (entry,) = ds.list_images()
        assert entry.labeled and entry.box_count == 2

    def test_replication_copies_the_rings(self, ds):
        entry = _add(ds, SQUARE)
        assert ds.replicate_image(entry.image_id, 2) == 2
        assert [len(ds.get_polygons(e.image_id)) for e in ds.list_images()] == [1, 1, 1]

    def test_removing_a_class_drops_and_renumbers_rings(self, ds):
        entry = _add(ds, SQUARE, name="bolt")
        other = _add(ds, U_SHAPE, name="nut")
        ds.remove_class(0)
        assert ds.get_classes() == ["nut"]
        assert ds.get_polygons(entry.image_id) == []
        assert [p.class_id for p in ds.get_polygons(other.image_id)] == [0]

    def test_the_split_writes_polygon_rows_on_tile_crops(self, ds):
        for _ in range(3):
            _add(ds, U_SHAPE)
        split = ds.build_split("job1", tile="auto")
        assert split.geometry == "tiles:auto" and split.stats.tiles > 0
        labels_dir = os.path.join(os.path.dirname(split.yaml_path), "train", "labels")
        rows = [row for name in os.listdir(labels_dir)
                for row in _rows(os.path.join(labels_dir, name))]
        assert rows and all(len(r) >= 7 and len(r) % 2 == 1 for r in rows)

    def test_the_export_reimports_as_segmentation_with_the_rings(self, ds, tmp_path):
        _add(ds, U_SHAPE)
        _add(ds, SQUARE)
        zip_path = str(tmp_path / "parts.zip")
        assert ds.export_zip(zip_path) == 2
        with zipfile.ZipFile(zip_path) as zf:
            assert len([n for n in zf.namelist() if n.startswith("labels/")]) == 2
        dest = tmp_path / "imported"
        classes, count = import_dataset_zip(zip_path, str(dest), img_size=0, task="segment")
        assert (classes, count) == (["bolt"], 2)
        tokens = sorted(len(r) for name in os.listdir(dest / "labels")
                        for r in _rows(os.path.join(dest, "labels", name)))
        assert tokens == [9, 17]


def _zip(tmp_path, label, image=None):
    path = tmp_path / "in.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("data.yaml", "names: [a]\n")
        zf.writestr("images/x.jpg", image or _jpeg(120, 80))
        zf.writestr("labels/x.txt", label)
    return str(path)


class TestImport:
    def test_box_rows_are_refused(self, tmp_path):
        with pytest.raises(DatasetImportError, match="detection box"):
            import_dataset_zip(_zip(tmp_path, "0 0.5 0.5 0.2 0.2\n"), str(tmp_path / "out"),
                               img_size=0, task="segment")

    def test_a_malformed_row_is_refused(self, tmp_path):
        with pytest.raises(DatasetImportError, match="x1 y1"):
            import_dataset_zip(_zip(tmp_path, "0 0.1 0.2 0.3 0.4 0.5\n"), str(tmp_path / "out"),
                               img_size=0, task="segment")

    def test_the_letterbox_transform_applies_to_every_point(self, tmp_path):
        # 120×80 → 64 px square: 64×43 at top 10, so y 0.1..0.9 → 0.2234..0.7609.
        label = "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n"
        dest = tmp_path / "out"
        assert import_dataset_zip(_zip(tmp_path, label), str(dest), img_size=64,
                                  task="segment")[1] == 1
        (name,) = os.listdir(dest / "labels")
        (row,) = _rows(os.path.join(dest, "labels", name))
        xs, ys = [float(v) for v in row[1::2]], [float(v) for v in row[2::2]]
        assert (min(xs), max(xs)) == (pytest.approx(0.1, abs=0.03), pytest.approx(0.9, abs=0.03))
        assert min(ys) == pytest.approx((0.1 * 43 + 10) / 64, abs=0.03)
        assert max(ys) == pytest.approx((0.9 * 43 + 10) / 64, abs=0.03)
