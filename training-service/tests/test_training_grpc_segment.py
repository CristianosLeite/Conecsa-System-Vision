# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""TrainingControl RPCs for segmentation datasets: the hub-record → dataset
path with polygons (class 0, an unknown name created, an unlabeled record), the
letterbox mapping of the points, the label RPCs and the rings of a SAM
result."""
from types import SimpleNamespace

import cv2
import grpc
import numpy as np
import pytest
import training_pb2 as pb
from service.dataset_service import DatasetService
from service.training_grpc import TrainingControlServicer

SQUARE = [0.25, 0.25, 0.75, 0.25, 0.75, 0.75, 0.25, 0.75]


class FakeContext:
    def __init__(self):
        self.code = None
        self.details = None

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


def _jpeg(width=64, height=48):
    ok, buf = cv2.imencode(".jpg", np.full((height, width, 3), 70, np.uint8))
    assert ok
    return buf.tobytes()


def _rig(tmp_path, size=0):
    dataset = DatasetService("d1", str(tmp_path / "d1"),
                             SimpleNamespace(MIN_IMAGES=1, runs_dir=str(tmp_path / "runs")))
    dataset.write_meta("parts", geometry={"letterbox": size} if size else {"native": True},
                       task="segment")
    app = SimpleNamespace(
        dataset_registry=SimpleNamespace(get=lambda dataset_id: dataset),
        config=SimpleNamespace(DATASET_IMG_SIZE=size),
        event_service=SimpleNamespace(publish=lambda *a, **k: None),
        training_service=SimpleNamespace(is_active=lambda: False),
        sam_service=SimpleNamespace(segment=lambda *a, **k: (
            [{"cx": 0.5, "cy": 0.5, "w": 0.5, "h": 0.5}], [0.9],
            [[[[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]]],
        )),
    )
    return SimpleNamespace(servicer=TrainingControlServicer(app), dataset=dataset)


def _ingest(rig, polygons=(), boxes=(), image=None):
    ctx = FakeContext()
    info = rig.servicer.AddDatasetImage(pb.LabeledImageUpload(
        dataset_id="d1", jpeg=image or _jpeg(), boxes=list(boxes), polygons=list(polygons),
    ), ctx)
    return info, ctx


class TestIngest:
    def test_a_record_lands_with_its_rings_and_new_classes(self, tmp_path):
        rig = _rig(tmp_path)
        rig.dataset.add_class("bolt")
        info, ctx = _ingest(rig, [
            pb.NamedPolygon(class_name="bolt", points=SQUARE),
            pb.NamedPolygon(class_name="nut", instance=1,
                            points=[0.0, 0.0, 0.2, 0.0, 0.2, 0.2, 0.0, 0.2]),
        ])
        assert ctx.code is None and info.labeled and info.box_count == 2
        assert rig.dataset.get_classes() == ["bolt", "nut"]
        assert sorted(p.class_id for p in rig.dataset.get_polygons(info.image_id)) == [0, 1]

    def test_a_letterboxed_dataset_maps_the_points_into_its_square(self, tmp_path):
        rig = _rig(tmp_path, size=64)
        info, ctx = _ingest(rig, [pb.NamedPolygon(class_name="bolt", points=SQUARE)],
                            image=_jpeg(64, 32))
        assert ctx.code is None
        (ring,) = rig.dataset.get_polygons(info.image_id)
        # 64×32 sits in the square between 16 px bands: y 0.25..0.75 → 0.375..0.625.
        ys = [p[1] for p in ring.points]
        assert (min(ys), max(ys)) == (pytest.approx(0.375, abs=0.02),
                                      pytest.approx(0.625, abs=0.02))

    def test_boxes_on_a_segmentation_dataset_are_invalid(self, tmp_path):
        rig = _rig(tmp_path)
        _, ctx = _ingest(rig, boxes=[pb.NamedBox(class_name="bolt", x1=0.1, y1=0.1,
                                                 x2=0.5, y2=0.5)])
        assert ctx.code == grpc.StatusCode.INVALID_ARGUMENT

    def test_an_unlabeled_record_is_allowed(self, tmp_path):
        rig = _rig(tmp_path)
        info, ctx = _ingest(rig)
        assert ctx.code is None and not info.labeled


class TestLabelRpcs:
    def _labeled(self, tmp_path):
        rig = _rig(tmp_path)
        rig.dataset.add_class("bolt")
        info, _ = _ingest(rig)
        return rig, info.image_id

    def test_set_then_get_round_trips_flat_points(self, tmp_path):
        rig, image_id = self._labeled(tmp_path)
        result = rig.servicer.SetLabels(pb.Labels(
            dataset_id="d1", image_id=image_id,
            polygons=[pb.Polygon(class_id=0, points=SQUARE)]), FakeContext())
        assert result.success, result.message
        labels = rig.servicer.GetLabels(pb.ImageId(dataset_id="d1", image_id=image_id),
                                        FakeContext())
        (polygon,) = labels.polygons
        assert polygon.class_id == 0
        assert list(polygon.points) == pytest.approx(SQUARE, abs=0.02)
        assert list(labels.boxes) == [] and not labels.HasField("image_class")

    def test_an_odd_point_list_is_a_failed_result(self, tmp_path):
        rig, image_id = self._labeled(tmp_path)
        result = rig.servicer.SetLabels(pb.Labels(
            dataset_id="d1", image_id=image_id,
            polygons=[pb.Polygon(class_id=0, points=[0.1, 0.2, 0.3])]), FakeContext())
        assert not result.success and "3 x, y" in result.message


class TestSam:
    def test_each_mask_ring_carries_the_index_of_its_box(self, tmp_path):
        rig = _rig(tmp_path)
        info, _ = _ingest(rig)
        result = rig.servicer.SamSegment(pb.SamRequest(
            dataset_id="d1", image_id=info.image_id, text_prompt="bolt"), FakeContext())
        assert result.success, result.message
        assert len(result.boxes) == 1
        (polygon,) = result.polygons
        assert polygon.instance == 0 and list(polygon.points) == pytest.approx(SQUARE)
