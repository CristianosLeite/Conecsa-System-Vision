# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for the training gRPC message -> JSON mappers."""
from types import SimpleNamespace

import pytest
import training_pb2 as trn_pb
from gateway.training import _job_dict, _meta_dict, _parse_named_boxes
from gateway.training.helpers import _labels_dict, _labels_message, _parse_named_polygons


def _job(**kw):
    base = dict(
        job_id="j1",
        status="running",
        progress=42,
        epoch=5,
        total_epochs=50,
        message="training",
        error="",
        model_name="best.pt",
        conversion_job_id="",
        metrics_json='{"mAP": 0.8}',
        started_at=1234.5,
        dataset_id="d1",
        federated=False,
        result_weights_id="",
    )
    base.update(kw)
    return SimpleNamespace(**base)


class TestJobDict:
    def test_full_job(self):
        d = _job_dict(_job())
        assert d["job_id"] == "j1"
        assert d["progress"] == 42
        assert d["metrics"] == {"mAP": 0.8}
        assert d["dataset_id"] == "d1"

    def test_empty_status_defaults_to_idle(self):
        assert _job_dict(_job(status=""))["status"] == "idle"

    def test_empty_metrics_json_is_empty_dict(self):
        assert _job_dict(_job(metrics_json=""))["metrics"] == {}

    def test_invalid_metrics_json_is_empty_dict(self):
        assert _job_dict(_job(metrics_json="{not json"))["metrics"] == {}

    def test_federated_fields_are_mapped(self):
        d = _job_dict(_job(federated=True, result_weights_id="abc123"))
        assert d["federated"] is True
        assert d["result_weights_id"] == "abc123"

    def test_geometry_is_mapped(self):
        assert _job_dict(_job(geometry="tiles:auto"))["geometry"] == "tiles:auto"

    def test_missing_geometry_defaults_to_empty(self):
        # Older stubs (or a job still preparing) carry no geometry.
        assert _job_dict(_job())["geometry"] == ""


class TestMetaDict:
    def test_full_meta(self):
        m = SimpleNamespace(
            dataset_id="d1",
            name="My Dataset",
            created_at=100.0,
            cover_image_id="img1",
            image_count=10,
            labeled_count=7,
            class_count=3,
            task="detect",
        )
        assert _meta_dict(m) == {
            "dataset_id": "d1",
            "name": "My Dataset",
            "created_at": 100.0,
            "cover_image_id": "img1",
            "image_count": 10,
            "labeled_count": 7,
            "class_count": 3,
            "task": "detect",
        }

    def test_a_meta_without_task_is_detect(self):
        # A training-service that predates application types sends none.
        m = SimpleNamespace(dataset_id="d1", name="n", created_at=0.0, cover_image_id="",
                            image_count=0, labeled_count=0, class_count=0)
        assert _meta_dict(m)["task"] == "detect"
        assert _meta_dict(SimpleNamespace(**{**vars(m), "task": ""}))["task"] == "detect"


class TestParseNamedBoxes:
    def test_valid_list(self):
        boxes = _parse_named_boxes(
            '[{"class_name": "cap", "x1": 0.1, "y1": 0.2, "x2": 0.5, "y2": 0.6}]'
        )
        assert len(boxes) == 1
        assert boxes[0].class_name == "cap"
        assert boxes[0].x1 == pytest.approx(0.1)
        assert boxes[0].y2 == pytest.approx(0.6)

    def test_empty_defaults_to_no_boxes(self):
        assert _parse_named_boxes("") == []
        assert _parse_named_boxes("[]") == []

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            _parse_named_boxes("{not json")

    def test_non_list_raises(self):
        with pytest.raises(ValueError):
            _parse_named_boxes('{"class_name": "cap"}')

    def test_non_dict_entry_raises(self):
        with pytest.raises(ValueError):
            _parse_named_boxes('["cap"]')

    def test_non_numeric_coord_raises(self):
        with pytest.raises(ValueError):
            _parse_named_boxes('[{"class_name": "cap", "x1": "left"}]')


class TestLabelsDict:
    def test_detection_labels_have_no_image_class(self):
        d = _labels_dict(trn_pb.Labels(image_id="i", boxes=[
            trn_pb.Box(class_id=1, cx=0.5, cy=0.5, w=0.25, h=0.25)]))
        assert d == {"image_id": "i", "image_class": None, "polygons": [], "boxes": [
            {"class_id": 1, "cx": 0.5, "cy": 0.5, "w": 0.25, "h": 0.25}]}

    def test_class_zero_is_a_class(self):
        # Presence, not the value: 0 is the first class, not "no class".
        d = _labels_dict(trn_pb.Labels(image_id="i", image_class=0))
        assert d["image_class"] == 0 and d["boxes"] == []


class TestLabelsMessage:
    def test_boxes(self):
        msg = _labels_message("d", "i", {"boxes": [
            {"class_id": 2, "cx": 0.1, "cy": 0.2, "w": 0.3, "h": 0.4}]})
        assert (msg.dataset_id, msg.image_id) == ("d", "i")
        assert msg.boxes[0].class_id == 2 and msg.boxes[0].h == pytest.approx(0.4)
        assert not msg.HasField("image_class")

    def test_image_class_zero(self):
        msg = _labels_message("d", "i", {"image_class": 0})
        assert msg.HasField("image_class") and msg.image_class == 0
        assert list(msg.boxes) == []

    def test_a_null_image_class_clears_the_label(self):
        msg = _labels_message("d", "i", {"image_class": None})
        assert not msg.HasField("image_class") and list(msg.boxes) == []

    @pytest.mark.parametrize("body, error", [
        ({}, "'polygons' list or an 'image_class'"),
        ([], "'polygons' list or an 'image_class'"),
        ({"boxes": "x"}, "'boxes' list"),
        ({"boxes": ["x"]}, "Malformed box entry"),
        ({"boxes": [{"cx": "left"}]}, "Malformed box entry"),
        ({"image_class": -1}, "class index"),
        ({"image_class": "cat"}, "class index"),
        ({"image_class": True}, "class index"),
    ])
    def test_malformed_bodies_raise(self, body, error):
        with pytest.raises(ValueError, match=error):
            _labels_message("d", "i", body)


def _flat(pairs):
    return [v for point in pairs for v in point]


TRIANGLE = [[0.1, 0.2], [0.5, 0.2], [0.5, 0.6]]


class TestParseNamedPolygons:
    def test_valid_list(self):
        (p,) = _parse_named_polygons(
            '[{"class_name": "bolt", "instance": 2, '
            '"points": [[0.1, 0.2], [0.5, 0.2], [0.5, 0.6]]}]')
        assert (p.class_name, p.instance) == ("bolt", 2)
        assert list(p.points) == pytest.approx(_flat(TRIANGLE))

    def test_omitted_instances_are_distinct_and_skip_explicit_ones(self):
        polygons = _parse_named_polygons(
            '[{"class_name": "bolt", "points": [[0.1, 0.2], [0.5, 0.2], [0.5, 0.6]]},'
            ' {"class_name": "bolt", "instance": 0, "points": [[0.1, 0.2], [0.5, 0.2], [0.5, 0.6]]},'
            ' {"class_name": "bolt", "points": [[0.1, 0.2], [0.5, 0.2], [0.5, 0.6]]}]')
        assert [p.instance for p in polygons] == [1, 0, 2]

    def test_empty_defaults_to_no_polygons(self):
        assert _parse_named_polygons("") == []

    @pytest.mark.parametrize("raw", [
        "nope",
        '{"a": 1}',
        "[1]",
        '[{"class_name": "a", "points": [[0.1, 0.2]]}]',
        '[{"class_name": "a", "points": [[0.1], [0.2], [0.3]]}]',
        '[{"class_name": "a", "points": [["x", 0], [1, 0], [1, 1]]}]',
        '[{"class_name": "a", "instance": -1, "points": [[0, 0], [1, 0], [1, 1]]}]',
    ])
    def test_malformed_raises(self, raw):
        with pytest.raises(ValueError):
            _parse_named_polygons(raw)


class TestPolygonLabels:
    def test_labels_dict_pairs_the_points(self):
        labels = trn_pb.Labels(image_id="i1", polygons=[
            trn_pb.Polygon(class_id=1, instance=3, points=_flat(TRIANGLE))])
        (polygon,) = _labels_dict(labels)["polygons"]
        assert (polygon["class_id"], polygon["instance"]) == (1, 3)
        assert _flat(polygon["points"]) == pytest.approx(_flat(TRIANGLE))

    def test_a_detection_image_has_no_polygons(self):
        assert _labels_dict(trn_pb.Labels(image_id="i1"))["polygons"] == []

    def test_labels_message_carries_the_rings(self):
        msg = _labels_message("d1", "i1", {"polygons": [{"class_id": 0, "points": TRIANGLE}]})
        (p,) = msg.polygons
        assert (p.class_id, p.instance) == (0, 0)
        assert list(p.points) == pytest.approx(_flat(TRIANGLE))
        assert list(msg.boxes) == [] and not msg.HasField("image_class")

    def test_labels_message_gives_each_ring_without_an_instance_its_own(self):
        msg = _labels_message("d1", "i1", {"polygons": [
            {"class_id": 0, "points": TRIANGLE},
            {"class_id": 0, "instance": 0, "points": TRIANGLE},
            {"class_id": 0, "instance": 2, "points": TRIANGLE},
            {"class_id": 0, "points": TRIANGLE},
            {"class_id": 0, "instance": None, "points": TRIANGLE},
        ]})
        assert [p.instance for p in msg.polygons] == [1, 0, 2, 3, 4]

    def test_an_empty_polygon_list_clears_the_labels(self):
        msg = _labels_message("d1", "i1", {"polygons": []})
        assert list(msg.polygons) == [] and list(msg.boxes) == []

    @pytest.mark.parametrize("body", [
        {"polygons": {}},
        {"polygons": [1]},
        {"polygons": [{"class_id": 0, "points": [[0, 0], [1, 1]]}]},
        {"polygons": [{"class_id": "x", "points": TRIANGLE}]},
        {"polygons": [{"class_id": -1, "points": TRIANGLE}]},
    ])
    def test_malformed_polygon_bodies_raise(self, body):
        with pytest.raises(ValueError):
            _labels_message("d1", "i1", body)
