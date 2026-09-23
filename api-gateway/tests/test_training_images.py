# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for the labeled-image upload route (bounded read, 413 over cap,
image-class pre-labels) and the label-editor routes."""
import io
from types import SimpleNamespace

import pytest
import training_pb2 as trn_pb
from flask import Flask
from gateway.config import settings
from gateway.training import images, training_bp


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(training_bp)
    return app.test_client()


@pytest.fixture
def training_stub(monkeypatch):
    calls = []

    def add(request):
        calls.append(request)
        info = trn_pb.ImageInfo(image_id="img-1", labeled=True)
        if request.image_class:
            info.image_class = 0
        return info

    def list_images(request):
        return trn_pb.ImageList(images=[
            trn_pb.ImageInfo(image_id="cls", labeled=True, image_class=0),
            trn_pb.ImageInfo(image_id="box", labeled=True, box_count=2),
            trn_pb.ImageInfo(image_id="new"),
        ])

    monkeypatch.setattr(images, "clients",
                        SimpleNamespace(training=SimpleNamespace(
                            AddDatasetImage=add, ListImages=list_images)))
    return calls


class TestImageUploadBound:
    def test_an_oversized_image_is_refused_with_413(self, client,
                                                    training_stub,
                                                    monkeypatch):
        monkeypatch.setattr(settings, "MAX_IMAGE_UPLOAD_BYTES", 1024)
        data = {"file": (io.BytesIO(b"j" * 2048), "frame.jpg")}
        resp = client.post("/api/v1/training/datasets/d1/images", data=data)
        assert resp.status_code == 413
        assert training_stub == [], "nothing may be relayed over the cap"

    def test_a_normal_image_is_relayed(self, client, training_stub,
                                       monkeypatch):
        monkeypatch.setattr(settings, "MAX_IMAGE_UPLOAD_BYTES", 1024)
        data = {"file": (io.BytesIO(b"j" * 100), "frame.jpg")}
        resp = client.post("/api/v1/training/datasets/d1/images", data=data)
        assert resp.status_code == 201
        assert training_stub[0].jpeg == b"j" * 100


class TestImageClassPreLabel:
    def test_the_image_class_is_relayed_by_name(self, client, training_stub):
        data = {"file": (io.BytesIO(b"j" * 10), "frame.jpg"), "image_class": " cat "}
        resp = client.post("/api/v1/training/datasets/d1/images", data=data)
        assert resp.status_code == 201
        assert training_stub[0].image_class == "cat"
        assert list(training_stub[0].boxes) == []

    def test_without_one_the_upload_carries_none(self, client, training_stub):
        data = {"file": (io.BytesIO(b"j" * 10), "frame.jpg")}
        resp = client.post("/api/v1/training/datasets/d1/images", data=data)
        assert training_stub[0].image_class == ""
        assert resp.get_json()["image_class"] is None

    def test_the_stored_image_reports_its_class(self, client, training_stub):
        data = {"file": (io.BytesIO(b"j" * 10), "frame.jpg"), "image_class": "cat"}
        resp = client.post("/api/v1/training/datasets/d1/images", data=data)
        assert resp.get_json()["image_class"] == 0


class TestImageList:
    def test_each_image_carries_its_class(self, client, training_stub):
        # The device gallery names a classify image's class from it; without
        # the key a labeled classify image read as "0 boxes".
        resp = client.get("/api/v1/training/datasets/d1/images")
        assert resp.status_code == 200
        by_id = {i["image_id"]: i for i in resp.get_json()["images"]}
        assert by_id["cls"]["image_class"] == 0, "class 0 is a real class"
        assert by_id["box"]["image_class"] is None and by_id["box"]["box_count"] == 2
        assert by_id["new"] == {"image_id": "new", "created_at": 0.0, "labeled": False,
                                "box_count": 0, "replica": False, "image_class": None}


@pytest.fixture
def labels_stub(monkeypatch):
    sent = []

    def get_labels(request):
        return trn_pb.Labels(image_id=request.image_id, image_class=0)

    def set_labels(message):
        sent.append(message)
        return trn_pb.Result(success=True, message="saved")

    monkeypatch.setattr(images, "clients", SimpleNamespace(training=SimpleNamespace(
        GetLabels=get_labels, SetLabels=set_labels)))
    return sent


class TestLabelRoutes:
    URL = "/api/v1/training/datasets/d1/images/i1/labels"

    def test_get_reports_the_image_class(self, client, labels_stub):
        resp = client.get(self.URL)
        assert resp.get_json() == {"image_id": "i1", "boxes": [], "polygons": [],
                                   "image_class": 0}

    def test_put_an_image_class(self, client, labels_stub):
        resp = client.put(self.URL, json={"image_class": 2})
        assert resp.status_code == 200
        assert labels_stub[0].image_class == 2
        assert (labels_stub[0].dataset_id, labels_stub[0].image_id) == ("d1", "i1")

    def test_put_boxes_still_works(self, client, labels_stub):
        resp = client.put(self.URL, json={"boxes": [
            {"class_id": 1, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}]})
        assert resp.status_code == 200
        assert labels_stub[0].boxes[0].class_id == 1
        assert not labels_stub[0].HasField("image_class")

    @pytest.mark.parametrize("body", [{}, {"boxes": "x"}, {"image_class": "cat"}])
    def test_a_malformed_body_is_refused_and_not_relayed(self, client, labels_stub, body):
        resp = client.put(self.URL, json=body)
        assert resp.status_code == 400
        assert labels_stub == []


class TestPolygonRoutes:
    """Segmentation pre-labels and label edits carry rings."""

    UPLOAD = "/api/v1/training/datasets/d1/images"
    LABELS = "/api/v1/training/datasets/d1/images/i1/labels"
    TRIANGLE = [[0.1, 0.2], [0.5, 0.2], [0.5, 0.6]]

    def test_an_upload_relays_its_rings_by_class_name(self, client, training_stub):
        data = {"file": (io.BytesIO(b"j" * 100), "frame.jpg"),
                "polygons": '[{"class_name": "bolt", "instance": 1, '
                            '"points": [[0.1, 0.2], [0.5, 0.2], [0.5, 0.6]]}]'}
        resp = client.post(self.UPLOAD, data=data)
        assert resp.status_code == 201, resp.get_json()
        (polygon,) = training_stub[0].polygons
        assert (polygon.class_name, polygon.instance) == ("bolt", 1)
        assert list(polygon.points) == pytest.approx([0.1, 0.2, 0.5, 0.2, 0.5, 0.6])
        assert list(training_stub[0].boxes) == []

    def test_malformed_rings_are_refused_and_not_relayed(self, client, training_stub):
        data = {"file": (io.BytesIO(b"j" * 100), "frame.jpg"),
                "polygons": '[{"class_name": "bolt", "points": [[0.1, 0.2]]}]'}
        assert client.post(self.UPLOAD, data=data).status_code == 400
        assert training_stub == []

    def test_put_polygons(self, client, labels_stub):
        resp = client.put(self.LABELS, json={"polygons": [
            {"class_id": 1, "points": self.TRIANGLE}]})
        assert resp.status_code == 200
        (polygon,) = labels_stub[0].polygons
        assert polygon.class_id == 1 and len(polygon.points) == 6
        assert list(labels_stub[0].boxes) == []
