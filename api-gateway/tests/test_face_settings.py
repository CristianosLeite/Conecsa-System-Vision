# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""POST /api/v1/face/settings and its status keys.

The route validates every field it is given before the inference-service is
called (numbers in range, never bools), forwards only the fields present,
announces a change on the ``thresholds`` event key, and ``/api/v1/status``
carries each value only when the inference-service reports it.
"""
from types import SimpleNamespace

import inference_pb2 as inf_pb
import pytest
from flask import Flask
from gateway.controllers import detection

PATH = "/api/v1/face/settings"


@pytest.fixture
def app():
    return Flask(__name__)


def _wire(monkeypatch, success=True):
    calls, published = [], []

    def set_face(request):
        calls.append({f.name: v for f, v in request.ListFields()})
        return SimpleNamespace(success=success, message="" if success else "out of range")

    monkeypatch.setattr(detection, "clients",
                        SimpleNamespace(detection=SimpleNamespace(SetFaceSettings=set_face)))

    def publish(resp, event, keys, **kwargs):
        published.append((event, keys, kwargs.get("data")))
        return resp

    monkeypatch.setattr(detection, "_publish_if_success", publish)
    return calls, published


@pytest.mark.parametrize("body", [
    {"match_threshold": 0.5},
    {"min_size_px": 0},
    {"max_faces": 20},
    {"match_threshold": 1, "min_size_px": 1024, "max_faces": 1},
])
def test_valid_fields_reach_the_service_and_are_announced(app, monkeypatch, body):
    calls, published = _wire(monkeypatch)
    with app.test_request_context(PATH, method="POST", json=body):
        resp = detection.set_face_settings()
    assert resp.status_code == 200
    assert calls == [pytest.approx(body)]
    assert published == [("thresholds_changed", ["status", "thresholds"],
                          {f"face_{k}": v for k, v in body.items()})]


@pytest.mark.parametrize("body", [
    {}, {"other": 1},
    {"match_threshold": -0.1}, {"match_threshold": 1.5}, {"match_threshold": "0.5"},
    {"match_threshold": True},
    {"min_size_px": -1}, {"min_size_px": 1025}, {"min_size_px": 40.5},
    {"max_faces": 0}, {"max_faces": 21}, {"max_faces": False},
    {"max_faces": 5, "min_size_px": -1},
])
def test_anything_else_is_refused_before_the_service(app, monkeypatch, body):
    calls, published = _wire(monkeypatch)
    with app.test_request_context(PATH, method="POST", json=body):
        resp = detection.set_face_settings()
    assert resp.status_code == 400
    assert calls == [] and published == []


@pytest.mark.parametrize("body", [[5], "5", 5, True])
def test_a_body_that_is_not_an_object_is_a_400(app, monkeypatch, body):
    calls, published = _wire(monkeypatch)
    with app.test_request_context(PATH, method="POST", json=body):
        resp = detection.set_face_settings()
    assert resp.status_code == 400
    assert calls == [] and published == []


def test_a_service_refusal_is_a_400_and_announces_nothing(app, monkeypatch):
    calls, published = _wire(monkeypatch, success=False)
    with app.test_request_context(PATH, method="POST", json={"max_faces": 3}):
        resp = detection.set_face_settings()
    assert resp.status_code == 400
    assert calls == [{"max_faces": 3}] and published == []


def test_status_reports_the_settings_only_when_the_service_does():
    status = inf_pb.StatusResponse(face_match_threshold=0.5, face_min_size_px=40,
                                   face_max_faces=5)
    assert detection._status_face(status) == {
        "face_match_threshold": 0.5, "face_min_size_px": 40, "face_max_faces": 5}
    assert detection._status_face(inf_pb.StatusResponse()) == {}
