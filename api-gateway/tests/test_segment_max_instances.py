# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""POST /api/v1/segment/max_instances and its status key.

The route validates before the inference-service is called (an integer 1..255,
never a bool), announces a change on the ``thresholds`` event key the device
UI already refreshes on, and ``/api/v1/status`` carries the value only when the
inference-service reports one.
"""
from types import SimpleNamespace

import inference_pb2 as inf_pb
import pytest
from flask import Flask
from gateway.controllers import detection

PATH = "/api/v1/segment/max_instances"


@pytest.fixture
def app():
    return Flask(__name__)


def _wire(monkeypatch, success=True):
    calls, published = [], []

    def set_limit(request):
        calls.append(request.max_instances)
        return SimpleNamespace(success=success, message="" if success else "out of range")

    monkeypatch.setattr(detection, "clients",
                        SimpleNamespace(detection=SimpleNamespace(SetSegmentMaxInstances=set_limit)))

    def publish(resp, event, keys, **kwargs):
        published.append((event, keys, kwargs.get("data")))
        return resp

    monkeypatch.setattr(detection, "_publish_if_success", publish)
    return calls, published


@pytest.mark.parametrize("value", [1, 12, 255])
def test_a_valid_limit_reaches_the_service_and_is_announced(app, monkeypatch, value):
    calls, published = _wire(monkeypatch)
    with app.test_request_context(PATH, method="POST", json={"max_instances": value}):
        resp = detection.set_segment_max_instances()
    assert resp.status_code == 200
    assert resp.get_json()["max_instances"] == value
    assert calls == [value]
    assert published == [("thresholds_changed", ["status", "thresholds"],
                          {"segment_max_instances": value})]


@pytest.mark.parametrize("body", [{}, {"max_instances": 0}, {"max_instances": 256},
                                  {"max_instances": 8.5}, {"max_instances": "8"},
                                  {"max_instances": True}])
def test_anything_else_is_refused_before_the_service(app, monkeypatch, body):
    calls, published = _wire(monkeypatch)
    with app.test_request_context(PATH, method="POST", json=body):
        resp = detection.set_segment_max_instances()
    assert resp.status_code == 400
    assert calls == [] and published == []


@pytest.mark.parametrize("body", [[8], "8", 8, True])
def test_a_body_that_is_not_an_object_is_a_400(app, monkeypatch, body):
    calls, published = _wire(monkeypatch)
    with app.test_request_context(PATH, method="POST", json=body):
        resp = detection.set_segment_max_instances()
    assert resp.status_code == 400
    assert calls == [] and published == []


def test_a_service_refusal_is_a_400_and_announces_nothing(app, monkeypatch):
    calls, published = _wire(monkeypatch, success=False)
    with app.test_request_context(PATH, method="POST", json={"max_instances": 8}):
        resp = detection.set_segment_max_instances()
    assert resp.status_code == 400
    assert calls == [8] and published == []


def test_status_reports_the_limit_only_when_the_service_does():
    assert detection._status_segment(inf_pb.StatusResponse(segment_max_instances=16)) == {
        "segment_max_instances": 16}
    assert detection._status_segment(inf_pb.StatusResponse()) == {}
