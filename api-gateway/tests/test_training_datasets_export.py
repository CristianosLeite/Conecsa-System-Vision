# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""GET /api/v1/training/datasets/<id>/export: the first chunk is pulled before
the response starts, so a training-service refusal (a face dataset never
leaves the device, a bad shard geometry) sets the HTTP status instead of
breaking a 200 stream."""
from types import SimpleNamespace

import grpc
import pytest
from flask import Flask
from gateway.grpc_clients import trn as trn_pb
from gateway.training import datasets, training_bp


class FakeRpcError(grpc.RpcError):
    def __init__(self, code, details):
        self._code, self._details = code, details

    def code(self):
        return self._code

    def details(self):
        return self._details


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(training_bp)
    return app.test_client()


def _wire(monkeypatch, chunks=(), error=None):
    def stream(request, timeout=None):
        if error is not None:
            raise error
        yield from (trn_pb.DatasetExportChunk(chunk=c) for c in chunks)

    monkeypatch.setattr(datasets, "clients", SimpleNamespace(training=SimpleNamespace(
        GetDataset=lambda r: trn_pb.DatasetInfo(dataset_id=r.dataset_id, name="staff"),
        ExportDataset=stream, ExportDatasetShard=stream)))


@pytest.mark.parametrize("query", ["", "?shards=2&index=0"])
def test_a_refused_export_answers_409_before_any_byte(client, monkeypatch, query):
    _wire(monkeypatch, error=FakeRpcError(
        grpc.StatusCode.FAILED_PRECONDITION, "Face datasets stay on the device"))
    resp = client.get(f"/api/v1/training/datasets/d1/export{query}")
    assert resp.status_code == 409
    assert resp.get_json() == {"error": "Face datasets stay on the device"}


@pytest.mark.parametrize("query,filename", [("", "staff.zip"),
                                            ("?shards=2&index=1", "staff-shard-1.zip")])
def test_an_export_streams_every_chunk(client, monkeypatch, query, filename):
    _wire(monkeypatch, chunks=[b"PK", b"rest"])
    resp = client.get(f"/api/v1/training/datasets/d1/export{query}")
    assert resp.status_code == 200
    assert resp.get_data() == b"PKrest"
    assert filename in resp.headers["Content-Disposition"]


def test_an_empty_export_is_still_a_200(client, monkeypatch):
    _wire(monkeypatch, chunks=[])
    resp = client.get("/api/v1/training/datasets/d1/export")
    assert resp.status_code == 200 and resp.get_data() == b""
