# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Upload routes under application types: a model upload without a size
defers to its task's default, and a dataset ZIP carries the device's task to
the import."""
import io
from types import SimpleNamespace

import grpc
import inference_pb2 as inf_pb
import pytest
import training_pb2 as trn_pb
from flask import Flask
from gateway.controllers import api_bp
from gateway.controllers import models as models_controller
from gateway.training import datasets, training_bp


class _Unimplemented(grpc.RpcError):
    """What an inference-service that predates application types answers."""

    def code(self):
        return grpc.StatusCode.UNIMPLEMENTED

    def details(self):
        return "unknown method"


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(api_bp)
    app.register_blueprint(training_bp)
    return app.test_client()


@pytest.fixture
def model_uploads(monkeypatch):
    metas = []

    def upload(stream):
        chunks = list(stream)
        metas.append(chunks[0].meta)
        return inf_pb.UploadResult(ok=True, http_status=202, json='{"status": "converting"}')

    monkeypatch.setattr(models_controller, "clients",
                        SimpleNamespace(model=SimpleNamespace(UploadModel=upload)))
    monkeypatch.setattr(models_controller, "_publish_event", lambda *a, **k: None)
    return metas


class TestModelUploadSize:
    @pytest.mark.parametrize("form, imgsz", [
        ({}, 0),                    # the inference-service picks 224 or 640 by task
        ({"imgsz": ""}, 0),
        ({"imgsz": "abc"}, 0),
        ({"imgsz": "-5"}, 0),
        ({"imgsz": "224"}, 224),
        ({"imgsz": "1280"}, 1280),
    ])
    def test_a_missing_size_defers_to_the_task_default(self, client, model_uploads,
                                                       form, imgsz):
        data = {"file": (io.BytesIO(b"pt"), "best.pt"), **form}
        resp = client.post("/api/v1/model", data=data)
        assert resp.status_code == 202
        assert model_uploads[0].imgsz == imgsz


def _wire_datasets(monkeypatch, *, task="classify", application_error=None):
    metas = []

    def get_application(_):
        if application_error is not None:
            raise application_error
        return inf_pb.ApplicationInfo(task=task, supported_tasks=["detect", "classify"])

    def upload(stream, timeout=None):
        chunks = list(stream)
        metas.append(chunks[0].meta)
        return trn_pb.DatasetUploadResult(
            success=True, message="imported",
            dataset=trn_pb.DatasetMeta(dataset_id="d1", name="pets", task=task or "detect"))

    monkeypatch.setattr(datasets, "clients", SimpleNamespace(
        management=SimpleNamespace(GetApplication=get_application),
        training=SimpleNamespace(UploadDataset=upload)))
    return metas


def _upload_zip(client):
    return client.post("/api/v1/training/datasets/upload",
                       data={"file": (io.BytesIO(b"PK"), "pets.zip"), "name": "pets"})


class TestDatasetUploadTask:
    def test_the_device_task_travels_with_the_archive(self, client, monkeypatch):
        metas = _wire_datasets(monkeypatch, task="classify")
        resp = _upload_zip(client)
        assert resp.status_code == 201
        assert (metas[0].name, metas[0].task) == ("pets", "classify")
        assert resp.get_json()["dataset"]["task"] == "classify"

    def test_refused_while_no_application_is_chosen(self, client, monkeypatch):
        metas = _wire_datasets(monkeypatch, task="")
        resp = _upload_zip(client)
        assert resp.status_code == 409
        assert metas == []

    def test_an_older_inference_service_imports_detection(self, client, monkeypatch):
        metas = _wire_datasets(monkeypatch, application_error=_Unimplemented())
        assert _upload_zip(client).status_code == 201
        assert metas[0].task == "detect"


class TestWeightsUploadTask:
    @staticmethod
    def _upload(client, monkeypatch, **form):
        from gateway.training import weights
        metas = []

        def upload(stream, timeout=None):
            chunks = list(stream)
            metas.append(chunks[0].meta)
            return trn_pb.WeightsUploadResult(success=True, weights_id="a" * 32, size=2)

        monkeypatch.setattr(weights, "clients",
                            SimpleNamespace(training=SimpleNamespace(UploadWeights=upload)))
        resp = client.post("/api/v1/training/weights",
                           data={"file": (io.BytesIO(b"pt"), "round.pt"), **form})
        return resp, metas

    def test_the_checkpoint_task_is_forwarded(self, client, monkeypatch):
        resp, metas = self._upload(client, monkeypatch, task=" classify ")
        assert resp.status_code == 201
        assert (metas[0].name, metas[0].task) == ("round.pt", "classify")

    def test_without_a_task_the_checkpoint_is_unchecked(self, client, monkeypatch):
        _, metas = self._upload(client, monkeypatch)
        assert metas[0].task == ""
