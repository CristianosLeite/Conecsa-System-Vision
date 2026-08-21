"""Unit tests for the labeled-image upload route (bounded read, 413 over cap)."""
import io
from types import SimpleNamespace

import pytest
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
        return SimpleNamespace(image_id="img-1", created_at="", labeled=True,
                               box_count=0, replica=False)

    monkeypatch.setattr(images, "clients",
                        SimpleNamespace(training=SimpleNamespace(
                            AddDatasetImage=add)))
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
