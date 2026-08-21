"""Tests for the streamed model upload (inference_grpc.UploadModel).

The chunks must go straight to a staged file on disk with a running byte cap
(never accumulated in memory) and the final name must appear atomically —
REFACTORING.md H7.
"""
import os
from types import SimpleNamespace

import api.inference_grpc as ig
import pytest
from api.config import Config
from api.services.model_service import ModelService


class StubDetectionService:
    def __init__(self):
        self.initialized = 0

    def stop(self):
        return False

    def initialize(self):
        self.initialized += 1

    def start(self):
        pass


class StubConversionService:
    pass


@pytest.fixture
def servicer(tmp_path):
    config = Config()
    config.MAX_MODEL_UPLOAD_BYTES = 1024  # small cap for the tests
    models = ModelService(config, str(tmp_path))
    models.attach_detection_service(StubDetectionService())
    models.attach_conversion_service(StubConversionService())
    app = SimpleNamespace(config=config, model_service=models)
    return ig.ModelControlServicer(app), str(tmp_path)


def _stream(filename, chunks, imgsz=640):
    yield ig.pb.ModelChunk(meta=ig.pb.ModelUploadMeta(filename=filename,
                                                      imgsz=imgsz))
    for chunk in chunks:
        yield ig.pb.ModelChunk(chunk=chunk)


def _leftover_parts(model_dir):
    return [f for f in os.listdir(model_dir) if f.endswith(".part")]


class TestUploadModel:
    def test_a_successful_upload_lands_atomically(self, servicer):
        control, model_dir = servicer
        result = control.UploadModel(
            _stream("m.engine", [b"abc", b"def"]), None)
        assert result.http_status == 200, result.json
        assert open(os.path.join(model_dir, "m.engine"), "rb").read() == b"abcdef"
        assert _leftover_parts(model_dir) == []

    def test_an_over_cap_upload_is_refused_and_leaves_nothing(self, servicer):
        control, model_dir = servicer
        result = control.UploadModel(
            _stream("big.engine", [b"x" * 600, b"y" * 600]), None)
        assert result.http_status == 413
        assert not os.path.exists(os.path.join(model_dir, "big.engine"))
        assert _leftover_parts(model_dir) == []

    def test_a_traversal_filename_is_refused_before_any_byte_lands(
            self, servicer, tmp_path):
        control, model_dir = servicer
        result = control.UploadModel(
            _stream("../escaped.engine", [b"payload"]), None)
        assert result.http_status == 400
        assert os.listdir(model_dir) == []
        assert not (tmp_path.parent / "escaped.engine").exists()

    def test_chunks_before_metadata_are_refused(self, servicer):
        control, model_dir = servicer

        def stream():
            yield ig.pb.ModelChunk(chunk=b"orphan")

        result = control.UploadModel(stream(), None)
        assert result.http_status == 400
        assert _leftover_parts(model_dir) == []

    def test_an_empty_stream_is_refused(self, servicer):
        control, model_dir = servicer
        result = control.UploadModel(iter(()), None)
        assert result.http_status == 400

    def test_the_active_model_cannot_be_overwritten(self, servicer):
        control, model_dir = servicer
        first = control.UploadModel(_stream("m.engine", [b"v1"]), None)
        assert first.http_status == 200
        control._app.model_service.select_model("m.engine")

        second = control.UploadModel(_stream("m.engine", [b"v2"]), None)
        assert second.http_status == 500
        assert "active" in second.json
        assert open(os.path.join(model_dir, "m.engine"), "rb").read() == b"v1"
        assert _leftover_parts(model_dir) == []
