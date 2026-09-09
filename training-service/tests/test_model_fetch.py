"""Unit tests for fetching device-model checkpoints through the gateway."""
import pytest
import service.model_fetch as mf
from service.dataset_service import DatasetError
from service.model_fetch import (
    fetch_weights,
    list_models_with_weights,
    model_stem,
    validate_model_ref,
)


class TestValidateModelRef:
    @pytest.mark.parametrize("name", ["Teste9.engine", "x.pt", "y.onnx", "z.plan"])
    def test_listed_model_names_pass(self, name):
        assert validate_model_ref(f" {name} ") == name

    @pytest.mark.parametrize("bad", ["", "Teste.txt", "../x.engine", "a/b.engine",
                                     ".hidden.engine", "x\"y.engine", "x\ny.engine"])
    def test_rejects(self, bad):
        with pytest.raises(DatasetError):
            validate_model_ref(bad)

    def test_stem(self):
        assert model_stem("Teste.engine") == "Teste"
        assert model_stem("a.b.pt") == "a.b"


class FakeResponse:
    def __init__(self, status_code=200, body=None, chunks=(), text=""):
        self.status_code = status_code
        self._body = body
        self._chunks = list(chunks)
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise mf.requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._body

    def iter_content(self, chunk_size):
        return iter(self._chunks)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestListModelsWithWeights:
    def test_keeps_only_entries_with_a_checkpoint(self, monkeypatch):
        body = {"models": [{"name": "a.engine", "has_weights": True},
                           {"name": "b.engine", "has_weights": False},
                           {"name": "c.onnx"}, "junk"]}
        monkeypatch.setattr(mf.requests, "get",
                            lambda url, timeout: FakeResponse(body=body))
        assert list_models_with_weights("http://gw") == ["a.engine"]

    def test_gateway_error_propagates(self, monkeypatch):
        monkeypatch.setattr(mf.requests, "get",
                            lambda url, timeout: FakeResponse(status_code=503))
        with pytest.raises(mf.requests.RequestException):
            list_models_with_weights("http://gw")


class TestFetchWeights:
    def test_downloads_the_checkpoint_as_the_model_stem(self, monkeypatch, tmp_path):
        seen = {}

        def fake_get(url, stream, timeout):
            seen["url"] = url
            return FakeResponse(chunks=[b"ab", b"", b"cd"])

        monkeypatch.setattr(mf.requests, "get", fake_get)
        dest = tmp_path / "base"
        path = fetch_weights("http://gw", "Teste.engine", str(dest))
        assert path == str(dest / "Teste.pt")
        assert (dest / "Teste.pt").read_bytes() == b"abcd"
        assert seen["url"] == "http://gw/api/v1/model/Teste.engine/weights"
        assert [p.name for p in dest.iterdir()] == ["Teste.pt"], "no temp file left behind"

    def test_percent_encodes_the_model_name_in_the_url(self, monkeypatch, tmp_path):
        seen = {}

        def fake_get(url, stream, timeout):
            seen["url"] = url
            return FakeResponse(chunks=[b"x"])

        monkeypatch.setattr(mf.requests, "get", fake_get)
        fetch_weights("http://gw", "a b?#%.engine", str(tmp_path / "base"))
        assert seen["url"] == "http://gw/api/v1/model/a%20b%3F%23%25.engine/weights"

    def test_earlier_bases_are_dropped(self, monkeypatch, tmp_path):
        monkeypatch.setattr(mf.requests, "get",
                            lambda url, stream, timeout: FakeResponse(chunks=[b"new"]))
        dest = tmp_path / "base"
        dest.mkdir()
        (dest / "Old.pt").write_bytes(b"old")
        (dest / "notes.txt").write_text("kept: not a checkpoint")
        # ultralytics' AMP check downloads its nano model into the trainer's
        # cwd (= this directory); it must survive so it is not re-fetched.
        (dest / "yolo26n.pt").write_bytes(b"amp check asset")
        fetch_weights("http://gw", "Teste.engine", str(dest))
        assert sorted(p.name for p in dest.iterdir()) == ["Teste.pt", "notes.txt", "yolo26n.pt"]

    def test_model_without_checkpoint_is_a_dataset_error(self, monkeypatch, tmp_path):
        monkeypatch.setattr(mf.requests, "get",
                            lambda url, stream, timeout: FakeResponse(status_code=404))
        with pytest.raises(DatasetError, match="no training checkpoint"):
            fetch_weights("http://gw", "nope.engine", str(tmp_path))

    def test_empty_download_is_rejected(self, monkeypatch, tmp_path):
        monkeypatch.setattr(mf.requests, "get",
                            lambda url, stream, timeout: FakeResponse(chunks=[]))
        with pytest.raises(DatasetError, match="empty"):
            fetch_weights("http://gw", "x.engine", str(tmp_path))
        assert not (tmp_path / "x.pt").exists()

    def test_name_is_validated_before_any_request(self, monkeypatch, tmp_path):
        monkeypatch.setattr(mf.requests, "get",
                            lambda *a, **k: pytest.fail("must not be called"))
        with pytest.raises(DatasetError):
            fetch_weights("http://gw", "../x.engine", str(tmp_path))
