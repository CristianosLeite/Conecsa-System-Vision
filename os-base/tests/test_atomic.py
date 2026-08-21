"""Tests for conecsa_common.atomic — power-cut-safe persistence."""
import json
import os

import pytest
from conecsa_common import atomic, atomic_write_bytes, atomic_write_json, read_json


class TestAtomicWrite:
    def test_round_trip(self, tmp_path):
        path = str(tmp_path / "state.json")
        atomic_write_json(path, {"model": "a.engine"})
        assert json.load(open(path)) == {"model": "a.engine"}

    def test_replaces_existing_content(self, tmp_path):
        path = str(tmp_path / "state.bin")
        atomic_write_bytes(path, b"old")
        atomic_write_bytes(path, b"new")
        assert open(path, "rb").read() == b"new"

    def test_the_old_file_survives_a_failed_write(self, tmp_path, monkeypatch):
        # A crash before the rename must leave the previous state intact and
        # no temp litter behind.
        path = str(tmp_path / "state.bin")
        atomic_write_bytes(path, b"old")

        def boom(fd):
            raise OSError("simulated power cut")

        monkeypatch.setattr(os, "fsync", boom)
        with pytest.raises(OSError):
            atomic_write_bytes(path, b"new")
        assert open(path, "rb").read() == b"old"
        leftovers = [f for f in os.listdir(tmp_path) if f.startswith(".tmp-")]
        assert leftovers == []

    def test_restrictive_mode(self, tmp_path):
        path = str(tmp_path / "secret.bin")
        atomic_write_bytes(path, b"s3cret", mode=0o600)
        assert oct(os.stat(path).st_mode & 0o777) == "0o600"

    def test_creates_missing_parent_directories(self, tmp_path):
        path = str(tmp_path / "nested" / "dir" / "state.json")
        atomic_write_json(path, [1, 2, 3])
        assert json.load(open(path)) == [1, 2, 3]


class TestReadJson:
    def test_missing_file_defaults_silently(self, tmp_path, caplog):
        result = read_json(str(tmp_path / "absent.json"), {"fresh": True})
        assert result == {"fresh": True}
        assert caplog.records == []

    def test_corrupt_file_is_quarantined_and_reported(self, tmp_path, caplog):
        path = str(tmp_path / "state.json")
        with open(path, "w") as fh:
            fh.write('{"trunc')
        with caplog.at_level("ERROR", logger=atomic.__name__):
            result = read_json(path, [])
        assert result == []
        assert not os.path.exists(path)
        assert os.path.exists(path + ".corrupt"), "evidence must survive"
        assert any("corrupt" in r.message for r in caplog.records)

    def test_valid_file_is_returned(self, tmp_path):
        path = str(tmp_path / "state.json")
        atomic_write_json(path, {"n": 1})
        assert read_json(path, None) == {"n": 1}


class TestLargePayloads:
    def test_a_multi_megabyte_write_round_trips_completely(self, tmp_path):
        # The write loop must persist every byte even when os.write returns
        # short (Copilot review finding on PR #66).
        path = str(tmp_path / "big.bin")
        payload = os.urandom(8 * 1024 * 1024)
        atomic_write_bytes(path, payload)
        assert open(path, "rb").read() == payload

    def test_short_writes_are_looped(self, tmp_path, monkeypatch):
        real_write = os.write

        def three_bytes_at_a_time(fd, view):
            return real_write(fd, bytes(view[:3]))

        monkeypatch.setattr(os, "write", three_bytes_at_a_time)
        path = str(tmp_path / "short.bin")
        atomic_write_bytes(path, b"0123456789abcdef")
        assert open(path, "rb").read() == b"0123456789abcdef"
