# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for the env-driven inference Config."""
import os

from api.config import Config


class TestConfigDefaults:
    def test_defaults(self, monkeypatch):
        for var in (
            "CONFIDENCE_THRESHOLD",
            "OVERLAY_THRESHOLD",
            "MODELS_DIR",
            "MODEL_PATH",
            "CLASSES_FILE_PATH",
            "SHM_NAME",
        ):
            monkeypatch.delenv(var, raising=False)
        cfg = Config()
        assert cfg.CONFIDENCE_THRESHOLD == 0.75
        assert cfg.OVERLAY_THRESHOLD == 0.45
        assert cfg.MODELS_DIR == "/data/models"
        assert cfg.MODEL_PATH == os.path.join("/data/models", "weights.engine")
        assert cfg.DEFAULT_LABELS == ["CLASS1", "CLASS2", "CLASS3"]
        assert cfg.SHM_NAME == "conecsa_frame_shm"

    def test_classes_file_derived_from_model_base(self, monkeypatch):
        monkeypatch.setenv("MODEL_PATH", "/data/models/custom.engine")
        monkeypatch.delenv("CLASSES_FILE_PATH", raising=False)
        cfg = Config()
        assert cfg.CLASSES_FILE_PATH == "/data/models/custom.txt"

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.3")
        monkeypatch.setenv("OVERLAY_THRESHOLD", "0.6")
        monkeypatch.setenv("SHM_NAME", "other_shm")
        cfg = Config()
        assert cfg.CONFIDENCE_THRESHOLD == 0.3
        assert cfg.OVERLAY_THRESHOLD == 0.6
        assert cfg.SHM_NAME == "other_shm"


class TestSetSegmentMaxInstances:
    def test_defaults_to_the_env_caps(self):
        assert Config().SEGMENT_MAX_INSTANCES is None

    def test_accepts_integers_in_range(self):
        cfg = Config()
        for limit in (1, 16, 255):
            assert cfg.set_segment_max_instances(limit) is True
            assert cfg.SEGMENT_MAX_INSTANCES == limit

    def test_rejects_anything_else(self):
        cfg = Config()
        for limit in (0, 256, -1, 8.5, True, "8", None):
            assert cfg.set_segment_max_instances(limit) is False
        assert cfg.SEGMENT_MAX_INSTANCES is None


class TestFaceSettings:
    def test_defaults(self, monkeypatch):
        for var in ("FACE_MATCH_THRESHOLD", "FACE_MIN_SIZE_PX", "FACE_MAX_FACES"):
            monkeypatch.delenv(var, raising=False)
        cfg = Config()
        assert cfg.FACE_MATCH_THRESHOLD == 0.363
        assert cfg.FACE_MIN_SIZE_PX == 40
        assert cfg.FACE_MAX_FACES == 5

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("FACE_MATCH_THRESHOLD", "0.5")
        monkeypatch.setenv("FACE_MIN_SIZE_PX", "80")
        monkeypatch.setenv("FACE_MAX_FACES", "3")
        cfg = Config()
        assert (cfg.FACE_MATCH_THRESHOLD, cfg.FACE_MIN_SIZE_PX, cfg.FACE_MAX_FACES) == (0.5, 80, 3)

    def test_an_out_of_range_or_unreadable_env_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv("FACE_MATCH_THRESHOLD", "2")
        monkeypatch.setenv("FACE_MIN_SIZE_PX", "-5")
        monkeypatch.setenv("FACE_MAX_FACES", "many")
        cfg = Config()
        assert (cfg.FACE_MATCH_THRESHOLD, cfg.FACE_MIN_SIZE_PX, cfg.FACE_MAX_FACES) == (
            0.363, 40, 5)

    def test_the_setters_accept_their_range(self):
        cfg = Config()
        for value in (0.0, 0.5, 1.0):
            assert cfg.set_face_match_threshold(value) is True
            assert cfg.FACE_MATCH_THRESHOLD == value
        for value in (0, 40, 1024):
            assert cfg.set_face_min_size(value) is True
            assert cfg.FACE_MIN_SIZE_PX == value
        for value in (1, 5, 20):
            assert cfg.set_face_max_faces(value) is True
            assert cfg.FACE_MAX_FACES == value

    def test_the_setters_reject_anything_else(self):
        cfg = Config()
        before = (cfg.FACE_MATCH_THRESHOLD, cfg.FACE_MIN_SIZE_PX, cfg.FACE_MAX_FACES)
        for value in (-0.1, 1.5, "0.5", True, None):
            assert cfg.set_face_match_threshold(value) is False
        for value in (-1, 1025, 40.5, True, "40", None):
            assert cfg.set_face_min_size(value) is False
        for value in (0, 21, 5.5, True, "5", None):
            assert cfg.set_face_max_faces(value) is False
        assert (cfg.FACE_MATCH_THRESHOLD, cfg.FACE_MIN_SIZE_PX, cfg.FACE_MAX_FACES) == before


class TestSetOverlayThreshold:
    def test_accepts_in_range(self):
        cfg = Config()
        assert cfg.set_overlay_threshold(0.5) is True
        assert cfg.OVERLAY_THRESHOLD == 0.5

    def test_accepts_boundaries(self):
        cfg = Config()
        assert cfg.set_overlay_threshold(0.0) is True
        assert cfg.set_overlay_threshold(1.0) is True

    def test_rejects_out_of_range(self):
        cfg = Config()
        cfg.OVERLAY_THRESHOLD = 0.45
        assert cfg.set_overlay_threshold(1.5) is False
        assert cfg.set_overlay_threshold(-0.1) is False
        assert cfg.OVERLAY_THRESHOLD == 0.45
