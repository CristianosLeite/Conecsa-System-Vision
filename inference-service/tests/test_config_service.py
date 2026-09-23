# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for ConfigService (get/update of the capture + inference config).

The generic patch is validated through the shared camera bounds, pushed to the
webcam-server first, and persisted only when the push was acknowledged —
the same contract as the dedicated camera path.
"""
from types import SimpleNamespace

import pytest
from api.services.config_service import ConfigService


class FakeVideo:
    def __init__(self, reachable=True):
        self.patches = []
        self.reachable = reachable

    def apply_webcam_server_config(self, patch):
        self.patches.append(patch)
        return self.reachable


class FakeSettings:
    def __init__(self):
        self.saved = 0

    def save(self):
        self.saved += 1


@pytest.fixture
def config():
    return SimpleNamespace(
        CAPTURE_DEVICE="/dev/video0",
        CAPTURE_RESOLUTION_X=640,
        CAPTURE_RESOLUTION_Y=480,
        CAPTURE_FRAMERATE=30,
        MODEL_PATH="/models/best.engine",
        CONFIDENCE_THRESHOLD=0.5,
    )


class TestLifecycleLock:
    def test_the_change_and_its_save_hold_the_model_lifecycle_lock(self, config):
        # A concurrent model activation must not swap the settings file between
        # the change and the save (the value would land in the wrong model).
        import threading
        lock = threading.RLock()
        held = []
        settings = SimpleNamespace(save=lambda: held.append(lock._is_owned()))  # type: ignore[attr-defined]
        svc = ConfigService(config, settings_service=settings, lifecycle_lock=lock)
        ok, _, status = svc.update_config({"confidence_threshold": 0.7})
        assert (ok, status) == (True, 200)
        assert held == [True] and config.CONFIDENCE_THRESHOLD == 0.7


class TestGetConfig:
    def test_maps_config_fields(self, config):
        svc = ConfigService(config)
        assert svc.get_config() == {
            "capture_device": "/dev/video0",
            "capture_resolution": [640, 480],
            "capture_framerate": 30,
            "model_path": "/models/best.engine",
            "confidence_threshold": 0.5,
        }


class TestUpdateConfig:
    def test_empty_data_is_rejected(self, config):
        ok, msg, status = ConfigService(config).update_config({})
        assert (ok, status) == (False, 400)
        assert msg == "No data provided"

    def test_device_path_is_stripped_to_camera_index(self, config):
        video = FakeVideo()
        ok, _, status = ConfigService(config, video_service=video).update_config(
            {"capture_device": "/dev/video2"}
        )
        assert (ok, status) == (True, 200)
        assert config.CAPTURE_DEVICE == "/dev/video2"
        assert video.patches == [{"camera_index": 2}]

    def test_bare_numeric_device_is_accepted(self, config):
        video = FakeVideo()
        ConfigService(config, video_service=video).update_config({"capture_device": "1"})
        assert video.patches == [{"camera_index": 1}]
        assert config.CAPTURE_DEVICE == "/dev/video1"

    def test_non_numeric_device_is_rejected_and_nothing_changes(self, config):
        video = FakeVideo()
        ok, msg, status = ConfigService(config, video_service=video).update_config(
            {"capture_device": "usb-cam"}
        )
        assert (ok, status) == (False, 400)
        assert "capture_device" in msg
        assert config.CAPTURE_DEVICE == "/dev/video0"
        assert video.patches == []

    def test_framerate_is_coerced_and_pushed(self, config):
        video = FakeVideo()
        ok, _, _ = ConfigService(config, video_service=video).update_config(
            {"capture_framerate": "60"}
        )
        assert ok is True
        assert config.CAPTURE_FRAMERATE == 60
        assert video.patches == [{"framerate": 60}]

    @pytest.mark.parametrize("bad", ["fast", 0, 241, -5, True])
    def test_out_of_range_framerate_is_rejected(self, config, bad):
        video = FakeVideo()
        ok, msg, status = ConfigService(config, video_service=video).update_config(
            {"capture_framerate": bad}
        )
        assert (ok, status) == (False, 400)
        assert "capture_framerate" in msg
        assert config.CAPTURE_FRAMERATE == 30
        assert video.patches == []

    def test_confidence_threshold_is_coerced_to_float(self, config):
        ok, _, _ = ConfigService(config).update_config({"confidence_threshold": "0.7"})
        assert ok is True
        assert config.CONFIDENCE_THRESHOLD == 0.7

    @pytest.mark.parametrize("bad", [-0.1, 1.5, "high"])
    def test_out_of_range_confidence_is_rejected(self, config, bad):
        ok, msg, status = ConfigService(config).update_config({"confidence_threshold": bad})
        assert (ok, status) == (False, 400)
        assert "confidence_threshold" in msg
        assert config.CONFIDENCE_THRESHOLD == 0.5

    def test_one_bad_field_rejects_the_whole_patch(self, config):
        video, settings = FakeVideo(), FakeSettings()
        ok, _, status = ConfigService(config, video, settings).update_config(
            {"capture_framerate": 60, "confidence_threshold": 7}
        )
        assert (ok, status) == (False, 400)
        assert config.CAPTURE_FRAMERATE == 30
        assert video.patches == []
        assert settings.saved == 0

    def test_an_unreachable_webcam_server_is_a_503_and_nothing_is_persisted(self, config):
        video, settings = FakeVideo(reachable=False), FakeSettings()
        ok, msg, status = ConfigService(config, video, settings).update_config(
            {"capture_framerate": 60, "confidence_threshold": 0.9}
        )
        assert (ok, status) == (False, 503)
        assert "webcam server" in msg
        assert config.CAPTURE_FRAMERATE == 30
        assert config.CONFIDENCE_THRESHOLD == 0.5
        assert settings.saved == 0

    def test_settings_are_persisted_after_update(self, config):
        settings = FakeSettings()
        ConfigService(config, settings_service=settings).update_config(
            {"confidence_threshold": 0.3}
        )
        assert settings.saved == 1

    def test_works_without_optional_collaborators(self, config):
        ok, msg, status = ConfigService(config).update_config({"capture_device": "/dev/video1"})
        assert (ok, status) == (True, 200)
        assert msg == "Configuration updated"
