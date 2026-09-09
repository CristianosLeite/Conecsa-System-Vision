"""The shared camera/config validation layer and its two consumers (review M2).

Both API surfaces — the dedicated camera path and the generic config patch —
must reject the same values with a 400 and apply the same values the same
way; the gRPC servicer turns the status into INVALID_ARGUMENT / UNAVAILABLE.
"""
from types import SimpleNamespace

import grpc
import pytest
from api.config import Config
from api.services.config_service import ConfigService
from api.services.config_validation import (
    CAMERA_INT_BOUNDS,
    ConfigValidationError,
    parse_capture_device,
    validate_camera_patch,
)
from api.services.video_service import VideoService


class TestBounds:
    @pytest.mark.parametrize("key,lo,hi", [(k, *b) for k, b in CAMERA_INT_BOUNDS.items()])
    def test_every_int_field_is_bounded(self, key, lo, hi):
        assert validate_camera_patch({key: lo}).webcam[key] == lo
        assert validate_camera_patch({key: str(hi)}).webcam[key] == hi
        for bad in (lo - 1, hi + 1, "x", None, True):
            with pytest.raises(ConfigValidationError, match=key):
                validate_camera_patch({key: bad})

    @pytest.mark.parametrize("key,bad", [
        ("stereo_blend_alpha", 1.1), ("stereo_offset", -0.6), ("stereo_offset_y", "far")])
    def test_stereo_floats_are_bounded(self, key, bad):
        with pytest.raises(ConfigValidationError, match=key):
            validate_camera_patch({key: bad})

    def test_unknown_or_empty_bodies_are_rejected(self):
        with pytest.raises(ConfigValidationError, match="No recognised"):
            validate_camera_patch({"colour": "blue"})
        with pytest.raises(ConfigValidationError, match="No data"):
            validate_camera_patch({})

    def test_capture_device_forms(self):
        assert parse_capture_device("/dev/video3") == 3
        assert parse_capture_device("4") == 4
        assert parse_capture_device(5) == 5
        for bad in ("usb-cam", "/dev/media0", "/dev/video99", ""):
            with pytest.raises(ConfigValidationError, match="capture_device"):
                parse_capture_device(bad)


class FakeCodec:
    def __init__(self):
        self.stereo = []

    def set_stereo_config(self, enabled, alpha, offset, offset_y):
        self.stereo.append((enabled, alpha, offset, offset_y))


class _Video(VideoService):
    """VideoService whose SHM push is scripted."""

    def __init__(self, reachable=True):
        super().__init__(SimpleNamespace(), FakeCodec())
        self.pushed = []
        self.reachable = reachable

    def apply_webcam_server_config(self, patch):
        self.pushed.append(patch)
        return self.reachable


class TestCameraPath:
    def test_applies_the_webcam_patch_then_the_stereo_settings(self):
        video = _Video()
        ok, _, status = video.apply_camera_update(
            {"width": "1280", "height": 720, "stereo_enabled": "true", "stereo_offset": 0.1})
        assert (ok, status) == (True, 200)
        assert video.pushed == [{"width": 1280, "height": 720}]
        assert video._codec.stereo == [(True, None, 0.1, None)]

    def test_an_unreachable_camera_leaves_the_stereo_settings_alone(self):
        video = _Video(reachable=False)
        ok, _, status = video.apply_camera_update({"framerate": 30, "stereo_enabled": True})
        assert (ok, status) == (False, 503)
        assert video._codec.stereo == []

    @pytest.mark.parametrize("body", [{"width": 8}, {"camera_index": 64}, {"framerate": 0}])
    def test_out_of_range_values_change_nothing(self, body):
        video = _Video()
        ok, _, status = video.apply_camera_update(body)
        assert (ok, status) == (False, 400)
        assert video.pushed == []


class TestParity:
    """The same framerate is judged the same way on both surfaces."""

    @pytest.mark.parametrize("value,accepted", [(1, True), (240, True), (0, False),
                                                (241, False), ("x", False)])
    def test_framerate(self, value, accepted):
        video = _Video()
        config = Config()
        _, _, camera_status = video.apply_camera_update({"framerate": value})
        _, _, config_status = ConfigService(config, _Video()).update_config(
            {"capture_framerate": value})
        assert (camera_status == 200) is accepted
        assert camera_status == config_status


class TestConfidence:
    def test_config_setter_bounds_the_threshold(self):
        config = Config()
        assert config.set_confidence_threshold(0.25) is True
        assert config.CONFIDENCE_THRESHOLD == 0.25
        assert config.set_confidence_threshold(1.5) is False
        assert config.CONFIDENCE_THRESHOLD == 0.25


class _AbortingContext:
    """gRPC ServicerContext stand-in: abort raises like the real one."""

    def __init__(self):
        self.aborted = None

    def abort(self, code, details):
        self.aborted = (code, details)
        raise RuntimeError("aborted")


class TestServicerStatuses:
    def _servicer(self, video=None, config_service=None):
        from api.inference_grpc import ManagementControlServicer
        app = SimpleNamespace(video_service=video, config_service=config_service,
                              model_settings_service=None)
        return ManagementControlServicer(app)

    def _call(self, method, body):
        import inference_pb2 as pb
        ctx = _AbortingContext()
        with pytest.raises(RuntimeError):
            method(pb.ConfigJson(json=body), ctx)
        assert ctx.aborted is not None
        return ctx.aborted

    def test_validation_failures_are_invalid_argument(self):
        servicer = self._servicer(video=_Video())
        code, details = self._call(servicer.UpdateCamera, '{"framerate": 0}')
        assert code == grpc.StatusCode.INVALID_ARGUMENT
        assert "framerate" in details

    def test_an_unreachable_webcam_server_is_unavailable(self):
        servicer = self._servicer(config_service=ConfigService(Config(), _Video(False)))
        code, _ = self._call(servicer.UpdateConfig, '{"capture_framerate": 30}')
        assert code == grpc.StatusCode.UNAVAILABLE

    def test_malformed_json_is_invalid_argument(self):
        servicer = self._servicer(video=_Video())
        code, _ = self._call(servicer.UpdateCamera, "{not json")
        assert code == grpc.StatusCode.INVALID_ARGUMENT

    def test_success_returns_a_result(self):
        import inference_pb2 as pb
        servicer = self._servicer(video=_Video())
        result = servicer.UpdateCamera(pb.ConfigJson(json='{"gain": 10}'), _AbortingContext())
        assert result.success is True
