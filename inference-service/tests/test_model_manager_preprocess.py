# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Host-side tests for ModelManager preprocessing (no TensorRT interpreter).

``ModelManager.__init__`` creates a TensorRT interpreter (and, with
TENSORRT_CONTEXTS>1, worker subprocesses), so the manager is built with
``__new__`` + ``_configure_preprocessing`` and fake ``input_details``, which is
all ``preprocess_image`` depends on.
"""
import pathlib

import cv2
import numpy as np
import pytest
from api.model_manager import (
    ModelManager,
    center_crop_offsets,
    classify_resize_size,
    input_size_from_shape,
    letterbox_pad_from_env,
    letterbox_to_square,
    resize_interp_from_env,
    tiling_mode_from_env,
    tiling_overlap_from_env,
    tiling_tile_from_env,
)


def _manager(size: int, dtype: type = np.float32, task: str = "detect"):
    mm = ModelManager.__new__(ModelManager)
    mm._configure_preprocessing(task)
    mm.input_details = [{"index": 0, "name": "images", "shape": (1, 3, size, size),
                         "dtype": dtype}]
    mm.input_size = input_size_from_shape(mm.input_details[0]["shape"])
    return mm


def _frame(h, w, value=200):
    return np.full((h, w, 3), value, dtype=np.uint8)


class TestEnvKnobs:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("INFER_LETTERBOX_PAD", raising=False)
        monkeypatch.delenv("INFER_RESIZE_INTERP", raising=False)
        assert letterbox_pad_from_env() == 0
        assert resize_interp_from_env() == cv2.INTER_NEAREST

    def test_pad_and_interp_are_read(self, monkeypatch):
        monkeypatch.setenv("INFER_LETTERBOX_PAD", "114")
        monkeypatch.setenv("INFER_RESIZE_INTERP", "area")
        assert letterbox_pad_from_env() == 114
        assert resize_interp_from_env() == cv2.INTER_AREA
        monkeypatch.setenv("INFER_RESIZE_INTERP", "Linear")
        assert resize_interp_from_env() == cv2.INTER_LINEAR

    @pytest.mark.parametrize("raw", ["abc", "-1", "256", ""])
    def test_invalid_pad_falls_back(self, monkeypatch, raw):
        monkeypatch.setenv("INFER_LETTERBOX_PAD", raw)
        assert letterbox_pad_from_env() == 0

    @pytest.mark.parametrize("raw", ["cubic", "", "2"])
    def test_invalid_interp_falls_back(self, monkeypatch, raw):
        monkeypatch.setenv("INFER_RESIZE_INTERP", raw)
        assert resize_interp_from_env() == cv2.INTER_NEAREST


class TestInputSizeFromShape:
    @pytest.mark.parametrize("size", [640, 1280])
    def test_channels_first_uses_spatial_dim(self, size):
        # The old code read shape[1] and reported 3 for every NCHW engine.
        assert input_size_from_shape((1, 3, size, size)) == size

    def test_channels_last_keeps_index_one(self):
        assert input_size_from_shape((1, 320, 320, 3)) == 320

    @pytest.mark.parametrize("size", [640, 1280])
    def test_finalize_interpreter_setup_reports_size(self, size):
        class FakeInterpreter:
            """Only the two detail getters are exercised by the setup step."""

            def get_input_details(self):
                return [{"index": 0, "name": "images", "shape": (1, 3, size, size),
                         "dtype": np.float32}]

            def get_output_details(self):
                return [{"index": 1, "name": "output0", "shape": (1, 300, 6),
                         "dtype": np.float32}]

            def allocate_tensors(self) -> None:
                raise AssertionError("not expected during setup")

            def set_tensor(self, tensor_index: int, value: np.ndarray) -> None:
                raise AssertionError("not expected during setup")

            def invoke(self) -> None:
                raise AssertionError("not expected during setup")

            def get_tensor(self, tensor_index: int) -> np.ndarray:
                raise AssertionError("not expected during setup")

        mm = ModelManager.__new__(ModelManager)
        mm._finalize_interpreter_setup(FakeInterpreter())
        assert mm.input_size == size


class TestLetterboxToSquare:
    @pytest.mark.parametrize("size,frame_hw,expected_top", [
        (640, (720, 1280), 140),
        (640, (480, 640), 80),
        (1280, (720, 1280), 280),
        (1280, (480, 640), 160),
    ])
    def test_geometry(self, size, frame_hw, expected_top):
        h, w = frame_hw
        img, scale, top = letterbox_to_square(_frame(h, w), size)
        assert img.shape == (size, size, 3)
        assert top == expected_top
        resized_h = int(size * h / w)
        assert scale == pytest.approx(h / resized_h)

    def test_pad_rows_take_the_pad_value(self):
        img, _, top = letterbox_to_square(_frame(720, 1280), 640, pad_value=114)
        assert np.all(img[:top] == 114)
        assert np.all(img[-top:] == 114)
        assert np.all(img[top:-top] == 200)


class TestPreprocessImage:
    @pytest.mark.parametrize("size", [640, 1280])
    @pytest.mark.parametrize("frame_hw,top_by_size", [
        ((720, 1280), {640: 140, 1280: 280}),
        ((480, 640), {640: 80, 1280: 160}),
    ])
    def test_tensor_shape_range_and_geometry(self, monkeypatch, size, frame_hw, top_by_size):
        monkeypatch.delenv("INFER_LETTERBOX_PAD", raising=False)
        monkeypatch.delenv("INFER_RESIZE_INTERP", raising=False)
        h, w = frame_hw
        mm = _manager(size)

        tensor, scale, top, actual = mm.preprocess_image(_frame(h, w))

        assert tensor.shape == (1, 3, size, size)
        assert tensor.dtype == np.float32
        assert tensor.min() >= 0.0 and tensor.max() <= 1.0
        assert actual == size
        assert top == top_by_size[size]
        assert scale == pytest.approx(h / int(size * h / w))

    def test_default_pad_is_black(self, monkeypatch):
        monkeypatch.delenv("INFER_LETTERBOX_PAD", raising=False)
        mm = _manager(640)
        tensor, _, top, _ = mm.preprocess_image(_frame(720, 1280))
        assert np.all(tensor[0, :, :top, :] == 0.0)
        assert np.all(tensor[0, :, -top:, :] == 0.0)
        assert np.all(tensor[0, :, top:-top, :] == pytest.approx(200 / 255.0))

    def test_pad_114_when_configured(self, monkeypatch):
        monkeypatch.setenv("INFER_LETTERBOX_PAD", "114")
        mm = _manager(1280)
        tensor, _, top, _ = mm.preprocess_image(_frame(720, 1280))
        assert top == 280
        assert np.all(tensor[0, :, :top, :] == pytest.approx(114 / 255.0))
        assert np.all(tensor[0, :, -top:, :] == pytest.approx(114 / 255.0))

    def test_env_is_read_once_at_configure_time(self, monkeypatch):
        monkeypatch.setenv("INFER_LETTERBOX_PAD", "114")
        mm = _manager(640)
        monkeypatch.setenv("INFER_LETTERBOX_PAD", "0")
        tensor, _, top, _ = mm.preprocess_image(_frame(720, 1280))
        assert np.all(tensor[0, :, :top, :] == pytest.approx(114 / 255.0))

    @pytest.mark.parametrize("interp", ["nearest", "linear", "area"])
    def test_interpolation_knob_is_accepted(self, monkeypatch, interp):
        monkeypatch.setenv("INFER_RESIZE_INTERP", interp)
        mm = _manager(640)
        # A frame with structure so interpolation actually has to do work.
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:, ::2] = 255
        tensor, scale, top, actual = mm.preprocess_image(frame)
        assert tensor.shape == (1, 3, 640, 640)
        assert (top, actual) == (140, 640)
        assert scale == pytest.approx(2.0)

    def test_bgr_to_rgb_channel_order(self, monkeypatch):
        monkeypatch.delenv("INFER_LETTERBOX_PAD", raising=False)
        mm = _manager(640)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:, :, 2] = 255  # pure red in BGR
        tensor, _, top, _ = mm.preprocess_image(frame)
        assert np.all(tensor[0, 0, top:-top, :] == 1.0)  # R first in RGB
        assert np.all(tensor[0, 1, top:-top, :] == 0.0)
        assert np.all(tensor[0, 2, top:-top, :] == 0.0)

    def test_uint8_engine_keeps_raw_bytes(self):
        mm = _manager(640, dtype=np.uint8)
        tensor, _, _, _ = mm.preprocess_image(_frame(720, 1280))
        assert tensor.dtype == np.uint8
        assert tensor.max() == 200


class TestTilingKnobs:
    def test_defaults_are_grid_with_an_auto_tile(self, monkeypatch):
        for var in ("TILING_MODE", "TILING_TILE", "TILING_OVERLAP"):
            monkeypatch.delenv(var, raising=False)
        assert tiling_mode_from_env() == "grid"
        assert tiling_tile_from_env() is None  # auto: the frame's short side
        assert tiling_overlap_from_env() == 0.2

    @pytest.mark.parametrize("raw,expected", [("auto", None), ("AUTO", None), ("720", 720)])
    def test_tile_is_auto_or_pixels(self, monkeypatch, raw, expected):
        monkeypatch.setenv("TILING_TILE", raw)
        assert tiling_tile_from_env() == expected

    def test_grid_is_read_case_insensitively(self, monkeypatch):
        monkeypatch.setenv("TILING_MODE", "Grid")
        assert tiling_mode_from_env() == "grid"

    @pytest.mark.parametrize("raw", ["on", "sahi", "1", ""])
    def test_unknown_mode_falls_back_to_grid(self, monkeypatch, raw):
        monkeypatch.setenv("TILING_MODE", raw)
        assert tiling_mode_from_env() == "grid"

    @pytest.mark.parametrize("raw", ["abc", "0", "-64", ""])
    def test_invalid_tile_falls_back_to_auto(self, monkeypatch, raw):
        monkeypatch.setenv("TILING_TILE", raw)
        assert tiling_tile_from_env() is None

    @pytest.mark.parametrize("raw", ["abc", "1.0", "-0.1", ""])
    def test_invalid_overlap_falls_back(self, monkeypatch, raw):
        monkeypatch.setenv("TILING_OVERLAP", raw)
        assert tiling_overlap_from_env() == 0.2


class TestPreprocessTiles:
    def test_off_wraps_preprocess_image_as_single_full_frame_tile(self, monkeypatch):
        monkeypatch.setenv("TILING_MODE", "off")
        mm = _manager(640)
        frame = _frame(720, 1280)
        tensors, metas = mm.preprocess_tiles(frame)
        assert not mm.tiling_active
        assert len(tensors) == 1 and len(metas) == 1
        meta = metas[0]
        assert (meta.ox, meta.oy, meta.width, meta.height) == (0, 0, 1280, 720)
        expected, scale, border_top, size = mm.preprocess_image(frame)
        assert (meta.scale, meta.border_top, meta.input_size) == (scale, border_top, size)
        np.testing.assert_array_equal(tensors[0], expected)

    @pytest.mark.parametrize("frame_hw,side,origins", [
        ((720, 1280), 720, [(0, 0), (560, 0)]),
        ((1080, 1920), 1080, [(0, 0), (840, 0)]),
        ((480, 640), 480, [(0, 0), (160, 0)]),
    ])
    def test_grid_defaults_give_two_columns_on_any_16x9_or_4x3_frame(
            self, monkeypatch, frame_hw, side, origins):
        # Default mode with the auto tile: the tile spans the frame's short
        # side, so the grid is two columns whatever the camera's pixel count
        # (the trailing tile slides back to the frame edge).
        monkeypatch.delenv("TILING_MODE", raising=False)
        monkeypatch.delenv("TILING_TILE", raising=False)
        mm = _manager(640)
        tensors, metas = mm.preprocess_tiles(_frame(*frame_hw))
        assert mm.tiling_active
        assert len(tensors) == 2
        assert [(m.ox, m.oy) for m in metas] == origins
        for tensor, meta in zip(tensors, metas, strict=True):
            assert tensor.shape == (1, 3, 640, 640)
            assert (meta.width, meta.height) == (side, side)
            assert meta.border_top == 0  # square crop: no letterbox bands
            assert meta.scale == pytest.approx(side / 640)
            assert meta.input_size == 640

    def test_explicit_pixel_tile_pins_the_grid(self, monkeypatch):
        monkeypatch.setenv("TILING_MODE", "grid")
        monkeypatch.setenv("TILING_TILE", "640")
        mm = _manager(640)
        tensors, metas = mm.preprocess_tiles(_frame(720, 1280))
        # Three columns x two rows, row-major — a pinned side ignores the aspect ratio.
        assert len(tensors) == 6
        assert [(m.ox, m.oy) for m in metas] == [
            (0, 0), (512, 0), (640, 0), (0, 80), (512, 80), (640, 80)]
        assert all((m.width, m.height) == (640, 640) for m in metas)

    def test_grid_on_a_square_frame_degenerates_to_one_tile(self, monkeypatch):
        monkeypatch.setenv("TILING_MODE", "grid")
        monkeypatch.delenv("TILING_TILE", raising=False)
        mm = _manager(640)
        tensors, metas = mm.preprocess_tiles(_frame(500, 500))
        assert len(tensors) == 1
        meta = metas[0]
        assert (meta.ox, meta.oy, meta.width, meta.height) == (0, 0, 500, 500)

    def test_pixel_tile_larger_than_the_frame_degenerates_to_one_tile(self, monkeypatch):
        monkeypatch.setenv("TILING_MODE", "grid")
        monkeypatch.setenv("TILING_TILE", "720")
        mm = _manager(640)
        tensors, metas = mm.preprocess_tiles(_frame(360, 640))
        assert len(tensors) == 1
        meta = metas[0]
        assert (meta.ox, meta.oy, meta.width, meta.height) == (0, 0, 640, 360)


FIXTURES = pathlib.Path(__file__).parent / "fixtures"
#: ultralytics 8.4.92 ``classify_transforms(32)`` on ``_parity_frame()`` (CHW, 0..1),
#: computed once on the workstation; CI has Pillow but no torch.
PARITY_FIXTURE = FIXTURES / "classify_preprocess_90x163_32.npy"


def _parity_frame():
    """The seeded 90x163 BGR noise frame the parity fixture was computed from.

    Noise is the worst case for a resize filter mismatch, and the 25 px crop
    excess exercises torchvision's half-to-even rounding (12.5 → 12).
    """
    return np.random.RandomState(7).randint(0, 256, size=(90, 163, 3), dtype=np.uint8)


class TestClassifyPreprocess:
    """Classification input = ultralytics ``classify_transforms``."""

    @pytest.mark.parametrize("w, h, size, expected", [
        (1280, 720, 224, (398, 224)),
        (720, 1280, 224, (224, 398)),
        (500, 500, 224, (224, 224)),
        (163, 90, 32, (57, 32)),
        (224, 224, 224, (224, 224)),
    ])
    def test_the_short_side_becomes_the_input_size(self, w, h, size, expected):
        assert classify_resize_size(w, h, size) == expected

    @pytest.mark.parametrize("excess, offset", [
        (0, 0), (1, 0), (2, 1), (3, 2), (5, 2), (25, 12), (174, 87),
    ])
    def test_crop_offsets_round_half_to_even_like_torchvision(self, excess, offset):
        assert center_crop_offsets(224 + excess, 224, 224) == (offset, 0)
        assert center_crop_offsets(224, 224 + excess, 224) == (0, offset)

    def test_matches_the_pinned_ultralytics_transform(self):
        mm = _manager(32, task="classify")
        tensor, scale, border_top, size = mm.preprocess_image(_parity_frame())
        assert tensor.shape == (1, 3, 32, 32) and tensor.dtype == np.float32
        assert (scale, border_top, size) == (1.0, 0, 32)
        np.testing.assert_allclose(tensor[0], np.load(PARITY_FIXTURE), atol=1e-6)

    def test_the_fixture_still_matches_the_installed_ultralytics(self):
        pytest.importorskip("torchvision")
        augment = pytest.importorskip("ultralytics.data.augment")
        from PIL import Image

        rgb = cv2.cvtColor(_parity_frame(), cv2.COLOR_BGR2RGB)
        reference = augment.classify_transforms(32)(Image.fromarray(rgb)).numpy()
        np.testing.assert_allclose(np.load(PARITY_FIXTURE), reference, atol=1e-6)

    def test_a_720p_frame_is_center_cropped_not_letterboxed(self):
        frame = np.zeros((720, 1280, 3), np.uint8)
        frame[:, :100] = 255  # a band the center crop (from x=87 of 398) cuts away
        tensor, _, border_top, _ = _manager(224, task="classify").preprocess_image(frame)
        assert tensor.shape == (1, 3, 224, 224) and border_top == 0
        assert float(tensor.max()) == 0.0

    def test_classification_is_never_tiled(self, monkeypatch):
        monkeypatch.setenv("TILING_MODE", "grid")
        mm = _manager(224, task="classify")
        assert mm.tiling_active is False
        tensors, metas = mm.preprocess_tiles(_frame(720, 1280))
        assert len(tensors) == 1
        # The trailing 0 is border_left: only the face path pads the X axis.
        assert metas[0] == (1.0, 0, 224, 0, 0, 1280, 720, 0)

    def test_detection_keeps_the_letterbox(self):
        _, _, border_top, _ = _manager(640).preprocess_image(_frame(360, 640))
        assert border_top == 140
