# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Face recognition building blocks: the YuNet decode and frame mapping, the
five-point alignment, the embedder input layout, the gallery file and its
matching, the face engine contract and the build-gated registry."""
import numpy as np
import pytest
from api import postprocess
from api.model_manager import ModelManager, TileMeta, letterbox_fit
from api.postprocess import _face_align, _face_assets, _face_gallery, _yunet
from api.postprocess.contract import ContractError, check, infer_task
from face_fixtures import face_landmarks, unit, write_gallery, yunet_details, yunet_outputs

INDICES = {name: i for i, name in enumerate(_yunet.OUTPUT_NAMES)}

# ── decode ──


def test_decode_recovers_box_landmarks_and_score():
    box = (96.0, 128.0, 224.0, 288.0)
    faces = _yunet.decode(yunet_outputs([(box, 0.8)]), INDICES, 640, 0.5)
    assert faces.boxes.tolist() == [pytest.approx(box, abs=1e-3)]
    assert faces.scores.tolist() == [pytest.approx(0.8)]
    assert np.allclose(faces.landmarks[0], face_landmarks(box), atol=1e-3)


def test_decode_scores_are_the_geometric_mean_of_cls_and_obj_clipped():
    outputs = yunet_outputs([((96, 128, 224, 288), 1.0)])
    outputs[INDICES["cls_32"]] *= 0.25
    outputs[INDICES["obj_32"]] *= 4.0            # clipped to 1
    faces = _yunet.decode(outputs, INDICES, 640, 0.0)
    assert faces.scores.tolist() == [pytest.approx(0.5)]


def test_decode_with_nothing_above_the_gate_is_empty():
    faces = _yunet.decode(yunet_outputs([((96, 128, 224, 288), 0.4)]), INDICES, 640, 0.4)
    assert faces.boxes.shape == (0, 4) and faces.landmarks.shape == (0, 5, 2)


@pytest.mark.parametrize("frame_w,frame_h,box", [
    (1280, 720, (499.0, 191.0, 794.0, 580.0)),    # the device camera
    (600, 800, (120.0, 150.0, 380.0, 520.0)),     # a portrait upload
    (640, 640, (100.0, 120.0, 260.0, 300.0)),     # square
    (2560, 720, (1500.0, 200.0, 1700.0, 460.0)),  # a stereo frame
])
def test_to_frame_is_the_inverse_of_the_face_letterbox(frame_w, frame_h, box):
    """A face at a known place in the frame must come back to that place.

    The forward transform is the pipeline's own ``letterbox_fit``, in every
    orientation, so neither a wrong scale direction nor a forgotten X band can
    pass: on a 1280x720 frame the factor is 2, and dividing instead of
    multiplying lands the box at a quarter of its coordinates.
    """
    _, scale, border_top, border_left = letterbox_fit(
        np.zeros((frame_h, frame_w, 3), np.uint8), 640)
    meta = TileMeta(scale, border_top, 640, 0, 0, frame_w, frame_h, border_left)
    x1, y1, x2, y2 = box
    to_input = lambda x, y: (x / scale + border_left, y / scale + border_top)  # noqa: E731
    corners = [*to_input(x1, y1), *to_input(x2, y2)]
    marks = np.array([to_input(x1 + 10, y1 + 10)] * 5, np.float32).reshape(1, 5, 2)

    mapped = _yunet.to_frame(
        _yunet.Faces(np.array([corners], np.float32), np.ones(1, np.float32), marks), meta)

    assert mapped.boxes[0].tolist() == pytest.approx(list(box), abs=1.0)
    assert mapped.landmarks[0, 0].tolist() == pytest.approx([x1 + 10, y1 + 10], abs=1.0)


def _face_manager(size: int = 640):
    """A ModelManager with face preprocessing and no TensorRT interpreter."""
    mm = ModelManager.__new__(ModelManager)
    mm._configure_preprocessing("face")
    mm.input_details = [{"index": 0, "name": "input", "shape": (1, 3, size, size),
                         "dtype": np.float32}]
    mm.input_size = size
    return mm


@pytest.mark.parametrize("frame_w,frame_h", [(1280, 720), (600, 800), (640, 640), (900, 900)])
def test_enrollment_photos_of_any_orientation_preprocess(frame_w, frame_h):
    """A portrait photo must not abort the build it belongs to.

    Enrollment feeds whatever the operator uploaded through the same manager
    the live pipeline uses; before ``letterbox_fit`` a portrait frame raised
    OpenCV's padding assertion and failed every photo in the package.
    """
    frame = np.zeros((frame_h, frame_w, 3), np.uint8)
    frame[:, :, 0] = 200          # blue channel, to prove BGR survives

    tensors, metas = _face_manager().preprocess_tiles(frame)

    assert tensors[0].shape == (1, 3, 640, 640) and tensors[0].dtype == np.float32
    assert tensors[0].max() > 1.0          # 0..255, not scaled like the YOLO paths
    assert tensors[0][0, 0].max() == 200   # channel 0 is still blue
    meta = metas[0]
    assert (meta.width, meta.height) == (frame_w, frame_h)
    assert meta.border_left >= 0 and meta.border_top >= 0
    assert (meta.border_left > 0) == (frame_h > frame_w)


@pytest.mark.parametrize("frame_w,frame_h", [(1280, 720), (600, 800), (640, 640), (2560, 720)])
def test_the_face_letterbox_fits_every_orientation(frame_w, frame_h):
    """A portrait photo used to abort the whole gallery build.

    ``letterbox_to_square`` scales the width to the input, so a taller-than-wide
    frame overflowed the square and OpenCV refused the negative padding.
    """
    image, scale, border_top, border_left = letterbox_fit(
        np.zeros((frame_h, frame_w, 3), np.uint8), 640)
    assert image.shape == (640, 640, 3)
    assert border_top >= 0 and border_left >= 0
    assert scale == pytest.approx(max(frame_w, frame_h) / 640, rel=0.01)


def test_to_frame_shifts_a_tile_into_frame_space():
    meta = TileMeta(1.0, 0, 640, 100, 50, 640, 640)
    faces = _yunet.Faces(np.array([[10, 20, 30, 40]], np.float32), np.ones(1, np.float32),
                         np.full((1, 5, 2), 60, np.float32))
    mapped = _yunet.to_frame(faces, meta)
    assert mapped.boxes[0].tolist() == [110, 70, 130, 90]
    assert mapped.landmarks[0, 0].tolist() == [160, 110]


def test_output_indices_are_found_by_name():
    order = list(reversed(_yunet.OUTPUT_NAMES))
    indices = _yunet.output_indices(yunet_details(order))
    assert indices["cls_8"] == order.index("cls_8")
    with pytest.raises(KeyError):
        _yunet.output_indices(yunet_details(order[1:]))

# ── alignment ──


def test_the_similarity_transform_maps_landmarks_onto_the_template():
    angle = np.deg2rad(20)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    src = (_face_align.TEMPLATE @ rotation.T) * 2.5 + np.array([300.0, 150.0])
    matrix = _face_align.similarity_matrix(src)
    mapped = src @ matrix[:, :2].T + matrix[:, 2]
    assert np.allclose(mapped, _face_align.TEMPLATE, atol=1e-6)


def test_align_crop_is_the_recognition_size():
    frame = np.zeros((480, 640, 3), np.uint8)
    crop = _face_align.align_crop(frame, face_landmarks((100, 100, 260, 300)))
    assert crop.shape == (112, 112, 3)


def test_the_embedder_input_is_rgb_nchw_in_0_255():
    crop = np.zeros((112, 112, 3), np.uint8)
    crop[..., 0] = 10   # blue
    crop[..., 2] = 200  # red
    batch = _face_align.embedder_input([crop])
    assert batch.shape == (1, 3, 112, 112) and batch.dtype == np.float32
    assert batch[0, 0, 0, 0] == 200 and batch[0, 2, 0, 0] == 10

# ── gallery ──


def test_a_gallery_round_trips_and_matches_by_each_persons_best_photo(tmp_path):
    path = str(tmp_path / "staff.gallery.npz")
    write_gallery(path, ["Ana", "Bruno"], [unit(1, 0), unit(0.6, 0.8), unit(0, 1)], [0, 0, 1])
    gallery = _face_gallery.load(path)
    assert gallery.names == ["Ana", "Bruno"] and gallery.dim == 2
    classes, sims = _face_gallery.match(gallery, np.stack([unit(0.6, 0.8), unit(0, 1)]))
    assert classes.tolist() == [0, 1]                # each person's own photo scores 1
    assert sims.tolist() == [pytest.approx(1.0), pytest.approx(1.0)]
    classes, sims = _face_gallery.match(gallery, np.stack([unit(0.7, 0.7)]))
    assert classes.tolist() == [0]                 # Ana's second photo is closest
    assert sims[0] == pytest.approx(float(unit(0.6, 0.8) @ unit(0.7, 0.7)), abs=1e-4)


def test_matching_no_faces_is_empty(tmp_path):
    path = str(tmp_path / "g.gallery.npz")
    write_gallery(path, ["Ana"], [unit(1, 0)], [0])
    classes, sims = _face_gallery.match(_face_gallery.load(path), np.zeros((0, 2)))
    assert classes.size == 0 and sims.size == 0


def test_broken_galleries_are_refused(tmp_path):
    with pytest.raises(_face_gallery.GalleryError, match="no face gallery"):
        _face_gallery.load(str(tmp_path / "missing.gallery.npz"))
    garbage = tmp_path / "garbage.gallery.npz"
    garbage.write_bytes(b"not a zip")
    with pytest.raises(_face_gallery.GalleryError):
        _face_gallery.load(str(garbage))
    wrong = str(tmp_path / "wrong.gallery.npz")
    write_gallery(wrong, ["Ana"], [unit(1, 0), unit(0, 1)], [0, 1])
    with pytest.raises(_face_gallery.GalleryError, match="fewer people"):
        _face_gallery.load(wrong)

# ── contract and registry ──


def _shapes():
    return [d["shape"] for d in yunet_details()]


def test_the_yunet_layout_passes_the_face_contract():
    check("face", _shapes(), [1, 3, 640, 640])
    assert infer_task(_shapes()) is None


@pytest.mark.parametrize("shapes,message", [
    (_shapes()[:-1], "has 12 outputs"),
    ([[1, 300, 6]], "has 12 outputs"),
    ([s[:2] + [7] for s in _shapes()], "carry"),
])
def test_other_layouts_are_refused_as_face(shapes, message):
    with pytest.raises(ContractError, match=message):
        check("face", shapes, [1, 3, 640, 640])


def test_anchors_must_fit_the_input_size():
    with pytest.raises(ContractError, match="do not fit"):
        check("face", _shapes(), [1, 3, 320, 320])


def test_a_face_detector_is_not_accepted_as_detect():
    with pytest.raises(ContractError):
        check("detect", _shapes(), [1, 3, 640, 640])


def test_face_is_supported_only_with_its_bundled_models(tmp_path, monkeypatch):
    monkeypatch.setenv("FACE_MODELS_DIR", str(tmp_path))
    assert "face" not in postprocess.supported_tasks()
    with pytest.raises(ContractError, match="cannot run 'face'"):
        postprocess.create("face", [], None)
    for name in (_face_assets.DETECTOR_ONNX, _face_assets.EMBEDDER_ONNX):
        (tmp_path / name).write_bytes(b"onnx")
    assert postprocess.supported_tasks() == ["detect", "classify", "segment", "face"]


def test_face_worker_ports_sit_past_the_labeling_worker(monkeypatch):
    monkeypatch.setenv("TENSORRT_WORKER_PORT", "6000")
    monkeypatch.delenv("TENSORRT_FACE_WORKER_PORT", raising=False)
    assert _face_assets.embed_worker_port() == 6017
    assert _face_assets.build_worker_ports() == (6018, 6019)


class TestPrivateWorkerPorts:
    """Every private worker sits past the live lanes whatever their count."""

    def _ports(self, monkeypatch, contexts, base=5501):
        from api.services.labeling_service import label_worker_port
        monkeypatch.setenv("TENSORRT_WORKER_PORT", str(base))
        monkeypatch.setenv("TENSORRT_CONTEXTS", str(contexts))
        for env in ("TENSORRT_LABEL_WORKER_PORT", "TENSORRT_FACE_WORKER_PORT",
                    "TENSORRT_FACE_BUILD_DETECTOR_PORT", "TENSORRT_FACE_BUILD_EMBEDDER_PORT"):
            monkeypatch.delenv(env, raising=False)
        return (label_worker_port(), _face_assets.embed_worker_port(),
                *_face_assets.build_worker_ports())

    def test_the_historical_layout_holds_up_to_sixteen_lanes(self, monkeypatch):
        assert self._ports(monkeypatch, 2) == (5517, 5518, 5519, 5520)
        assert self._ports(monkeypatch, 16) == (5517, 5518, 5519, 5520)

    def test_more_lanes_push_every_private_worker_past_them(self, monkeypatch):
        ports = self._ports(monkeypatch, 20)
        assert ports == (5521, 5522, 5523, 5524)
        lanes = {5501 + i for i in range(20)}
        assert not lanes & set(ports) and len(set(ports)) == 4

    def test_an_explicit_port_still_wins(self, monkeypatch):
        monkeypatch.setenv("TENSORRT_FACE_WORKER_PORT", "6000")
        monkeypatch.setenv("TENSORRT_CONTEXTS", "20")
        assert _face_assets.embed_worker_port() == 6000
