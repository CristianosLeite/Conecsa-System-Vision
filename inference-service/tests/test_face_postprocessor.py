# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Face recognition postprocess: the YuNet decode bound by output name, the
strict score gate, minimum size, NMS, largest-first cap, saved areas, the
known/unknown decision at the match threshold, the arrival counter and the
clean frame; activation refusals for a missing gallery, a foreign embedder
and a missing output."""
from types import SimpleNamespace

import numpy as np
import pytest
from api.config import Config
from api.model_manager import TileMeta
from api.postprocess import _face_assets, _yunet
from api.postprocess.contract import ContractError
from api.postprocess.face import UNKNOWN, FacePostprocessor, gallery_file_for_model
from face_fixtures import FakeEmbedder, unit, write_gallery, yunet_details, yunet_outputs

ANA, BRUNO = unit(1, 0), unit(0, 1)
#: A 640×480 frame letterboxed into 640: 80 pad rows on top, scale 1.
META = TileMeta(1.0, 80, 640, 0, 0, 640, 480)
FRAME = np.full((480, 640, 3), 30, np.uint8)
# Shuffled binding order: decoding must never rely on the index.
ORDER = list(reversed(_yunet.OUTPUT_NAMES))


@pytest.fixture
def model(tmp_path, monkeypatch):
    monkeypatch.setattr(_face_assets, "embedder_sha256", lambda: "sha-test")
    path = str(tmp_path / "staff.engine")
    write_gallery(gallery_file_for_model(path), ["Ana", "Bruno #00ff00"], [ANA, BRUNO], [0, 1])
    return path


def _config(model, **overrides):
    cfg = Config()
    cfg.MODEL_PATH = model
    cfg.CONFIDENCE_THRESHOLD = 0.5
    cfg.OVERLAY_THRESHOLD = 0.45
    cfg.FACE_MATCH_THRESHOLD = 0.4
    cfg.FACE_MIN_SIZE_PX = 20
    cfg.FACE_MAX_FACES = 5
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _strategy(model, vectors=(), **overrides):
    embedder = FakeEmbedder(vectors)
    pp = FacePostprocessor(["Ana", "Bruno #00ff00"], _config(model, **overrides),
                           embedder=embedder)
    pp.bind_outputs(yunet_details(ORDER))
    return pp, embedder


def _run(pp, faces, frame=FRAME):
    return pp.process([yunet_outputs(faces, ORDER)], frame, [META], False)


def test_a_known_face_is_named_and_mapped_back_to_the_frame(model):
    pp, _ = _strategy(model, [unit(0.9, 0.1)])
    out = _run(pp, [((100, 180, 200, 300), 0.9)])
    assert out.count == 1 and len(out.items) == 1
    face = out.items[0]
    assert face.class_name == "Ana" and face.class_id == 0
    assert face.bbox == (100, 100, 200, 220)          # 80 pad rows removed
    assert face.center == (150, 160)
    assert face.confidence == pytest.approx(float(unit(0.9, 0.1)[0]), abs=1e-4)


def test_the_color_follows_the_person_and_unknown_is_gray(model):
    pp, _ = _strategy(model, [BRUNO, unit(-1, -1)])
    out = _run(pp, [((100, 180, 200, 300), 0.9), ((400, 180, 480, 280), 0.9)])
    by_name = {d.class_name: d for d in out.items}
    assert by_name["Bruno"].color == "#00ff00"
    assert by_name[UNKNOWN].color == "#808080"
    assert by_name[UNKNOWN].class_id == -1


def test_the_match_threshold_is_strict(model):
    # Ana's own embedding: the similarity is exactly 1, the highest there is.
    pp, _ = _strategy(model, [ANA], FACE_MATCH_THRESHOLD=1.0)
    assert _run(pp, [((100, 180, 200, 300), 0.9)]).items[0].class_name == UNKNOWN
    pp, _ = _strategy(model, [ANA], FACE_MATCH_THRESHOLD=0.999)
    assert _run(pp, [((100, 180, 200, 300), 0.9)]).items[0].class_name == "Ana"


def test_the_score_gate_is_strict(model):
    pp, embedder = _strategy(model, [], CONFIDENCE_THRESHOLD=0.5)
    out = _run(pp, [((100, 180, 200, 300), 0.5)])
    assert out.count == 0 and out.items == [] and embedder.calls == []


def test_faces_smaller_than_the_minimum_side_are_dropped(model):
    pp, _ = _strategy(model, [ANA], FACE_MIN_SIZE_PX=50)
    out = _run(pp, [((100, 180, 140, 300), 0.9), ((300, 180, 400, 300), 0.9)])
    assert [d.bbox for d in out.items] == [(300, 100, 400, 220)]


def test_the_cap_keeps_the_largest_faces(model):
    pp, embedder = _strategy(model, [ANA, BRUNO], FACE_MAX_FACES=2)
    faces = [((20, 180, 60, 220), 0.9), ((200, 180, 320, 300), 0.9),
             ((400, 180, 480, 260), 0.9)]
    out = _run(pp, faces)
    assert embedder.calls == [2]
    assert [d.bbox for d in out.items] == [(200, 100, 320, 220), (400, 100, 480, 180)]


def test_overlapping_faces_are_suppressed(model):
    pp, embedder = _strategy(model, [ANA])
    # Two anchors (different cells) describing nearly the same face.
    out = _run(pp, [((96, 160, 226, 290), 0.9), ((100, 164, 230, 294), 0.8)])
    assert out.count == 1 and embedder.calls == [1]


def test_saved_areas_filter_and_tag_faces(model):
    pp, _ = _strategy(model, [ANA])
    area = SimpleNamespace(id="door", label="Door", shape="rectangle", is_editing=False,
                           x=0.0, y=0.0, width=0.5, height=1.0)
    pp.set_areas([area])
    out = _run(pp, [((100, 180, 200, 300), 0.9), ((400, 180, 500, 300), 0.9)])
    assert len(out.items) == 1
    assert out.items[0].area == {"id": "door", "label": "Door", "shape": "rectangle"}


#: Known people per frame → what the frame adds to the counter.
ARRIVALS = [
    ([["Ana"], ["Ana"]], [1, 0]),
    ([["Ana"], ["Ana", "Bruno"]], [1, 1]),
    ([["Ana"], [], ["Ana"]], [1, 0, 1]),
    ([["unknown"], ["unknown"]], [0, 0]),
]


@pytest.mark.parametrize("frames,increments", ARRIVALS)
def test_the_counter_counts_arrivals_of_known_people(model, frames, increments):
    vectors = {"Ana": ANA, "Bruno": BRUNO, "unknown": unit(-1, -1)}
    boxes = [((100, 180, 200, 300), 0.9), ((400, 180, 500, 300), 0.9)]
    queue = [vectors[name] for people in frames for name in people]
    pp, _ = _strategy(model, queue)
    got = [_run(pp, boxes[:len(people)]).count_increment for people in frames]
    assert got == increments


def test_a_live_rename_is_not_an_arrival(model):
    pp, _ = _strategy(model, [ANA, ANA])
    assert _run(pp, [((100, 180, 200, 300), 0.9)]).count_increment == 1
    pp.set_class_labels(["Ana Souza", "Bruno #00ff00"])
    out = _run(pp, [((100, 180, 200, 300), 0.9)])
    assert out.items[0].class_name == "Ana Souza" and out.count_increment == 0


def test_reset_state_makes_a_present_person_arrive_again(model):
    pp, _ = _strategy(model, [ANA, ANA])
    assert _run(pp, [((100, 180, 200, 300), 0.9)]).count_increment == 1
    pp.reset_state()
    assert _run(pp, [((100, 180, 200, 300), 0.9)]).count_increment == 1


def test_the_clean_frame_is_never_drawn_on(model):
    pp, _ = _strategy(model, [ANA])
    frame = FRAME.copy()
    out = _run(pp, [((100, 180, 200, 300), 0.9)], frame)
    assert np.array_equal(frame, FRAME)
    assert not np.array_equal(out.image, FRAME)


def test_no_face_returns_the_frame_undrawn(model):
    pp, embedder = _strategy(model)
    out = _run(pp, [])
    assert out.count == 0 and out.count_increment == 0 and embedder.calls == []
    assert np.array_equal(out.image, FRAME)


def test_a_missing_output_is_refused(model):
    pp, _ = _strategy(model)
    details = [d for d in yunet_details() if d["name"] != "kps_16"]
    with pytest.raises(ContractError, match="kps_16"):
        pp.bind_outputs(details)


def test_a_model_without_a_gallery_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(_face_assets, "embedder_sha256", lambda: "sha-test")
    with pytest.raises(ContractError, match="no face gallery"):
        FacePostprocessor([], _config(str(tmp_path / "none.engine")), embedder=FakeEmbedder())


def test_a_gallery_of_another_embedder_is_refused(model, monkeypatch):
    monkeypatch.setattr(_face_assets, "embedder_sha256", lambda: "another")
    with pytest.raises(ContractError, match="another face embedder"):
        FacePostprocessor([], _config(model), embedder=FakeEmbedder())


def test_names_missing_from_the_classes_sidecar_come_from_the_gallery(model):
    pp = FacePostprocessor([], _config(model), embedder=FakeEmbedder())
    assert pp.class_labels == ["Ana", "Bruno #00ff00"]
