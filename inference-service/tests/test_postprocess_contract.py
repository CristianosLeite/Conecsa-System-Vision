# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Engine output contract per task and the strategy registry."""
import pytest
from api import postprocess
from api.config import Config
from api.postprocess.contract import ContractError, check, check_probabilities, infer_task
from conecsa_common.tasks import TASKS

_SEG_OK = [[1, 300, 38], [1, 32, 160, 160]]


class TestInferTask:
    @pytest.mark.parametrize("shapes, task", [
        ([[1, 300, 6]], "detect"),
        ([[1, 84, 8400]], "detect"),
        ([[1, 5]], "classify"),
        (_SEG_OK, "segment"),
        (list(reversed(_SEG_OK)), "segment"),
        ([[1, 300, 6], [1, 300, 6]], None),
        ([[1, 2, 3, 4]], None),
        ([], None),
    ])
    def test_ranks_decide(self, shapes, task):
        assert infer_task(shapes) == task


class TestDetect:
    @pytest.mark.parametrize("shape", [
        [1, 300, 6],        # end-to-end (YOLO26)
        [1, 7, 6],          # end-to-end, few rows
        [1, 84, 8400],      # legacy one-to-many, features first
        [1, 8400, 84],      # legacy, detections first
        [1, 5, 8400],       # single-class legacy
        [-1, 300, 6],       # dynamic batch
    ])
    def test_accepts_today_s_layouts(self, shape):
        check("detect", [shape])

    def test_refuses_a_classification_head_and_says_so(self):
        with pytest.raises(ContractError, match="looks like a 'classify' model"):
            check("detect", [[1, 10]])

    def test_refuses_two_outputs(self):
        with pytest.raises(ContractError, match="looks like a 'segment' model"):
            check("detect", _SEG_OK)

    def test_refuses_too_few_features(self):
        with pytest.raises(ContractError, match="not a YOLO detection output"):
            check("detect", [[1, 4, 8400]])


class TestClassify:
    def test_accepts_one_probability_row(self):
        check("classify", [[1, 5]])

    def test_needs_two_classes(self):
        with pytest.raises(ContractError, match="at least 2 classes"):
            check("classify", [[1, 1]])

    def test_refuses_a_detection_head(self):
        with pytest.raises(ContractError, match="looks like a 'detect' model"):
            check("classify", [[1, 300, 6]])


class TestSegment:
    def test_accepts_the_end_to_end_head(self):
        check("segment", _SEG_OK, input_shape=[1, 3, 640, 640])

    def test_refuses_the_legacy_one_to_many_head(self):
        with pytest.raises(ContractError, match="end-to-end"):
            check("segment", [[1, 8400, 116], [1, 32, 160, 160]])

    def test_refuses_mismatched_mask_coefficients(self):
        with pytest.raises(ContractError, match="6 \\+ 32"):
            check("segment", [[1, 300, 40], [1, 32, 160, 160]])

    def test_refuses_prototypes_at_the_wrong_resolution(self):
        with pytest.raises(ContractError, match="expected 160"):
            check("segment", [[1, 300, 38], [1, 32, 80, 80]], input_shape=[1, 3, 640, 640])

    def test_refuses_a_single_output(self):
        with pytest.raises(ContractError, match="exactly two outputs"):
            check("segment", [[1, 300, 6]])


def test_unknown_task_is_refused():
    with pytest.raises(ContractError, match="unknown task"):
        check("pose", [[1, 300, 6]])


class TestProbabilities:
    @pytest.mark.parametrize("values", [
        [0.7, 0.2, 0.1],
        [1.0, 0.0],
        [0.5, 0.5045],       # fp16 rounding stays inside the tolerance
    ])
    def test_a_softmax_row_is_accepted(self, values):
        check_probabilities(values)

    @pytest.mark.parametrize("values", [
        [3.2, -1.0, 0.4],    # logits
        [0.2, 0.2],          # too low
        [0.5, 0.52],         # just outside ± 0.01
        [float("nan"), 0.5],
    ])
    def test_anything_else_is_refused(self, values):
        with pytest.raises(ContractError, match="sum to 1"):
            check_probabilities(values)


class TestRegistry:
    def test_supported_tasks_are_exactly_the_registered_strategies(self):
        assert postprocess.supported_tasks() == ["detect", "classify", "segment"]
        assert set(postprocess.supported_tasks()) <= set(TASKS)

    def test_detect_builds_the_detection_strategy(self):
        pp = postprocess.create("detect", ["a", "b"], Config())
        assert pp.task == "detect"
        assert pp.class_labels == ["a", "b"]

    def test_classify_builds_the_classification_strategy(self):
        pp = postprocess.create("classify", ["cat", "dog #00ff00"], Config())
        assert pp.task == "classify"
        assert pp.class_labels == ["cat", "dog"]

    def test_segment_builds_the_segmentation_strategy(self):
        pp = postprocess.create("segment", ["bolt", "nut #00ff00"], Config())
        assert pp.task == "segment"
        assert pp.class_labels == ["bolt", "nut"]

    @pytest.mark.parametrize("task", [t for t in TASKS
                                      if t not in ("detect", "classify", "segment")] + ["pose"])
    def test_every_other_task_is_refused(self, task):
        with pytest.raises(ContractError, match="cannot run"):
            postprocess.create(task, [], Config())

    @pytest.mark.parametrize("task, size", [
        ("classify", 224), ("detect", 640), ("segment", 640), (None, 640), ("", 640),
    ])
    def test_upload_input_size_defaults_follow_the_task(self, task, size):
        assert postprocess.default_imgsz(task) == size
