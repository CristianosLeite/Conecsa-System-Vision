# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""The .pt → .onnx converter's pure helpers (the export itself needs ultralytics)."""
import pytest
from api._pt_onnx_converter import checkpoint_imgsz, checkpoint_metrics


@pytest.mark.parametrize("ckpt, size", [
    ({"train_args": {"imgsz": 224}}, 224),
    ({"train_args": {"imgsz": 640, "epochs": 5}}, 640),
    ({"train_args": {"imgsz": [320, 640]}}, 640),
    ({"train_args": {"imgsz": []}}, None),
    ({"train_args": {"imgsz": 0}}, None),
    ({"train_args": {"imgsz": True}}, None),
    ({"train_args": {"imgsz": "640"}}, None),
    ({"train_args": {"imgsz": [320, "640"]}}, None),
    ({"train_args": {}}, None),
    ({"train_args": None}, None),
    ({}, None),
    (None, None),
])
def test_checkpoint_imgsz_reads_the_training_size(ckpt, size):
    assert checkpoint_imgsz(ckpt) == size


def test_checkpoint_metrics_reads_the_validation_scores():
    ckpt = {"train_metrics": {"metrics/mAP50-95(B)": 0.62, "metrics/mAP50-95(M)": 0.55,
                              "metrics/accuracy_top1": 0.9, "val/box_loss": 1.2,
                              "fitness": 0.6}}
    assert checkpoint_metrics(ckpt) == {"map50_95_box": 0.62, "map50_95_mask": 0.55,
                                        "accuracy_top1": 0.9}


@pytest.mark.parametrize("ckpt", [
    {"train_metrics": {"metrics/mAP50-95(B)": float("nan")}},
    {"train_metrics": {"metrics/mAP50-95(B)": 1.5}},
    {"train_metrics": {"metrics/mAP50-95(B)": -0.1}},
    {"train_metrics": {"metrics/mAP50-95(B)": "high"}},
    {"train_metrics": {"metrics/mAP50-95(B)": None}},
    {"train_metrics": {"metrics/mAP50-95(B)": True}},
    {"train_metrics": {"metrics/mAP50-95(B)": 10 ** 400}},
    {"train_metrics": None},
    {"train_args": {"imgsz": 640}},
    None,
])
def test_checkpoint_metrics_drops_what_is_not_a_score(ckpt):
    assert checkpoint_metrics(ckpt) == {}
