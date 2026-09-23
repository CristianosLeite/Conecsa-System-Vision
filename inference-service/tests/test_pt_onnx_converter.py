# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""The .pt → .onnx converter's pure helpers (the export itself needs ultralytics)."""
import pytest
from api._pt_onnx_converter import checkpoint_imgsz


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
