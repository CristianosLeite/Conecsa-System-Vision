# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Engine output contract per task.

Pure shape arithmetic, no numpy: the same rules classify an ONNX graph at
conversion time (``infer_task``) and verify an engine against its declared
task at activation (``check``). A mismatch is refused at activation, so
``ModelService.activate_model`` rolls back to the previous model instead of
decoding one task's tensors with another task's postprocess.

Shapes are sequences of ints; a negative or ``None`` dimension is dynamic
and matches anything.
"""
import math
from typing import Any, List, Optional, Sequence

#: End-to-end (YOLO26) heads emit at most 300 rows (ultralytics ``max_det``
#: default). Deliberately a constant and not ``YOLO_MAX_CANDIDATES``: that knob
#: is env-tunable and must not change how the output layout is classified.
E2E_MAX_DET = 300

Shape = Sequence[Optional[int]]


class ContractError(RuntimeError):
    """The engine's outputs do not match the layout its task requires."""


def _dims(shape: Any) -> List[Optional[int]]:
    """A shape as a list of ints, ``None`` for dynamic dimensions."""
    dims: List[Optional[int]] = []
    for d in list(shape):
        try:
            value = int(d)
        except (TypeError, ValueError):
            dims.append(None)
            continue
        dims.append(value if value >= 0 else None)
    return dims


def _is(value: Optional[int], expected: int) -> bool:
    return value is None or value == expected


def infer_task(shapes: Sequence[Shape]) -> Optional[str]:
    """Task an engine/graph was exported for, from its output shapes alone.

    One rank-2 output is a classification head, one rank-3 output a
    detection head, a rank-3 plus a rank-4 output (rows + mask prototypes) a
    segmentation head; anything else is not a YOLO layout this device knows.
    """
    ranks = sorted(len(_dims(s)) for s in shapes)
    if ranks == [2]:
        return "classify"
    if ranks == [3]:
        return "detect"
    if ranks == [3, 4]:
        return "segment"
    return None


def _check_detect(shapes: List[List[Optional[int]]]) -> None:
    if len(shapes) != 1:
        raise ContractError(
            f"a detection engine has exactly one output, this one has {len(shapes)}")
    shape = shapes[0]
    if len(shape) != 3 or not _is(shape[0], 1):
        raise ContractError(f"a detection output is [1, N, C] or [1, C, N], got {shape}")
    rows, cols = shape[1], shape[2]
    if rows is None or cols is None:
        return
    if cols == 6 and 6 < rows <= E2E_MAX_DET:
        return  # end-to-end rows [x1, y1, x2, y2, conf, cls]
    features = min(rows, cols)  # legacy layouts are transposed to features-first
    if features < 5:
        raise ContractError(f"not a YOLO detection output: {shape}")


def _check_classify(shapes: List[List[Optional[int]]]) -> None:
    if len(shapes) != 1:
        raise ContractError(
            f"a classification engine has exactly one output, this one has {len(shapes)}")
    shape = shapes[0]
    if len(shape) != 2 or not _is(shape[0], 1):
        raise ContractError(f"a classification output is [1, classes], got {shape}")
    if shape[1] is not None and shape[1] < 2:
        raise ContractError(f"a classification output needs at least 2 classes, got {shape}")


def _check_segment(shapes: List[List[Optional[int]]],
                   input_shape: Optional[List[Optional[int]]]) -> None:
    if len(shapes) != 2:
        raise ContractError(
            "a segmentation engine has exactly two outputs (rows and mask prototypes), "
            f"this one has {len(shapes)}")
    by_rank = {len(s): s for s in shapes}
    rows, protos = by_rank.get(3), by_rank.get(4)
    if rows is None or protos is None:
        raise ContractError(
            f"a segmentation engine outputs [1, N, 6 + nm] and [1, nm, h, w], got {shapes}")
    if not (_is(rows[0], 1) and _is(protos[0], 1)):
        raise ContractError(f"segmentation outputs must have batch 1, got {shapes}")
    nm = protos[1]
    if rows[1] is not None and not 0 < rows[1] <= E2E_MAX_DET:
        raise ContractError(
            "only the end-to-end (YOLO26) segmentation head is supported; "
            f"got {rows[1]} rows per image")
    if rows[2] is not None and nm is not None and rows[2] != 6 + nm:
        raise ContractError(
            f"segmentation rows carry {rows[2]} values, expected 6 + {nm} mask coefficients")
    if input_shape and len(input_shape) == 4:
        side = input_shape[2]
        if side is not None and not (_is(protos[2], side // 4) and _is(protos[3], side // 4)):
            raise ContractError(
                f"mask prototypes are {protos[2]}x{protos[3]}, expected {side // 4} "
                f"for a {side} px input")


#: YuNet strides and the last dimension of each per-stride output kind
#: (``cls``, ``obj``, ``bbox``, ``kps``).
FACE_STRIDES = (8, 16, 32)
FACE_OUTPUT_WIDTHS = (1, 1, 4, 10)


def _check_face(shapes: List[List[Optional[int]]],
                input_shape: Optional[List[Optional[int]]]) -> None:
    """The YuNet detector layout: 12 rank-3 outputs, four kinds per stride.

    Shapes alone cannot tell ``cls_*`` from ``obj_*``; the face strategy binds
    the outputs by name when it initializes.
    """
    expected = len(FACE_STRIDES) * len(FACE_OUTPUT_WIDTHS)
    if len(shapes) != expected:
        raise ContractError(
            f"a face detector (YuNet) has {expected} outputs, this one has {len(shapes)}")
    if any(len(s) != 3 or not _is(s[0], 1) for s in shapes):
        raise ContractError(f"face detector outputs are [1, anchors, k], got {shapes}")
    widths = sorted([w for w in (s[2] for s in shapes) if w is not None])
    if len(widths) == expected and widths != sorted(FACE_OUTPUT_WIDTHS * len(FACE_STRIDES)):
        raise ContractError(
            f"face detector outputs carry {widths} values, expected 1, 1, 4 and 10 per stride")
    side = input_shape[2] if input_shape and len(input_shape) == 4 else None
    rows = sorted([r for r in (s[1] for s in shapes) if r is not None])
    if side is not None and len(rows) == expected:
        want = sorted((side // stride) ** 2 for stride in FACE_STRIDES
                      for _ in FACE_OUTPUT_WIDTHS)
        if rows != want:
            raise ContractError(
                f"face detector anchors {sorted(set(rows))} do not fit a {side} px input")


def check(task: str, shapes: Sequence[Shape], input_shape: Optional[Shape] = None) -> None:
    """Raise :class:`ContractError` unless ``shapes`` fit ``task``'s layout.

    The message names what the engine looks like, so an operator can tell a
    model of another task from a corrupt or unsupported export.
    """
    dims = [_dims(s) for s in shapes]
    found = infer_task(shapes)
    try:
        if task == "detect":
            _check_detect(dims)
        elif task == "classify":
            _check_classify(dims)
        elif task == "segment":
            _check_segment(dims, _dims(input_shape) if input_shape is not None else None)
        elif task == "face":
            _check_face(dims, _dims(input_shape) if input_shape is not None else None)
        else:
            raise ContractError(f"unknown task '{task}'")
    except ContractError as exc:
        if found and found != task:
            raise ContractError(
                f"the engine looks like a '{found}' model, but it is declared as "
                f"'{task}' ({exc})") from None
        raise


#: How far a classification output may sum from 1 and still be probabilities.
PROBABILITY_SUM_TOLERANCE = 0.01


def check_probabilities(values: Sequence[float]) -> None:
    """Raise :class:`ContractError` unless a classification output sums to 1 (± 0.01).

    Ultralytics applies softmax inside the exported classification graph. An
    engine exported another way outputs logits: they pass the shape contract
    but make the confidence threshold meaningless, so activation runs one
    inference and refuses them.
    """
    total = math.fsum(float(v) for v in values)
    if not math.isfinite(total) or abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
        raise ContractError(
            "a classification engine outputs probabilities that sum to 1, this one sums "
            f"to {total:.3f} (logits?); export it with ultralytics, which applies softmax "
            "in the exported graph")
