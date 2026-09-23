# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Standalone .pt -> .onnx converter (ultralytics / torch).

Executed by conversion_service.ConversionService as a short-lived subprocess
so that the PyTorch caching allocator and ultralytics global state never
land in the long-lived inference-service process heap. When the subprocess
exits the kernel reclaims all of it, the reliable way to return that memory on
the Yocto host (glibc fragmentation).

Its CLI follows api.runtime_management._trt_engine_builder.

Usage:
    python3 -m api._pt_onnx_converter \\
        --pt /data/models/foo.pt \\
        --onnx /data/models/foo.onnx \\
        --imgsz 640 [--imgsz-from-checkpoint]
    python3 -m api._pt_onnx_converter \\
        --inspect-onnx /data/models/foo.onnx

With ``--imgsz-from-checkpoint`` the export uses the size the checkpoint was
trained at, and ``--imgsz`` only when the checkpoint records none.

Stdout (last line, JSON):
    {"class_names": ["person", ...], "task": "detect", "output_shapes": [[1, 300, 6]],
     "imgsz": 640}
    {"output_shapes": [[1, 300, 6]]}                      (--inspect-onnx)
``task`` is ultralytics' ``model.task`` (``null`` on the torch fallback);
``output_shapes`` is ``null`` when the ``onnx`` package cannot read the graph;
``imgsz`` is the size the export used.
"""
import argparse
import json
import logging
import os
import shutil
import sys
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)


def onnx_output_shapes(onnx_path: str) -> Optional[List[List[int]]]:
    """The graph's output shapes (``-1`` for a dynamic dimension), or ``None``."""
    try:
        # noinspection PyPackageRequirements
        import onnx  # type: ignore  # baked into os-base
    except ImportError:
        log.warning("onnx is not installed; the model's task is verified at activation")
        return None
    try:
        model = onnx.load(onnx_path, load_external_data=False)
    except Exception as exc:  # noqa: BLE001 - reported, the build decides
        log.warning("Could not read %s: %s", onnx_path, exc)
        return None
    shapes: List[List[int]] = []
    for output in model.graph.output:
        dims = output.type.tensor_type.shape.dim
        shapes.append([d.dim_value if d.HasField("dim_value") else -1 for d in dims])
    return shapes


def checkpoint_imgsz(ckpt: object) -> Optional[int]:
    """The input size a checkpoint was trained at, or ``None`` when it records none.

    ultralytics stores its training arguments (``train_args``) in every
    checkpoint it writes, and the FedAvg averager keeps its first input's
    container, so an averaged checkpoint carries them too. A ``[h, w]`` size
    gives its larger side.
    """
    args = ckpt.get("train_args") if isinstance(ckpt, dict) else None
    size = args.get("imgsz") if isinstance(args, dict) else None
    if isinstance(size, (list, tuple)):
        size = max(size, default=None) if all(type(s) is int for s in size) else None
    return size if type(size) is int and size > 0 else None


def convert_pt_to_onnx(pt_path: str, onnx_path: str, imgsz: int = 640,
                       from_checkpoint: bool = False) -> Tuple[List[str], Optional[str], int]:
    """Export a YOLO .pt to ONNX. Falls back to raw torch.onnx.export.

    Returns the class names, ultralytics' ``model.task`` (both empty on the
    torch fallback, which knows neither) and the size the export used:
    ``imgsz``, or with ``from_checkpoint`` the checkpoint's training size when
    it records one.
    """
    try:
        # noinspection PyPackageRequirements
        from ultralytics import YOLO  # type: ignore
        model = YOLO(pt_path)
        if from_checkpoint:
            imgsz = checkpoint_imgsz(getattr(model, "ckpt", None)) or imgsz
        log.info("Converting %s -> %s (imgsz=%d)", pt_path, onnx_path, imgsz)
        model.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True)

        class_names: List[str] = []
        if hasattr(model, "names") and isinstance(model.names, dict):
            class_names = [model.names[i] for i in sorted(model.names.keys())]
            log.info("Extracted %d class names: %s", len(class_names), class_names)
        task = getattr(model, "task", None)
        task = task if isinstance(task, str) else None

        auto_onnx = os.path.splitext(pt_path)[0] + ".onnx"
        if auto_onnx != onnx_path:
            shutil.move(auto_onnx, onnx_path)

        log.info("ONNX conversion complete: %s (task %s)", onnx_path, task)
        return class_names, task, imgsz

    except ImportError:
        log.warning("ultralytics not installed, trying torch.onnx.export fallback")

    # noinspection PyPackageRequirements
    import torch  # type: ignore

    log.info("Converting %s -> %s (imgsz=%d)", pt_path, onnx_path, imgsz)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(pt_path, map_location=device)
    model.eval()

    dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
    torch.onnx.export(
        model,
        (dummy,),
        onnx_path,
        opset_version=12,
        input_names=["images"],
        output_names=["output0"],
        dynamic_axes=None,
    )
    log.info("ONNX fallback export complete: %s", onnx_path)
    return [], None, imgsz


def main() -> None:
    """CLI entry point for the `.pt`→`.onnx` conversion subprocess."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(description=".pt -> .onnx converter (subprocess)")
    parser.add_argument("--pt", help="Path to .pt model")
    parser.add_argument("--onnx", help="Output .onnx path")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--imgsz-from-checkpoint", dest="imgsz_from_checkpoint",
                        action="store_true",
                        help="Export at the checkpoint's training size; --imgsz when it has none")
    parser.add_argument("--inspect-onnx", dest="inspect_onnx",
                        help="Only report the output shapes of this ONNX graph")
    args = parser.parse_args()

    if args.inspect_onnx:
        print(json.dumps({"output_shapes": onnx_output_shapes(args.inspect_onnx)}), flush=True)
        return
    if not args.pt or not args.onnx:
        parser.error("--pt and --onnx are required unless --inspect-onnx is given")

    try:
        class_names, task, imgsz = convert_pt_to_onnx(
            args.pt, args.onnx, args.imgsz, args.imgsz_from_checkpoint)
        shapes = onnx_output_shapes(args.onnx)
    except Exception as exc:
        log.exception("FATAL: %s", exc)
        print(json.dumps({"error": str(exc)}), flush=True)
        sys.exit(1)

    # Last stdout line is the machine-readable result for the parent.
    print(json.dumps({"class_names": class_names, "task": task, "output_shapes": shapes,
                      "imgsz": imgsz}),
          flush=True)


if __name__ == "__main__":
    main()
