"""Single authority for model filenames: the extension allowlist and the
traversal-safe name check.

Model names arrive from untrusted callers (multipart upload filenames, JSON
bodies, URL segments) and are joined onto the model directory, so every
consumer — upload, select, activate, delete, download, conversion outputs —
must validate through :func:`validate_model_filename`. Before this module the
allowlist existed in seven near-copies and only the download path checked for
traversal, which let ``../escaped.engine`` write outside the model directory.
"""
import os
from pathlib import Path
from typing import Tuple

# Everything the model API accepts and lists.
ALLOWED_MODEL_EXTENSIONS = ('.engine', '.plan', '.pt', '.onnx')

# What the TensorRT runtime can load or build from (no .pt — those convert first).
TENSORRT_MODEL_EXTENSIONS = ('.engine', '.plan', '.onnx')

# Prebuilt serialized engines (loadable without a build step).
ENGINE_FILE_EXTENSIONS = ('.engine', '.plan')


def validate_model_filename(name: str, model_root: str) -> Tuple[str, str]:
    """Validate an untrusted model filename against ``model_root``.

    Returns ``(absolute_path, "")`` when valid, ``("", error_message)``
    otherwise. Valid means: a plain basename (no separators, no traversal),
    no control characters or quotes (the name travels in HTTP headers), not a
    hidden/state file, an allow-listed extension (case-normalized), and a
    resolved path that stays under the resolved model root.
    """
    if not name or name != name.strip():
        return "", "Invalid model name"
    if os.path.basename(name) != name or "\\" in name or "/" in name:
        return "", "Invalid model name: path separators are not allowed"
    if any(ord(c) < 32 or c in ('"', "'") for c in name):
        return "", "Invalid model name: control characters are not allowed"
    if name.startswith("."):
        # Also covers the .current_model state file.
        return "", "Invalid model name: hidden names are reserved"
    if os.path.splitext(name)[1].lower() not in ALLOWED_MODEL_EXTENSIONS:
        return "", (
            "Invalid file type. Allowed: "
            + ", ".join(ALLOWED_MODEL_EXTENSIONS)
        )
    root = Path(model_root).resolve()
    resolved = (root / name).resolve()
    if not resolved.is_relative_to(root):
        return "", "Invalid model name: escapes the model directory"
    return str(resolved), ""
