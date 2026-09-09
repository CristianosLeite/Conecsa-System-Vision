"""Existing device models as fine-tuning bases and labeling assistants.

The inference-service owns the model directory and keeps, next to each
engine it converted from a ``.pt``, that checkpoint as a *weights sidecar*
(an on-device training run's ``best.pt``, a manual or federated ``.pt``
upload); ``GET /api/v1/models`` reports it as ``has_weights``. So "model X's
last best.pt" is X's sidecar, replaced whenever X is retrained. This service
has no access to that volume; it fetches the file through the gateway's
weights route, the same HTTP path the trainer already uses to hand its
result back.
"""
import logging
import os
import tempfile
from typing import List
from urllib.parse import quote

import requests

from .dataset_service import DatasetError

logger = logging.getLogger(__name__)

_LIST_TIMEOUT_S = 15.0
_FETCH_TIMEOUT_S = 300.0
# What the inference-service lists as a model (its model_paths allowlist).
_MODEL_EXTENSIONS = (".engine", ".plan", ".pt", ".onnx")


def validate_model_ref(name: str) -> str:
    """A model name exactly as the device model list reports it (``X.engine``).

    Mirrors the inference-service's filename rules: a plain basename with an
    allow-listed extension, no separators, control characters or quotes, not
    hidden. The name travels into a gateway URL and onto this service's disk.
    """
    name = (name or "").strip()
    if not name:
        raise DatasetError("Model name is required")
    if os.path.basename(name) != name or "/" in name or "\\" in name:
        raise DatasetError("Invalid model name: path separators are not allowed")
    if any(ord(c) < 32 or c in ('"', "'") for c in name):
        raise DatasetError("Invalid model name: control characters are not allowed")
    if name.startswith("."):
        raise DatasetError("Invalid model name: hidden names are reserved")
    if os.path.splitext(name)[1].lower() not in _MODEL_EXTENSIONS:
        raise DatasetError(f"Invalid model name '{name}'")
    return name


def model_stem(name: str) -> str:
    """``X.engine`` → ``X`` (the name the operator sees and the fetched file's stem)."""
    return os.path.splitext(name)[0]


def list_models_with_weights(gateway_addr: str) -> List[str]:
    """Device model names that keep a training checkpoint (``has_weights``)."""
    resp = requests.get(f"{gateway_addr}/api/v1/models", timeout=_LIST_TIMEOUT_S)
    resp.raise_for_status()
    try:
        models = resp.json().get("models", [])
    except ValueError as exc:
        raise DatasetError("Could not read the device model list") from exc
    return [
        str(m.get("name", ""))
        for m in models
        if isinstance(m, dict) and m.get("has_weights") and str(m.get("name", ""))
    ]


def fetch_weights(gateway_addr: str, model_name: str, dest_dir: str) -> str:
    """Download ``model_name``'s checkpoint into ``dest_dir`` as ``<stem>.pt``.

    Streams to a temp file in the same directory and renames it into place,
    so a torn download never leaves a half-written checkpoint under the
    final name. Always re-fetched: the file is tens of MB on a local link,
    and the inference-side sidecar is the source of truth (a retrain under
    the same name replaces it).
    """
    model_name = validate_model_ref(model_name)
    os.makedirs(dest_dir, exist_ok=True)
    # A filename, not a URL segment: "?", "#", "%" or a space are legal on
    # disk but must be percent-encoded here (the gateway decodes them back).
    url = f"{gateway_addr}/api/v1/model/{quote(model_name, safe='')}/weights"
    with requests.get(url, stream=True, timeout=_FETCH_TIMEOUT_S) as resp:
        if resp.status_code == 404:
            raise DatasetError(f"Model '{model_name}' has no training checkpoint on the device")
        if resp.status_code != 200:
            raise DatasetError(
                f"Model weights download failed (HTTP {resp.status_code}): {resp.text[:200]}"
            )
        fd, tmp_path = tempfile.mkstemp(prefix=".fetch-", suffix=".pt", dir=dest_dir)
        try:
            with os.fdopen(fd, "wb") as out:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        out.write(chunk)
            final = os.path.join(dest_dir, f"{model_stem(model_name)}.pt")
            os.replace(tmp_path, final)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    size = os.path.getsize(final)
    if size == 0:
        os.unlink(final)
        raise DatasetError(f"Model '{model_name}' weights downloaded empty")
    _drop_other_checkpoints(dest_dir, keep=final)
    logger.info("Fetched weights of %s (%.1f MB) into %s", model_name, size / 1e6, dest_dir)
    return final


def _drop_other_checkpoints(dest_dir: str, keep: str) -> None:
    """Keep ``dest_dir`` to the checkpoint just fetched.

    Every fetched base is re-downloaded on use, so earlier ones are dead
    weight (tens of MB each on the device's eMMC). Best-effort. Stock
    ``yolo*.pt`` files stay: the trainer runs with the weights' directory as
    cwd (``_yolo_trainer``), which is where ultralytics' AMP check drops its
    nano checkpoint — dropping it would cost a download (or an offline
    warning) on every fine-tune.
    """
    for entry in os.listdir(dest_dir):
        path = os.path.join(dest_dir, entry)
        if path == keep or not entry.endswith(".pt") or entry.startswith("yolo"):
            continue
        try:
            os.unlink(path)
        except OSError as exc:
            logger.warning("Could not remove stale base checkpoint %s: %s", path, exc)
