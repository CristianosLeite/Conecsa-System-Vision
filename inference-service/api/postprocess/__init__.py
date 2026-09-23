# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Per-task postprocess strategies.

``DetectionService`` builds one strategy per activated model from the model's
task; the registry below is also the source of truth for which application
types this build supports (``ApplicationInfo.supported_tasks``): a task is
supported exactly when a strategy for it is registered here. Face recognition
is registered only when the build carries its bundled models.
"""
from typing import Any, Callable, Dict, List, Optional

from conecsa_common.tasks import TASKS

from ._face_assets import assets_available
from .base import Postprocessor, PostprocessResult
from .classify import ClassifyPostprocessor
from .contract import ContractError
from .detect import DetectPostprocessor
from .face import FacePostprocessor
from .segment import SegmentPostprocessor

_REGISTRY: Dict[str, Callable[[List[str], Any], Postprocessor]] = {
    "detect": DetectPostprocessor,
    "classify": ClassifyPostprocessor,
    "segment": SegmentPostprocessor,
}


def _registry() -> Dict[str, Callable[[List[str], Any], Postprocessor]]:
    """The registered strategies, face included when its models are present."""
    if not assets_available():
        return _REGISTRY
    return {**_REGISTRY, "face": FacePostprocessor}

#: Model input side an upload is converted at when it does not say, per task
#: (ultralytics' own defaults: 224 for classification, 640 otherwise).
_DEFAULT_IMGSZ = {"classify": 224}
DEFAULT_IMGSZ = 640


def supported_tasks() -> List[str]:
    """Task ids this build can serve, in the canonical UI order."""
    registry = _registry()
    return [task for task in TASKS if task in registry]


def default_imgsz(task: Optional[str]) -> int:
    """The conversion input side for an upload of ``task`` that names none."""
    return _DEFAULT_IMGSZ.get(task or "", DEFAULT_IMGSZ)


def create(task: str, class_labels: List[str], config: Any) -> Postprocessor:
    """The strategy for ``task``; :class:`ContractError` when it is not supported."""
    factory = _registry().get(task)
    if factory is None:
        raise ContractError(
            f"this build cannot run '{task}' models "
            f"(supported: {', '.join(supported_tasks())})")
    return factory(class_labels, config)


__all__ = ["ClassifyPostprocessor", "ContractError", "DEFAULT_IMGSZ", "DetectPostprocessor",
           "FacePostprocessor", "PostprocessResult", "Postprocessor", "SegmentPostprocessor",
           "create", "default_imgsz", "supported_tasks"]
