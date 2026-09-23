# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: Apache-2.0

"""Application task ids shared by every service.

A device runs one application — object detection, image classification,
instance segmentation or face recognition — and every model, dataset and
detection record carries the task it belongs to. The first three ids equal
ultralytics' ``model.task`` values, so no mapping is ever needed; ``face``
models are galleries built on the device, never ultralytics exports. Data
recorded before application types existed carries no task and means
``detect``.
"""
import re
from typing import Optional

DETECT = "detect"
CLASSIFY = "classify"
SEGMENT = "segment"
FACE = "face"

#: Every task id, in the order the UI lists them.
TASKS = (DETECT, CLASSIFY, SEGMENT, FACE)

#: What a missing task means (legacy sidecars, datasets, status replies).
DEFAULT_TASK = DETECT

#: The class name a face recognition result carries when no enrolled person
#: matches. Reserved: no person may be enrolled under it, or a stranger and
#: that person would be indistinguishable in records and access-control flows.
UNKNOWN_FACE = "unknown"

_COLOR_SUFFIX = re.compile(r"\s*#[0-9a-fA-F]{6}$")

#: Characters a class entry may carry: the dataset name policy plus the
#: ``#`` of an optional ``" #rrggbb"`` colour suffix (safe in a single-quoted
#: YAML scalar, in a ``.txt`` sidecar line and in an HTTP header).
CLASS_NAME_SAFE = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-.#")
#: Longest class entry accepted anywhere.
CLASS_NAME_MAX = 64


def is_task(value: object) -> bool:
    """True when ``value`` is one of :data:`TASKS`."""
    return isinstance(value, str) and value in TASKS


def task_or_default(value: object) -> str:
    """The task a stored value stands for: missing or empty means ``detect``.

    An unknown non-empty string is returned unchanged (stripped): it comes
    from a newer build and must be refused or shown as unsupported by the
    caller, never silently read as ``detect``.
    """
    if not isinstance(value, str):
        return DEFAULT_TASK
    value = value.strip()
    return value or DEFAULT_TASK


def person_key(entry: object) -> str:
    """How two class entries are compared as *people*: without the optional
    colour suffix, surrounding whitespace or letter case, so ``Alice`` and
    ``ALICE #ff0000`` are one person."""
    if not isinstance(entry, str):
        return ""
    return _COLOR_SUFFIX.sub("", entry.strip()).strip().lower()


def is_safe_class_name(entry: object) -> bool:
    """True for a non-empty class entry within :data:`CLASS_NAME_MAX` made of
    :data:`CLASS_NAME_SAFE` characters only (no newline can split a sidecar)."""
    if not isinstance(entry, str):
        return False
    entry = entry.strip()
    return 0 < len(entry) <= CLASS_NAME_MAX and all(c in CLASS_NAME_SAFE for c in entry)


def is_reserved_face_name(name: object) -> bool:
    """True when ``name`` is the :data:`UNKNOWN_FACE` sentinel in any spelling.

    The comparison ignores case, surrounding whitespace and the optional
    ``" #rrggbb"`` colour suffix a class entry may carry, so ``"Unknown"`` and
    ``"UNKNOWN #ff0000"`` are refused as person names too.
    """
    if not isinstance(name, str):
        return False
    return person_key(name) == UNKNOWN_FACE


def optional_task(value: object) -> Optional[str]:
    """``None`` for a missing or empty value, the stripped string otherwise."""
    if not isinstance(value, str):
        return None
    return value.strip() or None
