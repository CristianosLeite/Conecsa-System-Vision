# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Shared-memory tensor transport between the service and a TensorRT worker.

Pickling the input tensor (4.9 MB for a 640 engine) and every output (3.3 MB
of mask prototypes per tile for segmentation) over the worker's localhost
connection would cost about a third of each lane's time on the Orin. Instead
the client creates one anonymous ``memfd`` per worker and hands it to the subprocess at spawn
(``pass_fds``). After a load, the worker lays the engine's input and
outputs out in it (:func:`layout_for`), sizes it, and answers with the
layout; an inference then only exchanges a small message, while the
tensors are written and read in place.

The connection message path is the fallback: without ``memfd`` (a
non-Linux host), for an engine with a dynamic shape, or when a tensor does
not match its slot.
"""
import mmap
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_ALIGN = 64


def _nbytes(shape: Sequence[int], dtype: np.dtype) -> int:
    count = 1
    for dim in shape:
        count *= int(dim)
    return count * dtype.itemsize


def _aligned(offset: int) -> int:
    return (offset + _ALIGN - 1) // _ALIGN * _ALIGN


def layout_for(input_details: Sequence[Dict[str, Any]],
               output_details: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The buffer layout for one engine, or ``None`` when it cannot have one.

    Only the first input is transported (the runtime supports single-input
    models). A shape with a non-positive dimension (dynamic) yields ``None``.
    Slots are 64-byte aligned; ``size`` is rounded up to whole pages.
    """
    if not input_details:
        return None
    slots: List[Dict[str, Any]] = []
    offset = 0
    for detail in [input_details[0], *output_details]:
        shape = [int(d) for d in detail["shape"]]
        if not shape or any(d <= 0 for d in shape):
            return None
        dtype = np.dtype(detail["dtype"])
        nbytes = _nbytes(shape, dtype)
        slots.append({"offset": offset, "nbytes": nbytes, "dtype": dtype.str, "shape": shape})
        offset = _aligned(offset + nbytes)
    page = mmap.PAGESIZE
    return {
        "size": (offset + page - 1) // page * page,
        "input": slots[0],
        "outputs": slots[1:],
    }


def create_fd(name: str) -> Optional[int]:
    """An anonymous, inheritable shared-memory file, or ``None`` without ``memfd``."""
    create = getattr(os, "memfd_create", None)
    if create is None:
        return None
    try:
        fd = create(name, 0)  # no MFD_CLOEXEC: the worker subprocess inherits it
    except OSError:
        return None
    os.set_inheritable(fd, True)
    return fd


class TensorBuffer:
    """Numpy views over one worker's shared buffer, per the layout."""

    def __init__(self, fd: int, layout: Dict[str, Any]) -> None:
        self.layout = layout
        self._map = mmap.mmap(fd, int(layout["size"]))
        self.input = self._view(layout["input"])
        self.outputs = [self._view(slot) for slot in layout["outputs"]]

    def _view(self, slot: Dict[str, Any]) -> np.ndarray:
        dtype = np.dtype(slot["dtype"])
        count = int(slot["nbytes"]) // dtype.itemsize
        return np.frombuffer(self._map, dtype=dtype, count=count,
                             offset=int(slot["offset"])).reshape(slot["shape"])

    def fits_input(self, value: np.ndarray) -> bool:
        """Whether ``value`` can be written to the input slot as it is."""
        return value.dtype == self.input.dtype and value.size == self.input.size

    def close(self) -> None:
        """Drop the views and unmap. Views handed out must not be used afterwards."""
        self.input = np.empty(0)
        self.outputs = []
        try:
            self._map.close()
        except BufferError:
            # A caller still holds a view; the mapping goes away with it.
            pass


def size_fd(fd: int, size: int) -> None:
    """Grow the shared file to at least ``size`` bytes; it never shrinks.

    The other side may still map a larger, earlier layout until it reads the
    new one, and touching a mapping past the end of its file faults (SIGBUS).
    """
    if os.fstat(fd).st_size < int(size):
        os.ftruncate(fd, int(size))
