# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""One reservation scheme for every private TensorRT worker.

The live pipeline's workers sit on ``TENSORRT_WORKER_PORT + i`` for every
lane ``i < TENSORRT_CONTEXTS``. The private workers (labeling, face embedder,
gallery build detector and embedder) come after them: past sixteen lanes by
default, or past the last configured lane when the deployment runs more, so
no context count can make a private worker bind or connect to a live lane.
An explicit ``TENSORRT_*_WORKER_PORT`` still wins for its worker.
"""
import os

#: Lanes always kept free before the first private worker (the historical
#: layout: labeling at +16, face embedder at +17, gallery build at +18/+19).
RESERVED_LANES = 16

#: Slot of each private worker after the reserved lanes.
LABEL_SLOT = 0
FACE_EMBED_SLOT = 1
FACE_BUILD_DETECTOR_SLOT = 2
FACE_BUILD_EMBEDDER_SLOT = 3


def base_port() -> int:
    return int(os.environ.get("TENSORRT_WORKER_PORT", "5501"))


def _lanes() -> int:
    try:
        return max(int(os.environ.get("TENSORRT_CONTEXTS", "1")), 1)
    except ValueError:
        return 1


def private_worker_port(slot: int, env: str = "") -> int:
    """Port of the private worker in ``slot``, unless ``env`` names one."""
    default = base_port() + max(_lanes(), RESERVED_LANES) + slot
    if not env:
        return default
    try:
        return int(os.environ.get(env, str(default)))
    except ValueError:
        return default
