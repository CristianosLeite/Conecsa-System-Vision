# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Boot-time check for a training job that holds the GPU.

Every other GPU consumer goes through the gateway's shared release, but the
inference-service's own boot restores the selected model and auto-starts
detection by itself. When only this container restarted in the middle of a
training, that would load TensorRT beside the trainer; so the boot asks the
training-service first.
"""
import importlib
import logging
import os

import grpc

logger = logging.getLogger(__name__)

#: Job statuses meaning the trainer holds the GPU (``gateway/gpu.py`` agrees).
ACTIVE_JOB_STATUSES = ("preparing", "training", "uploading")

_TIMEOUT_S = 2.0


def training_job_active(addr: str = "", timeout: float = _TIMEOUT_S) -> bool:
    """True while the training-service reports an active job.

    Fail-open: a training-service that does not answer (a whole-device boot,
    where it starts alongside this service and has no job yet) is no job.
    """
    addr = addr or os.environ.get("TRAINING_GRPC_ADDR", "training-service:50071")
    # The gRPC server module puts the generated stubs' directory on sys.path,
    # so they are imported only after it.
    from .. import inference_grpc  # noqa: F401
    trn = importlib.import_module("training_pb2")
    trn_grpc = importlib.import_module("training_pb2_grpc")

    try:
        with grpc.insecure_channel(addr) as channel:
            job = trn_grpc.TrainingControlStub(channel).GetTraining(trn.Empty(),
                                                                    timeout=timeout)
    except grpc.RpcError as exc:
        logger.info("Training probe at boot got %s; assuming no training job", exc.code())
        return False
    return job.status in ACTIVE_JOB_STATUSES
