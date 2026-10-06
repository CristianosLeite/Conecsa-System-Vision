# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Service-level refusals the gRPC servicers map to status codes.

``PreconditionFailed`` becomes ``FAILED_PRECONDITION`` (the gateway answers
409) and ``InvalidTask`` becomes ``INVALID_ARGUMENT`` (400). Any other
exception out of a service is an internal failure.
"""

#: Why detection cannot run on a device whose application type is unset.
NO_APPLICATION_MESSAGE = (
    "No application type is selected. An administrator must choose one "
    "before detection can start."
)


class PreconditionFailed(RuntimeError):
    """The request is valid but the device state forbids it right now."""


#: GPU holders that ``ModelService.free_gpu`` refuses to evict.
GPU_HOLDER_DETECTION = "detection"
GPU_HOLDER_CONVERSION = "conversion"


class GpuBusy(PreconditionFailed):
    """The GPU cannot be freed: detection runs or a model conversion does.

    ``holder`` is ``GPU_HOLDER_DETECTION`` or ``GPU_HOLDER_CONVERSION``; the
    gRPC layer reports it so the gateway can name the blocker.
    """

    def __init__(self, holder: str, message: str) -> None:
        super().__init__(message)
        self.holder = holder


class InvalidTask(ValueError):
    """An application task id that is not one of the known tasks."""
