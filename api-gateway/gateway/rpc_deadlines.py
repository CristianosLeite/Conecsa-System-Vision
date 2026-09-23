# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Default deadlines for every gateway-to-service gRPC call.

Waitress is thread-per-connection, so a call to a backend that stays connected
but stops answering would park its request thread forever, and a handful of
those would exhaust the pool. The channels are wrapped with this interceptor,
which fills in a deadline whenever a call has none. An explicit ``timeout=`` at
the call site always wins (the bulk transfers pass their own).

Deadline classes:

* unary control/read calls: ``GATEWAY_GRPC_TIMEOUT`` (12 s);
* slow unary calls that load a runtime or touch the whole dataset tree
  (``LONG_UNARY_METHODS``): ``GATEWAY_GRPC_LONG_TIMEOUT`` (120 s);
* client-streaming uploads: ``GATEWAY_GRPC_UPLOAD_TIMEOUT`` (600 s).

Server streams (``StreamEvents``, ``StreamStats``, the dataset/model
downloads) are deliberately left alone: the relays are meant to live for the
process, and the downloads already carry explicit timeouts.
"""
import collections
from typing import Iterable, Optional

import grpc

from .config import settings

# Unary methods (proto rpc names) that legitimately take longer than a control
# call: engine (de)serialization, GPU handover, an application switch (drains
# the pipeline), training start/stop, dataset tree removal, camera capture,
# SAM unload.
LONG_UNARY_METHODS = frozenset({
    "SelectModel", "ReloadModel", "DeleteModel",
    "Start", "ReleaseRuntime", "ResumeRuntime", "SetApplication",
    "StartTraining", "CancelTraining", "FinishTraining",
    "DeleteDataset", "CaptureImage", "ReplicateImage", "UnloadSam",
})


class _CallDetails(
    collections.namedtuple(
        "_CallDetails",
        ("method", "timeout", "metadata", "credentials", "wait_for_ready", "compression"),
    ),
    grpc.ClientCallDetails,
):
    """``grpc.ClientCallDetails`` with a replaced timeout."""


def _with_timeout(details: grpc.ClientCallDetails, seconds: float) -> grpc.ClientCallDetails:
    """Return *details* with ``seconds`` as its deadline unless one is already set."""
    if getattr(details, "timeout", None) is not None:
        return details
    return _CallDetails(
        details.method, seconds, getattr(details, "metadata", None),
        getattr(details, "credentials", None), getattr(details, "wait_for_ready", None),
        getattr(details, "compression", None),
    )


def _rpc_name(method: str) -> str:
    """``/package.Service/Rpc`` → ``Rpc``."""
    return method.rsplit("/", 1)[-1]


class DeadlineInterceptor(grpc.UnaryUnaryClientInterceptor, grpc.StreamUnaryClientInterceptor):
    """Fill in a default deadline for unary and client-streaming calls."""

    def __init__(self, default_s: Optional[float] = None, long_s: Optional[float] = None,
                 upload_s: Optional[float] = None,
                 long_methods: Iterable[str] = LONG_UNARY_METHODS) -> None:
        self._default = settings.GRPC_TIMEOUT if default_s is None else default_s
        self._long = settings.GRPC_LONG_TIMEOUT if long_s is None else long_s
        self._upload = settings.GRPC_UPLOAD_TIMEOUT if upload_s is None else upload_s
        self._long_methods = frozenset(long_methods)

    def deadline_for(self, method: str) -> float:
        """The unary deadline class for a full method name."""
        return self._long if _rpc_name(method) in self._long_methods else self._default

    def intercept_unary_unary(self, continuation, client_call_details, request):
        details = _with_timeout(client_call_details,
                                self.deadline_for(client_call_details.method))
        return continuation(details, request)

    def intercept_stream_unary(self, continuation, client_call_details, request_iterator):
        details = _with_timeout(client_call_details, self._upload)
        return continuation(details, request_iterator)


def with_deadlines(channel: grpc.Channel) -> grpc.Channel:
    """Wrap *channel* so every unary/client-streaming call gets a deadline."""
    return grpc.intercept_channel(channel, DeadlineInterceptor())
