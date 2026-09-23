# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Every unary gateway call gets a deadline.

A real in-process gRPC server whose GetStatus never answers proves the
interceptor releases the caller within the deadline; the unit tests pin the
deadline classes and that an explicit timeout at the call site wins.
"""
import threading
import time
from concurrent import futures
from types import SimpleNamespace
from typing import Any

import grpc
import inference_pb2 as inf_pb
import inference_pb2_grpc as inf_grpc
import pytest
from gateway.rpc_deadlines import LONG_UNARY_METHODS, DeadlineInterceptor, with_deadlines


class HangingDetection(inf_grpc.DetectionControlServicer):
    """GetStatus parks until the test releases it; Stop answers at once."""

    def __init__(self):
        self.release = threading.Event()

    def GetStatus(self, request, context):
        self.release.wait(10.0)
        return inf_pb.StatusResponse()

    def Stop(self, request, context):
        return inf_pb.Result(success=True, message="stopped")


@pytest.fixture
def backend():
    servicer = HangingDetection()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    inf_grpc.add_DetectionControlServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    yield servicer, f"127.0.0.1:{port}"
    servicer.release.set()
    server.stop(0)


class TestAgainstARealServer:
    def test_a_backend_that_never_answers_releases_the_caller(self, backend):
        _servicer, addr = backend
        channel = grpc.intercept_channel(
            grpc.insecure_channel(addr), DeadlineInterceptor(default_s=0.3, long_s=0.6))
        stub = inf_grpc.DetectionControlStub(channel)
        started = time.monotonic()
        with pytest.raises(grpc.RpcError) as info:
            stub.GetStatus(inf_pb.Empty())
        assert info.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
        assert time.monotonic() - started < 2.0

    def test_an_explicit_call_site_timeout_wins(self, backend):
        _servicer, addr = backend
        channel = grpc.intercept_channel(
            grpc.insecure_channel(addr), DeadlineInterceptor(default_s=5.0, long_s=5.0))
        stub = inf_grpc.DetectionControlStub(channel)
        started = time.monotonic()
        with pytest.raises(grpc.RpcError):
            stub.GetStatus(inf_pb.Empty(), timeout=0.2)
        assert time.monotonic() - started < 1.5

    def test_a_responsive_call_is_unaffected(self, backend):
        _servicer, addr = backend
        stub = inf_grpc.DetectionControlStub(with_deadlines(grpc.insecure_channel(addr)))
        assert stub.Stop(inf_pb.Empty()).success is True


def _details(method, timeout=None) -> Any:
    return SimpleNamespace(method=method, timeout=timeout, metadata=None,
                           credentials=None, wait_for_ready=None, compression=None)


def _capture(seen: list) -> Any:
    """A continuation that records the call details it is given."""
    return lambda details, _request: seen.append(details)


class TestDeadlineClasses:
    def test_control_calls_get_the_default(self):
        seen = []
        interceptor = DeadlineInterceptor(default_s=7.0, long_s=70.0, upload_s=700.0)
        interceptor.intercept_unary_unary(
            _capture(seen), _details("/conecsa.DetectionControl/GetStatus"), None)
        assert seen[0].timeout == 7.0

    @pytest.mark.parametrize("rpc", sorted(LONG_UNARY_METHODS))
    def test_slow_calls_get_the_long_class(self, rpc):
        interceptor = DeadlineInterceptor(default_s=7.0, long_s=70.0, upload_s=700.0)
        assert interceptor.deadline_for(f"/conecsa.Service/{rpc}") == 70.0

    def test_uploads_get_the_upload_class(self):
        seen = []
        interceptor = DeadlineInterceptor(default_s=7.0, long_s=70.0, upload_s=700.0)
        interceptor.intercept_stream_unary(
            _capture(seen), _details("/conecsa.ModelControl/UploadModel"), iter(()))
        assert seen[0].timeout == 700.0

    def test_an_existing_timeout_is_kept(self):
        seen = []
        interceptor = DeadlineInterceptor(default_s=7.0, long_s=70.0, upload_s=700.0)
        details = _details("/conecsa.ModelControl/SelectModel", timeout=3.0)
        interceptor.intercept_unary_unary(_capture(seen), details, None)
        assert seen[0] is details

    def test_server_streams_are_not_intercepted(self):
        # StreamEvents/StreamStats relays live for the process; the downloads
        # carry their own explicit deadlines.
        assert not isinstance(DeadlineInterceptor(1, 1, 1), grpc.UnaryStreamClientInterceptor)
        assert not isinstance(DeadlineInterceptor(1, 1, 1), grpc.StreamStreamClientInterceptor)

    def test_the_default_classes_come_from_settings(self):
        from gateway.config import settings
        interceptor = DeadlineInterceptor()
        assert interceptor.deadline_for("/x/GetStatus") == settings.GRPC_TIMEOUT
        assert interceptor.deadline_for("/x/SelectModel") == settings.GRPC_LONG_TIMEOUT
