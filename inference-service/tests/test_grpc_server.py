# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""serve_grpc binds on the caller's thread and fails loudly."""
import socket
from types import SimpleNamespace
from typing import Any

import grpc
import pytest
from api.inference_grpc import serve_grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_an_occupied_port_raises_instead_of_serving_nothing():
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen(1)
        addr = "127.0.0.1:%d" % holder.getsockname()[1]
        with pytest.raises(RuntimeError, match="could not bind"):
            serve_grpc(SimpleNamespace(), listen_addr=addr)


def test_a_started_server_reports_serving_on_the_health_service():
    addr = "127.0.0.1:%d" % _free_port()
    server = serve_grpc(SimpleNamespace(), listen_addr=addr)
    try:
        stub: Any = health_pb2_grpc.HealthStub(grpc.insecure_channel(addr))
        resp = stub.Check(health_pb2.HealthCheckRequest(service=""), timeout=5)
        assert resp.status == health_pb2.HealthCheckResponse.SERVING
    finally:
        server.stop(0)
