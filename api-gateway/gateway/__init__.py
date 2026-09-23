# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""API gateway package.

The api-gateway is a thin HTTP↔gRPC/SHM interface: it owns the external REST /
SSE / MJPEG contract while the real work lives in the headless services
(inference-service and training-service over gRPC, processed frames over SHM)
and the `os-base` hardware agent (network/Wi-Fi/GPIO over gRPC). Per-frame media
never crosses gRPC — the gateway reads the camera and processed-frame POSIX SHM
rings directly.
"""
import os as _os
import sys as _sys

# Make the compiled proto stubs importable (the generated *_pb2_grpc modules
# and several controllers do flat `import <name>_pb2`). Done here, at the
# package root before any submodule runs, so no module's import order can
# break stub resolution.
_PROTO_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "proto")
if _PROTO_DIR not in _sys.path:
    _sys.path.insert(0, _PROTO_DIR)
