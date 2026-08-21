"""API gateway package.

The api-gateway is a thin HTTP↔gRPC/SHM interface: it keeps the external REST /
SSE / MJPEG contract byte-compatible while the real work lives in the headless
inference-service (gRPC control + processed-frame SHM) and the `os` hardware
agent (network/Wi-Fi/GPIO over gRPC). Per-frame media never crosses gRPC — the
gateway reads the camera and processed-frame POSIX SHM rings directly.
"""
import os as _os
import sys as _sys

# Make the compiled proto stubs importable (the generated *_pb2_grpc modules
# and several controllers do flat `import <name>_pb2`). Done here — at the
# package root, before any submodule can run — so no module's import ORDER can
# break stub resolution: import sorting once hoisted `import detection_pb2`
# above the module that used to install this path, and the container died at
# boot while every test (which imported the modules in a luckier order) passed.
_PROTO_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "proto")
if _PROTO_DIR not in _sys.path:
    _sys.path.insert(0, _PROTO_DIR)
