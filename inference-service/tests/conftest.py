# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Shared pytest import setup for the inference-service suite.

Two things make the service importable on a plain host (no container, no
TensorRT):

* The generated ``*_pb2`` modules use flat imports (``import inference_pb2``), so
  their directory must be on ``sys.path``. In the container the stubs live next
  to ``api`` (``api/proto``, see Dockerfile.inference-service); for host test runs
  they come from ``scripts/compile-proto.sh``, which writes them under
  ``api-gateway/gateway/proto``.
* ``api/services/__init__.py`` eagerly imports every service, and a few of those
  do ``from ..proto import shm_pb2`` — a package-relative import that needs an
  ``api.proto`` package. The checkout has no ``api/proto`` dir, so we register a
  namespace package pointing at the flat proto dir. TensorRT/CUDA are imported
  lazily (inside methods), so the import chain stays host-safe.

Also puts ``os-base`` on the path for the shared ``conecsa_shm`` package.
"""
import os
import sys
import types

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
_PROTO_DIR = os.path.join(_REPO_ROOT, "api-gateway", "gateway", "proto")
_OS_BASE = os.path.join(_REPO_ROOT, "os-base")

for _path in (_OS_BASE, _PROTO_DIR):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

# Make `from ..proto import shm_pb2` resolve to the flat stubs above.
if os.path.isdir(_PROTO_DIR) and "api.proto" not in sys.modules:
    import api  # noqa: F401  (ensures the parent package is initialised first)

    _proto_pkg = types.ModuleType("api.proto")
    _proto_pkg.__path__ = [_PROTO_DIR]
    sys.modules["api.proto"] = _proto_pkg


@pytest.fixture(autouse=True)
def face_models_absent(monkeypatch):
    """Point ``FACE_MODELS_DIR`` at nothing unless a test provides the models.

    The face strategy is registered only when the bundled YuNet/SFace graphs
    are present (``postprocess._face_assets``), and a developer who ran
    ``scripts/fetch-face-models.sh`` has them in the working tree. Without this
    the registry — and therefore ``supported_tasks`` — would differ between
    that machine and CI. Face tests set the variable themselves.
    """
    monkeypatch.setenv("FACE_MODELS_DIR", os.path.join(os.sep, "nonexistent", "face-models"))
