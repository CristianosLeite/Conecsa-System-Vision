"""Shared pytest setup and fixtures for the api-gateway suite.

``gateway.media`` imports the shared ``conecsa_shm`` package (POSIX SHM rings),
which lives under ``os-base`` in the repo — put it on ``sys.path`` so the gateway
modules import on a plain host. The generated proto stubs already live under
``gateway/proto`` and are added to ``sys.path`` by ``gateway.grpc_clients`` itself
(the SHM readers construct lazily, so importing these modules never touches a
live segment).

Two request-context fixtures are shared here:

- ``trusted_proxy`` pins the trusted-proxy DNS resolution so ``_hub_verified``
  works without a live nginx terminator.
- ``real_app`` serves the *actual* ``gateway.app`` application — every
  blueprint plus the app-level ``before_request``/``after_request`` hooks and
  error handlers. Per-blueprint bare ``Flask`` apps (the older pattern in
  individual test files) never see those hooks, so anything enforced at the
  app level must be tested through ``real_app``.
"""
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
_OS_BASE = os.path.join(_REPO_ROOT, "os-base")

if os.path.isdir(_OS_BASE) and _OS_BASE not in sys.path:
    sys.path.insert(0, _OS_BASE)


# The address the pinned trusted-proxy resolution answers with; requests made
# with ``environ_base={"REMOTE_ADDR": TERMINATOR_IP}`` count as relayed by the
# mTLS terminator.
TERMINATOR_IP = "10.66.0.9"


@pytest.fixture
def trusted_proxy(monkeypatch):
    """Pin the trusted-proxy DNS resolution to TERMINATOR_IP, cold cache."""
    from gateway import helpers
    monkeypatch.setattr(helpers, "_resolve_proxy_ips",
                        lambda: frozenset({TERMINATOR_IP}))
    monkeypatch.setattr(helpers, "_proxy_cache",
                        {"ips": frozenset(), "at": float("-inf")})


def hub_request(role="admin", user="ana", **extra_headers):
    """Request kwargs for a hub-relayed (mTLS-verified) call.

    Usage: ``client.post("/api/v1/start", **hub_request(role="user"))``.
    Requires the ``trusted_proxy`` fixture so the pinned terminator address is
    actually trusted.
    """
    headers = {"X-Conecsa-Client-Verify": "SUCCESS",
               "X-Conecsa-User": user,
               "X-Conecsa-Role": role}
    headers.update(extra_headers)
    return {"headers": headers,
            "environ_base": {"REMOTE_ADDR": TERMINATOR_IP}}


@pytest.fixture
def real_app(monkeypatch, tmp_path, trusted_proxy):
    """The real gateway app, hermetic: audit ring on tmp storage, DNS pinned.

    The clock hook stays inert because no test sends the hub-time header.
    """
    from gateway import audit
    from gateway.app import app
    buf = audit.AuditBuffer(db_path=str(tmp_path / "audit.db"),
                            max_records=1000, max_bytes=1 << 20)
    monkeypatch.setattr(audit, "_buffer", buf)
    yield app
    buf._close()
