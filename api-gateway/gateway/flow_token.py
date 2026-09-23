# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Short-lived admin tokens for the embedded Node-RED editor.

Node-RED's editor and admin API live under ``/flow`` on the device, and
``adminAuth`` guards them: without it, any local process that could reach the
hub's loopback proxy port (or the device's plaintext network) could deploy
flows with no capability at all. The iframe cannot carry the hub's capability
header, but the Node-RED editor has its own token transport: it reads
``?access_token=`` from its URL into localStorage and sends
``Authorization: Bearer`` on every admin call and on the comms websocket.

So the gateway mints a token for the operator the hub vouched for, the
device UI opens the editor with it, and ``flow/admin-token.js`` (same
secret, same format) verifies it on the Node-RED side:

    v1.<base64url(JSON payload)>.<base64url(HMAC-SHA256 over "v1.<payload>")>

payload = ``{"sub": username, "role": role, "exp": unix_seconds}``. The
secret is ``FLOW_ADMIN_TOKEN_SECRET``, falling back to the already-required
``NODE_RED_CREDENTIAL_SECRET`` so no new deployment secret is needed.
"""
import base64
import hashlib
import hmac
import json
import time
from typing import Optional

from .config import settings

VERSION = "v1"


class FlowTokenError(ValueError):
    """The token is malformed, tampered with or expired."""


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64url(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def _signature(secret: str, signed_part: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), signed_part.encode("utf-8"),
                      hashlib.sha256).digest()
    return _b64url(digest)


def mint(username: str, role: str, secret: Optional[str] = None,
         ttl_sec: Optional[float] = None, now: Optional[float] = None) -> str:
    """Return a signed token for ``username``/``role`` valid for ``ttl_sec``."""
    secret = settings.FLOW_ADMIN_TOKEN_SECRET if secret is None else secret
    if not secret:
        raise FlowTokenError("no flow admin token secret configured")
    ttl = settings.FLOW_ADMIN_TOKEN_TTL_SEC if ttl_sec is None else ttl_sec
    issued = time.time() if now is None else now
    payload = json.dumps(
        {"sub": username, "role": role, "exp": int(issued + ttl)},
        separators=(",", ":"), sort_keys=True,
    )
    signed_part = f"{VERSION}.{_b64url(payload.encode('utf-8'))}"
    return f"{signed_part}.{_signature(secret, signed_part)}"


def verify(token: str, secret: Optional[str] = None, now: Optional[float] = None) -> dict:
    """Return the payload of a valid token; raise ``FlowTokenError`` otherwise.

    The Python verifier exists for the tests that prove the two
    implementations agree; production verification happens in Node-RED.
    """
    secret = settings.FLOW_ADMIN_TOKEN_SECRET if secret is None else secret
    if not secret:
        raise FlowTokenError("no flow admin token secret configured")
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != VERSION:
        raise FlowTokenError("malformed token")
    signed_part = f"{parts[0]}.{parts[1]}"
    if not hmac.compare_digest(_signature(secret, signed_part), parts[2]):
        raise FlowTokenError("bad signature")
    try:
        payload = json.loads(_unb64url(parts[1]))
    except (ValueError, UnicodeDecodeError) as exc:
        raise FlowTokenError("malformed payload") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("exp"), int):
        raise FlowTokenError("malformed payload")
    current = time.time() if now is None else now
    if payload["exp"] <= current:
        raise FlowTokenError("token expired")
    return payload
