# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Flow controller: admin tokens for the embedded Node-RED editor."""
import logging

from flask import request

from .. import audit_events, flow_token
from ..config import settings
from ..helpers import _hub_verified, _json, _json_error
from . import api_bp

logger = logging.getLogger(__name__)

# Identity minted when the request did not come through the hub: a caller on
# the internal compose network or the development stack, where — as in
# gateway/authz.py — the network boundary is the only one that exists.
_LOCAL_USER = "local"
_LOCAL_ROLE = "admin"


@api_bp.route('/api/v1/flow/token', methods=['POST'])
def flow_token_route():
    """POST /api/v1/flow/token — mint a Node-RED editor token for the caller.

    The device UI opens the editor iframe with it (``/flow/?access_token=``);
    Node-RED verifies it with the same secret and grants the editor
    permissions of the operator's role. The token carries the identity the
    hub vouched for when the request is hub-verified, or a local admin
    identity otherwise.
    """
    if not settings.FLOW_ADMIN_TOKEN_SECRET:
        return _json_error("flow editor tokens are not configured on this device", 503)
    username, role = _LOCAL_USER, _LOCAL_ROLE
    if _hub_verified():
        username, role = audit_events.actor(request)
        username = username or _LOCAL_USER
        role = role or _LOCAL_ROLE
    token = flow_token.mint(username, role)
    return _json({"token": token, "expires_in": int(settings.FLOW_ADMIN_TOKEN_TTL_SEC)})
