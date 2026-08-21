"""Audit backlog endpoints — the hub's drain protocol for the device's trail.

Both routes are hub-only: the trail names people, and on a device with no
authentication the mTLS terminator is the only gate there is.

The hub persists a page before acknowledging it, so an unacknowledged page is
simply re-delivered on the next pass. Delivery is therefore at-least-once and
the hub tolerates duplicates — the opposite trade would silently lose records.
"""
from flask import request

from .. import audit
from ..audit import DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT
from ..helpers import _hub_verified, _json, _json_error
from . import api_bp


@api_bp.route('/api/v1/audit/backlog', methods=['GET'])
def get_audit_backlog():
    """GET /api/v1/audit/backlog — one page of recorded actions, oldest first."""
    if not _hub_verified():
        return _json_error("Audit records are readable by the paired hub only", 403)
    limit = request.args.get("limit", type=int)
    if limit is None:
        limit = DEFAULT_PAGE_LIMIT
    elif limit < 0:
        return _json_error('"limit" must be >= 0', 400)
    elif limit > MAX_PAGE_LIMIT:
        limit = MAX_PAGE_LIMIT
    return _json(audit.buffer().list_backlog(limit))


@api_bp.route('/api/v1/audit/backlog/ack', methods=['POST'])
def ack_audit_backlog():
    """POST /api/v1/audit/backlog/ack — drop records the hub has persisted."""
    if not _hub_verified():
        return _json_error("Audit records are readable by the paired hub only", 403)
    ids = (request.get_json(silent=True) or {}).get("ids")
    if not isinstance(ids, list) or not all(type(i) is int for i in ids):
        return _json_error('Body must be {"ids": [int, ...]}', 400)
    deleted = audit.buffer().ack(ids)
    return _json({"success": True, "deleted": deleted})
