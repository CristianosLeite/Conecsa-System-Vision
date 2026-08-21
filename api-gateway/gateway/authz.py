"""Server-side role authorization for mutating routes.

The device authenticates nobody itself: the only identity that exists here is
the ``X-Conecsa-Role`` header the hub stamps onto requests it relays for a
logged-in operator, trusted only when the request came through the mTLS
terminator (``_hub_verified``). Until now that header was read for audit
attribution only, so the role checks in the device UI were purely cosmetic —
any operator could reach every mutating endpoint. This module makes the role
an enforced boundary.

Three request classes, decided per mutating request:

* **Not hub-verified** — a caller on the internal compose network or a dev
  deployment. No role exists to enforce; the pre-existing network boundary
  applies unchanged.
* **Hub-verified without a role header** — the hub acting on its own behalf
  (applying a recipe, draining backlogs, pairing). The hub's command layer
  authorizes those before issuing them, and its proxy always stamps a role on
  operator traffic, so an absent role cannot be forged by an operator.
* **Hub-verified with a role header** — an operator action relayed by the
  hub's device proxy. The role must be one of the three known roles and meet
  the route's policy; unknown roles and un-policied mutating routes are
  rejected outright.

``ROUTE_POLICIES`` is keyed like ``audit_events.ROUTE_EVENTS`` — by
``(method, registered werkzeug rule)`` — and `tests/test_authz.py` walks the
app's url_map to fail whenever a new mutating route lacks a policy.
"""
import logging
from urllib.parse import unquote

from .helpers import _hub_verified, _json_error

logger = logging.getLogger(__name__)

ROLE_USER = "user"
ROLE_ADMIN = "admin"
ROLE_OWNER = "owner"

_ROLE_RANK = {ROLE_USER: 0, ROLE_ADMIN: 1, ROLE_OWNER: 2}

MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

# Blueprints with their own authorization policy (enrollment uses the pairing
# token / un-enrolled state machine in gateway/enroll.py).
EXEMPT_BLUEPRINTS = frozenset({"enroll"})

# (method, registered rule) → minimum role. Operating the detector is a
# `user` action; anything that reconfigures the device, its models, network,
# GPIO, power, or training data requires `admin`.
ROUTE_POLICIES = {
    # ── detection control (operate) ──────────────────────────────────────────
    ("POST", "/api/v1/start"): ROLE_USER,
    ("POST", "/api/start"): ROLE_USER,
    ("POST", "/api/v1/stop"): ROLE_USER,
    ("POST", "/api/stop"): ROLE_USER,
    ("POST", "/api/v1/threshold"): ROLE_USER,
    ("POST", "/api/threshold"): ROLE_USER,
    ("POST", "/api/v1/overlay_threshold"): ROLE_USER,
    ("POST", "/api/overlay_threshold"): ROLE_USER,
    ("POST", "/api/v1/stats/reset"): ROLE_USER,
    ("POST", "/api/v1/counter/reset"): ROLE_USER,
    ("POST", "/api/v1/trigger/enable"): ROLE_USER,
    ("POST", "/api/v1/trigger/disable"): ROLE_USER,

    # ── hub housekeeping (normally issued without an operator) ───────────────
    ("POST", "/api/v1/detections/backlog/ack"): ROLE_ADMIN,
    ("POST", "/api/v1/audit/backlog/ack"): ROLE_ADMIN,

    # ── models / classes (configure) ─────────────────────────────────────────
    ("POST", "/api/v1/model"): ROLE_ADMIN,
    ("POST", "/api/v1/model/select"): ROLE_ADMIN,
    ("DELETE", "/api/v1/model/<model_name>"): ROLE_ADMIN,
    ("POST", "/api/v1/classes"): ROLE_ADMIN,
    ("POST", "/api/classes"): ROLE_ADMIN,
    ("DELETE", "/api/v1/classes"): ROLE_ADMIN,

    # ── detection areas ──────────────────────────────────────────────────────
    ("POST", "/api/v1/detection-areas"): ROLE_ADMIN,
    ("DELETE", "/api/v1/detection-areas/<area_id>"): ROLE_ADMIN,
    ("POST", "/api/v1/detection-areas/<area_id>/save"): ROLE_ADMIN,
    ("POST", "/api/v1/detection-areas/<area_id>/edit"): ROLE_ADMIN,
    ("POST", "/api/v1/detection-areas/<area_id>/discard"): ROLE_ADMIN,
    ("POST", "/api/v1/detection-areas/<area_id>/shape"): ROLE_ADMIN,
    ("POST", "/api/v1/detection-areas/<area_id>/command"): ROLE_ADMIN,

    # ── device configuration ─────────────────────────────────────────────────
    ("POST", "/api/v1/camera/config"): ROLE_ADMIN,
    ("PUT", "/api/v1/config"): ROLE_ADMIN,
    ("POST", "/api/v1/system/power"): ROLE_ADMIN,
    ("POST", "/api/v1/gpio/trigger"): ROLE_ADMIN,
    ("POST", "/api/v1/gpio/pin"): ROLE_ADMIN,
    ("POST", "/api/v1/network/config"): ROLE_ADMIN,
    ("POST", "/api/v1/network/wifi/connect"): ROLE_ADMIN,
    ("POST", "/api/v1/network/wifi/forget"): ROLE_ADMIN,

    # ── training session + jobs ──────────────────────────────────────────────
    ("POST", "/api/v1/training/enter"): ROLE_ADMIN,
    ("POST", "/api/v1/training/exit"): ROLE_ADMIN,
    ("POST", "/api/v1/training/heartbeat"): ROLE_ADMIN,
    ("POST", "/api/v1/training/train"): ROLE_ADMIN,
    ("POST", "/api/v1/training/train/cancel"): ROLE_ADMIN,
    ("POST", "/api/v1/training/train/finish"): ROLE_ADMIN,
    ("POST", "/api/v1/training/sam/load"): ROLE_ADMIN,
    ("POST", "/api/v1/training/sam/unload"): ROLE_ADMIN,
    ("POST", "/api/v1/training/sam/segment"): ROLE_ADMIN,

    # ── datasets / images / weights ──────────────────────────────────────────
    ("POST", "/api/v1/training/datasets"): ROLE_ADMIN,
    ("POST", "/api/v1/training/datasets/upload"): ROLE_ADMIN,
    ("PUT", "/api/v1/training/datasets/<dataset_id>"): ROLE_ADMIN,
    ("DELETE", "/api/v1/training/datasets/<dataset_id>"): ROLE_ADMIN,
    ("PUT", "/api/v1/training/datasets/<dataset_id>/cover"): ROLE_ADMIN,
    ("POST", "/api/v1/training/datasets/<dataset_id>/classes"): ROLE_ADMIN,
    ("PUT", "/api/v1/training/datasets/<dataset_id>/classes/<int:index>"): ROLE_ADMIN,
    ("DELETE", "/api/v1/training/datasets/<dataset_id>/classes/<int:index>"): ROLE_ADMIN,
    ("POST", "/api/v1/training/datasets/<dataset_id>/capture"): ROLE_ADMIN,
    ("POST", "/api/v1/training/datasets/<dataset_id>/images"): ROLE_ADMIN,
    ("DELETE", "/api/v1/training/datasets/<dataset_id>/images/<image_id>"): ROLE_ADMIN,
    ("POST", "/api/v1/training/datasets/<dataset_id>/images/<image_id>/replicate"):
        ROLE_ADMIN,
    ("PUT", "/api/v1/training/datasets/<dataset_id>/images/<image_id>/labels"):
        ROLE_ADMIN,
    ("POST", "/api/v1/training/weights"): ROLE_ADMIN,
    ("POST", "/api/v1/training/weights/average"): ROLE_ADMIN,
    ("DELETE", "/api/v1/training/weights/<weights_id>"): ROLE_ADMIN,
}


def enforce(request):
    """``before_request`` hook body: a 403 Response, or None to proceed.

    Runs after the clock hook (a rejected request still keeps the device's
    clock right) and before any handler; the audit ``after_request`` hook then
    records the rejected attempt with outcome "failed".
    """
    if request.method not in MUTATING_METHODS:
        return None
    if request.blueprint in EXEMPT_BLUEPRINTS:
        return None
    if not _hub_verified():
        return None
    raw = request.headers.get("X-Conecsa-Role", "")
    if not raw:
        # The hub acting on its own behalf — see the module docstring.
        return None
    role = unquote(raw)
    if role not in _ROLE_RANK:
        logger.warning("authz: rejected unknown role %r on %s %s",
                       role, request.method, request.path)
        return _forbidden()
    rule = request.url_rule.rule if request.url_rule is not None else None
    if rule is None:
        # No route matched; let the 404 handler answer.
        return None
    required = ROUTE_POLICIES.get((request.method, rule))
    if required is None:
        # Default-deny: a mutating route nobody assigned a policy to is a
        # policy bug, not an open door. test_authz.py keeps this unreachable.
        logger.error("authz: no policy for mutating route %s %s — denying",
                     request.method, rule)
        return _forbidden()
    if _ROLE_RANK[role] < _ROLE_RANK[required]:
        logger.info("authz: %s denied for role %s on %s %s",
                    required, role, request.method, request.path)
        return _forbidden()
    return None


def _forbidden():
    return _json_error("forbidden", 403)
