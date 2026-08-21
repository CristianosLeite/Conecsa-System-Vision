"""Turning a request into an audit record: who, what, and from where.

The gateway audits *per request* rather than per handler. Several routes are
served by more than one view — `controllers/aliases.py` re-exposes the v1
control endpoints under short paths — so a decorator on the handler would
silently miss half of them, while an ``after_request`` hook cannot.

Two rules keep the trail honest:

* **Whitelist the meaning, not the coverage.** ``ROUTE_EVENTS`` names the
  actions we understand. A mutating route that is missing from it is still
  recorded, under the generic ``device.request`` key with its method and path —
  a new endpoint appears in the trail as soon as it exists, rather than being
  invisible until someone remembers to register it.
* **Record what was done, never what it was done with.** ``detail`` is built by
  explicit per-route extraction of a single safe field. Bodies, uploads,
  passwords and Wi-Fi pre-shared keys never reach the buffer.
"""
import logging
from typing import Optional
from urllib.parse import unquote

logger = logging.getLogger(__name__)

# Routes that mutate state but say nothing about a user's intent: hub
# housekeeping, liveness pings, and one per-click interaction that would drown
# the trail (SAM segmentation fires on every click while labelling).
SKIP_RULES = frozenset({
    "/api/v1/detections/backlog/ack",
    "/api/v1/audit/backlog/ack",
    "/api/v1/training/heartbeat",
    "/api/v1/training/sam/segment",
})

# (method, registered rule) → stable event key. The *rule* is used, not the
# concrete path, so `/api/v1/training/datasets/<dataset_id>` is one entry
# rather than one per dataset.
ROUTE_EVENTS = {
    # ── detection control ────────────────────────────────────────────────────
    ("POST", "/api/v1/start"): "detection.start",
    ("POST", "/api/start"): "detection.start",
    ("POST", "/api/v1/stop"): "detection.stop",
    ("POST", "/api/stop"): "detection.stop",
    ("POST", "/api/v1/threshold"): "detection.threshold_changed",
    ("POST", "/api/threshold"): "detection.threshold_changed",
    ("POST", "/api/v1/overlay_threshold"): "detection.overlay_threshold_changed",
    ("POST", "/api/overlay_threshold"): "detection.overlay_threshold_changed",
    ("POST", "/api/v1/stats/reset"): "detection.stats_reset",
    ("POST", "/api/v1/counter/reset"): "detection.counter_reset",
    ("POST", "/api/v1/trigger/enable"): "trigger.enabled",
    ("POST", "/api/v1/trigger/disable"): "trigger.disabled",

    # ── models ───────────────────────────────────────────────────────────────
    ("POST", "/api/v1/model"): "model.uploaded",
    ("POST", "/api/v1/model/select"): "model.selected",
    ("DELETE", "/api/v1/model/<model_name>"): "model.deleted",

    # ── classes ──────────────────────────────────────────────────────────────
    ("POST", "/api/v1/classes"): "classes.set",
    ("POST", "/api/classes"): "classes.set",
    ("DELETE", "/api/v1/classes"): "classes.cleared",

    # ── detection areas ──────────────────────────────────────────────────────
    ("POST", "/api/v1/detection-areas"): "area.created",
    ("DELETE", "/api/v1/detection-areas/<area_id>"): "area.deleted",
    ("POST", "/api/v1/detection-areas/<area_id>/save"): "area.saved",
    ("POST", "/api/v1/detection-areas/<area_id>/edit"): "area.edited",
    ("POST", "/api/v1/detection-areas/<area_id>/discard"): "area.discarded",
    ("POST", "/api/v1/detection-areas/<area_id>/shape"): "area.shape_changed",
    ("POST", "/api/v1/detection-areas/<area_id>/command"): "area.command",

    # ── device configuration ─────────────────────────────────────────────────
    ("POST", "/api/v1/camera/config"): "camera.configured",
    ("PUT", "/api/v1/config"): "settings.changed",
    ("POST", "/api/v1/system/power"): "system.power",
    ("POST", "/api/v1/gpio/trigger"): "gpio.trigger_changed",
    ("POST", "/api/v1/gpio/pin"): "gpio.pin_set",
    ("POST", "/api/v1/network/config"): "network.configured",
    ("POST", "/api/v1/network/wifi/connect"): "network.wifi_connected",
    ("POST", "/api/v1/network/wifi/forget"): "network.wifi_forgotten",

    # ── training session + jobs ──────────────────────────────────────────────
    ("POST", "/api/v1/training/enter"): "training.entered",
    ("POST", "/api/v1/training/exit"): "training.exited",
    ("POST", "/api/v1/training/train"): "training.started",
    ("POST", "/api/v1/training/train/cancel"): "training.cancelled",
    ("POST", "/api/v1/training/train/finish"): "training.finished",
    ("POST", "/api/v1/training/sam/load"): "sam.loaded",
    ("POST", "/api/v1/training/sam/unload"): "sam.unloaded",

    # ── datasets ─────────────────────────────────────────────────────────────
    ("POST", "/api/v1/training/datasets"): "dataset.created",
    ("POST", "/api/v1/training/datasets/upload"): "dataset.uploaded",
    ("PUT", "/api/v1/training/datasets/<dataset_id>"): "dataset.renamed",
    ("DELETE", "/api/v1/training/datasets/<dataset_id>"): "dataset.deleted",
    ("PUT", "/api/v1/training/datasets/<dataset_id>/cover"): "dataset.cover_changed",
    ("POST", "/api/v1/training/datasets/<dataset_id>/classes"): "dataset.class_added",
    ("PUT", "/api/v1/training/datasets/<dataset_id>/classes/<int:index>"):
        "dataset.class_renamed",
    ("DELETE", "/api/v1/training/datasets/<dataset_id>/classes/<int:index>"):
        "dataset.class_removed",

    # ── dataset images ───────────────────────────────────────────────────────
    ("POST", "/api/v1/training/datasets/<dataset_id>/capture"): "image.captured",
    ("POST", "/api/v1/training/datasets/<dataset_id>/images"): "image.added",
    ("DELETE", "/api/v1/training/datasets/<dataset_id>/images/<image_id>"):
        "image.deleted",
    ("POST", "/api/v1/training/datasets/<dataset_id>/images/<image_id>/replicate"):
        "image.replicated",
    ("PUT", "/api/v1/training/datasets/<dataset_id>/images/<image_id>/labels"):
        "image.labels_changed",

    # ── weights ──────────────────────────────────────────────────────────────
    ("POST", "/api/v1/training/weights"): "weights.uploaded",
    ("POST", "/api/v1/training/weights/average"): "weights.averaged",
    ("DELETE", "/api/v1/training/weights/<weights_id>"): "weights.deleted",

    # ── enrollment ───────────────────────────────────────────────────────────
    ("POST", "/enroll/csr"): "enroll.csr_issued",
    ("POST", "/enroll/complete"): "enroll.completed",
    ("POST", "/enroll/reset"): "enroll.reset",
}

# Event key for a mutating route not named above.
GENERIC_EVENT = "device.request"

# rule → the single JSON body field worth recording, when there is one. Only
# these are ever read; anything not listed here contributes no detail at all.
_BODY_DETAIL_FIELDS = {
    "/api/v1/model/select": "model_name",
    "/api/v1/system/power": "action",
    "/api/v1/network/config": "method",
    # The SSID, never the pre-shared key that travels beside it.
    "/api/v1/network/wifi/connect": "ssid",
    "/api/v1/network/wifi/forget": "ssid",
    "/api/v1/training/datasets": "name",
    "/api/v1/training/datasets/<dataset_id>": "name",
    "/api/v1/training/train": "model_name",
}

# rule → the URL parameter worth recording, when the body has nothing better.
_VIEW_ARG_DETAIL = {
    "/api/v1/model/<model_name>": "model_name",
    "/api/v1/training/datasets/<dataset_id>": "dataset_id",
    "/api/v1/training/datasets/<dataset_id>/cover": "dataset_id",
    "/api/v1/training/datasets/<dataset_id>/classes": "dataset_id",
    "/api/v1/training/datasets/<dataset_id>/capture": "dataset_id",
    "/api/v1/training/datasets/<dataset_id>/images": "dataset_id",
    "/api/v1/training/weights/<weights_id>": "weights_id",
}

# Longest detail we keep; names are free text typed by the operator.
MAX_DETAIL_LEN = 120


def should_audit(request) -> bool:
    """Whether this request is a user action worth recording."""
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return False
    rule = _rule(request)
    if rule is None:
        # No route matched: a 404, not an action.
        return False
    return rule not in SKIP_RULES


def event_for(request) -> str:
    """The stable event key for a request, or the generic fallback."""
    rule = _rule(request)
    if rule is None:
        return GENERIC_EVENT
    return ROUTE_EVENTS.get((request.method, rule), GENERIC_EVENT)


def detail_for(request) -> str:
    """A short, safe description of what the action targeted.

    Only fields named in the tables above are read. An unrecognized route
    falls back to its method and path, which names the action without
    exposing anything that travelled in the body.
    """
    rule = _rule(request)
    if rule is None:
        return ""

    field = _BODY_DETAIL_FIELDS.get(rule)
    if field:
        from_body = _body_field(request, field)
        if from_body:
            return from_body[:MAX_DETAIL_LEN]

    arg = _VIEW_ARG_DETAIL.get(rule)
    if arg:
        from_url = (request.view_args or {}).get(arg)
        if from_url:
            return str(from_url)[:MAX_DETAIL_LEN]

    if (request.method, rule) not in ROUTE_EVENTS:
        return f"{request.method} {request.path}"[:MAX_DETAIL_LEN]
    return ""


def actor(request) -> tuple[str, str]:
    """The (username, role) the hub vouched for, or empty strings.

    The device authenticates nobody: these headers are the only identity that
    exists here, and they are trusted only when the request came through the
    mTLS terminator — which the caller establishes. An unattributed action is
    recorded as such rather than guessed at.
    """
    return (
        _decode(request.headers.get("X-Conecsa-User", "")),
        _decode(request.headers.get("X-Conecsa-Role", "")),
    )


def source_ip(request, from_proxy: bool) -> str:
    """Where the action came from.

    Prefers the origin the hub declares (it proxies on the operator's behalf,
    so its own address is the meaningful one), then the peer nginx saw, and
    only then the direct peer — which behind nginx is the terminator itself.

    Both headers are only read when ``from_proxy``: nginx sets X-Forwarded-For
    itself and clears the hub's origin header on the plaintext listener, so
    they mean something coming from it and nothing at all coming from a caller
    that reached the gateway directly. Such a caller is recorded by its actual
    address rather than by whichever one it claimed.
    """
    if from_proxy:
        declared = _decode(request.headers.get("X-Conecsa-Origin-Ip", ""))
        if declared:
            return declared
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.remote_addr or ""


def outcome_for(response) -> str:
    """Whether the action took effect."""
    return "ok" if response.status_code < 400 else "failed"


def _rule(request) -> Optional[str]:
    """The registered rule this request matched, or None."""
    return request.url_rule.rule if request.url_rule is not None else None


def _body_field(request, field: str) -> str:
    """One named field from a JSON or form body; "" when absent or unreadable."""
    try:
        if request.mimetype == "application/json":
            body = request.get_json(silent=True) or {}
            value = body.get(field)
        else:
            # Multipart uploads: read the form field only, never the file part.
            value = request.form.get(field)
    except (ValueError, TypeError, KeyError) as exc:
        # A malformed body is the request's problem, not the trail's.
        logger.debug("audit detail unavailable for %s: %s", field, exc)
        return ""
    return "" if value is None else str(value).strip()


def _decode(value: str) -> str:
    """Percent-decode a header the hub stamped (names may be non-ASCII)."""
    return unquote(value) if value else ""
