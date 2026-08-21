"""API gateway Flask app.

Mirrors the inference-service monolith's REST/SSE/MJPEG surface byte-for-byte,
but every handler is a thin translation to gRPC (inference control + os hardware
agent) or POSIX SHM (the two MJPEG feeds). The external contract — routes,
status codes, JSON shapes, protobuf negotiation, SSE envelopes — is unchanged so
the Leptos app and Node-RED flows need no edits.

This module only assembles the app: the handlers live in the `controllers`
package (per-resource, on `api_bp`), the training surface in `training` and
device enrollment in `enroll`; shared response/event helpers are in `helpers`.
"""
import logging

from flask import Flask, request
from werkzeug.exceptions import HTTPException

from .config import settings
from .helpers import _json

logger = logging.getLogger(__name__)

app = Flask(__name__)
# Bound multipart uploads (models, dataset ZIPs) before they hit the relays;
# the training-service enforces its own TRAINING_MAX_UPLOAD_MB on top.
app.config["MAX_CONTENT_LENGTH"] = 600 * 1024 * 1024

# No CORS: the frontend is served same-origin (through the device's nginx and the
# hub's reverse proxy), so cross-origin access is neither needed nor allowed.

# Inference/device surface (/api/v1/* and the /api/* aliases).
from .controllers import api_bp  # noqa: E402

app.register_blueprint(api_bp)

# Training-service surface (/api/v1/training/*) lives in its own package.
from .training import training_bp  # noqa: E402

app.register_blueprint(training_bp)

# Device enrollment surface (/enroll/*): the hub pairs the device and signs its
# server certificate. Kept outside /api so it stays reachable during bootstrap.
from .enroll import enroll_bp  # noqa: E402

app.register_blueprint(enroll_bp)

# Imported after the blueprints on purpose: clock needs the gRPC clients
# module loaded, audit opens its buffer once the routes it records exist, and
# authz's policy table is checked against the final url_map in tests.
from . import audit, authz, clock  # noqa: E402


@app.before_request
def sync_hub_clock():
    """Honour the hub's clock stamp on any verified mTLS request.

    Central rather than per-route so every hub call keeps the device's clock
    right — which is what keeps the mTLS channel itself working on a host with
    no RTC battery. See gateway/clock.py.
    """
    clock.sync_from_request_headers(request.headers)


@app.before_request
def enforce_role_policy():
    """Authorize mutating requests by the role the hub vouched for.

    Registered after the clock hook so a rejected request still corrects the
    device clock. See gateway/authz.py for the policy and trust model.
    """
    return authz.enforce(request)


@app.after_request
def record_audit(response):
    """Append every mutating request to the device's audit trail.

    Central rather than per-route: several control endpoints are served by more
    than one view (`controllers/aliases.py` re-exposes them under short paths),
    so a per-handler decorator would miss half of them. See gateway/audit.py.
    """
    audit.record_request(request, response)
    return response


@app.errorhandler(HTTPException)
def handle_http_exception(ex: HTTPException):
    """Preserve deliberate HTTP semantics with a JSON body.

    Without this, HTTPException subclasses fall into the catch-all below and
    e.g. the 413 for an over-limit upload (MAX_CONTENT_LENGTH) turned into a
    500. Werkzeug's name/description are generic, so they are safe to relay.
    """
    response = ex.get_response()
    body = _json({"error": ex.name, "message": ex.description},
                 response.status_code)
    body.headers.extend(
        (k, v) for k, v in response.headers.items()
        if k.lower() not in ("content-type", "content-length"))
    return body


@app.errorhandler(Exception)
def handle_exception(ex):
    """Catch-all: log the full traceback, answer with a stable generic body.

    Exception text and class names are internals — they belong in the log,
    not in the response. GATEWAY_DEBUG_ERRORS=true restores them for
    development.
    """
    logger.exception("Unhandled exception")
    body = {"error": "Internal server error"}
    if settings.DEBUG_ERRORS:
        body["message"] = str(ex)
        body["type"] = type(ex).__name__
    return _json(body, 500)


@app.errorhandler(404)
def not_found(_):
    """Flask error handler."""
    return _json({"error": "Route not found"}, 404)
