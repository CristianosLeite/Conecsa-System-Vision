# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Inference/device REST controllers (/api/v1/* and the /api/* aliases).

Every handler is a thin translation to gRPC (inference control + `os-base`
hardware agent) or POSIX SHM (the MJPEG feeds). Split per resource: detection,
streams (SSE), models, camera (feeds + config), trigger/counter, system
(config/health/power), gpio, network, classes,
areas and the simplified /api/* aliases. All submodules register onto the
single `api_bp` defined here; app.py registers the blueprint.
"""
from flask import Blueprint

api_bp = Blueprint("api", __name__)

# Importing the submodules registers their routes on `api_bp`.
from . import (  # noqa: E402,F401
    aliases,
    application,
    areas,
    audit,
    camera,
    classes,
    detection,
    flow,
    gpio,
    models,
    network,
    streams,
    system,
    trigger,
)
