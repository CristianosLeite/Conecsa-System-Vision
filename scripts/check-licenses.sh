#!/bin/bash

# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: Apache-2.0

#
# check-licenses.sh — license compliance gate (CI lint job and scripts/test.sh).
#
# 1. `reuse lint`: every file declares its copyright holder and SPDX license,
#    through a header or a REUSE.toml annotation, and every license in use has
#    its text under LICENSES/.
# 2. The Apache-2.0 Python layers never import AGPL code: neither
#    `ultralytics` nor the AGPL service packages (api, service, gateway, agent).
#
# The REUSE check is skipped with a warning when `reuse` is not installed
# (requirements-dev.txt pins it; CI installs it). Set REUSE to override the binary.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
status=0

REUSE="${REUSE:-}"
if [ -z "$REUSE" ]; then
    if [ -x .venv/bin/reuse ]; then
        REUSE=.venv/bin/reuse
    elif command -v reuse >/dev/null 2>&1; then
        REUSE=reuse
    fi
fi
if [ -n "$REUSE" ]; then
    if out="$("$REUSE" lint 2>&1)"; then
        echo "check-licenses: reuse lint ok"
    else
        echo "$out" >&2
        echo "check-licenses: reuse lint failed (add an SPDX header or a REUSE.toml annotation)" >&2
        status=1
    fi
else
    echo "check-licenses: reuse not installed, skipping the REUSE check (pip install -r requirements-dev.txt)" >&2
fi

APACHE_PYTHON=(os-base/conecsa_shm os-base/conecsa_common scripts docs)
if bad="$(grep -rnE --include='*.py' \
        '^[[:space:]]*(import|from)[[:space:]]+(ultralytics|api|service|gateway|agent)([.[:space:]]|$)' \
        "${APACHE_PYTHON[@]}" 2>/dev/null || true)"; [ -n "$bad" ]; then
    echo "check-licenses: Apache-2.0 code imports AGPL code:" >&2
    echo "$bad" >&2
    status=1
fi

if [ "$status" -eq 0 ]; then
    echo "check-licenses: ok"
fi
exit "$status"
