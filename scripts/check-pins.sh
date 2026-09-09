#!/bin/bash
# Reject moving build inputs (CI lint gate).
#
# 1. No Dockerfile or script may download from a GitHub `releases/latest`
#    URL: identical source revisions must build identical images, and a
#    download that is not pinned cannot be checksum-verified.
# 2. Every production requirements file uses exact `==` pins (pip-audit reads
#    them, and the base image bakes them in).
# 3. The Tailwind pin is the same in scripts/tailwind.pin and every Dockerfile
#    that mirrors it as ARGs.
# 4. CI installs Rust tools as pinned prebuilt binaries (taiki-e/install-action
#    with `tool: name@version` and `fallback: none`), never `cargo install`:
#    compiling wasm-pack and cargo-audit from source cost 470s of every run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
status=0

if grep -rn --include='Dockerfile*' --include='*.sh' --exclude=check-pins.sh \
        --exclude-dir=.git --exclude-dir=node_modules --exclude-dir=target \
        --exclude-dir=.venv --exclude-dir=yocto 'releases/latest/download' . ; then
    echo "check-pins: unpinned GitHub 'latest' download(s) above; pin a version and a SHA256" >&2
    status=1
fi

for req in os-base/requirements-common.txt os-base/agent/requirements.txt \
           api-gateway/requirements.txt inference-service/requirements.txt \
           training-service/requirements.txt; do
    # Ignore comments, blank lines, `-r` includes and pip options.
    if bad="$(grep -vE '^[[:space:]]*(#|$|-r |--)' "$req" | grep -vE '^[][A-Za-z0-9_.-]+==[^ ]+' || true)"; [ -n "$bad" ]; then
        echo "check-pins: $req has requirements without an exact '==' pin:" >&2
        echo "$bad" >&2
        status=1
    fi
done

# shellcheck source=tailwind.pin
. scripts/tailwind.pin
for df in system-vision/Dockerfile.system-vision system-vision/Dockerfile.system-vision.dev \
          hub-vision/Dockerfile.hub-builder; do
    [ -f "$df" ] || continue
    for var in TAILWIND_VERSION TAILWIND_SHA256_X86_64 TAILWIND_SHA256_AARCH64; do
        want="${!var}"
        if grep -q "^ARG ${var}=" "$df" && ! grep -q "^ARG ${var}=${want}$" "$df"; then
            echo "check-pins: $df pins ${var} differently from scripts/tailwind.pin" >&2
            status=1
        fi
    done
done

# The Trunk pin (scripts/trunk.pin, used by scripts/fetch-trunk.sh for the
# manual build) must agree with the ARGs the Dockerfiles carry; the hub builder
# writes the version with a leading "v".
# shellcheck source=trunk.pin
. scripts/trunk.pin
for df in system-vision/Dockerfile.system-vision system-vision/Dockerfile.system-vision.dev \
          hub-vision/Dockerfile.hub-builder; do
    [ -f "$df" ] || continue
    if grep -q '^ARG TRUNK_VERSION=' "$df" && ! grep -qE "^ARG TRUNK_VERSION=v?${TRUNK_VERSION}$" "$df"; then
        echo "check-pins: $df pins TRUNK_VERSION differently from scripts/trunk.pin" >&2
        status=1
    fi
    for var in TRUNK_SHA256_X86_64 TRUNK_SHA256_AARCH64; do
        want="${!var}"
        if grep -q "^ARG ${var}=" "$df" && ! grep -q "^ARG ${var}=${want}$" "$df"; then
            echo "check-pins: $df pins ${var} differently from scripts/trunk.pin" >&2
            status=1
        fi
    done
done

# CI tools: `cargo install <tool>` compiles the tool from source on every run
# (wasm-pack 148s, cargo-audit 325s before this gate existed). Use
# taiki-e/install-action with an explicit `tool: name@version` instead.
# The optional group lets both `run: cargo install x` and the body line of a
# `run: |` block match while a line whose first token is `#` never does — the
# workflows explain the rule in prose.
if bad="$(grep -rn --include='*.yml' --include='*.yaml' -E \
        '^[[:space:]]*([^#[:space:]][^#]*)?cargo install ' .github/workflows/ || true)"; [ -n "$bad" ]; then
    echo "check-pins: 'cargo install' in a workflow; use taiki-e/install-action with a pinned tool@version:" >&2
    echo "$bad" >&2
    status=1
fi

# Every taiki-e/install-action step must pin every tool (`name@version`, as a
# single value, a comma-separated list or a `tool: |` block) and set
# `fallback: none`: the action's default fallback quietly degrades a version it
# has not ingested yet to cargo-binstall/quickinstall or a from-source build —
# exactly the unpinned path the rule above forbids.
# shellcheck disable=SC2016  # awk program: $0 and the $ anchors belong to awk
install_action_check='
function check_entry(e, where) {
    gsub(/^[ \t"\047]+|[ \t"\047]+$/, "", e)   # \047 = single quote
    if (e == "") return
    if (e !~ /^[A-Za-z0-9_.-]+@[0-9][A-Za-z0-9_.+-]*$/)
        print where ": unpinned install-action tool \"" e "\""
}
function end_step() {
    if (in_step) {
        if (!have_tool) print step_where ": install-action step without a tool:"
        if (!have_fallback) print step_where ": install-action step without \"fallback: none\""
    }
    in_step = 0; in_block = 0
}
FNR == 1 { end_step() }
/^[ \t]*-?[ \t]*uses:[ \t]*taiki-e\/install-action/ {
    end_step(); in_step = 1; step_where = FILENAME ":" FNR; have_tool = 0; have_fallback = 0; next
}
!in_step { next }
/^[ \t]*-[ \t]+(name|uses|run|id|if|env|with):/ { end_step(); next }
in_block {
    match($0, /^[ \t]*/)
    if ($0 !~ /^[ \t]*$/ && RLENGTH > block_indent) { check_entry($0, FILENAME ":" FNR); next }
    in_block = 0
}
/^[ \t]*tool:/ {
    have_tool = 1
    v = $0; sub(/^[ \t]*tool:[ \t]*/, "", v)
    if (v ~ /^[|>]/) { in_block = 1; match($0, /^[ \t]*/); block_indent = RLENGTH; next }
    n = split(v, parts, ",")
    for (i = 1; i <= n; i++) check_entry(parts[i], FILENAME ":" FNR)
    next
}
/^[ \t]*fallback:[ \t]*none[ \t]*$/ { have_fallback = 1 }
END { end_step() }
'
if bad="$(find .github/workflows -name '*.yml' -o -name '*.yaml' | sort | xargs awk "$install_action_check")"; [ -n "$bad" ]; then
    echo "check-pins: taiki-e/install-action steps must pin every tool and set 'fallback: none':" >&2
    echo "$bad" >&2
    status=1
fi

if [ "$status" -eq 0 ]; then
    echo "check-pins: ok"
fi
exit "$status"
