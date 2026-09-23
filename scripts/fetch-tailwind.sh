#!/bin/bash

# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: Apache-2.0

# Ensure bin/tailwindcss is the pinned Tailwind CSS standalone CLI.
#
# Used by scripts/build.sh, scripts/build-hub.sh and scripts/dev.sh so every
# workstation runs the same verified Tailwind. Reads scripts/tailwind.pin,
# verifies the cached binary against the pinned SHA256 for this architecture,
# and (re)downloads and verifies it on any mismatch.
# Prints nothing on the happy path; exits non-zero when the download does not
# match the pin.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=tailwind.pin
. "$SCRIPT_DIR/tailwind.pin"

TAILWIND_BIN="${TAILWIND_BIN:-$PROJECT_ROOT/bin/tailwindcss}"

case "$(uname -m)" in
    x86_64)  asset="tailwindcss-linux-x64";   sum="$TAILWIND_SHA256_X86_64" ;;
    aarch64) asset="tailwindcss-linux-arm64"; sum="$TAILWIND_SHA256_AARCH64" ;;
    *) echo "fetch-tailwind: unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

verified() {
    [ -f "$TAILWIND_BIN" ] && echo "$sum  $TAILWIND_BIN" | sha256sum -c - >/dev/null 2>&1
}

if verified; then
    exit 0
fi

echo "Tailwind CSS v${TAILWIND_VERSION} (${asset}): downloading pinned binary..."
mkdir -p "$(dirname "$TAILWIND_BIN")"
tmp="$(mktemp "$(dirname "$TAILWIND_BIN")/.tailwindcss.XXXXXX")"
trap 'rm -f "$tmp"' EXIT
curl -fsSL -o "$tmp" \
    "https://github.com/tailwindlabs/tailwindcss/releases/download/v${TAILWIND_VERSION}/${asset}"
echo "$sum  $tmp" | sha256sum -c - >/dev/null
chmod +x "$tmp"
mv -f "$tmp" "$TAILWIND_BIN"
trap - EXIT
