#!/bin/bash
# Ensure bin/trunk is the pinned prebuilt Trunk release.
#
# Mirrors scripts/fetch-tailwind.sh for Trunk: the Dockerfiles already download
# the pinned tarball and verify it, but the host-side manual build
# (scripts/build-manual.sh) and CI had no equivalent — and `cargo install trunk`
# is forbidden (.agents/knowledge/ops/builds.md). Reads scripts/trunk.pin,
# checks the cached binary reports the pinned version, and otherwise downloads
# the tarball, verifies its SHA256 for this architecture and extracts it.
# Prints nothing on the happy path; exits non-zero on a checksum mismatch.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=trunk.pin
. "$SCRIPT_DIR/trunk.pin"

TRUNK_BIN="${TRUNK_BIN:-$PROJECT_ROOT/bin/trunk}"

case "$(uname -m)" in
    x86_64)  arch="x86_64";  sum="$TRUNK_SHA256_X86_64" ;;
    aarch64) arch="aarch64"; sum="$TRUNK_SHA256_AARCH64" ;;
    *) echo "fetch-trunk: unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

verified() {
    [ -x "$TRUNK_BIN" ] && [ "$("$TRUNK_BIN" --version 2>/dev/null)" = "trunk $TRUNK_VERSION" ]
}

if verified; then
    exit 0
fi

echo "Trunk v${TRUNK_VERSION} (${arch}): downloading pinned release..."
mkdir -p "$(dirname "$TRUNK_BIN")"
tmp="$(mktemp "$(dirname "$TRUNK_BIN")/.trunk.XXXXXX.tar.gz")"
trap 'rm -f "$tmp"' EXIT
curl -fsSL -o "$tmp" \
    "https://github.com/trunk-rs/trunk/releases/download/v${TRUNK_VERSION}/trunk-${arch}-unknown-linux-gnu.tar.gz"
echo "$sum  $tmp" | sha256sum -c - >/dev/null
tar -xzf "$tmp" -C "$(dirname "$TRUNK_BIN")" trunk
[ "$(basename "$TRUNK_BIN")" = trunk ] || mv -f "$(dirname "$TRUNK_BIN")/trunk" "$TRUNK_BIN"
chmod +x "$TRUNK_BIN"
trap - EXIT
verified || { echo "fetch-trunk: downloaded binary does not report trunk $TRUNK_VERSION" >&2; exit 1; }
