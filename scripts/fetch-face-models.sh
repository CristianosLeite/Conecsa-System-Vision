#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only
#
# Fetch the pinned face recognition models into inference-service/assets/face/.
#
# The face application type is served only by a build that carries both graphs
# (api/postprocess/_face_assets.py), so run this before building the
# inference-service image. Already-present files are verified, not re-fetched.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DEST="${FACE_MODELS_DIR:-$REPO_ROOT/inference-service/assets/face}"

# shellcheck source=face-models.pin
source "$SCRIPT_DIR/face-models.pin"

fetch_one() {
  local path="$1" file="$2" sha="$3"
  local target="$DEST/$file"
  if [ -f "$target" ] && echo "$sha  $target" | sha256sum --check --status; then
    echo "✓ $file (already present)"
    return
  fi
  echo "→ $file"
  curl -fsSL --retry 3 -o "$target.tmp" "$FACE_MODELS_BASE_URL/$path"
  if ! echo "$sha  $target.tmp" | sha256sum --check --status; then
    rm -f "$target.tmp"
    echo "✗ $file does not match its pinned SHA256 (scripts/face-models.pin)" >&2
    exit 1
  fi
  mv "$target.tmp" "$target"
  echo "✓ $file"
}

mkdir -p "$DEST"
fetch_one "$FACE_DETECTOR_PATH" "$FACE_DETECTOR_FILE" "$FACE_DETECTOR_SHA256"
fetch_one "$FACE_EMBEDDER_PATH" "$FACE_EMBEDDER_FILE" "$FACE_EMBEDDER_SHA256"
echo "Face models ready in $DEST"
