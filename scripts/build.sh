#!/bin/bash
#
# build.sh — production build of the web frontend.
#
# Downloads the Tailwind CLI if missing, builds minified CSS, compiles every
# Protocol Buffer (scripts/compile-proto.sh), then builds the Leptos/WASM app
# in release mode with Trunk into ../dist. This is the static bundle Nginx
# serves in the web deployment.

# Resolve the project root (one level above the scripts/ directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Add Cargo bin to PATH so tools like trunk are available
export PATH="$HOME/.cargo/bin:$PATH"

cd "$PROJECT_ROOT"

# Build CSS with Tailwind
echo "Building CSS..."
# Pinned + checksum-verified (scripts/tailwind.pin); re-downloads on mismatch.
TAILWIND_BIN="$PROJECT_ROOT/bin/tailwindcss"
"$PROJECT_ROOT/scripts/fetch-tailwind.sh"
"$TAILWIND_BIN" -i ./styles/input.css -o ./system-vision/styles.css --minify

# Compile protobuf files
echo "Compiling protobuf files..."
"$SCRIPT_DIR/compile-proto.sh"

# Build Rust/WASM with Trunk
echo "Building Rust application..."
cd "$PROJECT_ROOT/app" && trunk build --release --dist ../dist
