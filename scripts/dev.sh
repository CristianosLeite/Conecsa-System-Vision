#!/bin/bash
#
# dev.sh — local full-stack dev launcher.
#
# Starts the whole stack for development as background jobs with prefixed,
# interleaved logs, and tears everything down on Ctrl+C (single trap):
#
#   webcam-server   (Rust)      POSIX SHM producer (conecsa_frame_shm), no port
#   inference-service (Python)  gRPC :50061, reads frames / writes processed SHM
#   training-service  (Python)  gRPC :50071
#   api-gateway     (Python)    HTTP API :5000 (the only HTTP surface)
#   system-vision   (WASM)      Tailwind watch + trunk serve on :18080
#   tls terminator  (Docker)    nginx :443 — enroll/mTLS gate the hub talks to
#                               (dev twin of the system-vision container)
#   flow            (Docker)    Node-RED :1880 (production image, /flow proxy)
#
# All services start by default; use flags to skip the GPU-heavy ones on a
# machine without the TensorRT/PyCUDA stack:
#
#   ./scripts/dev.sh                          # everything
#   ./scripts/dev.sh --no-inference --no-training   # frontend + gateway only
#   ./scripts/dev.sh --gateway-only           # gateway + frontend, no webcam/GPU
#
# Run ./scripts/init.sh first to create the .venv and compile the protos.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export PATH="$HOME/.cargo/bin:$PATH"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# ---------------------------------------------------------------------------
# Arg parsing — default is "start everything".
# ---------------------------------------------------------------------------
RUN_WEBCAM=1
RUN_INFERENCE=1
RUN_TRAINING=1
RUN_APP=1
RUN_TLS=1
RUN_FLOW=1

usage() {
    cat <<'EOF'
Usage: ./scripts/dev.sh [options]

  --no-webcam      Don't start webcam-server.
  --no-inference   Don't start inference-service (GPU stack).
  --no-training    Don't start training-service (GPU stack).
  --no-app         Don't start the web frontend (Tailwind + trunk).
  --no-tls         Don't start the :443 mTLS terminator (nginx in Docker).
                   Without it the hub cannot discover/pair this machine.
  --no-flow        Don't start Node-RED (the production flow image, in Docker).
  --gateway-only   Only api-gateway + frontend (implies --no-webcam
                   --no-inference --no-training --no-flow).
  --help           Show this help.

The api-gateway always starts (it is the core HTTP surface).
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --no-webcam)    RUN_WEBCAM=0 ;;
        --no-inference) RUN_INFERENCE=0 ;;
        --no-training)  RUN_TRAINING=0 ;;
        --no-app)       RUN_APP=0 ;;
        --no-tls)       RUN_TLS=0 ;;
        --no-flow)      RUN_FLOW=0 ;;
        --gateway-only) RUN_WEBCAM=0; RUN_INFERENCE=0; RUN_TRAINING=0; RUN_FLOW=0 ;;
        --help|-h)      usage; exit 0 ;;
        *) echo -e "${RED}Unknown argument: $1${NC}" >&2; usage; exit 1 ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# venv guard — activate the root .venv if it isn't already active.
# ---------------------------------------------------------------------------
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$PROJECT_ROOT/.venv/bin/activate"
    else
        echo -e "${RED}No .venv found.${NC} Run ./scripts/init.sh first." >&2
        exit 1
    fi
fi

# The Python services import the shared `conecsa_shm` package (os-base/conecsa_shm),
# which the images copy into dist-packages. Locally, put os-base/ on the path so the
# gateway/inference/training imports resolve regardless of the .pth init.sh adds.
export PYTHONPATH="$PROJECT_ROOT/os-base${PYTHONPATH:+:$PYTHONPATH}"

# In production the services get Docker volumes mounted at /data/{models,
# training,detections}. Locally /data isn't writable, so mirror that layout
# under .dev-data/ and point the services' env overrides at it.
DEV_DATA_DIR="${DEV_DATA_DIR:-$PROJECT_ROOT/.dev-data}"
mkdir -p "$DEV_DATA_DIR/models" "$DEV_DATA_DIR/training" "$DEV_DATA_DIR/detections"
export MODELS_DIR="${MODELS_DIR:-$DEV_DATA_DIR/models}"
export TRAINING_DATA_DIR="${TRAINING_DATA_DIR:-$DEV_DATA_DIR/training}"
export DETECTIONS_DIR="${DETECTIONS_DIR:-$DEV_DATA_DIR/detections}"

# Enrollment/mTLS state: in production a Docker volume shared by the gateway
# (writes device key + hub-signed certs) and the nginx terminator (reads them).
# Locally the same dir backs both the gateway and the dev :443 container.
export CONECSA_CERT_DIR="${CONECSA_CERT_DIR:-$DEV_DATA_DIR/certs}"
mkdir -p "$CONECSA_CERT_DIR"
# The dev terminator runs with host networking, so hub calls it relays reach
# the gateway from loopback — trust it as the terminator peer.
if [ "$RUN_TLS" -eq 1 ]; then
    export TRUSTED_PROXY_HOST="${TRUSTED_PROXY_HOST:-localhost}"
fi

# The service configs default to Docker service hostnames (inference-service:50061,
# …). For local dev point the peers at localhost (override-able from the env).
export INFERENCE_GRPC_ADDR="${INFERENCE_GRPC_ADDR:-localhost:50061}"
export TRAINING_GRPC_ADDR="${TRAINING_GRPC_ADDR:-localhost:50071}"
export HARDWARE_AGENT_ADDR="${HARDWARE_AGENT_ADDR:-localhost:50051}"
export GATEWAY_ADDR="${GATEWAY_ADDR:-http://localhost:5000}"

# ---------------------------------------------------------------------------
# Process management — track each service's PID and, on exit, tear down each one
# together with its descendants (cargo's spawned binary, trunk's build procs,
# …). SIGTERM first, then SIGKILL any survivors after a short grace period
# (waitress/the gateway doesn't always stop promptly on SIGTERM alone). We kill
# specific subtrees rather than `kill 0` so the script stays alive long enough
# to escalate to SIGKILL.
# ---------------------------------------------------------------------------
PIDS=()

kill_tree() {
    local pid="$1" sig="$2" child
    for child in $(pgrep -P "$pid" 2>/dev/null); do
        kill_tree "$child" "$sig"
    done
    kill "-$sig" "$pid" 2>/dev/null
}

cleanup() {
    trap - INT TERM EXIT
    echo -e "\n${YELLOW}Shutting down...${NC}"
    for pid in "${PIDS[@]}"; do kill_tree "$pid" TERM; done
    [ "$RUN_TLS" -eq 1 ] && docker rm -f conecsa-dev-tls >/dev/null 2>&1
    [ "$RUN_FLOW" -eq 1 ] && docker rm -f conecsa-dev-flow >/dev/null 2>&1
    sleep 2
    for pid in "${PIDS[@]}"; do kill_tree "$pid" KILL; done
}
trap cleanup INT TERM EXIT

# run_svc NAME COLOR CMD...  — launch CMD as a background job, tagging each of
# its output lines with a colored [NAME] prefix (line-buffered via stdbuf), and
# record its PID for cleanup.
run_svc() {
    local name="$1"; local color="$2"; shift 2
    echo -e "${color}▶ starting ${name}${NC}"
    local tag
    tag="$(printf '%b' "${color}")[${name}]$(printf '%b' "${NC}") "
    "$@" > >(stdbuf -oL sed "s/^/${tag}/") 2>&1 &
    PIDS+=($!)
}

# ---------------------------------------------------------------------------
# Frontend prerequisite: Tailwind CLI (downloaded on first run).
# ---------------------------------------------------------------------------
if [ "$RUN_APP" -eq 1 ]; then
    TAILWIND_BIN="$PROJECT_ROOT/bin/tailwindcss"
    if [ ! -f "$TAILWIND_BIN" ]; then
        echo "Tailwind binary not found, downloading..."
        mkdir -p "$PROJECT_ROOT/bin"
        ARCH=$(uname -m)
        case "$ARCH" in
            x86_64)  TAILWIND_ASSET="tailwindcss-linux-x64" ;;
            aarch64) TAILWIND_ASSET="tailwindcss-linux-arm64" ;;
            *)       echo -e "${RED}Unsupported architecture: $ARCH${NC}"; exit 1 ;;
        esac
        curl -fsSL -o "$TAILWIND_BIN" \
            "https://github.com/tailwindlabs/tailwindcss/releases/latest/download/${TAILWIND_ASSET}"
        chmod +x "$TAILWIND_BIN"
    fi
fi

# ---------------------------------------------------------------------------
# Start services. Order matters: webcam-server produces the SHM ring that
# inference/training consume, so it goes first. (Consumers self-reconnect, so a
# strict barrier isn't required — a short head start is enough.)
# ---------------------------------------------------------------------------
if [ "$RUN_WEBCAM" -eq 1 ]; then
    run_svc "webcam" "$BLUE" \
        cargo run --release --manifest-path "$PROJECT_ROOT/webcam-server/Cargo.toml"
    sleep 1
fi

if [ "$RUN_INFERENCE" -eq 1 ]; then
    run_svc "inference" "$GREEN" \
        bash -c "cd '$PROJECT_ROOT/inference-service' && exec python3 -m main"
fi

if [ "$RUN_TRAINING" -eq 1 ]; then
    run_svc "training" "$YELLOW" \
        bash -c "cd '$PROJECT_ROOT/training-service' && exec python3 -m main"
fi

# api-gateway always runs — it's the HTTP surface the frontend talks to.
run_svc "gateway" "$BLUE" \
    python3 "$PROJECT_ROOT/api-gateway/main.py"

# :443 mTLS terminator — dev twin of the system-vision nginx container. The hub
# only ever talks to a device through this port (enroll bootstrap, then mTLS),
# so without it the machine is undiscoverable/unpairable from the hub.
if [ "$RUN_TLS" -eq 1 ]; then
    if ! docker info >/dev/null 2>&1; then
        echo -e "${RED}Docker unavailable — skipping the :443 terminator (hub pairing disabled).${NC}"
        RUN_TLS=0
    fi
fi
if [ "$RUN_TLS" -eq 1 ]; then
    # Snakeoil cert for the enrollment block, generated host-side (the nginx
    # image may not ship the openssl CLI). Same command as the prod entrypoint.
    if [ ! -f "$CONECSA_CERT_DIR/snakeoil.crt" ] || [ ! -f "$CONECSA_CERT_DIR/snakeoil.key" ]; then
        openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 \
            -keyout "$CONECSA_CERT_DIR/snakeoil.key" -out "$CONECSA_CERT_DIR/snakeoil.crt" \
            -days 3650 -nodes -subj "/CN=conecsa-enroll" >/dev/null 2>&1
    fi
    docker rm -f conecsa-dev-tls >/dev/null 2>&1
    run_svc "tls" "$MAGENTA" \
        docker run --rm --name conecsa-dev-tls --network host \
            -v "$CONECSA_CERT_DIR":/etc/conecsa/certs \
            -v "$PROJECT_ROOT/scripts/dev-nginx":/etc/nginx/conecsa:ro \
            -v "$PROJECT_ROOT/scripts/dev-nginx/maps.conf":/etc/nginx/conf.d/maps.conf:ro \
            --entrypoint sh nginx:alpine /etc/nginx/conecsa/entrypoint.sh
fi

# Node-RED (flow) — the production image (flow/Dockerfile) with host networking:
# listens on :1880, reached through the mTLS gate at /flow/ like on the device.
# Data persists in .dev-data/flow (the flow-data volume equivalent).
if [ "$RUN_FLOW" -eq 1 ] && ! docker info >/dev/null 2>&1; then
    echo -e "${RED}Docker unavailable — skipping Node-RED (flow).${NC}"
    RUN_FLOW=0
fi
if [ "$RUN_FLOW" -eq 1 ]; then
    mkdir -p "$DEV_DATA_DIR/flow"
    docker rm -f conecsa-dev-flow >/dev/null 2>&1
    run_svc "flow" "$MAGENTA" bash -c "\
        docker build -q -t conecsa-dev-flow '$PROJECT_ROOT/flow' && \
        exec docker run --rm --name conecsa-dev-flow --network host \
            -v '$DEV_DATA_DIR/flow':/data \
            -e TZ='${TZ:-America/Sao_Paulo}' \
            -e INFERENCE_URL='http://localhost:5000' \
            conecsa-dev-flow"
fi

if [ "$RUN_APP" -eq 1 ]; then
    # Tailwind watch: shared input styles/input.css → system-vision/styles.css
    # (the file index.html loads). Trunk serve proxies /api to the gateway :5000.
    run_svc "tailwind" "$YELLOW" \
        "$PROJECT_ROOT/bin/tailwindcss" \
        -i "$PROJECT_ROOT/styles/input.css" -o "$PROJECT_ROOT/system-vision/styles.css" --watch
    run_svc "system-vision" "$GREEN" \
        bash -c "cd '$PROJECT_ROOT/system-vision' && exec trunk serve --proxy-rewrite=/api --proxy-backend=http://localhost:5000 --port=18080"
fi

# ---------------------------------------------------------------------------
# Banner + block until interrupted.
# ---------------------------------------------------------------------------
echo ""
echo -e "${GREEN}=== conecsa dev stack running ===${NC}"
echo "  api-gateway   http://localhost:5000   (HTTP API)"
[ "$RUN_APP" -eq 1 ]       && echo "  web frontend  http://localhost:18080  (trunk dev server)"
[ "$RUN_INFERENCE" -eq 1 ] && echo "  inference     grpc://localhost:50061"
[ "$RUN_TRAINING" -eq 1 ]  && echo "  training      grpc://localhost:50071"
[ "$RUN_WEBCAM" -eq 1 ]    && echo "  webcam-server SHM producer (conecsa_frame_shm)"
[ "$RUN_TLS" -eq 1 ]       && echo "  mTLS gate     https://localhost:443  (hub pairing/API)"
[ "$RUN_FLOW" -eq 1 ]      && echo "  flow          http://localhost:1880/flow  (Node-RED)"
echo -e "${YELLOW}Press Ctrl+C to stop everything.${NC}"
echo ""

# Wait on all children; Ctrl+C triggers the trap above.
wait
