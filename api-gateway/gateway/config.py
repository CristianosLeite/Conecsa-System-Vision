"""Gateway configuration — all knobs come from the environment.

The gateway holds no business config; it only needs to know where its peers are
(inference gRPC, os hardware agent) and which SHM segments carry the frames.
"""
import os


def _env_float(name: str, default: float) -> float:
    """Read an environment variable as a float, falling back to *default*."""
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class Settings:
    """Environment-driven gateway configuration (gRPC peers, SHM names, ports)."""

    # gRPC peers (docker-compose service names + <SVC>_ADDR convention).
    INFERENCE_GRPC_ADDR = os.environ.get("INFERENCE_GRPC_ADDR", "inference-service:50061")
    HARDWARE_AGENT_ADDR = os.environ.get("HARDWARE_AGENT_ADDR", "os:50051")
    TRAINING_GRPC_ADDR = os.environ.get("TRAINING_GRPC_ADDR", "training-service:50071")

    # Stereo combine parameters for the training preview (same defaults as the
    # inference-service so the preview matches what gets captured/detected).
    STEREO_COMBINE = os.environ.get("STEREO_COMBINE", "none").strip().lower()
    STEREO_BLEND_ALPHA = _env_float("STEREO_BLEND_ALPHA", 0.5)
    STEREO_OFFSET = _env_float("STEREO_OFFSET", 0.0)
    STEREO_OFFSET_Y = _env_float("STEREO_OFFSET_Y", 0.0)

    # POSIX SHM rings (shared via the `ipc:` namespace with webcam-server +
    # inference-service). Camera ring is produced by the Rust webcam-server;
    # the processed ring is produced by inference-service's pipeline (Stage D).
    CAMERA_SHM_NAME = os.environ.get("SHM_NAME", "conecsa_frame_shm")
    PROCESSED_SHM_NAME = os.environ.get("PROCESSED_SHM_NAME", "conecsa_processed_shm")

    # Audit trail: the device's own record of user actions, drained by the hub.
    # The caps are ring limits — a hub that has been away for months must not
    # be able to fill the eMMC, but losing audit rows is bad enough that they
    # are set far above any realistic outage.
    AUDIT_DIR = os.environ.get("AUDIT_DIR", "/data/audit")
    AUDIT_MAX_RECORDS = int(os.environ.get("AUDIT_MAX_RECORDS", "50000"))
    AUDIT_MAX_BYTES = int(os.environ.get("AUDIT_MAX_BYTES", str(64 * 1024 * 1024)))

    # Include exception text/class in 500 bodies (development only; the
    # production default keeps internals out of responses).
    DEBUG_ERRORS = os.environ.get(
        "GATEWAY_DEBUG_ERRORS", "").strip().lower() in ("1", "true")

    # HTTP server.
    PORT = int(os.environ.get("GATEWAY_PORT", "5000"))
    WAITRESS_THREADS = int(os.environ.get("WAITRESS_THREADS", "32"))

    # A single labeled-image upload (one JPEG relayed to the training service
    # in one gRPC message). Far below the global 600 MB body limit — a JPEG
    # frame is a few MB; anything bigger is not a camera frame.
    MAX_IMAGE_UPLOAD_BYTES = int(os.environ.get(
        "MAX_IMAGE_UPLOAD_BYTES", str(20 * 1024 * 1024)))

    # Per-call gRPC deadlines (seconds). Control calls are quick; Wi-Fi
    # scan/connect block on the radio and are given longer in the hardware client.
    GRPC_TIMEOUT = float(os.environ.get("GATEWAY_GRPC_TIMEOUT", "12"))

    # Auto-exit training mode after this much client silence (the orphaned-
    # training watchdog; see gateway/training/orphan.py). 0 disables it.
    # The device UI heartbeats every 10s while the training page is mounted
    # (/training/heartbeat) and the hub coordinator polls every ~2s, so 120s
    # means ~12 missed beats; worst-case fire latency is timeout + the 30s
    # watchdog tick. Kept above 90s so a single long request (multi-GB dataset
    # upload/export touches the tracker only at request start) does not trip it.
    TRAINING_ORPHAN_TIMEOUT_SEC = float(
        os.environ.get("TRAINING_ORPHAN_TIMEOUT_SEC", "120"))

    # Hub-driven clock correction (gateway/clock.py). The host has no RTC
    # battery, so the hub's wall clock is the only time source on an isolated
    # LAN. Only step when the drift is worth a syscall, and — since the hub
    # polls every 2s — never attempt more often than the interval below (which
    # bounds the retries when the `os` agent is down, not the healthy path,
    # where a corrected clock stops matching the threshold).
    CLOCK_SYNC_THRESHOLD_SEC = _env_float("CLOCK_SYNC_THRESHOLD_SEC", 30.0)
    CLOCK_SYNC_MIN_INTERVAL_SEC = _env_float("CLOCK_SYNC_MIN_INTERVAL_SEC", 60.0)


settings = Settings()
