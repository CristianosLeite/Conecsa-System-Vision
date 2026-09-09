"""Headless training-service entry point.

Wires the services (composition root) and starts the TrainingControl gRPC
server (proto/training.proto :50071), then blocks. The api-gateway owns all
HTTP/SSE — there is no Flask here (same shape as the inference-service).
"""
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

from service.composition import Application  # noqa: E402
from service.training_grpc import serve_grpc  # noqa: E402

logger = logging.getLogger(__name__)

application = Application()
# A server that cannot bind its port is fatal: exit non-zero so the
# container's restart policy retries, instead of a live process nobody can reach.
try:
    grpc_server = serve_grpc(application)
except Exception as ex:  # noqa: BLE001 - report and exit non-zero
    logger.critical("Failed to start training gRPC server: %s", ex)
    sys.exit(1)

if __name__ == "__main__":
    logger.info("training-service running headless (gRPC only).")
    grpc_server.wait_for_termination()
