# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
gRPC server for the `os-base` hardware agent.

Serves HardwareService (proto/hardware.proto): network/Wi-Fi RPCs are backed by
NetworkAgent, GPIO RPCs by GpioAgent, system metrics by SystemAgent and clock
steps by the time agent.
"""
import logging
import os
import sys
from concurrent import futures

import grpc

# The generated *_pb2_grpc module does a flat `import hardware_pb2`, so the
# compiled proto directory must be on sys.path.
_PROTO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proto")
if _PROTO_DIR not in sys.path:
    sys.path.insert(0, _PROTO_DIR)

import hardware_pb2 as pb  # noqa: E402
import hardware_pb2_grpc as pb_grpc  # noqa: E402

from .ap_agent import ApAgent  # noqa: E402
from .clocks_agent import pin_performance_clocks  # noqa: E402
from .gpio_agent import GpioAgent  # noqa: E402
from .network_agent import AccessPointActive, NetworkAgent  # noqa: E402
from .system_agent import SystemAgent  # noqa: E402
from .time_agent import TimeAgent  # noqa: E402

logger = logging.getLogger(__name__)

LISTEN_ADDR = os.environ.get("HARDWARE_AGENT_LISTEN", "0.0.0.0:50051")


def _refuse(context, exc: AccessPointActive, reply):
    """Fail the current RPC with FAILED_PRECONDITION and return *reply* empty.

    ``set_code``/``set_details`` rather than ``abort``, like the other
    services: the client still raises ``RpcError`` with this code, which the
    gateway maps to a 409 carrying the message.
    """
    logger.info("refused: %s", exc)
    context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
    context.set_details(str(exc))
    return reply


def _iface_config_pb(cfg: dict) -> pb.InterfaceConfig:
    """Convert a NetworkAgent interface dict to an ``InterfaceConfig`` message."""
    return pb.InterfaceConfig(
        name=cfg.get("name", ""),
        method=cfg.get("method", "auto"),
        address=cfg.get("address", ""),
        prefix=int(cfg.get("prefix", 0) or 0),
        gateway=cfg.get("gateway", ""),
        dns=list(cfg.get("dns", []) or []),
        present=bool(cfg.get("present", False)),
    )


class HardwareServicer(pb_grpc.HardwareServiceServicer):
    """gRPC HardwareService implementation.

    Each RPC is a thin adapter that delegates to the backing agent
    (:class:`NetworkAgent`, :class:`GpioAgent`, :class:`SystemAgent`,
    :class:`TimeAgent`) and maps the result to/from the ``hardware.proto``
    messages.
    """

    def __init__(self, gpio: GpioAgent):
        self.ap = ApAgent(NetworkAgent._discover)
        self.network = NetworkAgent(ap=self.ap)
        self.gpio = gpio
        # Whatever the radio was doing before this process started, it is a
        # station now: the access point is never meant to outlive its agent.
        self.ap.reconcile()

    # ── Network / IP ────────────────────────────────────────────────────────────

    def GetNetworkConfig(self, request, context):
        """RPC: return the wired + Wi-Fi IPv4 config and Wi-Fi status."""
        cfg = self.network.get_network_config()
        status = cfg.get("wifi_status", {})
        return pb.NetworkConfig(
            wired=_iface_config_pb(cfg.get("wired", {})),
            wifi=_iface_config_pb(cfg.get("wifi", {})),
            wifi_status=pb.WifiStatus(
                ssid=status.get("ssid", ""),
                state=status.get("state", ""),
                signal=int(status.get("signal", 0) or 0),
            ),
        )

    def SetIpConfig(self, request, context):
        """RPC: apply an IPv4 config (auto/static) to the wired or Wi-Fi link."""
        interface = "wifi" if request.interface == pb.WIFI else "wired"
        try:
            res = self.network.set_ip_config(
                interface=interface,
                method=request.method,
                address=request.address,
                prefix=request.prefix,
                gateway=request.gateway,
                dns=list(request.dns),
            )
        except AccessPointActive as exc:
            return _refuse(context, exc, pb.Result())
        return pb.Result(success=res["success"], message=res["message"])

    # ── Wi-Fi ─────────────────────────────────────────────────────────────────────

    def ScanWifi(self, request, context):
        """RPC: scan for Wi-Fi networks."""
        nets = self.network.scan_wifi()
        return pb.WifiScanResult(networks=[
            pb.WifiNetwork(
                ssid=n["ssid"], signal=int(n["signal"]), security=n["security"],
                in_use=n["in_use"], saved=n["saved"],
            ) for n in nets
        ])

    def ConnectWifi(self, request, context):
        """RPC: connect to a Wi-Fi network by ``{ssid, password}``."""
        try:
            res = self.network.connect_wifi(request.ssid, request.password)
        except AccessPointActive as exc:
            return _refuse(context, exc, pb.WifiConnectResult())
        return pb.WifiConnectResult(
            success=res["success"], state=res["state"], message=res["message"],
        )

    def ForgetWifi(self, request, context):
        """RPC: remove a saved Wi-Fi network by ``{ssid}``."""
        try:
            res = self.network.forget_wifi(request.ssid)
        except AccessPointActive as exc:
            return _refuse(context, exc, pb.Result())
        return pb.Result(success=res["success"], message=res["message"])

    # ── Wi-Fi access point ──────────────────────────────────────────────────

    def GetApStatus(self, request, context):
        """RPC: the access point's state; never the passphrase."""
        st = self.ap.status()
        return pb.ApStatus(
            active=st["active"], ssid=st["ssid"], frequency_mhz=int(st["frequency_mhz"]),
            address=st["address"], prefix=int(st["prefix"]),
            stations=[pb.ApStation(address=s["address"], hostname=s["hostname"], signal=int(s["signal"]))
                      for s in st["stations"]],
            join_deadline_remaining_secs=int(st["join_deadline_remaining_secs"]),
            wired_ready=st["wired_ready"], message=st["message"],
            channels=[int(c) for c in st.get("channels", [])],
        )

    def StartAp(self, request, context):
        """RPC: start the access point (rollback-safe)."""
        res = self.ap.start(request.ssid, request.passphrase, int(request.channel))
        return pb.Result(success=res["success"], message=res["message"])

    def StopAp(self, request, context):
        """RPC: return the radio to station mode."""
        res = self.ap.stop()
        return pb.Result(success=res["success"], message=res["message"])

    # ── GPIO (config/output ops; per-frame trigger gate uses shared memory) ──────

    def GetGpioStatus(self, request, context):
        """RPC: return GPIO availability, trigger mode and output pin levels."""
        st = self.gpio.get_status()
        return pb.GpioStatus(
            available=st["available"],
            enabled=st["enabled"],
            pins=[pb.GpioPinState(pin=p, level=v) for p, v in st["pins"].items()],
        )

    def SetGpioTrigger(self, request, context):
        """RPC: enable/disable GPIO trigger mode."""
        res = self.gpio.set_enabled(request.enabled)
        return pb.Result(success=res["success"], message=res["message"])

    def SetGpioPin(self, request, context):
        """RPC: drive a single output pin HIGH/LOW."""
        res = self.gpio.set_pin(request.pin, request.level)
        return pb.Result(success=res["success"], message=res["message"])

    # ── System metrics ────────────────────────────────────────────────────────

    def GetSystemStatus(self, request, context):
        """RPC: return host metrics (CPU/RAM/disk + optional temp/GPU sensors)."""
        st = SystemAgent.get_system_status()
        msg = pb.SystemStatus(
            cpu_usage=st["cpu_usage"],
            ram_usage=st["ram_usage"],
            ram_total=int(st["ram_total"]),
            ram_used=int(st["ram_used"]),
            disk_usage=st["disk_usage"],
            disk_total=int(st["disk_total"]),
            disk_used=int(st["disk_used"]),
        )
        # proto3 `optional` floats: only set when the metric is available so the
        # gateway relays JSON `null` for absent sensors.
        for field in ("temperature", "gpu_usage", "gpu_temperature",
                      "gpu_freq_mhz", "gpu_max_freq_mhz"):
            value = st.get(field)
            if value is not None:
                setattr(msg, field, float(value))
        return msg

    # ── System power ─────────────────────────────────────────────────────────

    def SystemPower(self, request, context):
        """RPC: perform a host power action (e.g. shutdown/reboot)."""
        result = SystemAgent.system_power(request.action)
        return pb.SystemPowerResult(
            success=bool(result.get("success", False)),
            message=str(result.get("message", "")),
        )

    # ── System clock ─────────────────────────────────────────────────────────

    def SetSystemTime(self, request, context):
        """RPC: set the host clock to the hub's wall time (see TimeAgent)."""
        result = TimeAgent.set_system_time(request.epoch_millis, request.source)
        return pb.Result(
            success=bool(result.get("success", False)),
            message=str(result.get("message", "")),
        )


def serve() -> None:
    """Pin performance clocks, start the GPIO agent, and serve HardwareService.

    Blocks until termination, then cleans up the GPIO agent.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Pin the Jetson to performance clocks first — the dynamic governors leave the
    # GPU at its minimum for the bursty inference workload, ~2x-ing latency.
    pin_performance_clocks()
    # GpioAgent initializes the GPIO hardware and starts the shared-memory poll
    # loop before the server accepts calls.
    gpio = GpioAgent()
    from grpc_health.v1 import health, health_pb2, health_pb2_grpc
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8),
                         options=[("grpc.so_reuseport", 0)])
    pb_grpc.add_HardwareServiceServicer_to_server(HardwareServicer(gpio), server)
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    try:
        # A bind failure must not leave a live process with no listener: raise
        # so `python3 -m agent` exits non-zero and the restart policy retries.
        try:
            bound = server.add_insecure_port(LISTEN_ADDR)
        except RuntimeError as exc:
            raise RuntimeError(
                f"could not bind the hardware agent to {LISTEN_ADDR}: {exc}") from exc
        if bound == 0:
            raise RuntimeError(f"could not bind the hardware agent to {LISTEN_ADDR}")
        server.start()
        health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
        logger.info("Hardware agent gRPC server listening on %s", LISTEN_ADDR)
        server.wait_for_termination()
    finally:
        gpio.cleanup()
