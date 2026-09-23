# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Network controller: wired configuration and Wi-Fi scan/connect/forget via
the `os-base` hardware agent."""
import grpc
from flask import request

from .. import hardware
from ..enroll import device_id
from ..helpers import _grpc_error, _json, _publish_if_success
from . import api_bp


def _object_body() -> dict:
    """The request's JSON object, or an empty one when the body is not an
    object: the field checks then answer 400 instead of an AttributeError."""
    body = request.get_json(silent=True)
    return body if isinstance(body, dict) else {}


@api_bp.route('/api/v1/network/config', methods=['GET'])
def get_network_config():
    """GET /api/v1/network/config — gateway relay."""
    try:
        return _json(hardware.get_network_config())
    except grpc.RpcError as exc:
        return _grpc_error(exc, "hardware")
    except Exception as exc:  # noqa: BLE001
        return _json({"error": str(exc)}, 500)


@api_bp.route('/api/v1/network/config', methods=['POST'])
def set_network_config():
    """POST /api/v1/network/config — gateway relay."""
    body = _object_body()
    method = body.get("method")
    if not method:
        return _json({"error": "'method' field is required"}, 400)
    if method not in ("auto", "static"):
        return _json({"error": "'method' must be 'auto' or 'static'"}, 400)
    prefix = None
    if body.get("prefix") not in (None, ""):
        try:
            prefix = int(body["prefix"])
        except (TypeError, ValueError):
            return _json({"error": "'prefix' must be an integer"}, 400)
    dns = body.get("dns")
    if dns is not None and (not isinstance(dns, list)
                            or not all(isinstance(s, str) for s in dns)):
        return _json({"error": "'dns' must be a list of strings"}, 400)
    # The agent parses every address with `ipaddress` and rejects anything
    # else; the checks above only keep type errors from becoming a 500.
    try:
        result = hardware.set_network_config(
            interface=body.get("interface", "wired"), method=method,
            address=body.get("address"), prefix=prefix,
            gateway=body.get("gateway"), dns=dns)
    except grpc.RpcError as exc:
        return _grpc_error(exc, "hardware")
    except Exception as exc:  # noqa: BLE001
        return _json({"error": str(exc)}, 500)
    return _publish_if_success(_json(result), "network_config_changed", ["network"], data=result)


@api_bp.route('/api/v1/network/wifi/scan', methods=['GET'])
def scan_wifi():
    """GET /api/v1/network/wifi/scan — gateway relay."""
    try:
        return _json(hardware.scan_wifi())
    except grpc.RpcError as exc:
        return _grpc_error(exc, "hardware")
    except Exception as exc:  # noqa: BLE001
        return _json({"error": str(exc)}, 500)


@api_bp.route('/api/v1/network/wifi/connect', methods=['POST'])
def connect_wifi():
    """POST /api/v1/network/wifi/connect — gateway relay."""
    body = _object_body()
    ssid = body.get("ssid")
    if not ssid:
        return _json({"error": "'ssid' field is required"}, 400)
    try:
        result = hardware.connect_wifi(ssid, body.get("password", ""))
    except grpc.RpcError as exc:
        return _grpc_error(exc, "hardware")
    except Exception as exc:  # noqa: BLE001
        return _json({"error": str(exc)}, 500)
    return _publish_if_success(_json(result), "network_config_changed", ["network"], data=result)


@api_bp.route('/api/v1/network/wifi/forget', methods=['POST'])
def forget_wifi():
    """POST /api/v1/network/wifi/forget — gateway relay."""
    body = _object_body()
    ssid = body.get("ssid")
    if not ssid:
        return _json({"error": "'ssid' field is required"}, 400)
    try:
        result = hardware.forget_wifi(ssid)
    except grpc.RpcError as exc:
        return _grpc_error(exc, "hardware")
    except Exception as exc:  # noqa: BLE001
        return _json({"error": str(exc)}, 500)
    return _publish_if_success(_json(result), "network_config_changed", ["network"], data=result)


# ── Wi-Fi access point ───────────────────────────────────────────────────────
#
# The SSID is always the device id: the remote camera's owner finds the device by the
# same name the hub shows, and nothing has to be typed on the device side.
# The passphrase travels in the start body only; it is never returned, never
# published and never audited.

# The non-DFS 5 GHz block. Which of them the radio may start on right now is
# reported by the agent in the status (`channels`); 0 leaves the choice to it.
AP_CHANNELS = (36, 40, 44, 48)
AP_CHANNEL_AUTO = 0


@api_bp.route('/api/v1/network/ap', methods=['GET'])
def get_ap_status():
    """GET /api/v1/network/ap — gateway relay, plus the SSID a start would use."""
    try:
        status = hardware.get_ap_status()
    except grpc.RpcError as exc:
        return _grpc_error(exc, "hardware")
    except Exception as exc:  # noqa: BLE001
        return _json({"error": str(exc)}, 500)
    status["ssid"] = status["ssid"] or device_id()
    status.setdefault("channels", [])
    return _json(status)


@api_bp.route('/api/v1/network/ap/start', methods=['POST'])
def start_ap():
    """POST /api/v1/network/ap/start — `{passphrase, channel?}`; the SSID is the
    device id and an absent or 0 channel lets the agent choose a usable one."""
    body = _object_body()
    passphrase = body.get("passphrase")
    if not isinstance(passphrase, str) or not (8 <= len(passphrase) <= 63):
        return _json({"error": "'passphrase' must be 8 to 63 characters"}, 400)
    try:
        channel = int(body.get("channel", AP_CHANNEL_AUTO))
    except (TypeError, ValueError):
        return _json({"error": "'channel' must be an integer"}, 400)
    if channel != AP_CHANNEL_AUTO and channel not in AP_CHANNELS:
        return _json({"error": "'channel' must be 0 (automatic) or one of "
                      + ", ".join(map(str, AP_CHANNELS))}, 400)
    try:
        result = hardware.start_ap(device_id(), passphrase, channel)
    except grpc.RpcError as exc:
        return _grpc_error(exc, "hardware")
    except Exception as exc:  # noqa: BLE001
        return _json({"error": str(exc)}, 500)
    return _publish_if_success(_json(result), "network_config_changed", ["network"], data=result)


@api_bp.route('/api/v1/network/ap/stop', methods=['POST'])
def stop_ap():
    """POST /api/v1/network/ap/stop — gateway relay."""
    try:
        result = hardware.stop_ap()
    except grpc.RpcError as exc:
        return _grpc_error(exc, "hardware")
    except Exception as exc:  # noqa: BLE001
        return _json({"error": str(exc)}, 500)
    return _publish_if_success(_json(result), "network_config_changed", ["network"], data=result)
