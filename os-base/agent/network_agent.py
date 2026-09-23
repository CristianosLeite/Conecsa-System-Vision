# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
NetworkAgent — host network/Wi-Fi management for the `os-base` hardware agent.

IP configuration is done through systemd-networkd (via D-Bus + .network files);
Wi-Fi association is done through wpa_supplicant, speaking its control
protocol directly on the control socket (see wpa.py). The Yocto host has no
NetworkManager/nmcli.
"""
import ipaddress
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager

from conecsa_common.atomic import atomic_write_bytes

from . import networkd
from .wpa import CTRL_DIR, WpaCtrl, WpaError, encode_psk, encode_ssid

logger = logging.getLogger(__name__)

NETWORKD_DIR = "/etc/systemd/network"
MANAGED_PREFIX = "10-conecsa-"  # lower number wins over stock 20-/30- files

_CONNECT_TIMEOUT_S = 15.0
_POLL_INTERVAL_S = 0.5


class AccessPointActive(RuntimeError):
    """A Wi-Fi change was asked for while the radio is (or is becoming) an
    access point: the caller must stop the access point first. The servicer
    turns it into FAILED_PRECONDITION, the gateway into a 409.
    """

    def __init__(self):
        super().__init__("The access point is active; stop it before changing Wi-Fi")


class NetworkAgent:
    """Host network/Wi-Fi control surface backing the `os-base` hardware agent's gRPC RPCs.

    Wraps systemd-networkd (IPv4 config) and wpa_supplicant (Wi-Fi association)
    so the api-gateway never touches host networking directly. Stateless: each
    call rediscovers the wired/wireless interfaces.
    """

    def __init__(self, ap=None):
        # The access-point agent, when there is one: Wi-Fi changes are refused
        # while the radio is an access point (or on its way to becoming one)
        # and hold its ``radio`` lock meanwhile, so a start cannot slip in
        # between the check and the change.
        self._ap = ap

    def _refuse_while_ap(self) -> None:
        if self._ap is not None and self._ap.active:
            raise AccessPointActive()

    @contextmanager
    def _hold_radio(self, needed: bool = True) -> Iterator[None]:
        """Keep the access point off the radio for the length of a Wi-Fi change.

        The first check answers at once while a start or stop is running; the
        second, under the lock, catches one that claimed the radio meanwhile.
        """
        if not needed or self._ap is None:
            yield
            return
        self._refuse_while_ap()
        with self._ap.radio:
            self._refuse_while_ap()
            yield

    # ── interface discovery ────────────────────────────────────────────────────

    @staticmethod
    def _discover() -> dict[str, tuple[int, str]]:
        """Return {"wired": (ifindex, name), "wifi": (ifindex, name)} (missing
        kinds omitted). Wired = en*/eth*, wireless = wl*."""
        found: dict[str, tuple[int, str]] = {}
        try:
            for ifindex, name in networkd.list_links():
                if name.startswith(("en", "eth")) and "wired" not in found:
                    found["wired"] = (ifindex, name)
                elif name.startswith("wl") and "wifi" not in found:
                    found["wifi"] = (ifindex, name)
        except Exception as exc:  # noqa: BLE001
            logger.error("networkd ListLinks failed: %s", exc)
        return found

    @classmethod
    def _wifi_iface(cls) -> str | None:
        """Name of the wireless interface, or ``None``.

        Falls back to the wpa_supplicant control-socket filename when link
        discovery finds no ``wl*`` interface.
        """
        found = cls._discover().get("wifi")
        if found:
            return found[1]
        # Fallback: the wpa_supplicant control socket filename is the iface name.
        try:
            socks = os.listdir(CTRL_DIR)
            return socks[0] if socks else None
        except OSError:
            return None

    # ── read state ──────────────────────────────────────────────────────────────

    def get_network_config(self) -> dict:
        """Return the current wired + Wi-Fi IPv4 config and Wi-Fi status."""
        ifaces = self._discover()
        return {
            "wired": self._iface_config(ifaces.get("wired")),
            "wifi": self._iface_config(ifaces.get("wifi")),
            "wifi_status": self._wifi_status(ifaces.get("wifi")),
        }

    def _iface_config(self, link: tuple[int, str] | None) -> dict:
        """Resolve one interface's IPv4 config (method/address/gateway/dns).

        A managed `10-conecsa-*.network` file with ``DHCP=no`` is treated as the
        authoritative signal that the method is ``static``.
        """
        if not link:
            return {"name": "", "method": "auto", "address": "", "prefix": 0,
                    "gateway": "", "dns": [], "present": False}
        ifindex, name = link
        try:
            parsed = networkd.parse_ipv4(networkd.describe_link(ifindex))
        except Exception as exc:  # noqa: BLE001
            logger.error("DescribeLink(%s) failed: %s", name, exc)
            parsed = {"method": "auto", "address": "", "prefix": 0, "gateway": "", "dns": []}
        # A managed static file is the authoritative signal for the UI radio.
        if os.path.exists(self._managed_path(name)) and self._file_is_static(name):
            parsed["method"] = "static"
        parsed["name"] = name
        parsed["present"] = True
        return parsed

    def _wifi_status(self, link: tuple[int, str] | None) -> dict:
        """Return the Wi-Fi association status (ssid/state) for *link*."""
        if not link:
            return {"ssid": "", "state": "INACTIVE", "signal": 0}
        _ifindex, name = link
        try:
            st = WpaCtrl(name).status()
        except WpaError as exc:
            logger.error("wpa status failed: %s", exc)
            return {"ssid": "", "state": "DISCONNECTED", "signal": 0}
        return {
            "ssid": st.get("ssid", ""),
            "state": st.get("wpa_state", "DISCONNECTED"),
            "signal": 0,
        }

    # ── apply IP config ──────────────────────────────────────────────────────────

    def set_ip_config(self, interface: str, method: str, address: str = "",
                      prefix: int = 0, gateway: str = "", dns: list[str] | None = None) -> dict:
        """Apply an IPv4 config to the wired or Wi-Fi interface.

        Writes a managed `10-conecsa-*.network` file then reloads/reconfigures
        the link via networkd. ``method`` is ``auto`` (DHCP) or ``static``
        (``address`` + ``prefix`` required). Returns ``{success, message}``.

        Never strands the device: every static value is parsed as an IPv4
        address / prefix length before anything is written (so a value can
        neither be malformed nor smuggle an extra networkd directive), the
        file is replaced atomically, and if networkd refuses to reload or to
        reconfigure the link the previous file is put back and applied again.
        """
        if method not in ("auto", "static"):
            return {"success": False, "message": f"Invalid method: {method}"}
        kind = "wifi" if str(interface).lower() in ("wifi", "1") else "wired"
        with self._hold_radio(kind == "wifi"):
            return self._set_ip_config(kind, method, address, prefix, gateway, dns)

    def _set_ip_config(self, kind: str, method: str, address: str, prefix: int,
                       gateway: str, dns: list[str] | None) -> dict:
        link = self._discover().get(kind)
        if not link:
            return {"success": False, "message": f"No {kind} interface found"}
        ifindex, name = link

        dns = [server for server in (dns or []) if server]
        if method == "static":
            if not address or not prefix:
                return {"success": False,
                        "message": "Address and prefix are required for static IP"}
            error = self._validate_static(address, prefix, gateway, dns)
            if error:
                return {"success": False, "message": error}

        path = self._managed_path(name)
        previous = self._read_managed_file(path)
        try:
            self._write_network_file(name, method, address, prefix, gateway, dns)
            networkd.reload()
            networkd.reconfigure_link(ifindex)
        except Exception as exc:  # noqa: BLE001
            logger.error("set_ip_config failed: %s; restoring the previous file", exc)
            self._restore_managed_file(path, previous, ifindex)
            return {"success": False,
                    "message": f"{exc}; the previous network configuration was restored"}
        return {"success": True, "message": "Network configuration applied successfully"}

    @staticmethod
    def _validate_static(address: str, prefix: int, gateway: str,
                         dns: list[str]) -> str | None:
        """Return an error message for an invalid static config, else ``None``.

        Parsing through ``ipaddress`` is the whole defence: a value that is
        not exactly one dotted-quad (including anything carrying a newline or
        another control character) cannot become an ``Address=``/``Gateway=``/
        ``DNS=`` line.
        """
        try:
            ipaddress.IPv4Address(address)
        except (ValueError, TypeError):
            return f"Invalid IPv4 address: {address!r}"
        try:
            prefix_len = int(prefix)
        except (ValueError, TypeError):
            return f"Invalid prefix length: {prefix!r}"
        if not 1 <= prefix_len <= 32:
            return "Prefix length must be between 1 and 32"
        if gateway:
            try:
                ipaddress.IPv4Address(gateway)
            except (ValueError, TypeError):
                return f"Invalid gateway address: {gateway!r}"
        for server in dns:
            try:
                ipaddress.IPv4Address(server)
            except (ValueError, TypeError):
                return f"Invalid DNS server address: {server!r}"
        return None

    @staticmethod
    def _read_managed_file(path: str) -> bytes | None:
        """Current bytes of the managed file, or ``None`` when it does not exist."""
        try:
            with open(path, "rb") as fh:
                return fh.read()
        except FileNotFoundError:
            return None

    @staticmethod
    def _restore_managed_file(path: str, previous: bytes | None, ifindex: int) -> None:
        """Put the pre-change file back (or remove ours) and re-apply it.

        Best-effort: a failure here is logged, since the caller is already
        reporting the original error.
        """
        try:
            if previous is None:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
            else:
                atomic_write_bytes(path, previous, mode=0o644)
            networkd.reload()
            networkd.reconfigure_link(ifindex)
            logger.info("Previous network configuration restored")
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not restore the previous network configuration: %s", exc)

    def _managed_path(self, iface: str) -> str:
        """Path of the managed `.network` file for *iface*."""
        return os.path.join(NETWORKD_DIR, f"{MANAGED_PREFIX}{iface}.network")

    def _file_is_static(self, iface: str) -> bool:
        """True if the managed file for *iface* pins a static (``DHCP=no``) IP."""
        try:
            with open(self._managed_path(iface), encoding="utf-8") as fh:
                return "DHCP=no" in fh.read()
        except OSError:
            return False

    def _write_network_file(self, iface: str, method: str, address: str,
                           prefix: int, gateway: str, dns: list[str]) -> None:
        """Render and write the managed systemd-networkd `.network` file."""
        lines = ["[Match]", f"Name={iface}", "", "[Network]"]
        if method == "static":
            lines.append("DHCP=no")
            lines.append(f"Address={address}/{prefix}")
            if gateway:
                lines.append(f"Gateway={gateway}")
            for server in dns:
                if server:
                    lines.append(f"DNS={server}")
        else:
            lines.append("DHCP=yes")
            lines.append("LinkLocalAddressing=ipv4")
        content = "\n".join(lines) + "\n"
        # Atomic replace: networkd never sees a truncated or half-written file,
        # and a power cut cannot leave one behind. Readable by networkd (0644),
        # like the stock files next to it.
        atomic_write_bytes(self._managed_path(iface), content.encode("utf-8"), mode=0o644)

    # ── Wi-Fi ────────────────────────────────────────────────────────────────────

    def scan_wifi(self) -> list[dict]:
        """Scan for Wi-Fi networks, deduped by SSID (strongest signal wins).

        Each entry is ``{ssid, signal, security, in_use, saved}``, sorted by
        signal strength descending.
        """
        iface = self._wifi_iface()
        if not iface:
            return []
        wpa = WpaCtrl(iface)
        try:
            current_ssid = wpa.status().get("ssid", "")
            saved = {n["ssid"] for n in wpa.list_networks()}
            results = wpa.scan()
        except WpaError as exc:
            logger.error("wifi scan failed: %s", exc)
            return []
        # Dedup by SSID keeping the strongest signal.
        best: dict[str, dict] = {}
        for net in results:
            ssid = net["ssid"]
            if ssid not in best or net["signal"] > best[ssid]["signal"]:
                best[ssid] = net
        out = []
        for ssid, net in best.items():
            out.append({
                "ssid": ssid,
                "signal": net["signal"],
                "security": net["security"],
                "in_use": ssid == current_ssid,
                "saved": ssid in saved,
            })
        out.sort(key=lambda n: n["signal"], reverse=True)
        return out

    def connect_wifi(self, ssid: str, password: str) -> dict:
        """Associate with *ssid*, persisting only on success.

        On success the chosen network is made highest-priority and the config is
        saved. On failure the live supplicant state is rolled back with
        RECONFIGURE and **nothing is persisted**, so a wrong password can never
        strand the device. Returns ``{success, state, message}``.
        """
        with self._hold_radio():
            return self._connect_wifi(ssid, password)

    def _connect_wifi(self, ssid: str, password: str) -> dict:
        iface = self._wifi_iface()
        if not iface:
            return {"success": False, "state": "INACTIVE", "message": "No wireless interface"}
        # Serialize the credentials before touching the supplicant: a value
        # the control grammar cannot carry safely is refused here (hex SSID,
        # quoted/escaped passphrase or bare 64-hex PSK — see wpa.encode_*).
        try:
            ssid_arg = encode_ssid(ssid)
            psk_arg = encode_psk(password) if password else None
        except WpaError as exc:
            return {"success": False, "state": "INACTIVE", "message": str(exc)}

        wpa = WpaCtrl(iface)
        added = False
        try:
            # LIST_NETWORKS prints the SSID as text, so a printable SSID set in
            # hex form still matches here on the next connect.
            net_id = wpa.find_network_id(ssid)
            if net_id is None:
                net_id = wpa.add_network()
                added = True
            wpa.set_network(net_id, "ssid", ssid_arg)
            if psk_arg is not None:
                wpa.set_network(net_id, "psk", psk_arg)
            elif added:
                # Brand-new network with no password → treat as open.
                wpa.set_network(net_id, "key_mgmt", "NONE")
            # else: reusing a saved network without a new password — leave its
            # existing psk/key_mgmt untouched and just (re)select it below.
            wpa.enable_network(net_id)
            wpa.select_network(net_id)
        except WpaError as exc:
            logger.error("wifi connect setup failed for %s: %s", ssid, exc)
            self._restore(wpa)
            return {"success": False, "state": "DISCONNECTED", "message": str(exc)}

        state = self._await_connection(wpa, ssid)
        if state == "COMPLETED":
            try:
                # Keep every known network eligible for reboot auto-reconnect,
                # but make the chosen one strictly highest-priority so a later
                # SCAN does not make wpa_supplicant roam to another saved
                # network. Then persist the new working state.
                wpa.enable_all()
                self._prefer_network(wpa, net_id)
                wpa.save_config()
            except WpaError as exc:
                logger.warning("save_config after connect failed: %s", exc)
            return {"success": True, "state": state, "message": f"Connected to {ssid}"}

        # Failure: restore the device to its previous connection WITHOUT
        # persisting anything. We never SAVE_CONFIG here, so the on-disk config
        # still describes the known-good network(s); RECONFIGURE reloads it and
        # re-associates, so a wrong password can never strand the device.
        self._restore(wpa)
        return {"success": False, "state": state,
                "message": "Wrong password or could not connect"}

    @staticmethod
    def _prefer_network(wpa: WpaCtrl, net_id: str) -> None:
        """Give `net_id` the highest priority among all saved networks so
        wpa_supplicant stays on it (and does not switch on the next scan when
        several networks are saved). Lower-priority networks remain as fallback
        only when the preferred one is unreachable."""
        highest = 0
        for net in wpa.list_networks():
            highest = max(highest, wpa.get_network_int(net["id"], "priority"))
        try:
            wpa.set_network(net_id, "priority", str(highest + 1))
        except WpaError as exc:
            logger.warning("could not set priority on network %s: %s", net_id, exc)

    @staticmethod
    def _restore(wpa: WpaCtrl) -> None:
        """Roll back live supplicant state to the (unsaved) on-disk config."""
        try:
            wpa.reconfigure()
        except WpaError as exc:
            logger.error("wpa RECONFIGURE during rollback failed: %s", exc)

    def _await_connection(self, wpa: WpaCtrl, ssid: str) -> str:
        """Poll wpa_supplicant until *ssid* reaches COMPLETED or the timeout.

        Returns the last observed ``wpa_state``.
        """
        deadline = time.monotonic() + _CONNECT_TIMEOUT_S
        last = "DISCONNECTED"
        while time.monotonic() < deadline:
            try:
                st = wpa.status()
            except WpaError:
                time.sleep(_POLL_INTERVAL_S)
                continue
            last = st.get("wpa_state", last)
            if last == "COMPLETED" and st.get("ssid", "") == ssid:
                return "COMPLETED"
            time.sleep(_POLL_INTERVAL_S)
        return last

    def forget_wifi(self, ssid: str) -> dict:
        """Remove a saved Wi-Fi network and persist the change."""
        with self._hold_radio():
            return self._forget_wifi(ssid)

    def _forget_wifi(self, ssid: str) -> dict:
        iface = self._wifi_iface()
        if not iface:
            return {"success": False, "message": "No wireless interface"}
        wpa = WpaCtrl(iface)
        try:
            net_id = wpa.find_network_id(ssid)
            if net_id is None:
                return {"success": False, "message": f"No saved network for {ssid}"}
            wpa.remove_network(net_id)
            wpa.save_config()
        except WpaError as exc:
            return {"success": False, "message": str(exc)}
        return {"success": True, "message": f"Removed {ssid}"}
