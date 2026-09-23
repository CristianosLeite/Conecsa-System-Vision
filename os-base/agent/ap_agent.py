# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
ApAgent — the device's Wi-Fi access point for the remote camera.

The device has one radio. As an access point it leaves its Wi-Fi network, so
the hub reaches it over the wired port only: starting requires a wired link
with an IPv4 address — a link-local one counts, since a hub on a direct cable
with no DHCP server reaches the device by exactly that — and every step that
fails is undone (the volatile
network file removed, the wpa block removed, networkd reloaded, wpa_supplicant
``RECONFIGURE``d) so the radio returns to the persisted station configuration.

Nothing here is ever persisted: the access-point block never passes through
``SAVE_CONFIG`` and the networkd file lives under ``/run``. An agent restart
reconciles the radio back to station mode; a reboot does the same by itself.

``DescribeLink`` reports DHCP leases as
``DHCPServer.Leases[] = {ClientId: bytes, Address: bytes, Hostname: str,
ExpirationUSec: int}``. Client ids beginning with ``0x01`` carry the station's
MAC, which is how a lease is matched to an ``ALL_STA`` entry — and the MAC is
the one thing this module never returns or logs.
"""
import ipaddress
import logging
import os
import threading
import time
from typing import Callable

from conecsa_common.atomic import atomic_write_bytes

from . import networkd
from .wpa import WpaCtrl, WpaError, encode_psk, encode_ssid

logger = logging.getLogger(__name__)

RUN_NETWORKD_DIR = "/run/systemd/network"
AP_FILE_PREFIX = "05-conecsa-ap-"
# Documented default; a site whose wired network overlaps it sets the variable.
AP_ADDRESS_CIDR = os.environ.get("AP_ADDRESS_CIDR", "10.98.76.1/24")
# Channel → centre frequency, the non-DFS block of 5 GHz only. Whether a given
# channel may *initiate* a network at any moment is a regulatory flag (NO_IR)
# the driver lifts on a channel after hearing a beacon there and restores
# later, so the usable set changes over time and is read from the daemon at
# start time (``usable_frequencies``) rather than assumed here.
# ``AUTO_CHANNEL`` lets the agent choose.
CHANNELS = {36: 5180, 40: 5200, 44: 5220, 48: 5240}
AUTO_CHANNEL = 0
JOIN_DEADLINE_S = 300.0
_AP_TIMEOUT_S = 15.0
_ADDRESS_TIMEOUT_S = 10.0
_POLL_S = 0.5
_STATION_POLL_S = 5.0
_DHCP_POOL_OFFSET = 10
_DHCP_POOL_SIZE = 20


def _validate_request(ssid: str, passphrase: str, channel: int) -> str | None:
    """Return why the request is unusable, or ``None``."""
    if not ssid or len(ssid.encode("utf-8")) > 32:
        return "SSID must be 1 to 32 bytes"
    if not (8 <= len(passphrase) <= 63):
        return "passphrase must be 8 to 63 characters"
    if channel != AUTO_CHANNEL and channel not in CHANNELS:
        return "channel must be 0 (automatic) or one of " + ", ".join(str(c) for c in sorted(CHANNELS))
    return None


def _network_file(iface: str, cidr: str) -> bytes:
    """The volatile networkd file: taken only while the link is an access
    point, dropped by networkd itself when the radio returns to station mode."""
    return (
        "[Match]\n"
        f"Name={iface}\n"
        "WLANInterfaceType=ap\n"
        "\n"
        "[Network]\n"
        f"Address={cidr}\n"
        "DHCPServer=yes\n"
        "LinkLocalAddressing=no\n"
        "IPv6AcceptRA=no\n"
        "\n"
        "[DHCPServer]\n"
        f"PoolOffset={_DHCP_POOL_OFFSET}\n"
        f"PoolSize={_DHCP_POOL_SIZE}\n"
        # A network with no way out must not advertise resolvers.
        "EmitDNS=no\n"
    ).encode()


class ApAgent:
    """Start/stop/status of the access point, one transition at a time."""

    def __init__(self, discover: Callable[[], dict[str, tuple[int, str]]],
                 run_dir: str = RUN_NETWORKD_DIR, cidr: str = AP_ADDRESS_CIDR,
                 now: Callable[[], float] = time.monotonic,
                 wpa_factory: Callable[[str], WpaCtrl] = WpaCtrl):
        self._discover = discover
        self._run_dir = run_dir
        self._cidr = cidr
        self._now = now
        self._wpa = wpa_factory
        self._lock = threading.RLock()
        # Held for the whole of a start, a stop and (through ``NetworkAgent``)
        # a Wi-Fi change: the two never reconfigure the radio at the same time.
        self.radio = threading.RLock()
        self._active = False
        self._transition = False
        self._ssid = ""
        self._frequency = 0
        self._net_id: str | None = None
        self._joined = False
        self._deadline: float | None = None
        self._timer: threading.Timer | None = None
        self._watch_stop = threading.Event()
        self._message = ""

    # ── read ────────────────────────────────────────────────────────────────

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active or self._transition

    def status(self) -> dict:
        """The state the gateway relays; never the passphrase, never a MAC."""
        with self._lock:
            active = self._active
            ssid, frequency, message = self._ssid, self._frequency, self._message
            remaining = 0
            if active and not self._joined and self._deadline is not None:
                remaining = max(0, int(self._deadline - self._now() + 0.999))
        links = self._discover()
        out = {
            "active": active, "ssid": ssid if active else "", "frequency_mhz": frequency if active else 0,
            "address": "", "prefix": 0, "stations": [], "join_deadline_remaining_secs": remaining,
            "wired_ready": self._wired_ready(links), "message": message, "channels": [],
        }
        wifi = links.get("wifi")
        if wifi:
            out["channels"] = self._usable_channels(self._wpa(wifi[1])) or []
        if active and wifi:
            try:
                link = networkd.describe_link(wifi[0])
                address = self._ap_address(link)
                if address:
                    out["address"], out["prefix"] = address
                out["stations"] = self._stations(wifi[1], link)
            except Exception as exc:  # noqa: BLE001 - status must not raise
                logger.error("access point status read failed: %s", exc)
        return out

    def _ap_address(self, link: dict) -> tuple[str, int] | None:
        want = str(ipaddress.ip_interface(self._cidr).ip)
        for entry in link.get("Addresses", []) or []:
            if entry.get("Family") == networkd._AF_INET:
                ip = networkd._bytes_to_ipv4(entry.get("Address"))
                if ip is not None and ip == want:
                    return ip, int(entry.get("PrefixLength", 0) or 0)
        return None

    def _stations(self, iface: str, link: dict) -> list[dict]:
        """Joined stations with their leased address; matched on the client id's
        MAC, which stays inside this method."""
        leases: dict[str, dict] = {}
        for lease in ((link.get("DHCPServer") or {}).get("Leases") or []):
            client_id = bytes(int(b) & 0xFF for b in (lease.get("ClientId") or []))
            if len(client_id) == 7 and client_id[0] == 1:
                mac = ":".join(f"{b:02x}" for b in client_id[1:])
                leases[mac] = lease
        stations = []
        for sta in self._wpa(iface).all_sta():
            lease = leases.get(sta["mac"], {})
            stations.append({
                "address": networkd._bytes_to_ipv4(lease.get("Address")) or "",
                "hostname": str(lease.get("Hostname") or ""),
                "signal": int(sta.get("signal", 0) or 0),
            })
        return stations

    def _wired_ready(self, links: dict) -> bool:
        wired = links.get("wired")
        if not wired:
            return False
        try:
            link = networkd.describe_link(wired[0])
        except Exception:  # noqa: BLE001
            return False
        if str(link.get("CarrierState", "")).lower() not in ("carrier", "enslaved"):
            return False
        # Any IPv4 will do: on a direct cable the hub reaches the device over
        # the link-local address, which is the only one the port has.
        return bool(networkd.ipv4_addresses(link))

    # ── start ───────────────────────────────────────────────────────────────

    def start(self, ssid: str, passphrase: str, channel: int) -> dict:
        error = _validate_request(ssid, passphrase, channel)
        if error:
            return {"success": False, "message": error}
        # Refuse at once while a transition runs, instead of queueing behind
        # it on the radio; the check is repeated once the radio is held.
        with self._lock:
            refused = self._start_refusal()
        if refused:
            return {"success": False, "message": refused}
        with self.radio:
            with self._lock:
                refused = self._start_refusal()
                if refused:
                    return {"success": False, "message": refused}
                self._transition = True
            try:
                return self._start(ssid, passphrase, channel)
            finally:
                with self._lock:
                    self._transition = False

    def _start_refusal(self) -> str | None:
        """Why a start cannot claim the transition now; called under ``_lock``."""
        if self._transition:
            return "An access point transition is in progress"
        if self._active:
            return "The access point is already active"
        return None

    def _start(self, ssid: str, passphrase: str, channel: int) -> dict:
        links = self._discover()
        wifi = links.get("wifi")
        if not wifi:
            return self._fail("No wireless interface")
        if not self._wired_ready(links):
            return self._fail("The wired port needs a link and an address first: the access point "
                              "takes the radio off the Wi-Fi network")
        overlap = self._overlap(links)
        if overlap:
            return self._fail(overlap)
        try:
            ssid_arg, psk_arg = encode_ssid(ssid), encode_psk(passphrase)
        except WpaError as exc:
            return self._fail(str(exc))
        if not os.path.isdir(self._run_dir):
            return self._fail("This deployment cannot manage the access point network file")

        ifindex, iface = wifi
        wpa = self._wpa(iface)
        # The image sets a global regulatory country; ``country`` is not a
        # per-network field, so an unset one is refused, never patched here.
        country = self._country(wpa)
        if not country:
            return self._fail("wpa_supplicant has no regulatory country; the access point cannot start")
        picked, why = self._pick_channel(wpa, channel)
        if picked is None:
            return self._fail(why or "no usable channel")
        channel = picked
        net_id: str | None = None
        path = self._path(iface)
        try:
            # 1. The networkd file first: it matches only in AP mode, so it is
            #    inert until the radio switches, and ready the moment it does.
            #    World-readable on purpose: networkd runs as its own user and
            #    silently skips a file it cannot open, and the file holds an
            #    address and a pool, nothing secret.
            atomic_write_bytes(path, _network_file(iface, self._cidr), mode=0o644)
            networkd.reload()
            # 2. An unsaved wpa block for the access point.
            net_id = wpa.add_network()
            for key, value in (("ssid", ssid_arg), ("mode", "2"), ("frequency", str(CHANNELS[channel])),
                               ("key_mgmt", "WPA-PSK"), ("proto", "RSN"), ("pairwise", "CCMP"),
                               ("psk", psk_arg)):
                wpa.set_network(net_id, key, value)
            wpa.select_network(net_id)
            # 3. Wait for the radio to be an access point, then for the address.
            if not self._await(lambda: self._is_ap(wpa), _AP_TIMEOUT_S):
                raise WpaError("the radio did not become an access point in time")
            # 4. networkd does not re-run the match by itself when the link's
            #    type flips to AP, so a reconfigure makes it pick up the AP file.
            networkd.reconfigure_link(ifindex)
            if not self._await(lambda: self._has_address(ifindex), _ADDRESS_TIMEOUT_S):
                raise RuntimeError("networkd did not configure the access point address")
            if not self._await(lambda: self._dhcp_server_ready(ifindex), _ADDRESS_TIMEOUT_S):
                raise RuntimeError("networkd did not start the DHCP server")
        except Exception as exc:  # noqa: BLE001 - every failure rolls back
            logger.error("access point start failed: %s; rolling back", exc)
            self._rollback(wpa, ifindex, iface, net_id)
            return self._fail(f"{exc}; the radio is back in station mode")

        with self._lock:
            self._active, self._ssid, self._frequency, self._net_id = True, ssid, CHANNELS[channel], net_id
            self._joined = False
            self._deadline = self._now() + JOIN_DEADLINE_S
            self._message = "Access point started"
            self._arm_deadline()
            self._watch_stop = threading.Event()
            threading.Thread(target=self._watch_stations, args=(iface, self._watch_stop),
                             name="ap-stations", daemon=True).start()
        logger.info("access point %s up on %d MHz (country %s)", ssid, CHANNELS[channel], country)
        return {"success": True, "message": "Access point started"}

    def _fail(self, message: str) -> dict:
        with self._lock:
            self._message = message
        return {"success": False, "message": message}

    def _overlap(self, links: dict) -> str | None:
        """Why the access point subnet may not be used, or ``None``.

        A link whose addresses cannot be read counts as an overlap: starting
        blind could put the access point on the hub's own subnet."""
        ap_net = ipaddress.ip_interface(self._cidr).network
        for kind, (ifindex, _name) in links.items():
            try:
                link = networkd.describe_link(ifindex)
            except Exception as exc:  # noqa: BLE001
                logger.error("cannot describe the %s link before starting the access point: %s", kind, exc)
                return (f"The {kind} link's addresses cannot be read, so the access point subnet "
                        f"{self._cidr} cannot be checked against it")
            for ip, prefix in networkd.ipv4_addresses(link):
                if ipaddress.ip_interface(f"{ip}/{prefix}").network.overlaps(ap_net):
                    return f"The access point subnet {self._cidr} overlaps the {kind} network"
        return None

    def _is_ap(self, wpa: WpaCtrl) -> bool:
        try:
            st = wpa.status()
        except WpaError:
            return False
        return st.get("mode") == "AP" and st.get("wpa_state") == "COMPLETED"

    def _has_address(self, ifindex: int) -> bool:
        try:
            return self._ap_address(networkd.describe_link(ifindex)) is not None
        except Exception:  # noqa: BLE001
            return False

    def _dhcp_server_ready(self, ifindex: int) -> bool:
        """networkd reports a ``DHCPServer`` object on the link once the
        server is up; without it a joined phone would get no lease."""
        try:
            return isinstance(networkd.describe_link(ifindex).get("DHCPServer"), dict)
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _usable_channels(wpa: WpaCtrl) -> list[int] | None:
        """The channels of ``CHANNELS`` the radio may start a network on right
        now, or ``None`` when the daemon cannot say."""
        try:
            usable = wpa.usable_frequencies()
        except WpaError:
            return None
        return sorted(ch for ch, mhz in CHANNELS.items() if mhz in usable)

    def _pick_channel(self, wpa: WpaCtrl, channel: int) -> tuple[int | None, str | None]:
        """Resolve the request's channel against the live regulatory state.

        Automatic prefers the channel the station link is on — the radio has
        certainly heard a beacon there — and otherwise takes the lowest usable
        one. An explicit channel that is blocked right now is refused here,
        before the radio is touched, with the usable ones named: letting
        wpa_supplicant fail on it costs the full wait and says nothing.
        """
        usable = self._usable_channels(wpa)
        if channel == AUTO_CHANNEL:
            if usable is None:
                return min(CHANNELS), None
            if not usable:
                return None, ("no 5 GHz channel can start an access point right now: the radio has "
                              "not heard a beacon on any of them yet")
            try:
                current = int(wpa.status().get("freq", "") or 0)
            except (WpaError, ValueError):
                current = 0
            for ch in usable:
                if CHANNELS[ch] == current:
                    return ch, None
            return usable[0], None
        if usable is not None and channel not in usable:
            listed = ", ".join(str(c) for c in usable) or "none"
            return None, (f"channel {channel} cannot start an access point right now "
                          f"(usable: {listed}); leave the channel on automatic")
        return channel, None

    @staticmethod
    def _country(wpa: WpaCtrl) -> str:
        try:
            return wpa.get_country()
        except WpaError:
            return ""

    def _await(self, ready: Callable[[], bool], timeout: float) -> bool:
        deadline = self._now() + timeout
        while True:
            if ready():
                return True
            if self._now() >= deadline:
                return False
            time.sleep(_POLL_S)

    # ── stop / rollback ─────────────────────────────────────────────────────

    def stop(self) -> dict:
        """Return the radio to station mode. Idempotent."""
        refused = self._claim_stop()
        if refused:
            return {"success": False, "message": refused}
        return self._stop_claimed("Access point stopped")

    def _claim_stop(self, only_if_unjoined: bool = False) -> str | None:
        """Take the transition for a stop, or say why not.

        The join check and the claim happen under one lock: a station that
        associated before the deadline fired keeps the access point, one that
        associates after the claim goes down with it."""
        with self._lock:
            if self._transition:
                return "An access point transition is in progress"
            if only_if_unjoined and (not self._active or self._joined):
                return "a remote camera has joined"
            self._transition = True
            return None

    def _stop_claimed(self, message: str) -> dict:
        try:
            with self.radio:
                self._teardown()
            with self._lock:
                self._message = message
            return {"success": True, "message": message}
        finally:
            with self._lock:
                self._transition = False

    def reconcile(self) -> None:
        """Agent start-up: whatever the radio was doing, be a station now."""
        try:
            with self.radio:
                self._teardown()
        except Exception as exc:  # noqa: BLE001
            logger.error("access point reconcile failed: %s", exc)

    def _teardown(self) -> None:
        with self._lock:
            self._cancel_deadline()
            self._watch_stop.set()
            self._active, self._ssid, self._frequency, self._joined = False, "", 0, False
            self._deadline, self._net_id = None, None
        links = self._discover()
        wifi = links.get("wifi")
        if not wifi:
            return
        ifindex, iface = wifi
        self._rollback(self._wpa(iface), ifindex, iface, None)

    def _rollback(self, wpa: WpaCtrl, ifindex: int, iface: str, net_id: str | None) -> None:
        """Remove every trace of the access point; each step independent."""
        try:
            ids = [net_id] if net_id is not None else []
            for net in wpa.list_networks():
                if net["id"] not in ids and wpa.get_network(net["id"], "mode") == "2":
                    ids.append(net["id"])
            for an_id in ids:
                try:
                    wpa.remove_network(an_id)
                except WpaError as exc:
                    logger.warning("could not remove access point block %s: %s", an_id, exc)
        except WpaError as exc:
            logger.warning("could not list networks during rollback: %s", exc)
        for name in self._ap_files(iface):
            try:
                os.unlink(name)
            except OSError as exc:
                logger.warning("could not remove %s: %s", name, exc)
        for step, action in (("reload", networkd.reload),
                             ("reconfigure", lambda: networkd.reconfigure_link(ifindex))):
            try:
                action()
            except Exception as exc:  # noqa: BLE001
                logger.warning("networkd %s during rollback failed: %s", step, exc)
        try:
            wpa.reconfigure()
        except WpaError as exc:
            logger.error("wpa RECONFIGURE during rollback failed: %s", exc)

    def _path(self, iface: str) -> str:
        return os.path.join(self._run_dir, f"{AP_FILE_PREFIX}{iface}.network")

    def _ap_files(self, iface: str) -> list[str]:
        try:
            names = os.listdir(self._run_dir)
        except OSError:
            return []
        return [os.path.join(self._run_dir, n) for n in names if n.startswith(AP_FILE_PREFIX)]

    # ── join deadline ───────────────────────────────────────────────────────

    def _arm_deadline(self) -> None:
        self._cancel_deadline()
        self._timer = threading.Timer(JOIN_DEADLINE_S, self._deadline_fired)
        self._timer.daemon = True
        self._timer.start()

    def _cancel_deadline(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _deadline_fired(self) -> None:
        if self._claim_stop(only_if_unjoined=True) is not None:
            return
        logger.info("no remote camera joined the access point in %d s; stopping it", int(JOIN_DEADLINE_S))
        self._stop_claimed("No remote camera joined within five minutes; the access point stopped")

    def note_station(self) -> None:
        """A station has joined: the join deadline no longer applies."""
        with self._lock:
            if self._active and not self._joined:
                self._joined = True
                self._deadline = None
                self._cancel_deadline()

    def _watch_stations(self, iface: str, stop: threading.Event) -> None:
        wpa = self._wpa(iface)
        while not stop.wait(_STATION_POLL_S):
            try:
                if wpa.all_sta():
                    self.note_station()
                    return
            except WpaError:
                pass
