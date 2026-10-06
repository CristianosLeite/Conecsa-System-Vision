# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
ApAgent — the device's Wi-Fi access point for the remote camera.

The device has one radio. As an access point it leaves its Wi-Fi network, so
the hub reaches it over the wired port only: an admin start requires a wired
link with an IPv4 address — a link-local one counts, since a hub on a direct cable
with no DHCP server reaches the device by exactly that — and every step that
fails is undone (the volatile
network file removed, the wpa block removed, networkd reloaded, wpa_supplicant
``RECONFIGURE``d) so the radio returns to the persisted station configuration.

The radio configuration is never persisted: the access-point block never
passes through ``SAVE_CONFIG`` and the networkd file lives under ``/run``, so
an agent restart reconciles the radio back to station mode and a reboot does
the same by itself. What is persisted is the *intent*: a started access point
leaves ``AP_STATE_PATH`` (SSID, the derived WPA2 key — never the passphrase —
and the requested channel), only an admin stop removes it, and an agent start
that finds it brings the access point back whatever the wired port's state,
retrying until it succeeds or an admin start or stop takes over. Nothing turns
the access point off by itself.

``DescribeLink`` reports DHCP leases as
``DHCPServer.Leases[] = {ClientId: bytes, Address: bytes, Hostname: str,
ExpirationUSec: int}``. Client ids beginning with ``0x01`` carry the station's
MAC, which is how a lease is matched to an ``ALL_STA`` entry — and the MAC is
the one thing this module never returns or logs.
"""
import hashlib
import ipaddress
import logging
import os
import string
import threading
import time
from typing import Callable

from conecsa_common.atomic import atomic_write_bytes, atomic_write_json, fsync_dir, read_json

from . import networkd
from .wpa import WpaCtrl, WpaError, encode_psk, encode_ssid

logger = logging.getLogger(__name__)

RUN_NETWORKD_DIR = "/run/systemd/network"
AP_FILE_PREFIX = "05-conecsa-ap-"
# Survives a reboot: the directory also holds the clock floor (time_agent).
AP_STATE_PATH = "/var/lib/conecsa/access-point.json"
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
_AP_TIMEOUT_S = 15.0
_ADDRESS_TIMEOUT_S = 10.0
_POLL_S = 0.5
_DHCP_POOL_OFFSET = 10
_DHCP_POOL_SIZE = 20
# After a restart the access point is retried until it comes back: every
# _RESTORE_RETRY_S while boot settles (the wpa socket and a usable channel
# arrive late), then every _RESTORE_SLOW_RETRY_S, since each failed attempt
# that reached the radio bounces its station link.
_RESTORE_WINDOW_S = 120.0
_RESTORE_RETRY_S = 10.0
_RESTORE_SLOW_RETRY_S = 60.0
_RESTORING = "Restoring the access point after a restart"
# ``read_json``'s answer for a missing file: a file holding JSON ``null`` must
# still reach validation (and removal) rather than pass for an absent one.
_MISSING = object()


def _derive_psk(ssid: str, passphrase: str) -> str:
    """The WPA2 pre-shared key (IEEE 802.11i PBKDF2), as 64 hex digits.

    The radio is configured with it rather than the passphrase, so the state
    file can hold the key a restart needs without the passphrase itself."""
    return hashlib.pbkdf2_hmac("sha1", passphrase.encode("utf-8"), ssid.encode("utf-8"), 4096, 32).hex()


def _valid_state(state: object) -> dict | None:
    """The persisted intent if it is well formed, else ``None``."""
    if not isinstance(state, dict):
        return None
    ssid, psk, channel = state.get("ssid"), state.get("psk"), state.get("channel")
    if not isinstance(ssid, str):
        return None
    try:
        # Valid JSON can still carry a lone surrogate, which has no UTF-8 form.
        ssid_bytes = len(ssid.encode("utf-8"))
    except UnicodeEncodeError:
        return None
    if not 1 <= ssid_bytes <= 32:
        return None
    if not isinstance(psk, str) or len(psk) != 64 or any(c not in string.hexdigits for c in psk):
        return None
    if isinstance(channel, bool) or not isinstance(channel, int) or \
            (channel != AUTO_CHANNEL and channel not in CHANNELS):
        return None
    return {"ssid": ssid, "psk": psk, "channel": channel}


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
                 wpa_factory: Callable[[str], WpaCtrl] = WpaCtrl,
                 state_path: str = AP_STATE_PATH,
                 spawn: Callable[[Callable[[], None]], None] | None = None):
        self._discover = discover
        self._run_dir = run_dir
        self._state_path = state_path
        self._spawn = spawn or (lambda fn: threading.Thread(target=fn, name="ap-restore", daemon=True).start())
        self._restore_cancel: threading.Event | None = None
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
        links = self._discover()
        out = {
            "active": active, "ssid": ssid if active else "", "frequency_mhz": frequency if active else 0,
            "address": "", "prefix": 0, "stations": [],
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
        try:
            encode_psk(passphrase)
        except WpaError as exc:
            return self._fail(str(exc))
        # An admin start supersedes a restore still waiting after a restart.
        self._cancel_restore()
        return self._locked_start(ssid, _derive_psk(ssid, passphrase), channel, restored=False)

    def _locked_start(self, ssid: str, psk: str, channel: int, restored: bool,
                      cancel: threading.Event | None = None) -> dict:
        # Refuse at once while a transition runs, instead of queueing behind
        # it on the radio; the check is repeated once the radio is held.
        with self._lock:
            refused = self._start_refusal(cancel)
        if refused:
            return {"success": False, "message": refused}
        with self.radio:
            with self._lock:
                refused = self._start_refusal(cancel)
                if refused:
                    return {"success": False, "message": refused}
                self._transition = True
            try:
                return self._start(ssid, psk, channel, restored)
            finally:
                with self._lock:
                    self._transition = False

    def _start_refusal(self, cancel: threading.Event | None = None) -> str | None:
        """Why a start cannot claim the transition now; called under ``_lock``.

        A restore passes its ``cancel`` event: an admin start or stop sets it
        under this same lock, so a restore that lost the race to one of them
        never claims the radio after it, even if it passed its own loop check
        just before."""
        if cancel is not None and cancel.is_set():
            return "An admin start or stop superseded the restore"
        if self._transition:
            return "An access point transition is in progress"
        if self._active:
            return "The access point is already active"
        return None

    def _start(self, ssid: str, psk: str, channel: int, restored: bool) -> dict:
        requested = channel
        links = self._discover()
        wifi = links.get("wifi")
        if not wifi:
            return self._fail("No wireless interface")
        # An admin start needs the wired port, the hub's only way in once the
        # radio leaves the Wi-Fi network. A restore does not wait for it: the
        # access point comes back whatever the cable's state.
        if not restored and not self._wired_ready(links):
            return self._fail("The wired port needs a link and an address first: the access point "
                              "takes the radio off the Wi-Fi network")
        overlap = self._overlap(links)
        if overlap:
            return self._fail(overlap)
        try:
            ssid_arg, psk_arg = encode_ssid(ssid), encode_psk(psk)
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

        message = "Access point restored after a restart" if restored else "Access point started"
        # A fresh start records the intent; a restore found it already there.
        # An intent that cannot be written must not leave an older one behind
        # for the next restart to bring back instead.
        if not restored and not self._save_state(ssid, psk, requested):
            if self._clear_state():
                message = ("Access point started, but it will not come back after a restart: "
                           "its state could not be saved")
            else:
                message = ("Access point started, but its state could not be saved or cleared: "
                           "a restart may bring back an earlier access point")
        with self._lock:
            self._active, self._ssid, self._frequency, self._net_id = True, ssid, CHANNELS[channel], net_id
            self._message = message
        logger.info("access point %s %s on %d MHz (country %s)", ssid, "restored" if restored else "up",
                    CHANNELS[channel], country)
        return {"success": True, "message": message}

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

        When nothing fits on the first read, one scan limited to the four
        frequencies is tried before refusing: a beacon received during it
        lifts ``NO_IR`` on that channel (the kernel's beacon hint), which is
        the only way a stock world-wide channel plan ever opens 5 GHz.
        """
        usable = self._usable_channels(wpa)
        if usable is not None and (not usable if channel == AUTO_CHANNEL else channel not in usable):
            logger.info("no usable 5 GHz channel on first read (usable: %s); scanning once", usable)
            try:
                wpa.scan_frequencies(CHANNELS.values())
            except WpaError as exc:
                logger.warning("scan before refusing failed: %s", exc)
            # Keep the first read's answer unless the re-read says something:
            # a transient control-socket failure must not turn "blocked" into
            # "unknown" and let the request through unchecked.
            rescanned = self._usable_channels(wpa)
            if rescanned is not None:
                usable = rescanned
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
        """Return the radio to station mode and forget the intent, so a
        restart leaves it a station too. Idempotent."""
        self._cancel_restore()
        refused = self._claim_stop()
        if refused:
            return {"success": False, "message": refused}
        if self._clear_state():
            return self._stop_claimed("Access point stopped")
        return self._stop_claimed("Access point stopped, but it may come back after a restart: "
                                  "its saved state could not be removed")

    def _claim_stop(self) -> str | None:
        """Take the transition for a stop, or say why not."""
        with self._lock:
            if self._transition:
                return "An access point transition is in progress"
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
        """Agent start-up: whatever the radio was doing, be a station now;
        then, if the access point was on before the restart, bring it back
        in the background."""
        try:
            with self.radio:
                self._teardown()
        except Exception as exc:  # noqa: BLE001
            logger.error("access point reconcile failed: %s", exc)
        state = self._load_state()
        if state is None:
            return
        cancel = threading.Event()
        with self._lock:
            self._restore_cancel = cancel
            self._message = _RESTORING
        logger.info("the access point was on before the restart; restoring it")
        self._spawn(lambda: self._restore(state, cancel))

    def _restore(self, state: dict, cancel: threading.Event) -> None:
        """Start the persisted access point, retrying until it is up.

        It never gives up on its own: only an admin start or stop (``cancel``)
        ends it. At boot the wpa socket and a usable channel arrive late, so
        the first ``_RESTORE_WINDOW_S`` retry quickly and later attempts slow
        down. The wired port is not waited for — see ``_start``."""
        slow_after = self._now() + _RESTORE_WINDOW_S
        logged = ""
        while not cancel.is_set():
            try:
                result = self._locked_start(state["ssid"], state["psk"], state["channel"], restored=True,
                                            cancel=cancel)
                if result["success"]:
                    return
                why = result["message"]
            except Exception as exc:  # noqa: BLE001 - a restore attempt must not kill the thread
                why = str(exc)
            if why != logged:
                logger.warning("access point restore attempt failed, retrying: %s", why)
                logged = why
            with self._lock:
                if not cancel.is_set():
                    self._message = f"{_RESTORING}; the last attempt failed: {why}"
            wait = _RESTORE_RETRY_S if self._now() < slow_after else _RESTORE_SLOW_RETRY_S
            if cancel.wait(wait):
                return

    def _cancel_restore(self) -> None:
        with self._lock:
            if self._restore_cancel is not None:
                self._restore_cancel.set()
                self._restore_cancel = None

    # ── persisted intent ────────────────────────────────────────────────────

    def _save_state(self, ssid: str, psk: str, channel: int) -> bool:
        try:
            atomic_write_json(self._state_path, {"ssid": ssid, "psk": psk, "channel": channel})
            return True
        except OSError as exc:
            logger.error("could not save the access point state to %s: %s", self._state_path, exc.strerror)
            return False

    def _clear_state(self) -> bool:
        """Remove the intent; ``True`` when none is left on disk.

        The removal is made durable like the write: a power cut before the
        directory entry reaches the disk would bring a stopped access point
        back at the next start."""
        try:
            os.unlink(self._state_path)
        except FileNotFoundError:
            return True
        except OSError as exc:
            logger.error("could not remove the access point state %s: %s", self._state_path, exc.strerror)
            return False
        fsync_dir(os.path.dirname(os.path.abspath(self._state_path)))
        return True

    def _load_state(self) -> dict | None:
        raw = read_json(self._state_path, _MISSING)
        if raw is _MISSING:
            return None
        state = _valid_state(raw)
        if state is None:
            logger.error("ignoring a malformed access point state in %s", self._state_path)
            self._clear_state()
        return state

    def _teardown(self) -> None:
        with self._lock:
            self._active, self._ssid, self._frequency, self._net_id = False, "", 0, None
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
