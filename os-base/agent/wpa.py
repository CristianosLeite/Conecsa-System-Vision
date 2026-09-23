# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Direct client for the wpa_supplicant control interface.

wpa_supplicant runs on the host with its control socket at
`/run/wpa_supplicant/<iface>`. That directory is bind-mounted into the agent
container. We do NOT use the `wpa_cli` binary: it creates its reply socket under
`/tmp`, which lives in the container's mount namespace, so the host daemon's
`sendto()` to that path never arrives and every command times out.

Instead we speak the (simple) control protocol over an AF_UNIX/SOCK_DGRAM socket
and bind our reply socket *inside* the shared `/run/wpa_supplicant` mount — the
same path on host and container — so the daemon can reach it. Commands are the
raw upper-case control verbs (STATUS, SCAN, SCAN_RESULTS, …).

NOTE: never log PSKs — SET_NETWORK psk arguments are not echoed to the logger.
"""
import itertools
import logging
import os
import re
import socket
import string
import time

logger = logging.getLogger(__name__)

CTRL_DIR = "/run/wpa_supplicant"

_counter = itertools.count()

_SSID_MAX_BYTES = 32
_PASSPHRASE_MIN, _PASSPHRASE_MAX = 8, 63
_PSK_HEX_LEN = 64


class WpaError(RuntimeError):
    """A wpa_supplicant control command failed, timed out, or returned FAIL."""


# ── credential serialization ────────────────────────────────────────────────
# SET_NETWORK values are parsed by wpa_supplicant's config grammar: a
# double-quoted string with backslash escapes, or a bare hex string. The
# operator's SSID/passphrase must never be able to change that grammar.

# One line of ``GET_CAPABILITY freq``: " 36 = 5180 MHz (NO_IR)" or " 44 = 5220 MHz".
_FREQ_LINE = re.compile(r"^\s*(\d+)\s*=\s*(\d+)\s*MHz(.*)$")


def encode_ssid(ssid: str) -> str:
    """Serialize an SSID for ``SET_NETWORK <id> ssid`` as its hex byte form.

    Hex sidesteps quoting entirely, so quotes, backslashes, spaces and
    non-ASCII characters all round-trip exactly. Raises :class:`WpaError` for
    an empty SSID or one over the 802.11 limit of 32 bytes.
    """
    raw = ssid.encode("utf-8")
    if not raw:
        raise WpaError("SSID is required")
    if len(raw) > _SSID_MAX_BYTES:
        raise WpaError(f"SSID must be at most {_SSID_MAX_BYTES} bytes")
    return raw.hex()


def encode_psk(password: str) -> str:
    """Serialize a passphrase (or a raw PSK) for ``SET_NETWORK <id> psk``.

    A 64-digit hex string is the pre-shared key itself and goes bare;
    anything else is a WPA passphrase, which the standard restricts to 8–63
    printable ASCII characters, serialized as a quoted string with ``\\`` and
    ``"`` escaped the way wpa_supplicant's parser expects. Anything outside
    those rules raises :class:`WpaError` instead of reaching the socket.
    """
    if len(password) == _PSK_HEX_LEN and all(c in string.hexdigits for c in password):
        return password
    if not _PASSPHRASE_MIN <= len(password) <= _PASSPHRASE_MAX:
        raise WpaError(
            f"Wi-Fi passphrase must be {_PASSPHRASE_MIN}-{_PASSPHRASE_MAX} characters "
            f"(or a {_PSK_HEX_LEN}-digit hex key)")
    if any(not 0x20 <= ord(c) <= 0x7E for c in password):
        raise WpaError("Wi-Fi passphrase must be printable ASCII")
    escaped = password.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


class WpaCtrl:
    """Stateless control-socket client for a single wireless interface.

    Speaks the ``wpa_ctrl`` protocol over the socket in ``CTRL_DIR``.
    """

    def __init__(self, iface: str):
        self.iface = iface

    # ── transport ───────────────────────────────────────────────────────────────

    def _cmd(self, command: str, timeout: float = 5.0) -> str:
        """Send one control verb and return the daemon's raw reply.

        Binds a short-lived reply socket inside the shared ``/run/wpa_supplicant``
        mount so the host daemon can reach it. Raises :class:`WpaError` on
        timeout or socket error.
        """
        daemon = os.path.join(CTRL_DIR, self.iface)
        if not os.path.exists(daemon):
            raise WpaError(f"control socket {daemon} not found")
        local = os.path.join(CTRL_DIR, f".conecsa-{os.getpid()}-{next(_counter)}")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            try:
                os.unlink(local)
            except OSError:
                pass
            sock.bind(local)
            sock.settimeout(timeout)
            sock.connect(daemon)
            sock.send(command.encode())
            return sock.recv(32768).decode(errors="replace")
        except socket.timeout as exc:
            raise WpaError(f"timeout running '{command.split()[0]}'") from exc
        except OSError as exc:
            raise WpaError(f"control error on '{command.split()[0]}': {exc}") from exc
        finally:
            sock.close()
            try:
                os.unlink(local)
            except OSError:
                pass

    def _cmd_ok(self, command: str, timeout: float = 5.0) -> None:
        """Run *command* and raise :class:`WpaError` unless it replies ``OK``."""
        reply = self._cmd(command, timeout=timeout).strip()
        last = reply.splitlines()[-1].strip() if reply else ""
        if last != "OK":
            raise WpaError(f"'{command.split()[0]}' returned {reply or 'no reply'}")

    # ── status / scan ─────────────────────────────────────────────────────────────

    def status(self) -> dict[str, str]:
        """Return the parsed STATUS key/value map (ssid, wpa_state, …)."""
        out = self._cmd("STATUS")
        result: dict[str, str] = {}
        for line in out.splitlines():
            key, sep, value = line.partition("=")
            if sep:
                result[key.strip()] = value.strip()
        return result

    def usable_frequencies(self) -> set[int]:
        """Frequencies (MHz) the radio may *start* a network on right now.

        ``GET_CAPABILITY freq`` lists every channel with the driver's live
        regulatory flags. ``NO_IR`` (no initiating radiation) is not fixed per
        channel: the driver lifts it on a channel once it has heard a beacon
        there and puts it back after a while, so the set changes from one hour
        to the next and is read at start time, never cached. ``DISABLED`` and
        ``RADAR`` (DFS) channels are out as well. Raises :class:`WpaError` when
        the daemon cannot say.
        """
        reply = self._cmd("GET_CAPABILITY freq")
        if reply.strip() in ("", "FAIL"):
            raise WpaError("GET_CAPABILITY freq returned " + (reply.strip() or "no reply"))
        usable: set[int] = set()
        for line in reply.splitlines():
            match = _FREQ_LINE.match(line)
            if not match:
                continue
            flags = match.group(3).upper()
            if "NO_IR" in flags or "DISABLED" in flags or "RADAR" in flags:
                continue
            usable.add(int(match.group(2)))
        return usable

    def get_country(self) -> str:
        """The regulatory country the daemon runs under (``GET country``);
        empty when none is set."""
        reply = self._cmd("GET country").strip()
        last = reply.splitlines()[-1].strip() if reply else ""
        return "" if last == "FAIL" else last

    def scan(self) -> list[dict]:
        """Trigger a scan and return parsed results (one entry per BSS)."""
        try:
            self._cmd("SCAN")
        except WpaError:
            pass  # FAIL-BUSY when a scan is already running — reuse results
        time.sleep(2.0)
        out = self._cmd("SCAN_RESULTS")
        networks: list[dict] = []
        for line in out.splitlines()[1:]:  # skip header row
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            _bssid, _freq, signal, flags, ssid = parts[0], parts[1], parts[2], parts[3], parts[4]
            if not ssid:
                continue
            networks.append({
                "ssid": ssid,
                "signal": _safe_int(signal),
                "security": _security_from_flags(flags),
            })
        return networks

    def list_networks(self) -> list[dict]:
        """Return saved networks as ``{id, ssid, flags}`` dicts."""
        out = self._cmd("LIST_NETWORKS")
        nets: list[dict] = []
        for line in out.splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            nets.append({
                "id": parts[0].strip(),
                "ssid": parts[1].strip(),
                "flags": parts[3] if len(parts) > 3 else "",
            })
        return nets

    # ── connect / forget ───────────────────────────────────────────────────────────

    def find_network_id(self, ssid: str) -> str | None:
        """Return the saved-network id for *ssid*, or ``None`` if not saved."""
        for net in self.list_networks():
            if net["ssid"] == ssid:
                return net["id"]
        return None

    def add_network(self) -> str:
        """Create an empty network and return its id."""
        reply = self._cmd("ADD_NETWORK").strip()
        last = reply.splitlines()[-1].strip() if reply else ""
        if not last.isdigit():
            raise WpaError(f"ADD_NETWORK returned {reply or 'no reply'}")
        return last

    def set_network(self, net_id: str, key: str, value: str) -> None:
        """Set one network variable (SET_NETWORK). PSK values are never logged."""
        self._cmd_ok(f"SET_NETWORK {net_id} {key} {value}")

    def get_network_int(self, net_id: str, key: str, default: int = 0) -> int:
        """Read a numeric network variable (e.g. priority). Unset vars reply
        FAIL, which maps to `default`."""
        reply = self._cmd(f"GET_NETWORK {net_id} {key}").strip()
        if not reply or reply.splitlines()[-1].strip() == "FAIL":
            return default
        return _safe_int(reply.strip().strip('"'), default)

    def select_network(self, net_id: str) -> None:
        """Select *net_id* and disable the others (SELECT_NETWORK)."""
        self._cmd_ok(f"SELECT_NETWORK {net_id}")

    def enable_network(self, net_id: str) -> None:
        """Mark *net_id* eligible for (re)association (ENABLE_NETWORK)."""
        self._cmd_ok(f"ENABLE_NETWORK {net_id}")

    def remove_network(self, net_id: str) -> None:
        """Delete a saved network (REMOVE_NETWORK)."""
        self._cmd_ok(f"REMOVE_NETWORK {net_id}")

    def save_config(self) -> None:
        """Persist the in-memory config to disk (SAVE_CONFIG).

        Access-point blocks (``mode=2``) are removed first: they are volatile
        by design, and one that reached the file would bring the access point
        back on the next RECONFIGURE or reboot.
        """
        for net in self.list_networks():
            try:
                if self.get_network(net["id"], "mode") == "2":
                    self.remove_network(net["id"])
            except WpaError as exc:
                raise WpaError(f"could not drop access point block {net['id']} before saving: {exc}") from exc
        self._cmd_ok("SAVE_CONFIG")

    def reconfigure(self) -> None:
        """Re-read the on-disk config and re-associate. Used to roll back a
        failed connect attempt: since we never SAVE_CONFIG on failure, the file
        still describes the previously-working network(s), so this restores the
        device to its prior connection without stranding it."""
        self._cmd_ok("RECONFIGURE")

    def enable_all(self) -> None:
        """Re-enable every saved network (SELECT_NETWORK disables the others),
        so all known networks stay eligible for auto-reconnect on reboot.

        Access-point blocks (``mode=2``) are skipped: enabling one would turn
        the radio into an access point on the next scan.
        """
        for net in self.list_networks():
            try:
                if self.get_network(net["id"], "mode") == "2":
                    continue
                self.enable_network(net["id"])
            except WpaError:
                pass

    def get_network(self, net_id: str, key: str) -> str | None:
        """Read one network variable as text (GET_NETWORK); ``None`` when unset."""
        reply = self._cmd(f"GET_NETWORK {net_id} {key}").strip()
        if not reply or reply.splitlines()[-1].strip() == "FAIL":
            return None
        return reply.splitlines()[-1].strip().strip('"')

    def all_sta(self) -> list[dict]:
        """Return the stations joined to this access point (ALL_STA).

        Each entry is ``{mac, signal}``; ``signal`` is 0 when the driver does not
        report it. The MAC is needed to match a DHCP lease and must never leave
        the agent.
        """
        out = self._cmd("ALL_STA")
        stations: list[dict] = []
        current: dict | None = None
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            if "=" not in line:
                current = {"mac": line.lower(), "signal": 0}
                stations.append(current)
            elif current is not None:
                key, _, value = line.partition("=")
                if key.strip() == "signal":
                    current["signal"] = _safe_int(value.strip())
        return stations


def _safe_int(value: str, default: int = 0) -> int:
    """Parse *value* as int, returning *default* on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _security_from_flags(flags: str) -> str:
    """Map a scan_results flags string to a coarse security label."""
    f = flags.upper()
    if "SAE" in f or "WPA3" in f:
        return "WPA3"
    if "WPA2" in f or "RSN" in f:
        return "WPA2"
    if "WPA" in f:
        return "WPA2"
    if "WEP" in f:
        return "WEP"
    return "OPEN"
