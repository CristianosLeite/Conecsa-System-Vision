# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""The Wi-Fi access point: start, rollback, stop, deadline, reconcile.

Runs the real ``ApAgent`` against a fake wpa control, a fake networkd whose
``DescribeLink`` returns the real lease shapes, a temp ``/run``
directory and a controlled clock. No sockets, no D-Bus.
"""
import logging
import threading
from typing import Any

import pytest
from agent import ap_agent as mod
from agent.ap_agent import ApAgent
from agent.network_agent import AccessPointActive, NetworkAgent
from agent.wpa import WpaError

WIFI, WIRED = (3, "wlP1p1s0"), (2, "enP8p1s0")
PASS = "correct-horse-battery"
STA_MAC = "02:00:5e:00:53:01"  # documentation-range MAC, not a real station


def ipv4(text):
    return [int(b) for b in text.split(".")]


def addr(text, prefix=24, source="DHCPv4"):
    return {"Family": 2, "Address": ipv4(text), "PrefixLength": prefix, "ConfigSource": source}


def lease(mac, address, hostname="camera"):
    client_id = [1] + [int(b, 16) for b in mac.split(":")]
    return {"ClientId": client_id, "Address": ipv4(address), "Hostname": hostname, "ExpirationUSec": 1}


class FakeWpa:
    """Becomes an access point when the AP block is selected; scripted failures."""

    def __init__(self, iface):
        self.iface = iface
        self.networks = {"0": {"ssid": "Plant", "mode": "0"}}
        self.selected = None
        self.reconfigured = 0
        self.saved = 0
        self.stations = []
        self.fail_add = False
        self.never_ap = False
        self.fail_reconfigure = False
        self.country = "BR"
        self.block_select: threading.Event | None = None
        self.commands = []
        # ``GET_CAPABILITY freq`` as parsed: 48 is NO_IR unless a test lifts it,
        # and the station link sits on channel 44.
        self.usable = {5180, 5200, 5220}
        self.capability_fails = False
        self.station_freq = "5220"

    def add_network(self):
        if self.fail_add:
            raise WpaError("ADD_NETWORK returned FAIL")
        net_id = str(len(self.networks))
        self.networks[net_id] = {}
        return net_id

    def set_network(self, net_id, key, value):
        self.commands.append((net_id, key, value))
        self.networks[net_id][key] = value

    def get_network(self, net_id, key):
        return self.networks.get(net_id, {}).get(key)

    def select_network(self, net_id):
        self.selected = net_id
        if self.block_select is not None:
            self.block_select.wait(5.0)

    def remove_network(self, net_id):
        self.networks.pop(net_id)

    def list_networks(self):
        return [{"id": i, "ssid": n.get("ssid", ""), "flags": ""} for i, n in self.networks.items()]

    def status(self):
        block = self.networks.get(self.selected or "", {})
        if block.get("mode") == "2" and not self.never_ap:
            return {"wpa_state": "COMPLETED", "mode": "AP", "freq": block.get("frequency", "")}
        return {"wpa_state": "COMPLETED", "mode": "station", "ssid": "Plant", "freq": self.station_freq}

    def usable_frequencies(self):
        if self.capability_fails:
            raise WpaError("GET_CAPABILITY freq returned FAIL")
        return set(self.usable)

    def reconfigure(self):
        self.reconfigured += 1
        if self.fail_reconfigure:
            raise WpaError("RECONFIGURE returned FAIL")
        for net_id in [i for i, n in self.networks.items() if n.get("mode") == "2"]:
            self.networks.pop(net_id)
        self.selected = None

    def save_config(self):
        self.saved += 1

    def all_sta(self):
        return [{"mac": m, "signal": -50} for m in self.stations]

    def enable_network(self, net_id):
        pass

    def enable_all(self):
        pass

    def get_country(self):
        return self.country


class FakeNetworkd:
    """The real DescribeLink lease shapes; the AP address appears
    after a reload while an AP block is selected."""

    def __init__(self, wpa_holder):
        self.wpa_holder = wpa_holder
        self.wired_addresses = [addr("192.168.10.20")]
        self.wired_carrier = "carrier"
        self.leases = []
        self.calls = []
        self.fail_reload = False
        self.fail_reconfigure_link = False
        self.no_ap_address = False
        self.no_dhcp_server = False
        self.undescribable: set[int] = set()

    def describe_link(self, ifindex):
        if ifindex in self.undescribable:
            raise RuntimeError("networkd: DescribeLink failed")
        if ifindex == WIRED[0]:
            return {"Type": "ether", "CarrierState": self.wired_carrier, "Addresses": self.wired_addresses,
                    "Routes": [], "DNS": []}
        wpa = self.wpa_holder[0]
        if wpa is not None and wpa.status().get("mode") == "AP" and not self.no_ap_address:
            link = {"Type": "wlan", "WirelessLanInterfaceTypeString": "ap",
                    "Addresses": [addr("10.98.76.1", 24, "static")]}
            if not self.no_dhcp_server:
                link["DHCPServer"] = {"PoolOffset": 10, "PoolSize": 20, "Leases": self.leases}
            return link
        return {"Type": "wlan", "WirelessLanInterfaceTypeString": "station",
                "Addresses": [addr("192.168.5.7", 22)], "Routes": [], "DNS": []}

    def reload(self):
        self.calls.append("reload")
        if self.fail_reload:
            self.fail_reload = False
            raise RuntimeError("networkd: Reload failed")

    def reconfigure_link(self, ifindex):
        self.calls.append(("reconfigure", ifindex))
        if self.fail_reconfigure_link:
            raise RuntimeError("networkd: ReconfigureLink failed")


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def rig(monkeypatch, tmp_path):
    holder: list[Any] = [None]

    def make_wpa(iface: str) -> Any:
        if holder[0] is None:
            holder[0] = FakeWpa(iface)
        return holder[0]

    networkd = FakeNetworkd(holder)
    monkeypatch.setattr(mod.networkd, "describe_link", networkd.describe_link)
    monkeypatch.setattr(mod.networkd, "reload", networkd.reload)
    monkeypatch.setattr(mod.networkd, "reconfigure_link", networkd.reconfigure_link)
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
    monkeypatch.setattr(mod, "_STATION_POLL_S", 0.01)
    clock = Clock()
    agent = ApAgent(lambda: {"wired": WIRED, "wifi": WIFI}, run_dir=str(tmp_path), now=clock,
                    wpa_factory=make_wpa)
    yield agent, networkd, holder, clock, tmp_path
    agent._teardown()


def ap_file(tmp_path):
    return tmp_path / f"05-conecsa-ap-{WIFI[1]}.network"


class TestStart:
    def test_a_good_start_writes_the_file_selects_the_block_and_reports_active(self, rig):
        agent, networkd, holder, clock, tmp_path = rig
        result = agent.start("conecsa-000001", PASS, 36)
        assert result["success"], result
        wpa = holder[0]
        text = ap_file(tmp_path).read_text()
        assert "WLANInterfaceType=ap" in text and "Address=10.98.76.1/24" in text
        assert "DHCPServer=yes" in text and "EmitDNS=no" in text
        # networkd runs as its own user: a 0600 file is skipped without a word.
        assert ap_file(tmp_path).stat().st_mode & 0o777 == 0o644
        block = wpa.networks[wpa.selected]
        assert block["mode"] == "2" and block["frequency"] == "5180"
        assert block["key_mgmt"] == "WPA-PSK" and block["pairwise"] == "CCMP"
        assert block["ssid"] == "conecsa-000001".encode().hex()
        assert wpa.saved == 0, "an access point block never reaches the saved config"
        assert networkd.calls[0] == "reload"

        st = agent.status()
        assert st["active"] and st["ssid"] == "conecsa-000001" and st["frequency_mhz"] == 5180
        assert (st["address"], st["prefix"]) == ("10.98.76.1", 24)
        assert st["join_deadline_remaining_secs"] == 300
        assert st["stations"] == [] and st["wired_ready"]

    @pytest.mark.parametrize("ssid,passphrase,channel,fragment", [
        ("", PASS, 36, "SSID"),
        ("x" * 33, PASS, 36, "SSID"),
        ("dev", "short", 36, "passphrase"),
        ("dev", "p" * 64, 36, "passphrase"),
        ("dev", PASS, 52, "channel"),
        ("dev", PASS, 100, "channel"),
    ])
    def test_bad_requests_touch_nothing(self, rig, ssid, passphrase, channel, fragment):
        agent, networkd, holder, _clock, tmp_path = rig
        result = agent.start(ssid, passphrase, channel)
        assert not result["success"] and fragment in result["message"]
        assert holder[0] is None and networkd.calls == [] and not ap_file(tmp_path).exists()

    def test_without_a_wired_link_the_radio_is_left_alone(self, rig):
        agent, networkd, holder, _clock, tmp_path = rig
        networkd.wired_carrier = "no-carrier"
        result = agent.start("dev", PASS, 36)
        assert not result["success"] and "wired" in result["message"]
        networkd.wired_carrier = "carrier"
        networkd.wired_addresses = []
        result = agent.start("dev", PASS, 36)
        assert not result["success"] and "wired" in result["message"], "carrier alone is no link to the hub"
        assert holder[0] is None and not ap_file(tmp_path).exists()

    def test_a_link_local_wired_address_is_a_link_to_the_hub(self, rig):
        # A hub on a direct cable, with no DHCP server between them, reaches
        # the device over 169.254/16: that cable is exactly what the rule is for.
        agent, networkd, holder, _clock, _tmp = rig
        networkd.wired_addresses = [addr("169.254.183.45", 16)]
        assert agent.status()["wired_ready"]
        assert agent.start("dev", PASS, 36)["success"]
        assert holder[0] is not None

    def test_a_blocked_channel_is_refused_before_the_radio_is_touched(self, rig):
        # 48 is a valid request but NO_IR right now: the refusal names the
        # usable channels instead of costing the full wait on a silent failure.
        agent, networkd, holder, _clock, tmp_path = rig
        result = agent.start("dev", PASS, 48)
        assert not result["success"] and "channel 48" in result["message"]
        assert "36, 40, 44" in result["message"] and "automatic" in result["message"]
        assert holder[0].selected is None and networkd.calls == [] and not ap_file(tmp_path).exists()

    def test_a_channel_becomes_usable_when_the_flag_lifts(self, rig):
        agent, _networkd, holder, _clock, _tmp = rig
        holder[0] = FakeWpa(WIFI[1])
        holder[0].usable.add(5240)
        result = agent.start("dev", PASS, 48)
        assert result["success"], result["message"]
        assert agent.status()["frequency_mhz"] == 5240

    def test_automatic_prefers_the_station_link_channel(self, rig):
        agent, _networkd, holder, _clock, _tmp = rig
        assert agent.start("dev", PASS, 0)["success"]
        assert agent.status()["frequency_mhz"] == 5220, "the station sat on 44: a beacon was heard there"
        assert ("1", "frequency", "5220") in holder[0].commands

    def test_automatic_takes_the_lowest_usable_channel_otherwise(self, rig):
        agent, _networkd, holder, _clock, _tmp = rig
        holder[0] = FakeWpa(WIFI[1])
        holder[0].station_freq = "5240"
        assert agent.start("dev", PASS, 0)["success"]
        assert agent.status()["frequency_mhz"] == 5180

    def test_automatic_with_nothing_usable_refuses_and_says_why(self, rig):
        agent, networkd, holder, _clock, tmp_path = rig
        holder[0] = FakeWpa(WIFI[1])
        holder[0].usable = set()
        result = agent.start("dev", PASS, 0)
        assert not result["success"] and "beacon" in result["message"]
        assert holder[0].selected is None and networkd.calls == [] and not ap_file(tmp_path).exists()

    def test_when_the_daemon_cannot_list_channels_the_request_is_taken_as_is(self, rig):
        agent, _networkd, holder, _clock, _tmp = rig
        holder[0] = FakeWpa(WIFI[1])
        holder[0].capability_fails = True
        assert agent.status()["channels"] == [], "unknown, not empty-by-choice"
        assert agent.start("dev", PASS, 48)["success"], "nothing to check against: the daemon decides"
        assert agent.status()["frequency_mhz"] == 5240

    def test_status_reports_the_usable_channels(self, rig):
        agent, *_ = rig
        assert agent.status()["channels"] == [36, 40, 44]

    def test_networkd_is_asked_to_re_match_the_link_once_the_radio_is_an_access_point(self, rig):
        # networkd does not re-match a link whose type flips to AP: without
        # a reconfigure the AP address never arrives.
        agent, networkd, _holder, _clock, _tmp = rig
        assert agent.start("dev", PASS, 0)["success"]
        assert networkd.calls == ["reload", ("reconfigure", WIFI[0])]

    def test_a_reconfigure_that_fails_after_the_radio_switched_rolls_back(self, rig):
        agent, networkd, holder, _clock, tmp_path = rig
        networkd.fail_reconfigure_link = True
        result = agent.start("dev", PASS, 0)
        assert not result["success"] and "ReconfigureLink" in result["message"]
        assert "station mode" in result["message"]
        assert holder[0].reconfigured == 1 and holder[0].selected is None
        assert not ap_file(tmp_path).exists()

    def test_an_overlapping_subnet_is_refused(self, rig):
        agent, networkd, holder, _clock, _tmp = rig
        networkd.wired_addresses = [addr("10.98.76.20", 24)]
        result = agent.start("dev", PASS, 36)
        assert not result["success"] and "overlaps the wired" in result["message"]
        assert holder[0] is None

    def test_a_link_whose_addresses_cannot_be_read_is_not_assumed_clear(self, rig):
        agent, networkd, holder, _clock, _tmp = rig
        networkd.undescribable.add(WIFI[0])
        result = agent.start("dev", PASS, 36)
        assert not result["success"] and "cannot be read" in result["message"]
        assert holder[0] is None, "refused before the radio was touched"

    def test_a_start_waits_for_a_wifi_change_that_holds_the_radio(self, rig):
        agent, networkd, holder, _clock, _tmp = rig
        held, release = threading.Event(), threading.Event()

        def wifi_change():
            with agent.radio:
                held.set()
                release.wait(5.0)

        threading.Thread(target=wifi_change, daemon=True).start()
        assert held.wait(1.0)
        outcome = {}
        worker = threading.Thread(target=lambda: outcome.update(agent.start("dev", PASS, 36)))
        worker.start()
        worker.join(0.3)
        assert worker.is_alive() and holder[0] is None, "the radio is left alone until the change is done"
        assert not agent.active, "waiting for the radio is not a transition"
        release.set()
        worker.join(5.0)
        assert outcome["success"], outcome

    def test_a_second_start_is_refused_while_active(self, rig):
        agent, *_ = rig
        assert agent.start("dev", PASS, 36)["success"]
        assert "already active" in agent.start("dev", PASS, 36)["message"]

    def test_an_absent_carrier_state_is_not_ready(self, rig):
        agent, networkd, holder, _clock, _tmp = rig
        networkd.wired_carrier = ""
        result = agent.start("dev", PASS, 36)
        assert not result["success"] and "wired" in result["message"]
        assert not agent.status()["wired_ready"] and holder[0].selected is None

    def test_a_missing_country_refuses_to_start_before_touching_the_radio(self, rig):
        agent, networkd, holder, _clock, tmp_path = rig
        holder[0] = FakeWpa(WIFI[1])
        holder[0].country = ""
        result = agent.start("dev", PASS, 36)
        assert not result["success"] and "regulatory country" in result["message"]
        assert holder[0].commands == [] and holder[0].selected is None
        assert networkd.calls == [] and not ap_file(tmp_path).exists()
        assert all(n.get("mode") != "2" for n in holder[0].networks.values())

    def test_a_missing_run_directory_is_refused_before_any_change(self, rig, tmp_path):
        agent, networkd, holder, _clock, _tmp = rig
        agent._run_dir = str(tmp_path / "absent")
        result = agent.start("dev", PASS, 36)
        assert not result["success"] and "network file" in result["message"]
        assert holder[0] is None and networkd.calls == []

    def test_a_start_or_stop_during_a_transition_is_refused(self, rig):
        agent, networkd, holder, _clock, _tmp = rig
        holder[0] = FakeWpa(WIFI[1])
        gate = threading.Event()
        holder[0].block_select = gate
        outcome = {}
        worker = threading.Thread(target=lambda: outcome.update(agent.start("dev", PASS, 36)))
        worker.start()
        for _ in range(500):
            if holder[0].selected is not None:
                break
            threading.Event().wait(0.01)
        assert holder[0].selected is not None, "the start never reached the radio"
        assert "in progress" in agent.start("dev", PASS, 36)["message"]
        assert "in progress" in agent.stop()["message"]
        assert agent.active and not agent.status()["active"], "busy, but not yet an access point"
        gate.set()
        worker.join(5.0)
        assert outcome["success"], outcome
        assert agent.status()["active"]


class TestRollback:
    def _rolled_back(self, holder, networkd, tmp_path):
        wpa = holder[0]
        assert not ap_file(tmp_path).exists()
        assert all(n.get("mode") != "2" for n in wpa.networks.values())
        assert wpa.reconfigured >= 1
        assert "reload" in networkd.calls and ("reconfigure", WIFI[0]) in networkd.calls

    def test_the_radio_never_becoming_an_ap_rolls_everything_back(self, rig):
        agent, networkd, holder, clock, tmp_path = rig
        holder[0] = FakeWpa(WIFI[1])
        holder[0].never_ap = True
        ticking = iter(range(1, 100))
        agent._now = lambda: clock.t + next(ticking) * 2.0  # the wait times out
        result = agent.start("dev", PASS, 40)
        assert not result["success"] and "station mode" in result["message"]
        self._rolled_back(holder, networkd, tmp_path)
        assert not agent.status()["active"]

    def test_a_missing_address_rolls_back(self, rig):
        agent, networkd, holder, clock, tmp_path = rig
        networkd.no_ap_address = True
        ticking = iter(range(1, 100))
        agent._now = lambda: clock.t + next(ticking) * 2.0
        result = agent.start("dev", PASS, 36)
        assert not result["success"] and "address" in result["message"]
        self._rolled_back(holder, networkd, tmp_path)

    def test_a_failed_reload_rolls_back_before_any_wpa_change(self, rig):
        agent, networkd, holder, _clock, tmp_path = rig
        networkd.fail_reload = True
        result = agent.start("dev", PASS, 36)
        assert not result["success"]
        assert not ap_file(tmp_path).exists()
        assert holder[0] is not None and holder[0].selected is None

    def test_a_failed_add_network_rolls_back(self, rig):
        agent, networkd, holder, _clock, tmp_path = rig
        holder[0] = FakeWpa(WIFI[1])
        holder[0].fail_add = True
        assert not agent.start("dev", PASS, 36)["success"]
        self._rolled_back(holder, networkd, tmp_path)

    def test_a_missing_dhcp_server_rolls_back(self, rig):
        agent, networkd, holder, clock, tmp_path = rig
        networkd.no_dhcp_server = True
        ticking = iter(range(1, 100))
        agent._now = lambda: clock.t + next(ticking) * 2.0
        result = agent.start("dev", PASS, 36)
        assert not result["success"] and "DHCP server" in result["message"]
        self._rolled_back(holder, networkd, tmp_path)
        assert not agent.status()["active"]

    def test_a_failed_file_write_rolls_back(self, rig, monkeypatch):
        agent, networkd, holder, _clock, tmp_path = rig

        def refuse(path, data):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(mod, "atomic_write_bytes", refuse)
        result = agent.start("dev", PASS, 36)
        assert not result["success"] and "station mode" in result["message"]
        assert not ap_file(tmp_path).exists()
        assert holder[0] is not None and holder[0].selected is None
        assert holder[0].reconfigured == 1 and "reload" in networkd.calls

    def test_a_failed_networkd_reconfigure_during_rollback_is_logged_not_raised(self, rig, caplog):
        agent, networkd, holder, _clock, tmp_path = rig
        holder[0] = FakeWpa(WIFI[1])
        holder[0].fail_add = True
        networkd.fail_reconfigure_link = True
        with caplog.at_level(logging.WARNING):
            result = agent.start("dev", PASS, 36)
        assert not result["success"]
        assert "reconfigure during rollback failed" in caplog.text
        assert holder[0].reconfigured == 1, "the wpa rollback still ran"
        assert not ap_file(tmp_path).exists()

    def test_a_failed_wpa_reconfigure_during_rollback_is_logged(self, rig, caplog):
        agent, networkd, holder, _clock, tmp_path = rig
        assert agent.start("dev", PASS, 36)["success"]
        holder[0].fail_reconfigure = True
        with caplog.at_level(logging.ERROR):
            result = agent.stop()
        assert result["success"], "the agent has done all it can; the radio is reported as stopped"
        assert "RECONFIGURE during rollback failed" in caplog.text
        assert not ap_file(tmp_path).exists() and not agent.status()["active"]
        assert all(n.get("mode") != "2" for n in holder[0].networks.values()), "the block was removed first"

    def test_the_passphrase_is_never_logged(self, rig, caplog):
        agent, networkd, holder, _clock, _tmp = rig
        holder[0] = FakeWpa(WIFI[1])
        holder[0].fail_add = True
        with caplog.at_level(logging.DEBUG):
            agent.start("dev", PASS, 36)
            agent.status()
        assert PASS not in caplog.text
        assert PASS not in str(agent.status())


class TestStop:
    def test_stop_returns_to_station_mode_and_is_idempotent(self, rig):
        agent, networkd, holder, _clock, tmp_path = rig
        assert agent.start("dev", PASS, 44)["success"]
        assert agent.stop()["success"]
        wpa = holder[0]
        assert not ap_file(tmp_path).exists() and wpa.reconfigured == 1
        assert all(n.get("mode") != "2" for n in wpa.networks.values())
        st = agent.status()
        assert not st["active"] and st["ssid"] == "" and st["join_deadline_remaining_secs"] == 0
        assert agent.stop()["success"], "a second stop is a no-op, not an error"
        assert wpa.saved == 0

    def test_reconcile_removes_a_leftover_block_and_file(self, rig):
        agent, networkd, holder, _clock, tmp_path = rig
        holder[0] = FakeWpa(WIFI[1])
        holder[0].networks["7"] = {"ssid": "old", "mode": "2"}
        ap_file(tmp_path).write_text("[Match]\n")
        agent.reconcile()
        assert "7" not in holder[0].networks and not ap_file(tmp_path).exists()
        assert holder[0].reconfigured == 1


class TestStations:
    def test_stations_are_reported_by_leased_address_never_by_mac(self, rig):
        agent, networkd, holder, _clock, _tmp = rig
        assert agent.start("dev", PASS, 36)["success"]
        wpa = holder[0]
        wpa.stations = [STA_MAC, "02:00:5e:00:53:02"]
        networkd.leases = [lease(STA_MAC, "10.98.76.11", "pixel")]
        st = agent.status()
        assert st["stations"] == [
            {"address": "10.98.76.11", "hostname": "pixel", "signal": -50},
            {"address": "", "hostname": "", "signal": -50},
        ]
        assert "02:00:5e" not in str(st)

    def test_a_joined_station_cancels_the_join_deadline_for_good(self, rig):
        agent, networkd, holder, clock, _tmp = rig
        assert agent.start("dev", PASS, 36)["success"]
        clock.t += 120
        assert agent.status()["join_deadline_remaining_secs"] == 180
        agent.note_station()
        assert agent.status()["join_deadline_remaining_secs"] == 0
        agent._deadline_fired()  # the timer fires anyway: nothing happens
        assert agent.status()["active"]
        holder[0].stations = []
        agent._deadline_fired()
        assert agent.status()["active"], "a later disconnect does not stop the access point"

    def test_the_watcher_notices_a_station(self, rig):
        agent, networkd, holder, _clock, _tmp = rig
        assert agent.start("dev", PASS, 36)["success"]
        holder[0].stations = [STA_MAC]
        deadline = threading.Event()
        for _ in range(200):
            if agent.status()["join_deadline_remaining_secs"] == 0:
                deadline.set()
                break
            threading.Event().wait(0.01)
        assert deadline.is_set()

    def test_the_deadline_stops_an_access_point_nobody_joined(self, rig):
        agent, networkd, holder, clock, tmp_path = rig
        assert agent.start("dev", PASS, 36)["success"]
        clock.t += mod.JOIN_DEADLINE_S + 1
        agent._deadline_fired()
        st = agent.status()
        assert not st["active"] and "No remote camera joined" in st["message"]
        assert not ap_file(tmp_path).exists()

    def test_a_station_joining_as_the_deadline_fires_keeps_the_access_point(self, rig, monkeypatch):
        agent, networkd, holder, clock, tmp_path = rig
        assert agent.start("dev", PASS, 36)["success"]
        reconfigured_before = holder[0].reconfigured
        real_claim = agent._claim_stop

        def joins_first(only_if_unjoined=False):
            # The station associates in the window between the timer firing
            # and the stop being claimed: the claim must see it.
            agent.note_station()
            return real_claim(only_if_unjoined)

        monkeypatch.setattr(agent, "_claim_stop", joins_first)
        clock.t += mod.JOIN_DEADLINE_S + 1
        agent._deadline_fired()
        st = agent.status()
        assert st["active"] and st["join_deadline_remaining_secs"] == 0
        assert holder[0].reconfigured == reconfigured_before and ap_file(tmp_path).exists()
        assert not agent._transition, "the refused claim left nothing held"


class TestWifiConflict:
    def test_wifi_changes_are_refused_while_the_access_point_is_up(self, rig, monkeypatch):
        agent, *_ = rig
        assert agent.start("dev", PASS, 36)["success"]
        network = NetworkAgent(ap=agent)
        monkeypatch.setattr(NetworkAgent, "_discover", staticmethod(lambda: {"wired": WIRED, "wifi": WIFI}))
        for change in (lambda: network.connect_wifi("Plant", "secret12"),
                       lambda: network.forget_wifi("Plant"),
                       lambda: network.set_ip_config("wifi", "auto")):
            with pytest.raises(AccessPointActive, match="access point is active"):
                change()


def test_the_network_file_matches_only_an_access_point_link():
    text = mod._network_file("wlan9", "10.98.76.1/24").decode()
    assert text.startswith("[Match]\nName=wlan9\nWLANInterfaceType=ap\n")
    assert "PoolOffset=10" in text and "PoolSize=20" in text
