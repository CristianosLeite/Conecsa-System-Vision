# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""The control-socket client's own guards, run against a scripted daemon:
no access-point block ever reaches the saved config, ``enable_all`` never
enables one, and the regulatory country is read as text."""
import pytest
from agent.wpa import WpaCtrl, WpaError


class ScriptedWpa(WpaCtrl):
    """A daemon with one station block and one leftover access-point block."""

    def __init__(self, country="BR"):
        super().__init__("wlan0")
        self.networks = {"0": {"ssid": "Plant", "mode": "0"}, "1": {"ssid": "conecsa-000001", "mode": "2"}}
        self.country = country
        self.sent = []

    def _cmd(self, command, timeout=5.0):
        self.sent.append(command)
        verb, *args = command.split()
        if verb == "LIST_NETWORKS":
            rows = [f"{i}\t{n['ssid']}\tany\t" for i, n in self.networks.items()]
            return "network id / ssid / bssid / flags\n" + "\n".join(rows)
        if verb == "GET_NETWORK":
            value = self.networks.get(args[0], {}).get(args[1])
            return "FAIL" if value is None else value
        if verb == "REMOVE_NETWORK":
            return "OK" if self.networks.pop(args[0], None) is not None else "FAIL"
        if verb in ("SAVE_CONFIG", "ENABLE_NETWORK"):
            return "OK"
        if verb == "GET" and args == ["country"]:
            return self.country or "FAIL"
        if verb == "GET_CAPABILITY" and args == ["freq"]:
            return self.capability
        return "FAIL"

    # `GET_CAPABILITY freq` as wpa_supplicant 2.10 prints it; the flags are
    # the driver's live regulatory state and change over time.
    capability = (
        "Mode[A] Channels:\n"
        " 36 = 5180 MHz (NO_IR)\n"
        " 40 = 5200 MHz (NO_IR)\n"
        " 44 = 5220 MHz\n"
        " 48 = 5240 MHz\n"
        " 52 = 5260 MHz (NO_IR) (RADAR)\n"
        " 100 = 5500 MHz (DISABLED)\n"
        "Mode[G] Channels:\n"
        " 1 = 2412 MHz\n"
    )


def test_save_config_drops_access_point_blocks_before_saving():
    wpa = ScriptedWpa()
    wpa.save_config()
    assert "1" not in wpa.networks and "0" in wpa.networks
    assert wpa.sent.index("REMOVE_NETWORK 1") < wpa.sent.index("SAVE_CONFIG")


def test_a_block_that_cannot_be_dropped_stops_the_save():
    wpa = ScriptedWpa()
    wpa.networks["1"]["ssid"] = "stuck"
    original = wpa._cmd

    def refuse_removal(command, timeout=5.0):
        if command.startswith("REMOVE_NETWORK"):
            wpa.sent.append(command)
            return "FAIL"
        return original(command, timeout)

    wpa._cmd = refuse_removal
    with pytest.raises(WpaError, match="access point block"):
        wpa.save_config()
    assert "SAVE_CONFIG" not in wpa.sent


def test_usable_frequencies_keeps_only_channels_the_radio_may_initiate_on():
    wpa = ScriptedWpa()
    assert wpa.usable_frequencies() == {5220, 5240, 2412}


def test_usable_frequencies_raises_when_the_daemon_cannot_say():
    wpa = ScriptedWpa()
    wpa.capability = "FAIL"
    with pytest.raises(WpaError, match="GET_CAPABILITY"):
        wpa.usable_frequencies()


def test_enable_all_skips_access_point_blocks():
    wpa = ScriptedWpa()
    wpa.enable_all()
    enabled = [c for c in wpa.sent if c.startswith("ENABLE_NETWORK")]
    assert enabled == ["ENABLE_NETWORK 0"]


@pytest.mark.parametrize("reply,expected", [("BR", "BR"), ("", ""), (None, "")])
def test_get_country_reads_the_regulatory_domain_or_nothing(reply, expected):
    wpa = ScriptedWpa(country=reply)
    assert wpa.get_country() == expected
