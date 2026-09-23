# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Wi-Fi credentials never reach the wpa_supplicant control grammar unescaped:
SSIDs go as hex, passphrases as escaped quoted strings, raw PSKs
bare, and anything the grammar cannot carry is refused before the socket."""
import pytest
from agent import network_agent as mod
from agent.network_agent import NetworkAgent
from agent.wpa import WpaError, encode_psk, encode_ssid


class TestEncodeSsid:
    @pytest.mark.parametrize("ssid", ['Caf" Wi-Fi', "back\\slash", "with space",
                                      "Café ☕", "plain"])
    def test_any_ssid_is_hex(self, ssid):
        encoded = encode_ssid(ssid)
        assert bytes.fromhex(encoded).decode("utf-8") == ssid
        assert '"' not in encoded and " " not in encoded

    def test_limits(self):
        with pytest.raises(WpaError, match="required"):
            encode_ssid("")
        assert encode_ssid("x" * 32)
        with pytest.raises(WpaError, match="32 bytes"):
            encode_ssid("x" * 33)
        with pytest.raises(WpaError, match="32 bytes"):
            encode_ssid("☕" * 11)  # 33 bytes of UTF-8


class TestEncodePsk:
    def test_a_plain_passphrase_is_quoted(self):
        assert encode_psk("correct horse") == '"correct horse"'

    def test_quotes_and_backslashes_are_escaped(self):
        assert encode_psk('pa"ss\\word') == '"pa\\"ss\\\\word"'

    def test_a_64_hex_key_goes_bare(self):
        key = "ab" * 32
        assert encode_psk(key) == key
        assert encode_psk(key.upper()) == key.upper()

    def test_64_non_hex_characters_are_a_passphrase_and_too_long(self):
        with pytest.raises(WpaError, match="8-63"):
            encode_psk("g" * 64)

    @pytest.mark.parametrize("bad", ["short", "x" * 7])
    def test_short_passphrases_are_refused(self, bad):
        with pytest.raises(WpaError, match="8-63"):
            encode_psk(bad)

    @pytest.mark.parametrize("bad", ["pass\nword", "pass\x00word", "senha-café"])
    def test_control_and_non_ascii_bytes_are_refused(self, bad):
        with pytest.raises(WpaError, match="printable ASCII"):
            encode_psk(bad)


class FakeWpaCtrl:
    """Records SET_NETWORK arguments; association completes immediately."""

    instances = []

    def __init__(self, iface):
        self.iface = iface
        self.set_calls = []
        self.saved = False
        self._ssid = ""
        FakeWpaCtrl.instances.append(self)

    def find_network_id(self, ssid):
        return None

    def add_network(self):
        return "0"

    def set_network(self, net_id, key, value):
        self.set_calls.append((net_id, key, value))
        if key == "ssid":
            self._ssid = bytes.fromhex(value).decode("utf-8")

    def enable_network(self, net_id):
        pass

    def select_network(self, net_id):
        pass

    def status(self):
        return {"wpa_state": "COMPLETED", "ssid": self._ssid}

    def list_networks(self):
        return [{"id": "0", "ssid": self._ssid, "flags": ""}]

    def get_network_int(self, net_id, key, default=0):
        return default

    def enable_all(self):
        pass

    def save_config(self):
        self.saved = True

    def reconfigure(self):
        pass


@pytest.fixture
def agent(monkeypatch):
    FakeWpaCtrl.instances.clear()
    monkeypatch.setattr(mod, "WpaCtrl", FakeWpaCtrl)
    monkeypatch.setattr(NetworkAgent, "_wifi_iface", classmethod(lambda cls: "wlP1p1s0"))
    monkeypatch.setattr(mod, "_POLL_INTERVAL_S", 0.0)
    return NetworkAgent()


class TestConnectWifi:
    def test_sends_the_hex_ssid_and_the_escaped_passphrase(self, agent):
        result = agent.connect_wifi('Caf" Wi-Fi', 'pa"ss\\word')
        assert result["success"] is True
        wpa = FakeWpaCtrl.instances[0]
        assert wpa.set_calls[:2] == [
            ("0", "ssid", 'Caf" Wi-Fi'.encode().hex()),
            ("0", "psk", '"pa\\"ss\\\\word"'),
        ]
        assert wpa.saved is True

    def test_a_raw_psk_goes_bare(self, agent):
        key = "0f" * 32
        assert agent.connect_wifi("plain", key)["success"] is True
        assert ("0", "psk", key) in FakeWpaCtrl.instances[0].set_calls

    @pytest.mark.parametrize("ssid,password,fragment", [
        ("", "longenough", "SSID is required"),
        ("x" * 33, "longenough", "32 bytes"),
        ("net", "short", "8-63"),
        ("net", "bad\nline", "printable"),
    ])
    def test_invalid_credentials_never_reach_the_supplicant(self, agent, ssid, password,
                                                            fragment):
        result = agent.connect_wifi(ssid, password)
        assert result["success"] is False
        assert fragment in result["message"]
        assert FakeWpaCtrl.instances == []
