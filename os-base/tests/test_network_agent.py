"""Static IP configuration: validation, atomic write and rollback (review H2).

Runs the real ``NetworkAgent.set_ip_config`` against a temp networkd
directory, a fixed interface and recorded ``reload``/``reconfigure_link``
calls. No D-Bus, no networkd.
"""
import os

import pytest
from agent import network_agent as mod
from agent.network_agent import NetworkAgent

IFINDEX, IFACE = 2, "enP8p1s0"


class Networkd:
    """Records the reload/reconfigure sequence; either call can be scripted to fail."""

    def __init__(self):
        self.calls = []
        self.fail_reload = False
        self.fail_reconfigure = False

    def reload(self):
        self.calls.append("reload")
        if self.fail_reload:
            self.fail_reload = False  # fail once: the rollback's reload succeeds
            raise RuntimeError("networkd: Reload failed")

    def reconfigure_link(self, ifindex):
        self.calls.append(("reconfigure", ifindex))
        if self.fail_reconfigure:
            self.fail_reconfigure = False
            raise RuntimeError("networkd: ReconfigureLink failed")


@pytest.fixture
def rig(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "NETWORKD_DIR", str(tmp_path))
    monkeypatch.setattr(NetworkAgent, "_discover",
                        staticmethod(lambda: {"wired": (IFINDEX, IFACE)}))
    networkd = Networkd()
    monkeypatch.setattr(mod.networkd, "reload", networkd.reload)
    monkeypatch.setattr(mod.networkd, "reconfigure_link", networkd.reconfigure_link)
    path = tmp_path / f"10-conecsa-{IFACE}.network"
    return NetworkAgent(), networkd, path


def _static(agent, **overrides):
    kwargs = dict(interface="wired", method="static", address="192.168.10.20",
                  prefix=24, gateway="192.168.10.1", dns=["1.1.1.1", "8.8.8.8"])
    kwargs.update(overrides)
    return agent.set_ip_config(**kwargs)


class TestApply:
    def test_static_writes_the_managed_file_and_applies_it(self, rig):
        agent, networkd, path = rig
        result = _static(agent)
        assert result["success"] is True
        assert path.read_text() == (
            f"[Match]\nName={IFACE}\n\n[Network]\nDHCP=no\n"
            "Address=192.168.10.20/24\nGateway=192.168.10.1\n"
            "DNS=1.1.1.1\nDNS=8.8.8.8\n")
        assert oct(path.stat().st_mode & 0o777) == "0o644"
        assert networkd.calls == ["reload", ("reconfigure", IFINDEX)]
        assert [f for f in os.listdir(path.parent) if ".tmp-" in f] == []

    def test_auto_writes_a_dhcp_file(self, rig):
        agent, _, path = rig
        assert agent.set_ip_config("wired", "auto")["success"] is True
        assert "DHCP=yes" in path.read_text()

    def test_empty_dns_entries_are_skipped(self, rig):
        agent, _, path = rig
        assert _static(agent, dns=["", "9.9.9.9"])["success"] is True
        assert path.read_text().count("DNS=") == 1


class TestValidation:
    @pytest.mark.parametrize("field,value,fragment", [
        ("address", "192.168.10", "address"),
        ("address", "10.0.0.1\nDNS=6.6.6.6", "address"),
        ("address", "example.com", "address"),
        ("prefix", 0, "Address and prefix are required"),
        ("prefix", 33, "Prefix length"),
        ("prefix", "24\nGateway=6.6.6.6", "prefix"),
        ("gateway", "gw.local", "gateway"),
        ("gateway", "192.168.10.1\nDNS=6.6.6.6", "gateway"),
        ("dns", ["1.1.1.1", "not-an-ip"], "DNS"),
        ("dns", ["1.1.1.1\nAddress=6.6.6.6/8"], "DNS"),
    ])
    def test_bad_values_are_rejected_before_anything_is_written(
            self, rig, field, value, fragment):
        agent, networkd, path = rig
        result = _static(agent, **{field: value})
        assert result["success"] is False
        assert fragment.lower() in result["message"].lower()
        assert not path.exists()
        assert networkd.calls == []

    def test_unknown_method_is_rejected(self, rig):
        agent, networkd, path = rig
        assert agent.set_ip_config("wired", "manual")["success"] is False
        assert not path.exists()


class TestRollback:
    def test_a_failed_reload_restores_the_previous_file_and_reapplies_it(self, rig):
        agent, networkd, path = rig
        assert _static(agent)["success"] is True
        before = path.read_bytes()
        networkd.calls.clear()

        networkd.fail_reload = True
        result = _static(agent, address="192.168.10.99")
        assert result["success"] is False
        assert "restored" in result["message"]
        assert path.read_bytes() == before
        # Failed apply, then the rollback's own reload + reconfigure.
        assert networkd.calls == ["reload", "reload", ("reconfigure", IFINDEX)]

    def test_a_failed_reconfigure_restores_the_previous_file(self, rig):
        agent, networkd, path = rig
        assert agent.set_ip_config("wired", "auto")["success"] is True
        before = path.read_bytes()

        networkd.fail_reconfigure = True
        result = _static(agent)
        assert result["success"] is False
        assert path.read_bytes() == before
        assert networkd.calls[-2:] == ["reload", ("reconfigure", IFINDEX)]

    def test_a_failure_with_no_previous_file_removes_the_new_one(self, rig):
        agent, networkd, path = rig
        networkd.fail_reload = True
        assert _static(agent)["success"] is False
        assert not path.exists()
        assert [f for f in os.listdir(path.parent) if ".tmp-" in f] == []
