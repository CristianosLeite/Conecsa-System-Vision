"""Unit tests for device enrollment identity + crypto helpers."""
import os
from datetime import datetime, timezone

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from gateway import enroll


@pytest.fixture
def cert_dir(tmp_path, monkeypatch):
    """Point the module's cert paths at a temp dir for isolated key generation."""
    d = tmp_path / "certs"
    monkeypatch.setattr(enroll, "CERT_DIR", str(d))
    monkeypatch.setattr(enroll, "KEY_PATH", str(d / "device.key"))
    monkeypatch.setattr(enroll, "CERT_PATH", str(d / "device.crt"))
    monkeypatch.setattr(enroll, "CA_PATH", str(d / "ca.crt"))
    return d


class TestDeviceId:
    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("DEVICE_ID", "cam-42")
        assert enroll.device_id() == "cam-42"

    def test_reads_host_hostname_file(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DEVICE_ID", raising=False)
        hostfile = tmp_path / "host_hostname"
        hostfile.write_text("jetson-01\n")
        monkeypatch.setattr(enroll, "HOST_HOSTNAME_PATH", str(hostfile))
        assert enroll.device_id() == "jetson-01"

    def test_falls_back_to_container_hostname(self, monkeypatch):
        monkeypatch.delenv("DEVICE_ID", raising=False)
        monkeypatch.setattr(enroll, "HOST_HOSTNAME_PATH", "/nonexistent/path")
        # Falls back to socket.gethostname() — just assert it's a non-empty string.
        assert isinstance(enroll.device_id(), str)
        assert enroll.device_id() != ""


class TestLogicalName:
    def test_format(self, monkeypatch):
        monkeypatch.setenv("DEVICE_ID", "cam-42")
        assert enroll.logical_name() == "device-cam-42.conecsa.local"


class TestIsEnrolled:
    def test_false_when_certs_absent(self, cert_dir):
        assert enroll.is_enrolled() is False

    def test_true_when_both_present(self, cert_dir):
        cert_dir.mkdir(parents=True, exist_ok=True)
        (cert_dir / "device.crt").write_text("x")
        (cert_dir / "ca.crt").write_text("x")
        assert enroll.is_enrolled() is True


class TestPublicFingerprint:
    def test_generates_key_and_stable_hex(self, cert_dir):
        fp1 = enroll.public_fingerprint()
        assert len(fp1) == 64  # SHA-256 hex
        assert all(c in "0123456789abcdef" for c in fp1)
        # Key is persisted with 0600 perms, and the fingerprint is stable.
        assert os.path.exists(str(cert_dir / "device.key"))
        assert oct(os.stat(str(cert_dir / "device.key")).st_mode & 0o777) == "0o600"
        assert enroll.public_fingerprint() == fp1


class TestBuildCsr:
    def test_csr_has_logical_common_name(self, cert_dir, monkeypatch):
        monkeypatch.setenv("DEVICE_ID", "cam-42")
        pem = enroll._build_csr()
        csr = x509.load_pem_x509_csr(pem)
        cn = csr.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
        assert cn == "device-cam-42.conecsa.local"
        san = csr.extensions.get_extension_for_class(x509.SubjectAlternativeName)
        assert san.value.get_values_for_type(x509.DNSName) == [
            "device-cam-42.conecsa.local"
        ]

    def test_csr_signature_is_valid(self, cert_dir):
        csr = x509.load_pem_x509_csr(enroll._build_csr())
        assert csr.is_signature_valid


class TestTokenRequired:
    def test_no_token_by_default(self, monkeypatch):
        monkeypatch.delenv("DEVICE_PAIR_TOKEN", raising=False)
        assert enroll.token_required() is False

    def test_token_required_when_set(self, monkeypatch):
        monkeypatch.setenv("DEVICE_PAIR_TOKEN", "secret")
        assert enroll.token_required() is True


class TestCompleteAdoptsTheHubClock:
    """Pairing is the one channel that works while the device clock is wrong
    (nothing is validated on it), so `/enroll/complete` must take the hub's time
    BEFORE the certificates land — installing them flips nginx into enforcing
    mode, and from then on a clock older than the CA's not_before rejects every
    hub call."""

    @pytest.fixture
    def client(self, cert_dir, monkeypatch):
        from flask import Flask
        monkeypatch.setenv("DEVICE_ID", "cam-42")
        monkeypatch.delenv("DEVICE_PAIR_TOKEN", raising=False)
        app = Flask(__name__)
        app.register_blueprint(enroll.enroll_bp)
        return app.test_client()

    @staticmethod
    def _hub_signed_cert() -> tuple:
        """(device_cert, ca_cert) PEMs for the device's own CSR, as the hub sends."""
        csr = x509.load_pem_x509_csr(enroll._build_csr())
        ca_key = ec.generate_private_key(ec.SECP256R1())
        ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test-hub")])
        span = {
            "not_valid_before": datetime(2020, 1, 1, tzinfo=timezone.utc),
            "not_valid_after": datetime(2050, 1, 1, tzinfo=timezone.utc),
        }
        ca_cert = (
            x509.CertificateBuilder()
            .subject_name(ca_name).issuer_name(ca_name)
            .public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(span["not_valid_before"])
            .not_valid_after(span["not_valid_after"])
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .sign(ca_key, hashes.SHA256())
        )
        leaf = (
            x509.CertificateBuilder()
            .subject_name(csr.subject).issuer_name(ca_name)
            .public_key(csr.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(span["not_valid_before"])
            .not_valid_after(span["not_valid_after"])
            .add_extension(
                csr.extensions.get_extension_for_class(x509.SubjectAlternativeName).value,
                critical=False)
            .sign(ca_key, hashes.SHA256())
        )
        pem = serialization.Encoding.PEM
        return (leaf.public_bytes(pem).decode(), ca_cert.public_bytes(pem).decode())

    def _post(self, client, monkeypatch, apply_result=True):
        """Run a full pairing, recording the order of the two side effects."""
        device_cert, ca_cert = self._hub_signed_cert()
        order = []

        def fake_apply(raw, source, force=False):
            order.append(("clock", raw, source, force))
            return apply_result

        real_install = enroll._install_certs
        monkeypatch.setattr(enroll.clock, "apply_hub_time", fake_apply)
        monkeypatch.setattr(enroll, "_install_certs",
                            lambda c, a: (order.append(("install",)), real_install(c, a)))
        resp = client.post("/enroll/complete", json={
            "device_cert": device_cert,
            "ca_cert": ca_cert,
            "hub_time": "2026-08-03T10:00:00.000Z",
        })
        return resp, order

    def test_clock_is_set_before_the_certificates_are_installed(
            self, client, monkeypatch):
        resp, order = self._post(client, monkeypatch)
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "enrolled"
        assert [step[0] for step in order] == ["clock", "install"]
        # Forced: the bootstrap step bypasses the poll-path threshold and rate
        # limit, so pairing always leaves a clock floor behind.
        assert order[0][1:] == ("2026-08-03T10:00:00.000Z", "pairing", True)
        assert enroll.is_enrolled() is True

    def test_pairing_still_completes_when_the_clock_cannot_be_set(
            self, client, monkeypatch):
        resp, order = self._post(client, monkeypatch, apply_result=False)
        assert resp.status_code == 200
        assert [step[0] for step in order] == ["clock", "install"]
        assert enroll.is_enrolled() is True

    def test_an_older_hub_omitting_hub_time_still_pairs(self, client, monkeypatch):
        device_cert, ca_cert = self._hub_signed_cert()
        seen = []
        monkeypatch.setattr(enroll.clock, "apply_hub_time",
                            lambda raw, source, force=False: seen.append(raw))
        resp = client.post("/enroll/complete", json={
            "device_cert": device_cert, "ca_cert": ca_cert})
        assert resp.status_code == 200
        assert seen == [None]


class TestResetAuthorized:
    """Unpair authorization must not be spoofable: the mTLS header only counts
    when the request was relayed by the nginx terminator itself."""

    TERMINATOR_IP = "10.66.0.9"

    def _ctx(self, remote_addr, headers=None):
        from flask import Flask
        return Flask(__name__).test_request_context(
            "/enroll/reset", method="POST", headers=headers or {},
            environ_base={"REMOTE_ADDR": remote_addr})

    @pytest.fixture(autouse=True)
    def _pin_terminator(self, monkeypatch):
        from gateway import helpers
        monkeypatch.delenv("DEVICE_PAIR_TOKEN", raising=False)
        monkeypatch.setattr(helpers, "_resolve_proxy_ips",
                            lambda: frozenset({self.TERMINATOR_IP}))
        monkeypatch.setattr(helpers, "_proxy_cache",
                            {"ips": frozenset(), "at": float("-inf")})

    def test_header_via_the_terminator_authorizes(self):
        with self._ctx(self.TERMINATOR_IP,
                       {"X-Conecsa-Client-Verify": "SUCCESS"}):
            assert enroll._reset_authorized() is True

    def test_spoofed_header_from_another_container_is_rejected(self):
        with self._ctx("172.20.0.5", {"X-Conecsa-Client-Verify": "SUCCESS"}):
            assert enroll._reset_authorized() is False

    def test_pairing_token_still_authorizes_without_mtls(self, monkeypatch):
        monkeypatch.setenv("DEVICE_PAIR_TOKEN", "secret")
        from flask import Flask
        with Flask(__name__).test_request_context(
                "/enroll/reset", method="POST", json={"token": "secret"},
                environ_base={"REMOTE_ADDR": "172.20.0.5"}):
            assert enroll._reset_authorized() is True
