"""Node-RED editor tokens (review L2): minting, verification and the route."""
import pytest
from flask import Flask
from gateway import flow_token
from gateway.config import settings
from gateway.controllers import api_bp
from gateway.flow_token import FlowTokenError, mint, verify

SECRET = "test-secret"


class TestRoundTrip:
    def test_mint_and_verify(self):
        token = mint("ana", "admin", secret=SECRET, ttl_sec=60, now=1000.0)
        assert token.startswith("v1.")
        assert verify(token, secret=SECRET, now=1010.0) == {
            "sub": "ana", "role": "admin", "exp": 1060}

    def test_expiry(self):
        token = mint("ana", "admin", secret=SECRET, ttl_sec=60, now=1000.0)
        with pytest.raises(FlowTokenError, match="expired"):
            verify(token, secret=SECRET, now=1060.0)

    def test_another_secret_or_a_tampered_token_is_refused(self):
        token = mint("ana", "admin", secret=SECRET, ttl_sec=60, now=1000.0)
        with pytest.raises(FlowTokenError, match="signature"):
            verify(token, secret="other", now=1010.0)
        head, payload, sig = token.split(".")
        forged = mint("ana", "owner", secret=SECRET, ttl_sec=60, now=1000.0).split(".")[1]
        with pytest.raises(FlowTokenError, match="signature"):
            verify(f"{head}.{forged}.{sig}", secret=SECRET, now=1010.0)

    @pytest.mark.parametrize("bad", ["", "v1.x", "v2.a.b", "v1.a.b.c"])
    def test_malformed(self, bad):
        with pytest.raises(FlowTokenError):
            verify(bad, secret=SECRET)

    def test_no_secret_refuses_to_mint(self):
        with pytest.raises(FlowTokenError, match="secret"):
            mint("ana", "admin", secret="")

    def test_matches_the_node_verifier_format(self):
        # flow/admin-token.js signs "v1.<payload>" with HMAC-SHA256 and reads a
        # compact key-sorted JSON payload; pin the exact bytes.
        import base64
        import hashlib
        import hmac
        token = mint("ana", "admin", secret=SECRET, ttl_sec=60, now=1000.0)
        head, payload, sig = token.split(".")
        assert base64.urlsafe_b64decode(payload + "==") == b'{"exp":1060,"role":"admin","sub":"ana"}'
        expected = hmac.new(SECRET.encode(), f"{head}.{payload}".encode(), hashlib.sha256).digest()
        assert base64.urlsafe_b64decode(sig + "=" * (-len(sig) % 4)) == expected


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(api_bp)
    return app.test_client()


class TestRoute:
    def test_mints_a_local_admin_token_off_the_hub_path(self, client, monkeypatch):
        monkeypatch.setattr(settings, "FLOW_ADMIN_TOKEN_SECRET", SECRET)
        resp = client.post("/api/v1/flow/token")
        assert resp.status_code == 200
        body = resp.get_json()
        payload = verify(body["token"], secret=SECRET)
        assert (payload["sub"], payload["role"]) == ("local", "admin")
        assert body["expires_in"] == int(settings.FLOW_ADMIN_TOKEN_TTL_SEC)

    def test_carries_the_identity_the_hub_vouched_for(self, client, monkeypatch):
        from gateway.controllers import flow as flow_controller
        monkeypatch.setattr(settings, "FLOW_ADMIN_TOKEN_SECRET", SECRET)
        monkeypatch.setattr(flow_controller, "_hub_verified", lambda: True)
        resp = client.post("/api/v1/flow/token", headers={
            "X-Conecsa-User": "ana", "X-Conecsa-Role": "user"})
        payload = verify(resp.get_json()["token"], secret=SECRET)
        assert (payload["sub"], payload["role"]) == ("ana", "user")

    def test_503_without_a_secret(self, client, monkeypatch):
        monkeypatch.setattr(settings, "FLOW_ADMIN_TOKEN_SECRET", "")
        assert client.post("/api/v1/flow/token").status_code == 503
        assert flow_token.VERSION == "v1"
