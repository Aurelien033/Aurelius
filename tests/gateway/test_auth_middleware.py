"""Integration tests for aurelius_api.py require_api_key middleware."""

from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


def _load_app(api_key: str = "test-secret-key"):
    """Import app with a given AURELIUS_API_KEY env value."""
    with patch.dict("os.environ", {"AURELIUS_API_KEY": api_key}, clear=False):
        import gateway.aurelius_api as _mod

        importlib.reload(_mod)
        return _mod.app


@pytest.fixture()
def client():
    app = _load_app("test-secret-key")
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def empty_key_client():
    app = _load_app("")
    return TestClient(app, raise_server_exceptions=False)


class TestPublicEndpoints:
    """Unauthenticated requests to exempt paths must succeed."""

    def test_health_no_key(self, client):
        r = client.get("/health")
        assert r.status_code != 401

    def test_health_ready_no_key(self, client):
        r = client.get("/health/ready")
        assert r.status_code != 401

    def test_root_no_key(self, client):
        r = client.get("/")
        assert r.status_code != 401


class TestProtectedEndpoints:
    """Non-exempt endpoints require a valid API key."""

    def test_missing_key_returns_401(self, client):
        r = client.get("/v1/models")
        assert r.status_code == 401

    def test_wrong_key_returns_401(self, client):
        r = client.get("/v1/models", headers={"X-API-Key": "wrong"})
        assert r.status_code == 401

    def test_correct_key_passes_auth(self, client):
        r = client.get("/v1/models", headers={"X-API-Key": "test-secret-key"})
        # Not 401 or 503 — auth layer passed (may still 404 if route absent)
        assert r.status_code not in (401, 503)

    def test_empty_key_env_returns_503(self, empty_key_client):
        """If AURELIUS_API_KEY is unset, the server should refuse all protected requests."""
        r = empty_key_client.get("/v1/models")
        assert r.status_code == 503
