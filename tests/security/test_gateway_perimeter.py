"""Gateway and tools perimeter checks (H-13 extension)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def gateway_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("AURELIUS_METRICS_API_KEY", "metrics-secret")
    import gateway.aurelius_api as api

    api._rate_limiter = lambda _ip: True
    return TestClient(api.app, base_url="http://localhost")


class TestGatewayMetricsAuth:
    def test_metrics_rejects_missing_key_when_configured(self, gateway_client: TestClient):
        res = gateway_client.get("/metrics")
        assert res.status_code == 401

    def test_metrics_accepts_valid_key(self, gateway_client: TestClient):
        res = gateway_client.get("/metrics", headers={"X-API-Key": "metrics-secret"})
        assert res.status_code == 200
        assert "text/plain" in res.headers.get("content-type", "")

    def test_metrics_unconfigured_returns_503(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("AURELIUS_METRICS_API_KEY", raising=False)
        import gateway.aurelius_api as api

        api._rate_limiter = lambda _ip: True
        client = TestClient(api.app, base_url="http://localhost")
        res = client.get("/metrics")
        assert res.status_code == 503


class TestToolsEgressGuard:
    def test_http_client_blocks_metadata_ip(self):
        from tools.http_client import _is_safe_url

        safe, reason = _is_safe_url("http://169.254.169.254/latest/meta-data/")
        assert not safe
        assert reason

    def test_web_tool_exposes_fetch(self):
        from tools.web_tool import WebTool

        tool = WebTool()
        assert callable(getattr(tool, "fetch", None))
