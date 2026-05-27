"""
Evidence test for T3: Gateway fail-closed perimeter (C-03, C-04, M-06)

Proves:
1. Rate limiter is fail-closed: denies requests when uninitialized (C-04)
2. /metrics endpoint requires authentication (C-03, M-06)
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import os


@pytest.fixture
def client():
    """Create test client with fresh app state."""
    # Configure allowed hosts to include testserver
    with patch.dict(os.environ, {'AURELIUS_ALLOWED_HOSTS': 'localhost,127.0.0.1,testserver'}):
        # Import fresh to avoid state pollution
        import importlib
        import gateway.aurelius_api as api_module
        importlib.reload(api_module)
        # Initialize rate limiter to avoid 503 on all requests
        api_module._rate_limiter = lambda ip: True
        yield TestClient(api_module.app)


class TestRateLimiterFailClosed:
    """C-04: Rate limiter must deny requests when uninitialized."""

    def test_uninitialized_rate_limiter_denies_request(self, client):
        """When _rate_limiter is None, requests must be rejected (fail-closed)."""
        import gateway.aurelius_api as api_module
        
        # Ensure rate limiter is uninitialized
        with patch.object(api_module, '_rate_limiter', None):
            response = client.get("/health")
            
            # Should be 503 (service unavailable) or 429, not 200
            assert response.status_code in [429, 503], (
                f"Rate limiter fail-open: uninitialized limiter allowed request "
                f"(status {response.status_code}). Must deny when not initialized."
            )

    def test_rate_limiter_allows_when_initialized(self, client):
        """When rate limiter is initialized, requests should pass normally."""
        import gateway.aurelius_api as api_module
        
        # Mock initialized rate limiter that allows requests
        mock_limiter = MagicMock(return_value=True)
        with patch.object(api_module, '_rate_limiter', mock_limiter):
            response = client.get("/health")
            # Should succeed (200) since limiter allows
            assert response.status_code == 200


class TestMetricsAuthentication:
    """C-03, M-06: /metrics must require authentication."""

    def test_metrics_requires_auth(self, client):
        """GET /metrics without auth header must return 401 or 403."""
        # Set metrics API key so endpoint is configured but auth is required
        with patch.dict(os.environ, {'AURELIUS_METRICS_API_KEY': 'test-metrics-key'}):
            response = client.get("/metrics")
        
        # Should be 401 (unauthorized) or 403 (forbidden), not 200
        assert response.status_code in [401, 403], (
            f"Unauthenticated /metrics access allowed (status {response.status_code}). "
            f"Must require authentication to prevent information disclosure."
        )

    def test_metrics_with_valid_auth(self, client):
        """GET /metrics with valid auth should return metrics."""
        # This test will pass once we add auth support
        # For now, document the expected behavior
        pytest.skip("Auth implementation pending - will be added with fix")
