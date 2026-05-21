"""Tests for CORS validator."""

from __future__ import annotations

import pytest

from src.security.cors_validator import CORSPolicy, CORSValidator


class TestCORSValidator:
    def test_default_allows_all(self):
        cv = CORSValidator()
        assert cv.check_origin("http://evil.com") is True

    def test_restrictive_policy(self):
        cv = CORSValidator()
        cv.add_policy("/api", CORSPolicy(allowed_origins=["https://app.example.com"]))
        assert cv.check_origin("https://app.example.com", "/api") is True
        assert cv.check_origin("http://evil.com", "/api") is False

    def test_method_check(self):
        cv = CORSValidator()
        cv.add_policy("/api", CORSPolicy(allowed_methods=["GET"]))
        assert cv.check_method("GET", "/api") is True
        assert cv.check_method("POST", "/api") is False

    def test_to_headers(self):
        cv = CORSValidator()
        cv.add_policy(
            "/api",
            CORSPolicy(
                allowed_origins=["http://localhost"],
                allowed_methods=["GET"],
                allow_credentials=True,
            ),
        )
        headers = cv.to_headers("/api")
        assert "Access-Control-Allow-Origin" in headers
        assert "Access-Control-Allow-Credentials" in headers
        assert headers["Access-Control-Allow-Origin"] == "http://localhost"

    def test_wildcard_with_credentials_raises(self):
        with pytest.raises(ValueError):
            CORSPolicy(allowed_origins=["*"], allow_credentials=True)

    def test_empty_origin_with_credentials_raises(self):
        with pytest.raises(ValueError):
            CORSPolicy(allowed_origins=[], allow_credentials=True)

    def test_credentials_with_specific_origin_allowed(self):
        policy = CORSPolicy(allowed_origins=["https://example.com"], allow_credentials=True)
        assert policy.allow_credentials is True
        assert "https://example.com" in policy.allowed_origins

    def test_default_methods(self):
        policy = CORSPolicy()
        assert "GET" in policy.allowed_methods
        assert "POST" in policy.allowed_methods

    def test_default_headers(self):
        policy = CORSPolicy()
        assert "Content-Type" in policy.allowed_headers
