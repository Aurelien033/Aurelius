"""Unit tests for gateway.aurelius_api.validate_config()."""

from __future__ import annotations

import pytest

from gateway.aurelius_api import validate_config


def _run(env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    validate_config()


class TestValidateConfig:
    def test_defaults_pass(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AURELIUS_MODEL_PATH", "checkpoints/aurelius_1.3b")
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("CORS_ORIGINS", raising=False)
        monkeypatch.delenv("CORS_ORIGIN", raising=False)
        validate_config()

    def test_hf_model_id_accepted(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AURELIUS_MODEL_PATH", "org/my-model")
        monkeypatch.delenv("DATABASE_URL", raising=False)
        validate_config()

    def test_invalid_port_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AURELIUS_PORT", "99999")
        with pytest.raises(RuntimeError, match="AURELIUS_PORT"):
            validate_config()

    def test_non_integer_port_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AURELIUS_PORT", "not-a-number")
        with pytest.raises(RuntimeError, match="AURELIUS_PORT"):
            validate_config()

    def test_invalid_database_url_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DATABASE_URL", "not-a-url")
        with pytest.raises(RuntimeError, match="DATABASE_URL"):
            validate_config()

    def test_valid_database_url_passes(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/db")
        monkeypatch.setenv("AURELIUS_MODEL_PATH", "org/model")
        validate_config()

    def test_invalid_cors_origin_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CORS_ORIGIN", "not-a-url")
        with pytest.raises(RuntimeError, match="CORS_ORIGINS"):
            validate_config()

    def test_valid_cors_origin_passes(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CORS_ORIGIN", "http://localhost:5173")
        monkeypatch.setenv("AURELIUS_MODEL_PATH", "org/model")
        monkeypatch.delenv("DATABASE_URL", raising=False)
        validate_config()

    def test_tensor_parallel_size_zero_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("TENSOR_PARALLEL_SIZE", "0")
        with pytest.raises(RuntimeError, match="TENSOR_PARALLEL_SIZE"):
            validate_config()

    def test_tensor_parallel_size_non_integer_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("TENSOR_PARALLEL_SIZE", "two")
        with pytest.raises(RuntimeError, match="TENSOR_PARALLEL_SIZE"):
            validate_config()

    def test_bad_model_path_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AURELIUS_MODEL_PATH", "/no/such/path/here/nope")
        with pytest.raises(RuntimeError, match="AURELIUS_MODEL_PATH"):
            validate_config()
