"""Tests for encryptor — hardened.

Covers: missing-key, invalid-key, empty/multibyte roundtrip, explicit-key-over-env,
empty-plaintext, and module-level singleton load contract.
"""

from __future__ import annotations

import os

import pytest

# Set a valid Fernet key before importing the encryptor module.
# AURELIUS_ENCRYPTION_KEY is required by SimpleEncryptor.__post_init__.
os.environ.setdefault(
    "AURELIUS_ENCRYPTION_KEY",
    "eraA96t0Jt605u3a6it1Z58dZXraqjM22HCNv4RYb7U=",
)

from src.security.encryptor import SimpleEncryptor, _get_default_encryptor  # noqa: E402

# Detect whether cryptography is available.
_CRYPTOGRAPHY_AVAILABLE = True
try:
    _get_default_encryptor()
except ImportError:
    _CRYPTOGRAPHY_AVAILABLE = False

_VALID_KEY = "eraA96t0Jt605u3a6it1Z58dZXraqjM22HCNv4RYb7U="
_INVALID_KEY = "not-a-valid-fernet-key!!!"


skip_no_crypto = pytest.mark.skipif(
    not _CRYPTOGRAPHY_AVAILABLE, reason="cryptography not installed"
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    import src.security.encryptor as _enc_mod

    _enc_mod.SIMPLE_ENCRYPTOR = None
    yield
    _enc_mod.SIMPLE_ENCRYPTOR = None


# ── Key validation ────────────────────────────────────────────────────────────


@skip_no_crypto
class TestKeyValidation:
    def test_explicit_valid_key_constructor(self):
        enc = SimpleEncryptor(key=_VALID_KEY.encode())
        assert enc.key == _VALID_KEY.encode()

    def test_missing_env_key_raises_runtime_error(self, monkeypatch):
        monkeypatch.delenv("AURELIUS_ENCRYPTION_KEY", raising=False)
        with pytest.raises(RuntimeError, match="AURELIUS_ENCRYPTION_KEY"):
            SimpleEncryptor()

    def test_invalid_key_bytes_raises(self):
        with pytest.raises(ValueError, match="not a valid Fernet key"):
            SimpleEncryptor(key=_INVALID_KEY.encode())

    def test_invalid_key_str_also_raises(self):
        with pytest.raises(ValueError, match="not a valid Fernet key"):
            SimpleEncryptor(key=_INVALID_KEY)

    def test_explicit_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("AURELIUS_ENCRYPTION_KEY", _INVALID_KEY)
        enc = SimpleEncryptor(key=_VALID_KEY.encode())  # valid key overrides poisoned env
        ct = enc.encrypt("override")
        assert enc.decrypt(ct) == "override"

    def test_explicit_key_str_overrides_env(self, monkeypatch):
        monkeypatch.setenv("AURELIUS_ENCRYPTION_KEY", _INVALID_KEY)
        enc = SimpleEncryptor(key=_VALID_KEY)  # str form also accepted
        ct = enc.encrypt("override-str")
        assert enc.decrypt(ct) == "override-str"

    def test_bytes_key_accepted(self):
        enc = SimpleEncryptor(key=_VALID_KEY.encode())
        ct = enc.encrypt("bytes-key")
        assert enc.decrypt(ct) == "bytes-key"


# ── Payload roundtrip ─────────────────────────────────────────────────────────


@skip_no_crypto
class TestRoundtrip:
    def test_encrypt_decrypt_roundtrip(self):
        enc = SimpleEncryptor()
        ct = enc.encrypt("hello world")
        pt = enc.decrypt(ct)
        assert pt == "hello world"

    def test_different_plaintexts_differ(self):
        enc = SimpleEncryptor()
        ct1 = enc.encrypt("abc")
        ct2 = enc.encrypt("xyz")
        assert ct1 != ct2

    def test_empty_plaintext_roundtrip(self):
        enc = SimpleEncryptor()
        ct = enc.encrypt("")
        assert ct != ""
        assert enc.decrypt(ct) == ""

    def test_same_plaintext_produces_different_ciphertext(self):
        """Fernet includes a random nonce — identical plaintext must produce
        different ciphertext on each call."""
        enc = SimpleEncryptor()
        ct1 = enc.encrypt("constant")
        ct2 = enc.encrypt("constant")
        assert ct1 != ct2

    def test_non_ascii_roundtrip(self):
        enc = SimpleEncryptor()
        payload = "日本語テスト 🎉 café résumé naïve 𝄞"
        ct = enc.encrypt(payload)
        assert enc.decrypt(ct) == payload

    def test_very_long_plaintext_roundtrip(self):
        enc = SimpleEncryptor()
        payload = "A" * 100_000
        ct = enc.encrypt(payload)
        assert enc.decrypt(ct) == payload

    def test_newline_and_null_roundtrip(self):
        enc = SimpleEncryptor()
        payload = "line1\nline2\r\n\x00binary-ish"
        ct = enc.encrypt(payload)
        assert enc.decrypt(ct) == payload


# ── Module-level singleton ────────────────────────────────────────────────────


@skip_no_crypto
class TestDefaultEncryptor:
    def test_get_default_encryptor_returns_simple_encryptor(self):
        enc = _get_default_encryptor()
        assert isinstance(enc, SimpleEncryptor)

    def test_default_encryptor_reuses_singleton(self):
        a = _get_default_encryptor()
        b = _get_default_encryptor()
        assert a is b

    def test_default_encryptor_can_encrypt_decrypt(self):
        enc = _get_default_encryptor()
        ct = enc.encrypt("singleton")
        assert enc.decrypt(ct) == "singleton"
