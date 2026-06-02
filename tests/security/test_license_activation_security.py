"""Regression tests for C2: license activation must not derive the API
key from the user-supplied license secret.

After C1 decommissioning, the only active license-activation handler
lives in the Python unified server (``gateway/aurelius_server.py``).
The legacy Node route at ``server_legacy/src/routes/license.ts`` is
now a stub that throws on import. These tests therefore focus on the
Python handler (the live surface) and the frontend storage path.

The C2 vulnerability, in both runtimes before remediation, accepted a
user-supplied ``license_key`` and used it (or a truncation/HMAC of it)
as the global ``api_key``. After C2:

* ``_handle_license_activate`` does NOT mutate ``runtime_config`` at
  all from the license path. The API key is operator-supplied via
  ``AURELIUS_API_KEY`` and rotated out of band.
* ``_handle_license_activate`` does NOT flip the auth-policy toggle
  (operator policy, not a license decision).
* ``_handle_license_activate`` rejects any input that attempts to
  influence the credential surface (extra fields like ``api_key``,
  ``require_auth``, ``role``, ``scopes``, ``admin``).
* The frontend does not write the user-supplied license key into
  ``localStorage`` as the API key; activation only records the
  public tier label and lets the login flow issue the real token.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PY_GATEWAY = REPO_ROOT / "gateway" / "aurelius_server.py"
FRONTEND_GATE = REPO_ROOT / "frontend" / "src" / "components" / "LicenseGate.tsx"


def _read(path: Path) -> str:
    assert path.exists(), f"required file missing: {path}"
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Python gateway (the only live license handler post-C1)
# ---------------------------------------------------------------------------


def test_python_gateway_license_activate_does_not_slice_key_as_api_key() -> None:
    """``_handle_license_activate`` must not set
    ``runtime_config['api_key']`` to any slice or transformation of the
    user-supplied ``key`` (or ``payload``)."""
    body = _read(PY_GATEWAY)
    m = re.search(
        r"def _handle_license_activate\(self.*?(?=\n    def |\nclass |\Z)",
        body,
        re.DOTALL,
    )
    assert m, "_handle_license_activate not found in gateway."
    fn_body = m.group(0)
    forbidden = [
        r'runtime_config\[["\']api_key["\']\]\s*=\s*key\[',
        r'runtime_config\[["\']api_key["\']\]\s*=\s*payload\[',
        r'runtime_config\[["\']api_key["\']\]\s*=\s*hmac\.',
    ]
    for pat in forbidden:
        assert not re.search(pat, fn_body), (
            f"C2 violation: _handle_license_activate contains /{pat}/. "
            "API key must be generated server-side, not derived from user input."
        )


def test_python_gateway_license_activate_does_not_flip_auth_policy() -> None:
    """``_handle_license_activate`` must not MUTATE the auth-policy
    toggle. The test only checks for assignment-style patterns; it
    allows the function to mention the field name in a
    reject-forbidden-field list (which is required defense)."""
    body = _read(PY_GATEWAY)
    m = re.search(
        r"def _handle_license_activate\(self.*?(?=\n    def |\nclass |\Z)",
        body,
        re.DOTALL,
    )
    assert m, "_handle_license_activate not found in gateway."
    fn_body = m.group(0)
    # Assignment to the auth-policy field is the dangerous pattern.
    mutating_patterns = [
        r'runtime_config\[["\']require_auth["\']\]\s*=',
        r'self\.server\.runtime_config\[["\']require_auth["\']\]\s*=',
    ]
    for pat in mutating_patterns:
        assert not re.search(pat, fn_body), (
            f"C2 violation: _handle_license_activate matches /{pat}/. "
            "Auth posture is operator policy, not a license-driven decision."
        )
    # The function must reject the field by name (defense-in-depth:
    # the client cannot smuggle the field through to runtime_config).
    assert '"require_auth"' in fn_body or "'require_auth'" in fn_body, (
        "C2 suspicion: activation handler does not explicitly reject the "
        "require_auth field in the input payload. Add a reject loop."
    )


def test_python_gateway_license_activate_rejects_credential_surface_fields() -> None:
    """The activation handler must reject payload fields that influence
    the credential surface (``api_key``, ``require_auth``, ``role``,
    ``scopes``, ``admin``)."""
    body = _read(PY_GATEWAY)
    m = re.search(
        r"def _handle_license_activate\(self.*?(?=\n    def |\nclass |\Z)",
        body,
        re.DOTALL,
    )
    assert m, "_handle_license_activate not found in gateway."
    fn_body = m.group(0)
    for field in ("api_key", "require_auth", "role", "scopes", "admin"):
        assert field in fn_body, (
            f"C2 suspicion: activation handler does not reference the "
            f"forbidden field {field!r}; either the rejection is missing "
            f"or the wording has changed — verify intentional."
        )


def test_python_gateway_license_activate_returns_no_api_key_field() -> None:
    """The activation response must not echo or invent an API key.

    The response body must contain only ``success`` and ``tier``; the
    API key is delivered through a separate, server-issued channel
    (e.g. an admin bootstrap endpoint) and is rotated by the operator.
    """
    body = _read(PY_GATEWAY)
    m = re.search(
        r"def _handle_license_activate\(self.*?(?=\n    def |\nclass |\Z)",
        body,
        re.DOTALL,
    )
    assert m, "_handle_license_activate not found"
    fn = m.group(0)
    success_response = re.search(r"self\._send_json\(\s*200.*?\)", fn, re.DOTALL)
    assert success_response, "activation handler has no 200 response"
    payload = success_response.group(0).lower()
    assert "api_key" not in payload, (
        "C2 perimeter violation: license activation response contains an "
        "api_key field. Clients should not be able to learn the admin "
        "credential via license activation."
    )


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------


def test_frontend_license_gate_does_not_store_user_key_as_api_key() -> None:
    """The frontend must not write the user-supplied license key into
    ``localStorage`` as the API key. Activation must round-trip via a
    server-issued session token, not the raw secret."""
    body = _read(FRONTEND_GATE)
    forbidden = re.search(
        r"localStorage\.setItem\(\s*['\"]aurelius-api-key['\"]\s*,\s*key",
        body,
    )
    assert not forbidden, (
        "C2 violation: frontend LicenseGate stores the user-supplied "
        "license key as the API key in localStorage. The API key must "
        "come from a server-issued session/credential, not the raw secret."
    )
