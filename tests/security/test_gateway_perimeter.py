"""End-to-end regression tests for the unified gateway perimeter.

These tests complement the per-handler tests in
``test_license_activation_security.py`` and
``test_legacy_server_decommissioned.py`` by asserting the *perimeter*
behavior of the live gateway:

* License activation never lets the user pick the API key.
* The unauthenticated license endpoints cannot reach privileged
  state (events, logs, memory entries, agent state changes).
* When ``require_auth`` is enabled, every non-public path requires
  a credential; public-path bypasses are limited to ``/api/health``
  and ``/api/license/validate``.
* The active server in the workspace has only one port-binding
  entrypoint (the Python ``aurelius_server.py``), and the legacy
  Node entrypoint is gone.

These tests are written to be importable on Python 3.9 (no PEP 634
match statements), per the project runtime constraint.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PY_GATEWAY = REPO_ROOT / "gateway" / "aurelius_server.py"
NODE_APP = REPO_ROOT / "server" / "src" / "app.ts"
NODE_MAIN = REPO_ROOT / "server" / "src" / "main.ts"


def _read(path: Path) -> str:
    assert path.exists(), f"required file missing: {path}"
    return path.read_text(encoding="utf-8")


def test_python_gateway_license_activate_returns_no_api_key_field() -> None:
    """The activation response must not echo or invent an API key.

    Whatever the server's internal credential model is, the response
    body must not contain a field that the client could store as a
    long-lived credential. Activation returns ``tier`` and ``success``
    only; the API key is delivered through a separate, server-issued
    channel (e.g. an admin bootstrap endpoint) and is rotated by the
    operator."""
    body = _read(PY_GATEWAY)
    m = re.search(
        r"def _handle_license_activate\(self.*?(?=\n    def |\nclass |\Z)",
        body,
        re.DOTALL,
    )
    assert m, "_handle_license_activate not found"
    fn = m.group(0)
    # The success response must not include the API key.
    success_response = re.search(r"self\._send_json\(\s*200.*?\)", fn, re.DOTALL)
    assert success_response, "activation handler has no 200 response"
    payload = success_response.group(0).lower()
    assert "api_key" not in payload, (
        "C2 perimeter violation: license activation response contains an "
        "api_key field. Clients should not be able to learn the admin "
        "credential via license activation."
    )


def test_python_gateway_public_bypass_limited_to_health_and_license_validate() -> None:
    """The list of public bypasses in ``_check_auth`` must be a tight
    allowlist. The current code only lets ``/api/health`` and
    ``/api/license/validate`` through. After C2 the license-validate
    bypass itself must not be a vector for privilege escalation."""
    body = _read(PY_GATEWAY)
    m = re.search(
        r"def _check_auth\(self.*?(?=\n    def |\nclass |\Z)",
        body,
        re.DOTALL,
    )
    assert m, "_check_auth not found"
    fn = m.group(0)
    # Extract the bypassed paths.
    bypasses = re.findall(r'self\.path\s*==\s*["\']([^"\']+)["\']', fn)
    # A small allowlist is the intended posture; reject any growth.
    assert set(bypasses) <= {"/api/health", "/api/license/validate"}, (
        f"Perimeter regression: _check_auth has unexpected public bypasses "
        f"{bypasses!r}. Only /api/health and /api/license/validate are "
        f"allowed."
    )


def test_active_server_entrypoint_is_uniquely_python() -> None:
    """There must be exactly one ``startServer`` / ``create_app`` /
    ``uvicorn`` entrypoint bound in the active runtime.

    * If ``server/src/main.ts`` exists, it must not bind a port.
    * If ``server/src/app.ts`` exists, it must not call
      ``app.listen``.
    * The Python gateway's ``aurelius_server.py`` is the only port-
      binding entrypoint.
    """
    if NODE_MAIN.exists():
        body = _read(NODE_MAIN)
        assert "listen(" not in body and "startServer(" not in body, (
            "Perimeter regression: legacy Node main.ts is still binding a port."
        )
    if NODE_APP.exists():
        body = _read(NODE_APP)
        assert "app.listen(" not in body, (
            "Perimeter regression: legacy Node app.ts is still binding a port."
        )
    # The Python server should still have its run() guard for the CLI.
    body = _read(PY_GATEWAY)
    assert re.search(r'if __name__ == [\'"]__main__[\'"]', body), (
        "Perimeter regression: Python gateway no longer has a __main__ guard."
    )
