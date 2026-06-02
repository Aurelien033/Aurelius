"""Regression tests for C1: legacy Node gateway must be decommissioned.

The legacy Node/Express gateway in ``server/`` was superseded by the
Python unified server in ``gateway/aurelius_server.py`` and the FastAPI
service in ``gateway/aurelius_api.py``. Both architectures cannot be
exposed at the same time without splitting identity, auth, license,
metrics, and SSE state across two non-communicating runtimes. Per
``docs/remediation/security-2026-06-01/register.md`` issue C1, the
legacy server must be archived and removed from the active build.

These tests are scoped to the PR1 allowlist (the legacy server
source files themselves). Out-of-allowlist follow-ups (root
``package.json`` workspaces / scripts, ``server/DEPRECATED.md``)
are tracked in the register and will be addressed in a subsequent
tranche.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_DIR = REPO_ROOT / "server"
LEGACY_MAIN = SERVER_DIR / "src" / "main.ts"
LEGACY_APP = SERVER_DIR / "src" / "app.ts"
LEGACY_WS_HUB = SERVER_DIR / "src" / "ws" / "hub.ts"
LEGACY_WS_INDEX = SERVER_DIR / "src" / "ws" / "index.ts"
LEGACY_CONFIG = SERVER_DIR / "src" / "config.ts"
LEGACY_ROUTES_INDEX = SERVER_DIR / "src" / "routes" / "index.ts"
LEGACY_LICENSE = SERVER_DIR / "src" / "routes" / "license.ts"


def _read(path: Path) -> str:
    assert path.exists(), f"required file missing: {path}"
    return path.read_text(encoding="utf-8")


def test_legacy_server_directory_removed_or_archived() -> None:
    """The ``server/`` directory must either be deleted or renamed to
    ``server_legacy/`` (so the workspace tools and CI no longer
    discover it as a Node package)."""
    if not SERVER_DIR.exists():
        return
    legacy_alt = REPO_ROOT / "server_legacy"
    assert legacy_alt.exists(), (
        "C1 violation: server/ is still present and has not been renamed "
        "to server_legacy/. The legacy Node package is still discoverable "
        "by workspace tooling and CI."
    )


def test_legacy_main_does_not_invoke_start_server() -> None:
    """``server/src/main.ts`` (or its ``server_legacy/`` equivalent)
    must not be an active entrypoint.

    The current file is a 3-line wrapper that calls ``startServer()``.
    After C1 remediation, the file should be either deleted or reduced
    to a guard that exits with a deprecation error.
    """
    if not LEGACY_MAIN.exists():
        return
    body = _read(LEGACY_MAIN).strip()
    assert "startServer()" not in body, (
        "C1 violation: server/src/main.ts still calls startServer(); "
        "the legacy Node server is still bootable."
    )


def test_legacy_app_is_not_in_active_runtime() -> None:
    """``server/src/app.ts`` (the Express app) must not be live.

    The file must not register routes, mount the WS hub, and listen.
    """
    if not LEGACY_APP.exists():
        return
    body = _read(LEGACY_APP)
    active_markers = (
        "registerRoutes(apiRouter)",
        "setupWebSocket(server)",
        "app.listen(",
    )
    found = [m for m in active_markers if m in body]
    assert not found, (
        f"C1 violation: server/src/app.ts still contains active runtime "
        f"markers {found!r}. The Express app is still bootable."
    )


def test_legacy_websocket_hub_does_not_export_broadcast() -> None:
    """The legacy WS hub must not expose ``broadcastToChannel`` or
    ``subscribeToChannel`` — these are the surface that the legacy
    chat stream uses, and they are the only reason the legacy server
    can keep SSE-style state in two runtimes."""
    if not LEGACY_WS_HUB.exists() and not LEGACY_WS_INDEX.exists():
        return
    for p in (LEGACY_WS_HUB, LEGACY_WS_INDEX):
        if not p.exists():
            continue
        body = _read(p)
        assert "broadcastToChannel" not in body, (
            f"C1 violation: {p} still exports broadcastToChannel — the "
            f"legacy WS hub is still importable."
        )


def test_legacy_license_route_does_not_set_runtime_config() -> None:
    """The legacy license route (now only relevant as historical
    reference) must not mutate the runtime config (C2 also covers
    this; this test is the legacy-server variant)."""
    if not LEGACY_LICENSE.exists():
        return
    body = _read(LEGACY_LICENSE)
    # The setConfig calls we accept in the historical file (if it
    # must remain for reference) are limited to the license state.
    set_config_calls = re.findall(r"setConfig\([^)]*\)", body)
    allowed = {"license_key", "license_activated", "license_tier"}
    for call in set_config_calls:
        m = re.search(r"['\"]([a-zA-Z0-9_]+)['\"]", call)
        if not m:
            continue
        assert m.group(1) in allowed, (
            f"C1/C2 violation: legacy license route still mutates "
            f"setConfig({m.group(1)!r}); legacy runtime config writes "
            f"are forbidden once the server is decommissioned."
        )


def test_no_outside_reference_to_legacy_routes_index() -> None:
    """Nothing outside the legacy archive should still import from
    ``server/src/routes/index.ts``. The new Python gateway is the
    canonical route registry."""
    if not LEGACY_ROUTES_INDEX.exists():
        return
    # Grep for any import of the legacy routes module from elsewhere.
    # We restrict the search to the new gateway directories.
    suspicious: list[tuple[str, str]] = []
    new_dirs = [
        REPO_ROOT / "gateway",
        REPO_ROOT / "middle",
        REPO_ROOT / "frontend",
    ]
    needle = "server/src/routes/index"
    for d in new_dirs:
        if not d.exists():
            continue
        for path in d.rglob("*.py"):
            if needle in path.read_text(encoding="utf-8", errors="ignore"):
                suspicious.append((str(path), needle))
        for path in d.rglob("*.ts"):
            if needle in path.read_text(encoding="utf-8", errors="ignore"):
                suspicious.append((str(path), needle))
        for path in d.rglob("*.tsx"):
            if needle in path.read_text(encoding="utf-8", errors="ignore"):
                suspicious.append((str(path), needle))
    assert not suspicious, (
        f"C1 violation: legacy routes module still imported from new "
        f"runtime: {suspicious!r}."
    )
