from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.client import HTTPResponse
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from src.serving.aurelius_server import AureliusServer, create_aurelius_server


@contextmanager
def _running_server() -> Iterator[AureliusServer]:
    srv = create_aurelius_server("127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.05)
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()


def _url(server: AureliusServer, path: str) -> str:
    host, port = server.server_address
    return f"http://{host}:{port}{path}"


def _decode(resp: HTTPResponse | HTTPError) -> dict:
    raw = resp.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def _get(
    server: AureliusServer, path: str, headers: dict[str, str] | None = None
) -> tuple[int, dict]:
    req = Request(_url(server, path), headers=headers or {})  # noqa: S310
    try:
        with urlopen(req, timeout=5) as resp:  # noqa: S310
            return resp.status, _decode(resp)
    except HTTPError as exc:
        return exc.code, _decode(exc)


def _post(
    server: AureliusServer,
    path: str,
    payload: dict,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict]:
    req = Request(  # noqa: S310
        _url(server, path),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urlopen(req, timeout=5) as resp:  # noqa: S310
            return resp.status, _decode(resp)
    except HTTPError as exc:
        return exc.code, _decode(exc)


def test_health_is_public_but_status_fails_closed_by_default() -> None:
    with _running_server() as server:
        health_status, health = _get(server, "/api/health")
        status_status, status = _get(server, "/api/status")

    assert health_status == 200
    assert health["status"] == "ok"
    assert status_status == 401
    assert status["error"] == "Unauthorized"


def test_api_key_comparison_allows_valid_key() -> None:
    with _running_server() as server:
        server.runtime_config["api_key"] = "server-secret"
        denied_status, _ = _get(server, "/api/status", headers={"X-API-Key": "server-secret-extra"})
        allowed_status, data = _get(server, "/api/status", headers={"X-API-Key": "server-secret"})

    assert denied_status == 401
    assert allowed_status == 200
    assert "agents" in data


def test_license_activation_mints_unpredictable_api_key() -> None:
    license_key = "AURELIUS-" + "A" * 32
    predictable_suffix = license_key[-16:]

    with _running_server() as server:
        activate_status, activated = _post(
            server,
            "/api/license/activate",
            {"license_key": license_key, "tier": "pro"},
        )
        new_api_key = activated["api_key"]
        suffix_status, _ = _get(server, "/api/status", headers={"X-API-Key": predictable_suffix})
        key_status, _ = _get(server, "/api/status", headers={"X-API-Key": new_api_key})

    assert activate_status == 200
    assert new_api_key != predictable_suffix
    assert len(new_api_key) > 32
    assert suffix_status == 401
    assert key_status == 200


def test_config_cannot_disable_auth_or_leak_api_key() -> None:
    with _running_server() as server:
        server.runtime_config["api_key"] = "server-secret"
        headers = {"X-API-Key": "server-secret"}
        config_status, config = _get(server, "/api/config", headers=headers)
        update_status, update = _post(
            server,
            "/api/config",
            {"config": {"require_auth": False}},
            headers=headers,
        )

    assert config_status == 200
    assert config["config"]["api_key"] == "[REDACTED]"
    assert update_status == 403
    assert "auth settings" in update["error"]
