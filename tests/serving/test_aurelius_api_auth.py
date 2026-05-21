from __future__ import annotations

import importlib
from pathlib import Path

from fastapi.testclient import TestClient


def _client(monkeypatch, tmp_path: Path) -> TestClient:
    monkeypatch.setenv("AURELIUS_API_KEY", "test-secret")
    monkeypatch.setenv("AURELIUS_ALLOWED_HOSTS", "testserver,localhost,127.0.0.1")
    monkeypatch.setenv("AURELIUS_WORKSPACE_ROOT", str(tmp_path))
    import gateway.aurelius_api as aurelius_api

    module = importlib.reload(aurelius_api)
    return TestClient(module.app)


def test_health_and_readiness_are_public(monkeypatch, tmp_path: Path) -> None:
    client = _client(monkeypatch, tmp_path)

    health = client.get("/health")
    assert health.status_code == 200

    readiness = client.get("/health/ready")
    assert readiness.status_code in {200, 503}
    assert readiness.status_code != 401


def test_non_probe_routes_require_api_key(monkeypatch, tmp_path: Path) -> None:
    client = _client(monkeypatch, tmp_path)

    response = client.get("/workspaces")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_valid_api_key_allows_protected_route(monkeypatch, tmp_path: Path) -> None:
    client = _client(monkeypatch, tmp_path)

    response = client.get("/workspaces", headers={"X-API-Key": "test-secret"})

    assert response.status_code == 200
    assert response.json() == [{"id": "default", "path": str(Path.cwd())}]


def test_workspace_paths_are_confined_to_workspace_root(monkeypatch, tmp_path: Path) -> None:
    client = _client(monkeypatch, tmp_path)
    headers = {"X-API-Key": "test-secret"}

    accepted = client.post("/workspaces", json={"path": "project-a"}, headers=headers)
    assert accepted.status_code == 200
    assert accepted.json()["path"] == str(tmp_path / "project-a")

    rejected = client.post("/workspaces", json={"path": "../outside"}, headers=headers)
    assert rejected.status_code == 403


def test_chat_completion_route_requires_auth(monkeypatch, tmp_path: Path) -> None:
    client = _client(monkeypatch, tmp_path)
    payload = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 1}

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 401
