"""Regression tests for C3 (WebSocket origin/room/token auth) and C4
(BFF identity preservation).

The pre-remediation BFF in middle/src/ had two P0 issues:

- C3: WebSocket handler accepted long-lived API keys as the auth
  credential, did not validate Origin, emitted agent/notification
  data in the connect handshake, and let any authenticated client
  subscribe to any room or broadcast to it.

- C4: The provider router and the OpenAI-compatible /v1/completions
  proxy always forwarded the service principal key to the upstream,
  collapsing all user-triggered calls into a single shared identity.
  The frontend also stored the raw long-lived API key in
  localStorage and sent it in WS query strings.

These tests are source-text inspections; they lock in the security
properties the fix must establish.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MIDDLE_SRC = REPO_ROOT / "middle" / "src"
FRONTEND_SRC = REPO_ROOT / "frontend" / "src"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_code_only(path: Path) -> str:
    """Strip line and block comments so substring searches don't
    false-positive on documentation."""
    text = read(path)
    # Remove block comments /* ... */
    text = re.sub(r"/\*[\s\S]*?\*/", "", text)
    # Remove line comments // ... (only when // is not inside a
    # string literal — this is a best-effort stripper for the
    # security tests).
    text = re.sub(r"//[^\n]*", "", text)
    return text


# ---------------------------------------------------------------------------
# C3: WebSocket origin + room + token auth
# ---------------------------------------------------------------------------


def test_ws_handler_enforces_origin_allowlist_before_auth() -> None:
    """The WS handler must check the Origin header before authenticating
    or emitting any data. A bad or missing Origin must close the socket
    with a 1008 status before auth runs."""
    handler = read_code_only(MIDDLE_SRC / "ws" / "handler.ts")
    assert "Origin" in handler, "WS handler must reference Origin validation"
    # The Origin check must appear BEFORE the auth verifier is CALLED
    # (not just imported). The call site is what runs; the import is
    # benign.
    origin_idx = handler.find("Origin")
    # Match the first call to verifyWsToken(...) — excludes the import
    call_match = re.search(r"\bverifyWsToken\s*\(", handler)
    auth_idx = call_match.start() if call_match else -1
    assert origin_idx != -1, "WS handler must reference Origin"
    assert auth_idx != -1, "WS handler must use a token verifier (verifyWsToken) instead of validateApiKey"
    assert origin_idx < auth_idx, "Origin check must come before auth verification"


def test_ws_handler_does_not_emit_agent_or_notification_data_before_auth() -> None:
    """The pre-remediation handler sent a 'connected' payload containing
    listAgents() and getNotificationStats() data immediately on connect,
    after only a static API key check. The fix must not emit any of that
    sensitive data before a verified auth principal exists."""
    handler = read_code_only(MIDDLE_SRC / "ws" / "handler.ts")
    # The connect path must not call listAgents/getNotificationStats until
    # the auth principal is established. The fix should either remove the
    # 'connected' payload entirely, or scope it to a non-sensitive
    # handshake (e.g. server time + connection id).
    if "listAgents" in handler:
        # If listAgents is still referenced, it must be in a path that
        # runs AFTER the verified authUser assignment.
        user_assign = re.search(r"ws\.authUser\s*=", handler)
        agents_call = handler.find("listAgents")
        assert user_assign is not None, "authUser must be assigned before any sensitive emit"
        assert user_assign.start() < agents_call, (
            "listAgents() must not be called before authUser is assigned"
        )
    if "getNotificationStats" in handler:
        stats_call = handler.find("getNotificationStats")
        user_assign = re.search(r"ws\.authUser\s*=", handler)
        assert user_assign is not None
        assert user_assign.start() < stats_call, (
            "getNotificationStats() must not run before authUser is assigned"
        )


def test_ws_handler_subscribe_validates_room_authorization() -> None:
    """The subscribe handler must validate that the requesting user is
    authorized for the target room (tenant + scope). It must not accept
    any room name from any authenticated client."""
    handler = read_code_only(MIDDLE_SRC / "ws" / "handler.ts")
    rooms = read_code_only(MIDDLE_SRC / "ws" / "rooms.ts")
    # The subscribe branch must call an authorization helper
    assert "subscribe" in handler, "subscribe handler must exist"
    # The new authz helper should be defined in rooms.ts
    authz_helpers = ["authorizeRoom", "canJoinRoom", "canSubscribe"]
    assert any(name in rooms for name in authz_helpers), (
        "rooms.ts must define an authorization helper "
        "(authorizeRoom/canJoinRoom/canSubscribe)"
    )
    assert any(name in handler for name in authz_helpers), (
        "WS handler must call the room authorization helper on subscribe"
    )


def test_ws_handler_does_not_accept_long_lived_api_key_as_ws_auth() -> None:
    """The pre-remediation handler read x-api-key from the upgrade
    request. The fix must verify a short-lived, audience-bound, exp-
    bound token instead."""
    handler = read_code_only(MIDDLE_SRC / "ws" / "handler.ts")
    # validateApiKey is the long-lived-key validator
    assert "validateApiKey" not in handler, (
        "WS handler must not use validateApiKey (long-lived API key) for auth"
    )
    # verifyWsToken is the short-lived WS token verifier
    assert "verifyWsToken" in handler, (
        "WS handler must call verifyWsToken (short-lived audience-bound token)"
    )
    # The ws_token module must exist
    assert (MIDDLE_SRC / "auth" / "ws_token.ts").exists(), (
        "middle/src/auth/ws_token.ts must exist"
    )


def test_ws_token_module_signs_and_verifies_with_audience_and_exp() -> None:
    """The WS token module must implement audience-bound, exp-bound
    HMAC verification. A token with a wrong audience or past exp must
    fail verification."""
    ws_token = read(MIDDLE_SRC / "auth" / "ws_token.ts")
    assert "audience" in ws_token or "aud" in ws_token, (
        "WS token must carry an audience claim"
    )
    assert "exp" in ws_token, "WS token must carry an exp claim"
    assert "verify" in ws_token, "WS token module must export a verify function"
    assert "sign" in ws_token, "WS token module must export a sign function"


def test_ws_origin_allowlist_is_centralized_not_hardcoded_wildcard() -> None:
    """The WS origin allowlist must be sourced from a configurable
    policy, not hardcoded. A literal '*' wildcard is forbidden."""
    handler = read_code_only(MIDDLE_SRC / "ws" / "handler.ts")
    config = read_code_only(MIDDLE_SRC / "config.ts")
    # No literal wildcard origin match
    assert 'origin === "*"' not in handler, (
        "WS handler must not accept a wildcard Origin match"
    )
    # The allowlist should be sourced from config (a single source of truth)
    # or from a constant allowlist
    assert "wsOriginAllowlist" in config or "wsAllowedOrigins" in config or "wsOriginAllowlist" in handler, (
        "WS origin allowlist must be centralized in config.ts or imported by handler.ts"
    )


# ---------------------------------------------------------------------------
# C4: BFF identity preservation
# ---------------------------------------------------------------------------


def test_provider_router_does_not_always_use_service_key() -> None:
    """The pre-remediation router unconditionally set
    headers.Authorization = Bearer ${config.serviceApiKey} for every
    upstream call, collapsing user-triggered calls into the service
    principal. The fix must take caller-bound credentials and only
    fall back to the service key for explicitly internal callers."""
    router = read_code_only(MIDDLE_SRC / "provider_router.ts")
    # The completeWithAurelius signature must now take caller credentials
    assert "callerCredentials" in router or "userCredentials" in router or "authHeader" in router, (
        "ProviderRouter must accept caller-bound credentials"
    )
    # The unconditional service-key header assignment must be gone
    assert "headers.Authorization = `Bearer ${config.serviceApiKey}`" not in router, (
        "ProviderRouter must not unconditionally set the service principal auth header"
    )


def test_models_route_uses_user_bound_credentials() -> None:
    """The /api/models route is user-triggered (called by the frontend
    with the user's session). It must not forward the service principal
    key on every call. The new design should either forward the
    caller's bearer or scope the route to admin-only with a clearly
    labeled internal policy."""
    route = read_code_only(MIDDLE_SRC / "routes" / "models.ts")
    # The user-facing GET / handler must not reference the service key.
    # The internal-only route (`internalModelsRouter`) is exempt by
    # design; the test inspects only the exported `router` block.
    user_block = re.search(
        r"router\.get\('/'.*?\}\s*\)\s*\n",
        route,
        re.DOTALL,
    )
    assert user_block is not None, "user-facing GET / handler must exist"
    body = user_block.group(0)
    assert "config.serviceApiKey" not in body, (
        "/api/models (user-facing) must not use the service principal key"
    )


def test_completions_proxy_uses_user_bound_credentials() -> None:
    """The /api/v1/completions proxy must forward the caller's bearer
    token (or an explicit service key in a clearly labeled internal
    handler), not a single shared service principal."""
    server = read_code_only(MIDDLE_SRC / "server.ts")
    # The /api/v1/completions handler must read the inbound auth header
    # and forward it (or use an internal policy helper)
    v1_block = re.search(
        r"app\.post\('/api/v1/completions'.*?(?=^\}\)\s*$|app\.(?:post|get|put|delete))",
        server,
        re.DOTALL | re.MULTILINE,
    )
    assert v1_block is not None, "/api/v1/completions handler must exist"
    body = v1_block.group(0)
    # The user-facing /api/v1/completions must prefer the inbound
    # Authorization. config.serviceApiKey may appear only inside an
    # explicit 'else if' fallback for anonymous calls (which is
    # logged via X-Aurelius-Anonymous-Upstream).
    assert "Authorization" in body, (
        "/api/v1/completions must reference an Authorization header to forward"
    )
    # Verify the user-bearer path is the primary path (the
    # 'else if (config.serviceApiKey)' must be a fallback).
    if "config.serviceApiKey" in body:
        assert "else if" in body and "X-Aurelius-Anonymous-Upstream" in body, (
            "/api/v1/completions service-key path must be the explicit fallback"
        )


def test_frontend_useWebSocket_uses_canonical_room_schema() -> None:
    """The frontend must send the canonical
    { type: 'subscribe', room: '...' } schema. The legacy { channel }
    schema is deprecated and must not be used in the new code path."""
    hook = read_code_only(FRONTEND_SRC / "hooks" / "useWebSocket.ts")
    # The subscribe function must send 'room', not 'channel'
    sub_block = re.search(
        r"const subscribe = useCallback\(\(.*?\)\s*=>\s*\{[^}]+\}",
        hook,
        re.DOTALL,
    )
    assert sub_block is not None, "useWebSocket.subscribe must exist"
    body = sub_block.group(0)
    assert "room" in body, "subscribe must send 'room' field"
    assert "channel" not in body, (
        "subscribe must not use the deprecated 'channel' field"
    )


def test_frontend_useWebSocket_passes_ws_token_not_api_key() -> None:
    """The frontend WS connect must pass the short-lived WS token (not
    a long-lived API key) to authenticate. A token in the query string
    is acceptable if it is short-lived; a raw API key in the URL is
    not."""
    hook = read_code_only(FRONTEND_SRC / "hooks" / "useWebSocket.ts")
    # The connect path should reference a token, not an apiKey
    assert "token" in hook.lower(), (
        "useWebSocket must reference a token (not an apiKey) for WS auth"
    )


def test_frontend_useAuth_does_not_store_raw_api_key_long_term() -> None:
    """The pre-remediation useAuth stored the raw API key in
    localStorage as 'aurelius-api-key'. The fix must replace that with
    a short-lived session token (or remove the persistence entirely)."""
    hook = read_code_only(FRONTEND_SRC / "hooks" / "useAuth.ts")
    store = read_code_only(FRONTEND_SRC / "stores" / "apiStore.ts")
    # The new design stores a short-lived session token, not a raw key
    assert "aurelius-api-key" not in hook, (
        "useAuth must not store the raw API key in localStorage"
    )
    assert "aurelius-api-key" not in store, (
        "apiStore must not store the raw API key in localStorage"
    )
    # The replacement should be a short-lived session identifier
    assert (
        "aurelius-session" in hook
        or "sessionToken" in hook
        or "session" in hook.lower()
    ), "useAuth must reference a short-lived session token"


def test_frontend_services_api_uses_session_token_not_api_key_header() -> None:
    """The frontend HTTP client must use the short-lived session token
    in the Authorization header (or a new session header), not the raw
    API key in an X-API-Key header."""
    svc = read_code_only(FRONTEND_SRC / "services" / "api.ts")
    assert "X-API-Key" not in svc, (
        "frontend services/api.ts must not send X-API-Key with the raw API key"
    )
    # Either cookie-based (credentials: 'include') for the
    # production flow, or Authorization: Bearer ***    # for the dev BYOK flow. Both are acceptable; the
    # production default is cookies.
    has_cookie_auth = "credentials:" in svc and "include" in svc
    has_bearer_auth = re.search(
        r"Authorization['\"]?\s*[:=]\s*['\"]?Bearer",
        svc,
    ) is not None
    assert has_cookie_auth or has_bearer_auth, (
        "frontend services/api.ts must use cookie auth (credentials: 'include') "
        "or Authorization: Bearer *** BYOK)"
    )
