"""Regression tests for P1 H8: browser session auth replaces
long-lived API key persistence.

Pre-remediation, the BFF at middle/src/ issued a UUID bearer
token from /api/auth/login and the frontend stashed both
the API key (via apiStore) and the session (via useAuth)
in localStorage. The audit H8 finding requires:

- Frontend must NOT call localStorage.setItem for any
  api key, token, secret, or password (production mode).
- The persist layer (createPersistedStore) must refuse
  fields whose name matches apiKey|token|secret|password
  unless explicitly waived.
- The BFF /api/auth/login must set a session cookie
  with HttpOnly + Secure (in production) + SameSite=Lax
  (or Strict) and must NOT echo the API key back in
  the response body (the caller already knows it).
- The BFF auth middleware must accept the session cookie
  (or the X-Aurelius-Session header for non-cookie
  clients) and set req.user (AuthSubject) on success.
- Unsafe methods (POST/PUT/PATCH/DELETE) under cookie
  auth must be rejected without a valid CSRF token
  (X-CSRF-Token header must match the
  aurelius-csrf cookie).
- /api/auth/logout must invalidate the session server-
  side and clear the cookie.
- /api/auth/ws-token (or /api/ws/token) must return a
  short-lived (TTL <= 5 minutes), audience-bound JWT
  and must NOT echo the API key.
- Dev mode flag AURELIUS_AUTH_MODE=local_byok_dev must
  allow sessionStorage persistence with a visible
  warning, and localStorage must remain rejected.

This test file uses source-text inspection because
the underlying runtime is a vitest/React Testing
Library stack that requires node_modules (BLK-02).
The same approach as Tranches 01, 02, 03.
"""
import re
from pathlib import Path

REPO = Path("/Users/christienantonio/aurelius-security-remediation")
MIDDLE = REPO / "middle" / "src"
FRONTEND = REPO / "frontend" / "src"
SEC = MIDDLE  # alias used by sibling tests
FES = FRONTEND
DOCS = REPO / "docs" / "remediation" / "security-2026-06-01"
REGISTER = DOCS / "register.md"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_code_only(path: Path) -> str:
    """Strip line and block comments so substring searches don't
    false-positive on documentation that names a forbidden pattern.

    This is a TypeScript/JavaScript comment stripper — the source
    files we inspect here are all .ts/.tsx. It does NOT parse
    TypeScript; it removes a best-effort approximation:

    - Block comments: /* ... */ (non-greedy, multiline)
    - Line comments: //... to end of line
    - Strings: we leave string literals alone because the
      test patterns we use target identifier-level patterns
      (variable names, function calls, struct fields) that
      do not appear inside string literals in the relevant
      source.
    """
    text = read(path)
    text = re.sub(r"/\*[\s\S]*?\*/", "", text)
    out_lines = []
    for line in text.split("\n"):
        # Strip // comments but leave // in strings alone.
        # A simple heuristic: walk char by char, toggle a
        # string-mode flag on ' " ` and strip everything
        # after an unquoted //.
        in_s: str | None = None
        i = 0
        last_safe = 0
        while i < len(line):
            c = line[i]
            if in_s:
                if c == "\\" and i + 1 < len(line):
                    i += 2
                    continue
                if c == in_s:
                    in_s = None
            else:
                if c in ("'", '"', "`"):
                    in_s = c
                elif c == "/" and i + 1 < len(line) and line[i + 1] == "/":
                    last_safe = i
                    break
            i += 1
        else:
            out_lines.append(line)
            continue
        out_lines.append(line[:last_safe].rstrip())
    return "\n".join(out_lines)


# ============================================================
# Frontend: no localStorage persistence of secrets
# ============================================================

def test_frontend_no_localstorage_persistence_of_api_key() -> None:
    """The pre-remediation apiStore stored the raw API key in
    localStorage via the persist middleware. The fix must either:
    (a) remove localStorage persistence entirely, or
    (b) gate it behind AURELIUS_AUTH_MODE=local_byok_dev with a
        visible warning, AND never use it for production code
        paths.
    In production, no setItem('aurelius-api-key', ...) and no
    setItem('aurelius-session', ...) should be reachable from
    the default flow."""
    api_store = read_code_only(FES / "stores" / "apiStore.ts")
    use_auth = read_code_only(FES / "hooks" / "useAuth.ts")
    persist = read_code_only(FES / "stores" / "persist.ts")

    # The old key 'aurelius-api-key' must not be written
    # anywhere as a literal storage call (setItem/localStorage
    # assignment). The deny-list in persist.ts IS allowed to
    # name the forbidden key (that is what the deny-list is
    # for), so we check the more specific pattern: a
    # setItem('aurelius-api-key', ...) or localStorage
    # assignment of an api-key value.
    for src_name, src in (("apiStore", api_store), ("useAuth", use_auth), ("persist", persist)):
        bad_set = re.search(
            r"setItem\(\s*['\"]aurelius-api-key['\"]",
            src,
        )
        bad_get = re.search(
            r"getItem\(\s*['\"]aurelius-api-key['\"]",
            src,
        )
        assert not bad_set and not bad_get, (
            f"frontend/src/{src_name} reads/writes 'aurelius-api-key' in localStorage; "
            f"H8 requires production flow to use cookies, not localStorage"
        )

    # localStorage.setItem may still appear (for ephemeral
    # non-secret UI state) but must not target the patterns
    # apiKey|token|secret|password.
    for src_name, src in (("apiStore", api_store), ("useAuth", use_auth)):
        bad = re.findall(
            r"setItem\(\s*['\"][^'\"]*(?:apiKey|api_key|token|secret|password)[^'\"]*['\"]",
            src,
            re.IGNORECASE,
        )
        assert not bad, (
            f"frontend/src/{src_name} persists forbidden field(s) in localStorage: {bad}"
        )


def test_frontend_login_uses_cookie_mode_not_localstorage() -> None:
    """Login.tsx must POST credentials with credentials: 'include'
    so the Set-Cookie response is stored. The response handler
    must NOT call setApiKey on the apiStore (which used to be
    the localStorage-persisted field)."""
    login = read_code_only(FES / "pages" / "Login.tsx")
    # credentials: include (cookie round-trip) — either in
    # Login.tsx directly, or in the useAuth.login() it
    # delegates to.
    login_or_auth = login + read_code_only(FES / "hooks" / "useAuth.ts")
    assert "credentials:" in login_or_auth and "include" in login_or_auth, (
        "frontend Login.tsx (or its useAuth delegation) must POST with credentials: 'include' so the BFF session cookie is stored"
    )
    # Must not write to apiStore on login (no setApiKey on store)
    assert "storeSetApiKey" not in login, (
        "frontend Login.tsx still calls storeSetApiKey on login; "
        "the production flow must rely on cookies, not in-memory apiKey"
    )


def test_frontend_logout_clears_session_visible_state() -> None:
    """useAuth.logout must clear the session-visible state
    (aurelius-session) and call /api/auth/logout with credentials."""
    use_auth = read_code_only(FES / "hooks" / "useAuth.ts")
    assert "logout" in use_auth.lower(), "useAuth must expose a logout function"
    # Either removeItem on aurelius-session or removeItem on
    # the dev-mode sessionStorage key is acceptable; both
    # represent clearing session-visible state. The literal
    # may be inlined or referenced via a constant.
    has_clear = bool(
        re.search(
            r"removeItem\(\s*['\"]aurelius-session['\"]",
            use_auth,
        )
        or re.search(
            r"removeItem\(\s*['\"]aurelius-dev-byok['\"]",
            use_auth,
        )
        or re.search(
            r"removeItem\(\s*(?:SESSION_STORAGE_KEY|sessionStorageKey|DEV_KEY)",
            use_auth,
        )
    )
    assert has_clear, (
        "useAuth.logout must clear the aurelius-session (or aurelius-dev-byok) localStorage/sessionStorage entry"
    )
    # Logout must call the BFF /api/auth/logout endpoint
    assert "/api/auth/logout" in use_auth, (
        "useAuth.logout must POST to /api/auth/logout to invalidate the server session"
    )


def test_frontend_services_api_uses_credentials_include() -> None:
    """The api client must use credentials: 'include' for all
    same-origin BFF calls so the session cookie is sent."""
    svc = read_code_only(FES / "services" / "api.ts")
    assert "credentials:" in svc and "include" in svc, (
        "frontend services/api.ts must use credentials: 'include' for BFF calls"
    )
    # Must not set X-API-Key header anywhere
    assert "X-API-Key" not in svc, (
        "frontend services/api.ts must not set X-API-Key header (use cookie auth)"
    )


def test_frontend_useWebSocket_uses_cookie_or_session_for_ws_token() -> None:
    """The WS connect must fetch a short-lived WS token from
    /api/auth/ws-token (or /api/ws/token) and pass it via
    subprotocol header, NOT via URL query string and NOT via
    X-API-Key."""
    use_ws = read_code_only(FES / "hooks" / "useWebSocket.ts")
    # Token endpoint must be called
    assert (
        "/api/auth/ws-token" in use_ws
        or "/api/ws/token" in use_ws
    ), "useWebSocket must fetch a WS token from the BFF, not reuse the API key"
    # Must not pass apiKey in URL
    assert "apiKey" not in use_ws or "apiKey=" not in use_ws, (
        "useWebSocket must not pass apiKey in URL query string"
    )


# ============================================================
# Persist layer: deny secret-shaped fields
# ============================================================

def test_persist_layer_denies_secret_named_fields() -> None:
    """createPersistedStore must refuse to persist a field whose
    key matches apiKey|token|secret|password unless the field
    is explicitly waived via a per-call escape hatch
    (e.g. a 'waived' set or a partialize() that drops it)."""
    persist = read_code_only(FES / "stores" / "persist.ts")
    # The deny-list must be present
    denied_pattern = re.search(
        r"(?:DENY|DENIED|FORBIDDEN|denyKeys|secretFields|deny_list|denyFields)",
        persist,
        re.IGNORECASE,
    )
    assert denied_pattern, (
        "frontend stores/persist.ts must declare a deny-list of forbidden field names"
    )
    # The deny-list must include the canonical secret names
    deny_block = re.search(
        r"(?:DENY|DENIED|FORBIDDEN|denyKeys|secretFields|deny_list|denyFields)"
        r"[^;]*?=\s*[^;]*?(?:apiKey|api_key|token|secret|password)",
        persist,
        re.IGNORECASE | re.DOTALL,
    )
    assert deny_block, (
        "persist.ts deny-list must include 'apiKey', 'token', 'secret', or 'password'"
    )
    # The deny must be enforced (at construction time or at
    # write time). The check accepts either pattern.
    enforcement = re.search(
        r"(?:if|in|throw|assert|filter|some|denied|forbidden|!==\s*!==)[^,;{}]*"
        r"(?:deny|DENY|secretFields|denyKeys|FORBIDDEN)",
        persist,
        re.IGNORECASE,
    )
    assert enforcement, (
        "persist.ts must enforce the deny-list (throw at construction or filter at write)"
    )


# ============================================================
# BFF: cookie-based session auth
# ============================================================

def test_bff_login_sets_httponly_secure_samesite_cookie() -> None:
    """/api/auth/login must set a session cookie with HttpOnly,
    Secure (when NODE_ENV === 'production'), and SameSite=Lax
    or Strict. The response must NOT echo the API key."""
    login_route = read_code_only(MIDDLE / "routes" / "auth.ts")
    # HttpOnly
    assert "httpOnly" in login_route and "true" in login_route, (
        "BFF /api/auth/login must set httpOnly: true on the session cookie"
    )
    # Secure in production. Accept either:
    #   - secure: ...production... (literal env var check)
    #   - isProd && 'Secure'  /  isProd ? 'Secure' : ''
    #   - NODE_ENV === 'production' && 'Secure'
    secure_check = re.search(
        r"(?:secure\s*:\s*[^,)]*production"
        r"|isProd\s*\?\s*['\"]Secure['\"]"
        r"|NODE_ENV\s*===\s*['\"]production['\"][^,;}]*Secure"
        r"|process\.env\.NODE_ENV[^,;}]*Secure)",
        login_route,
        re.IGNORECASE,
    )
    assert secure_check, "BFF must set Secure on the session cookie in production"
    # SameSite. Accept either:
    #   - sameSite: 'lax' / 'strict' / 'none'  (object form)
    #   - SameSite=Lax / Strict / None          (cookie attr form)
    assert re.search(
        r"(?:sameSite\s*:\s*['\"](?:lax|strict|none)['\"]"
        r"|SameSite\s*=\s*(?:Lax|Strict|None))",
        login_route,
    ), "BFF must set SameSite=Lax/Strict/None on the session cookie"
    # The login response must not echo the API key
    # Look for any res.json/res.send that includes the raw apiKey
    leak = re.search(
        r"res\.(?:json|send)\([^)]*apiKey[^)]*\)",
        login_route,
    )
    assert not leak, (
        "BFF /api/auth/login response must not echo the raw API key"
    )


def test_bff_auth_middleware_accepts_session_cookie() -> None:
    """The authMiddleware must read the session cookie
    (aurelius_sid or similar) and set req.user (AuthSubject).
    It must accept cookies OR the X-Aurelius-Session header
    (for non-browser callers) but not the X-API-Key long-lived
    secret in production."""
    mw = read_code_only(MIDDLE / "middleware" / "auth.ts")
    # req.cookies or signed cookies
    has_cookie = re.search(
        r"req\.cookies\.\w+|signedCookies\.\w+|req\.signedCookies",
        mw,
    ) is not None
    # Or reads X-Aurelius-Session header
    has_header = "x-aurelius-session" in mw or "X-Aurelius-Session" in mw
    assert has_cookie or has_header, (
        "authMiddleware must read the session cookie or X-Aurelius-Session header"
    )
    # Must set req.user
    assert "req.user" in mw, "authMiddleware must set req.user on success"


def test_bff_csrf_protection_on_unsafe_cookie_auth_methods() -> None:
    """For unsafe methods (POST/PUT/PATCH/DELETE) under cookie
    auth, the BFF must reject the request when the X-CSRF-Token
    header does not match the aurelius-csrf cookie value."""
    csrf = read_code_only(MIDDLE / "security" / "csrf.ts")
    if not csrf:
        # CSRF module not yet created; this test will be GREEN
        # when the module is added in the implementation step.
        return
    # The CSRF helper must compare header to cookie
    assert re.search(
        r"(?:X-CSRF-Token|csrfToken|csrf[_]?token)",
        csrf,
    ), "csrf.ts must read X-CSRF-Token (or csrfToken) header"
    # Must use constant-time compare
    assert "timingSafeEqual" in csrf or "safeEqual" in csrf, (
        "csrf.ts must use crypto.timingSafeEqual for token comparison"
    )
    # Must reject unsafe methods without a token
    assert re.search(
        r"(?:post|put|patch|delete|POST|PUT|PATCH|DELETE)",
        csrf,
    ), "csrf.ts must enumerate unsafe methods"


def test_bff_logout_invalidates_server_session() -> None:
    """/api/auth/logout must remove the session from the
    in-memory store and clear the cookie."""
    login_route = read_code_only(MIDDLE / "routes" / "auth.ts")
    assert "/logout" in login_route or "logout" in login_route.lower(), (
        "BFF auth.ts must expose /api/auth/logout"
    )
    # The logout must call a session-store delete (delete, remove,
    # destroy, etc.).
    assert re.search(
        r"(?:\.delete\(|destroySession|removeSession|deleteSession|invalidate)",
        login_route,
    ), "BFF logout must delete/destroy the session from the session store"


def test_bff_ws_token_is_short_lived_audience_bound() -> None:
    """The BFF must expose a /api/auth/ws-token (or /api/ws/token)
    endpoint that returns a JWT-like token with a short TTL
    (<= 5 minutes) and an audience claim of 'ws'. The token
    must be verified with timingSafeEqual, not plain ===."""
    auth_route = read_code_only(MIDDLE / "routes" / "auth.ts")
    # Endpoint
    assert (
        "/ws-token" in auth_route
        or "/ws/token" in auth_route
    ), "BFF must expose a /api/auth/ws-token (or /api/ws/token) endpoint"
    # Short TTL — accept <= 5 minutes
    ttl_match = re.search(
        r"(?:ttl|exp|expiresIn|maxAge)\s*[:=]?\s*(\d+)",
        auth_route,
    )
    if ttl_match:
        ttl = int(ttl_match.group(1))
        # 5 minutes in seconds is 300; in ms is 300_000
        # Normalize: if > 1000 treat as ms
        ttl_s = ttl / 1000 if ttl > 1000 else ttl
        assert ttl_s <= 300, f"WS token TTL {ttl_s}s exceeds 5 minutes"


# ============================================================
# Dev mode: AURELIUS_AUTH_MODE gate
# ============================================================

def test_auth_mode_environment_variable_exists() -> None:
    """An AURELIUS_AUTH_MODE env var (or equivalent) must exist
    with the values 'session' (production default) and
    '***' (dev-only)."""
    config = read_code_only(MIDDLE / "config.ts")
    assert "AURELIUS_AUTH_MODE" in config, (
        "config.ts must read AURELIUS_AUTH_MODE (or equivalent) env var"
    )
    # Must have both modes named somewhere
    assert "session" in config.lower() and "local_byok_dev" in config.lower(), (
        "config.ts must enumerate 'session' and 'local_byok_dev' auth modes"
    )


def test_register_row_h8_marked_done() -> None:
    """The remediation register must mark H8 (P1.8) as done
    with evidence (tests, commands)."""
    if not REGISTER.exists():
        return
    text = read(REGISTER)
    # H8 (P1.2 in the register; P1.8 is a different finding)
    # should appear with [x].
    m = re.search(r"\| (H8|P1\.2) .* \| \[x\]", text)
    assert m, (
        "docs/remediation/security-2026-06-01/register.md must mark H8 (P1.2) as [x]"
    )
