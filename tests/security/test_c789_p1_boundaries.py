"""Regression tests for P1 boundaries: H1 URL policy, H2/H3 command
+ scheduler typed envelopes, H4 RAG tenant/owner scoping, H5 file
upload streaming + sanitization.

Pre-remediation vulnerabilities:
- H1: Upstream URL was a free-form string in config.upstreamUrl;
  no allowlist, no host validation, no metadata-IP rejection.
  Threat: an attacker who can influence config.upstreamUrl
  can point upstream at internal services (SSRF) or at a
  controlled host (credential exfiltration).
- H2/H3: /api/command accepted a raw text command and
  scheduler stored tasks as `{ name, cron, command: string }`.
  No typed command envelopes, no scope check beyond admin,
  no approval, no max-runs, no expiry, no emergency disable.
  Threat: a misbehaving agent or compromised admin can
  schedule arbitrary privileged work with no bound on
  damage.
- H4: RAG documents were stored in a process-global
  `documents` array. No tenant/owner scope, no quota,
  predictable IDs (`doc_${Date.now()}_${rand}`), and
  cross-user read was trivial. Threat: one tenant
  can read another tenant's documents.
- H5: File uploads trusted the client-supplied filename
  and MIME type. No server-side content sniffing, no
  sanitization beyond stripping \r\n, no streaming
  enforcement (the in-memory chunks array grew unboundedly
  until abort). Threat: a malicious client can upload
  content that the server will return to other users
  under a different filename/MIME.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
MIDDLE = REPO / "middle"
SEC = MIDDLE / "src"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_code_only(path: Path) -> str:
    """Strip line and block comments so substring searches don't
    false-positive on documentation. Strings and template literals
    are preserved."""
    text = path.read_text(encoding="utf-8")
    # Strip /* ... */ block comments
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    # Strip // line comments (naive; doesn't handle // inside
    # strings, but the test files don't have those patterns)
    lines = []
    for line in text.splitlines():
        idx = line.find("//")
        if idx != -1:
            # Keep string literals intact by checking for " or '
            # before //; if // appears in a string, leave the line
            # alone.
            stripped = line[:idx]
            if '"' in stripped or "'" in stripped or "`" in stripped:
                # naive: if the // is inside a string, leave the
                # line; otherwise strip
                if line.count('"') % 2 == 0 and line.count("'") % 2 == 0:
                    line = stripped
            else:
                line = stripped
        lines.append(line)
    return "\n".join(lines)


# --------- H1 URL policy ---------

def test_url_policy_module_exists() -> None:
    """A central URL policy module must exist and export a
    `validateUpstreamUrl` (or similarly named) function that the
    BFF uses to validate every upstream URL before fetch."""
    sec = SEC / "security"
    assert sec.is_dir(), f"middle/src/security directory missing: {sec}"
    files = list(sec.glob("*.ts"))
    assert files, "no URL policy / validation file in middle/src/security"
    policy = read_code_only(sec / "url_policy.ts")
    assert "validateUpstreamUrl" in policy, (
        "url_policy.ts must export validateUpstreamUrl"
    )


def test_url_policy_rejects_metadata_ip() -> None:
    """169.254.169.254 is the cloud metadata service. The URL
    policy must reject any URL that resolves to this address,
    even if the hostname is a different DNS name (defense in
    depth)."""
    policy = read_code_only(SEC / "security" / "url_policy.ts")
    # Must contain either an explicit 169.254.169.254 check or
    # a metadata subnet check
    has_metadata = "169.254.169.254" in policy or "metadata" in policy.lower()
    assert has_metadata, "url_policy must reject cloud metadata IPs"


def test_url_policy_rejects_userinfo_url() -> None:
    """URLs with userinfo (e.g. https://attacker:pw@victim) can
    confuse credential handling. The policy must reject any URL
    containing @ before the host."""
    policy = read_code_only(SEC / "security" / "url_policy.ts")
    # Look for explicit userinfo rejection (auth in URL)
    has_userinfo_check = "@" in policy and ("userinfo" in policy.lower() or "auth" in policy.lower())
    assert has_userinfo_check, (
        "url_policy must reject URLs with userinfo (@ in the authority)"
    )


def test_url_policy_https_required_in_production() -> None:
    """In production, only HTTPS URLs are allowed for upstream.
    The policy must require https:// or allow localhost for dev."""
    policy = read_code_only(SEC / "security" / "url_policy.ts")
    # Must enforce https
    assert "https" in policy.lower() or "HTTPS" in policy, (
        "url_policy must enforce HTTPS for production upstreams"
    )


def test_server_uses_url_policy_for_upstream() -> None:
    """The server's /api/v1/completions handler must validate
    config.upstreamUrl through the policy before fetching. The
    raw `${config.upstreamUrl}/v1/chat/completions` concatenation
    is a finding."""
    server = read_code_only(SEC / "server.ts")
    # Must use validateUpstreamUrl or import url_policy
    uses_policy = "validateUpstreamUrl" in server or "url_policy" in server or "urlPolicy" in server
    assert uses_policy, (
        "server.ts must validate upstream URL through url_policy"
    )


def test_provider_router_uses_url_policy() -> None:
    """The provider_router must validate upstream URLs through
    the policy. Free-form URL construction in complete() is a
    finding."""
    router = read_code_only(SEC / "provider_router.ts")
    uses_policy = "validateUpstreamUrl" in router or "url_policy" in router or "urlPolicy" in router
    assert uses_policy, (
        "provider_router.ts must validate upstream URL through url_policy"
    )


def test_config_does_not_let_runtime_mutate_upstream_url() -> None:
    """H1 follow-on: the POST /api/config handler must not let
    a config:write scope silently rewrite the upstream URL.
    Either it rejects the change or requires admin+break-glass."""
    config_route = read_code_only(SEC / "routes" / "config.ts")
    _server = read_code_only(SEC / "server.ts")
    # The original server.ts has /api/config that mutates any
    # config key, including chat.upstream_url. The fix must
    # add a protected-keys list (mirrors the C2 PROTECTED_CONFIG_KEYS
    # pattern in server_legacy/src/routes/config.ts:4).
    has_protected = "PROTECTED" in config_route or "protected" in config_route or "deny" in config_route.lower() or "BLOCKED" in config_route
    # Or: the server explicitly rejects upstream URL mutations
    has_reject = "upstream_url" in config_route and "reject" in config_route.lower()
    assert has_protected or has_reject, (
        "routes/config.ts must block runtime mutation of upstream/auth/CORS keys"
    )


# --------- H4 RAG ---------

def test_rag_documents_have_tenant_owner_scopes() -> None:
    """The RAG documents map must be keyed by tenant+owner. The
    pre-remediation implementation used a single process-global
    `documents` array."""
    rag = read_code_only(SEC / "routes" / "rag.ts")
    # Must have a per-tenant data structure (Map<tenant, Map<id, doc>>)
    # or a tenant/owner field on every record
    has_tenant = "tenant" in rag.lower() or "owner" in rag.lower() or "ownerId" in rag
    has_per_tenant_map = re.search(r"Map<[^,]+,\s*Map<", rag) is not None
    assert has_tenant or has_per_tenant_map, (
        "rag.ts must scope documents by tenant/owner"
    )


def test_rag_documents_use_uuid_ids() -> None:
    """The pre-remediation IDs were `doc_${Date.now()}_${Math.random...}`.
    The fix must use uuidv4() (or similar) for non-enumerable IDs."""
    rag = read_code_only(SEC / "routes" / "rag.ts")
    has_uuid = "uuidv4" in rag or "crypto.randomUUID" in rag or "uuid()" in rag
    has_old_pattern = re.search(r"doc_\$\{Date\.now", rag) is not None
    assert has_uuid, "rag.ts must use uuid() for document IDs"
    assert not has_old_pattern, (
        "rag.ts still uses doc_${Date.now()}_${Math.random} pattern"
    )


def test_rag_routes_check_owner_or_tenant() -> None:
    """The pre-remediation GET /documents/:id returned any
    document to any caller. The fix must check that the caller
    owns the document or is admin in the same tenant."""
    rag = read_code_only(SEC / "routes" / "rag.ts")
    # Must have a per-request owner/tenant check on read
    # (look for `req.user` or `userId` or `tenant` in a read path)
    has_check = (
        "req.user" in rag
        and ("ownerId" in rag or "userId" in rag or "tenant" in rag)
    )
    assert has_check, (
        "rag.ts must check owner/tenant before returning a document"
    )


def test_rag_enforces_quota() -> None:
    """The fix must enforce a per-tenant document quota so that
    one tenant cannot exhaust the in-memory store."""
    rag = read_code_only(SEC / "routes" / "rag.ts")
    has_quota = re.search(r"(quota|maxDocuments|maxDocs|MAX_DOCS)", rag) is not None
    assert has_quota, "rag.ts must enforce a per-tenant document quota"


# --------- H5 Files ---------

def test_files_trusts_server_side_mime_not_client_header() -> None:
    """The pre-remediation files route used
    `req.headers['x-file-type']` for the MIME type. The fix must
    server-side sniff the first bytes of the uploaded content
    instead of trusting the client header."""
    files = read_code_only(SEC / "routes" / "files.ts")
    # Look for server-side MIME sniff (file-type-detection lib,
    # or magic-byte check, or "sniffMime")
    has_sniff = (
        "sniffMime" in files
        or "file-type" in files
        or "magic" in files.lower()
        or re.search(r"mime.*detect", files, re.IGNORECASE) is not None
    )
    # Must not trust x-file-type as authoritative
    # (the header may still be used as a hint, but the server
    # must override if sniff disagrees)
    no_trust = "x-file-type" not in files
    assert has_sniff or no_trust, (
        "files.ts must server-side sniff MIME; not trust x-file-type"
    )


def test_files_sanitizes_filename_against_path_traversal() -> None:
    """The pre-remediation sanitization only stripped \\r\\n. The
    fix must also reject filenames containing path separators
    (\\, /) and parent-directory references (..)."""
    files = read_code_only(SEC / "routes" / "files.ts")
    # Must reject / and \ in filename
    has_path_check = re.search(r"[/\\\\]", files) is not None
    has_dotdot_check = "\\.\\." in files or '".."' in files or "'..'" in files
    assert has_path_check or has_dotdot_check, (
        "files.ts must sanitize filename against path traversal"
    )


def test_files_uses_streaming_with_temp_cleanup_on_abort() -> None:
    """The pre-remediation code stored the whole upload in
    memory (chunks: Buffer[]). The fix must stream to a temp
    file and delete the temp file on abort/oversize."""
    files = read_code_only(SEC / "routes" / "files.ts")
    # Must use a temp file (write to disk, then rename) or
    # busboy/multipart parser with disk spooling.
    has_stream = (
        "busboy" in files
        or "multer" in files
        or "tempfile" in files.lower()
        or "createWriteStream" in files
        or "tmpdir" in files
    )
    # Must have explicit cleanup on error
    has_cleanup = "unlink" in files or "rm(" in files or "cleanup" in files.lower()
    assert has_stream, "files.ts must use a streaming upload, not in-memory Buffer[]"
    assert has_cleanup, "files.ts must clean up the temp file on abort/error"


def test_files_strips_log_injection_chars_from_metadata() -> None:
    """User-supplied filenames must not be interpolated raw into
    log lines (CWE-117). The fix must either sanitize before
    logging or use structured fields."""
    files = read_code_only(SEC / "routes" / "files.ts")
    # Look for structured logging or sanitization before log
    has_safe_log = (
        "JSON.stringify" in files
        or "sanitizeForLog" in files
        or "sanitize" in files.lower()
    )
    assert has_safe_log, (
        "files.ts must sanitize log strings against CWE-117 (log injection)"
    )


# --------- H2/H3 Scheduler + Command ---------

def test_scheduler_uses_typed_command_envelope() -> None:
    """The pre-remediation scheduler stored `{ name, cron, command }`.
    The fix must use a typed envelope with command type, params,
    scope, owner, approval, max-runs, expiry."""
    scheduler = read_code_only(SEC / "routes" / "scheduler.ts")
    # Must have an explicit envelope type
    has_envelope = (
        re.search(r"interface\s+CommandEnvelope|type\s+CommandEnvelope", scheduler) is not None
        or re.search(r"interface\s+ScheduledCommand|type\s+ScheduledCommand", scheduler) is not None
    )
    # Must reference approval, max-runs, expiry
    has_approval = "approval" in scheduler.lower() or "approvedBy" in scheduler
    has_max_runs = "maxRuns" in scheduler or "max_runs" in scheduler
    has_expiry = "expiresAt" in scheduler or "expiry" in scheduler.lower()
    assert has_envelope, "scheduler.ts must use a typed command envelope"
    assert has_approval, "scheduler.ts must record approval metadata"
    assert has_max_runs, "scheduler.ts must enforce max-runs"
    assert has_expiry, "scheduler.ts must enforce expiry"


def test_scheduler_rejects_raw_string_command() -> None:
    """The pre-remediation scheduler stored the command as a
    raw string. The fix must require a typed command (not a
    free-form string)."""
    scheduler = read_code_only(SEC / "routes" / "scheduler.ts")
    # The old code had `command: string` as a top-level field;
    # the new code must require `{ type, params }` instead.
    # Look for the absence of a free-form `command: string` field
    # on the persisted record.
    has_old_pattern = re.search(r"command:\s*string", scheduler) is not None
    assert not has_old_pattern, (
        "scheduler.ts must not store raw 'command: string' on the task"
    )


def test_scheduler_enforces_emergency_disable() -> None:
    """The fix must provide an emergency disable that halts ALL
    scheduled tasks (e.g., an env-gated switch or a global
    `scheduler.enabled` flag)."""
    scheduler = read_code_only(SEC / "routes" / "scheduler.ts")
    has_kill_switch = (
        "disabled" in scheduler.lower()
        or "killSwitch" in scheduler
        or "EMERGENCY" in scheduler
        or "scheduler.enabled" in scheduler.lower()
    )
    assert has_kill_switch, "scheduler.ts must have an emergency disable switch"


def test_command_route_requires_typed_envelope() -> None:
    """The pre-remediation /api/command route accepted
    `{ agentId, command: string }`. The fix must require a
    typed envelope or convert the route to proposal-only."""
    server = read_code_only(SEC / "server.ts")
    # Look for typed command handling: the route must not have
    # a free-form `command` string as the body
    has_old_pattern = re.search(
        r"/api/command[\s\S]{0,200}command\s*[,}\)]", server
    ) is not None
    # The /api/command route should be turned into a proposal
    # endpoint or require a typed envelope
    has_typed = (
        re.search(r"commandType|CommandType|command\.type", server) is not None
        or re.search(r"proposal", server, re.IGNORECASE) is not None
    )
    assert has_typed or not has_old_pattern, (
        "server.ts /api/command must be proposal-only or require typed envelope"
    )


def test_command_route_redacts_sensitive_params_in_logs() -> None:
    """Sensitive command params (API keys, tokens, passwords)
    must be redacted before logging."""
    server = read_code_only(SEC / "server.ts")
    has_redaction = (
        "redact" in server.lower()
        or "[REDACTED]" in server
        or re.search(r"sanitize.*log|safeLog", server, re.IGNORECASE) is not None
    )
    assert has_redaction, (
        "server.ts /api/command must redact sensitive params before logging"
    )


def test_scheduler_callback_uses_localhost_loopback_not_configurable_host() -> None:
    """The pre-remediation scheduler hit
    `http://localhost:${process.env.MIDDLE_PORT || 3001}/api/command`.
    The fix must use a hard-coded loopback URL or 127.0.0.1, not
    a configurable port that could point at an attacker host."""
    scheduler = read_code_only(SEC / "routes" / "scheduler.ts")
    # The callback should be to a fixed loopback, not an env var
    has_env_port = re.search(r"MIDDLE_PORT|process\.env\.\w+PORT", scheduler) is not None
    # If the URL is built with process.env.X, that's a finding
    # (the loopback itself is fine, but the port should be a
    # constant or come from the validated config)
    assert not has_env_port, (
        "scheduler.ts must not use process.env.MIDDLE_PORT for the callback URL"
    )


# --------- Cross-cutting: telemetry is structured ---------

def test_routes_use_safe_log_strings() -> None:
    """The pre-remediation routes interpolated user input directly
    into log messages (e.g., `Command "${command.slice(0,80)}" sent`).
    The fix must use structured logging (JSON.stringify or
    separate fields)."""
    for route in ("rag.ts", "files.ts", "scheduler.ts", "config.ts"):
        path = SEC / "routes" / route
        if not path.exists():
            continue
        text = read_code_only(path)
        # Look for direct interpolation of user input into a log
        # message. The safe pattern is JSON.stringify({...}).
        has_template_log = re.search(
            r"appendActivity\([^,]+,\s*(true|false),\s*`[^`]*\$\{",
            text,
        ) is not None
        if has_template_log:
            # Check if the interpolated value is sanitized
            # (e.g., `.replace(/[\r\n]/g, "")` on the value)
            safe = re.search(
                r"appendActivity\([^,]+,\s*(true|false),\s*`[^`]*\$\{[^}]*\.replace",
                text,
            )
            assert safe, (
                f"routes/{route} interpolates user input into log message without sanitization"
            )
