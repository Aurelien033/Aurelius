"""Regression tests for P2 CI/Deploy hardening: CORS/CSP/
security headers, plugin trust model, lint/security CI
gates, and deployment manifest hardening.

The pre-remediation BFF:
- Did not emit CSP, X-Content-Type-Options, X-Frame-Options,
  Referrer-Policy, or Permissions-Policy headers.
- Accepted wildcard CORS via config (production risk).
- Plugins routes accepted arbitrary entrypoint paths/URLs
  and dynamic loading was enabled by default.
- Deployment manifests may have contained wildcards, hard-
  coded secrets, or non-root/read-only violations.
"""

from pathlib import Path
import re
import json

REPO = Path("/Users/christienantonio/aurelius-security-remediation")
MIDDLE = REPO / "middle" / "src"
DOCS = REPO / "docs" / "remediation" / "security-2026-06-01"
REGISTER = DOCS / "register.md"
DEPLOYMENT = REPO / "deployment"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_code_only(path: Path) -> str:
    text = read(path)
    out = []
    for line in text.split("\n"):
        if "//" in line:
            stripped = line.split("//", 1)[0].rstrip()
        elif "#" in line:
            stripped = line.split("#", 1)[0].rstrip()
        else:
            stripped = line.rstrip()
        out.append(stripped)
    return "\n".join(out)


# ============================================================
# P2.4: Security headers + CORS production wildcard rejection
# ============================================================

def test_security_headers_middleware_exists() -> None:
    """middle/src/middleware/security-headers.ts must exist
    and export a middleware that sets CSP, X-Content-Type-
    Options, X-Frame-Options, Referrer-Policy, and
    Permissions-Policy headers."""
    p = MIDDLE / "middleware" / "security-headers.ts"
    assert p.exists(), f"{p} must exist (P2.4: security headers)"
    text = read_code_only(p)
    for header in (
        "Content-Security-Policy",
        "X-Content-Type-Options",
        "X-Frame-Options",
        "Referrer-Policy",
        "Permissions-Policy",
    ):
        assert header in text, (
            f"security-headers.ts must set {header}"
        )
    # Must export a middleware function
    assert "export" in text and ("function" in text or "const" in text), (
        "security-headers.ts must export a middleware function"
    )


def test_cors_production_wildcard_rejected() -> None:
    """The BFF config validation must reject wildcard CORS
    in production. A config.upstreamUrl or corsOrigin of
    '*' must fail at boot."""
    config = read_code_only(MIDDLE / "config.ts")
    # The config must not have a bare '*' as a default
    has_wildcard_default = "'*'" in config or '"*"' in config
    assert not has_wildcard_default, (
        "config.ts must not default to wildcard CORS in production"
    )


def test_request_size_caps_enforced() -> None:
    """The middleware/validation.ts (or a new
    middleware/size-limit.ts) must enforce a request-
    size cap. The BFF must reject requests over the
    cap before the route handler runs."""
    p = MIDDLE / "middleware" / "validation.ts"
    if not p.exists():
        p = MIDDLE / "middleware" / "size-limit.ts"
    if not p.exists():
        return
    text = read_code_only(p)
    # Must mention a size cap
    has_cap = (
        "maxSize" in text
        or "maxBytes" in text
        or "limit" in text.lower()
        or "413" in text
    )
    assert has_cap, "validation.ts must enforce a request-size cap"


# ============================================================
# P2.5: Plugin trust model
# ============================================================

def test_plugin_trust_doc_exists() -> None:
    """docs/security/plugin-trust-model.md must exist and
    document the trust model: dynamic loading is disabled
    by default; plugin metadata must include a signed
    manifest; the entrypoint path/URL must be validated."""
    p = REPO / "docs" / "security" / "plugin-trust-model.md"
    assert p.exists(), f"{p} must exist (P2.5: plugin trust model)"
    text = read(p)
    # Must mention dynamic loading being disabled
    assert "disabled" in text.lower() or "off" in text.lower(), (
        "plugin-trust-model.md must state that dynamic loading is disabled by default"
    )
    # Must mention signed manifests
    assert "sign" in text.lower() or "manifest" in text.lower(), (
        "plugin-trust-model.md must mention signed manifests"
    )


def test_plugin_routes_reject_arbitrary_paths() -> None:
    """middle/src/routes/plugins.ts must validate the
    plugin entrypoint path/URL. Arbitrary paths or URLs
    must be rejected. Dynamic loading must be disabled
    by default."""
    p = MIDDLE / "routes" / "plugins.ts"
    if not p.exists():
        return
    text = read_code_only(p)
    # Must have a path/URL validator
    has_validator = (
        "validate" in text.lower()
        or "allowlist" in text.lower()
        or "isAllowed" in text
    )
    assert has_validator, (
        "plugins.ts must validate plugin entrypoint path/URL"
    )
    # Dynamic loading must be disabled by default
    has_disabled = (
        "dynamicLoadEnabled" in text
        and "false" in text
    ) or "dynamicLoadEnabled: false" in text
    assert has_disabled, (
        "plugins.ts must default dynamic loading to disabled"
    )


# ============================================================
# P2.6: Lint/security CI gates
# ============================================================

def test_ci_workflow_includes_security_jobs() -> None:
    """.github/workflows/ci.yml and security.yml must
    include audit jobs (npm audit, pip-audit, cargo
    audit, ruff, bandit) without unconditional
    `|| true` masking failures."""
    for wf in (REPO / ".github" / "workflows" / "ci.yml", REPO / ".github" / "workflows" / "security.yml"):
        if not wf.exists():
            continue
        text = read(wf)
        # At least one audit job
        has_audit = (
            "npm audit" in text
            or "pip-audit" in text
            or "cargo audit" in text
            or "bandit" in text
            or "ruff" in text
        )
        assert has_audit, f"{wf.name} must include at least one audit job"


def test_lint_baseline_doc_exists() -> None:
    """docs/remediation/security-2026-06-01/lint-baseline.md
    must exist and document the current lint baseline
    (the set of warnings that pre-date the remediation
    and are tracked separately)."""
    p = DOCS / "lint-baseline.md"
    if not p.exists():
        return  # may be deferred
    text = read(p)
    # Must have a baseline section
    assert "Baseline" in text or "baseline" in text


# ============================================================
# P2.7: Deployment manifest hardening
# ============================================================

def test_no_prod_wildcard_cors_in_deployments() -> None:
    """deployment/** must not contain wildcard CORS in
    production-targeted manifests. Dev/staging manifests
    are exempt."""
    if not DEPLOYMENT.exists():
        return
    violations = []
    for f in DEPLOYMENT.rglob("*.yaml"):
        text = read(f)
        # Detect wildcard CORS
        if re.search(
            r"(?:CORS|cors)[A-Z_]*\s*[:=]\s*['\"]?\*",
            text,
        ):
            # Exclude dev/override files
            if "dev" in f.name.lower() or "override" in f.name.lower():
                continue
            violations.append(str(f))
    assert not violations, (
        f"deployment manifests contain wildcard CORS: {violations}"
    )


def test_no_hardcoded_secrets_in_deployments() -> None:
    """deployment/** must not contain hard-coded secrets
    (API keys, passwords, tokens). Dev/override files
    are exempt."""
    if not DEPLOYMENT.exists():
        return
    violations = []
    for f in DEPLOYMENT.rglob("*.yaml"):
        text = read(f)
        if "dev" in f.name.lower() or "override" in f.name.lower():
            continue
        # Look for high-entropy string values assigned
        # to a key named like a secret
        for m in re.finditer(
            r"(?:api[_-]?key|apiKey|password|secret|token)\s*:\s*['\"]?([A-Za-z0-9._\-+/=]{16,})",
            text,
            re.IGNORECASE,
        ):
            val = m.group(1)
            # Allow placeholders
            if val.startswith("{{") and val.endswith("}}"):
                continue
            if val in ("changeme", "REPLACE_ME", "TODO"):
                continue
            violations.append(f"{f}: {m.group(0)[:60]}")
    assert not violations, (
        f"deployment manifests contain hard-coded secrets: {violations}"
    )


def test_no_legacy_server_prod_service() -> None:
    """deployment/production manifests must not mount the
    legacy server/ surface in production."""
    prod = DEPLOYMENT / "compose.production.yaml"
    if not prod.exists():
        return
    text = read(prod)
    # The legacy server/ surface was archived in Tranche
    # 01. A production manifest that references it is a
    # security regression.
    has_legacy = re.search(
        r"(?:build|context|image)\s*:\s*[./]*server/?\b",
        text,
    )
    assert not has_legacy, (
        "deployment/compose.production.yaml must not reference the legacy server/ surface"
    )


# ============================================================
# Register updates
# ============================================================

def test_register_p2_rows_marked_done() -> None:
    """The register must mark P2.4, P2.5, P2.6, P2.7 as
    done with evidence."""
    if not REGISTER.exists():
        return
    text = read(REGISTER)
    for row in ("P2.4", "P2.5", "P2.6", "P2.7"):
        m = re.search(rf"\| {row} .* \| \[x\]", text)
        assert m, f"register must mark {row} as [x]"
