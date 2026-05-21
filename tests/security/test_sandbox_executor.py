"""Tests for src.security.sandbox_executor."""

from __future__ import annotations

import pytest

from src.security.sandbox_executor import (
    SandboxConfig,
    SandboxExecutor,
    SandboxViolation,
)


def test_simple_arithmetic_executes():
    runner = SandboxExecutor()
    result = runner.execute("x = 2 + 3\nprint(x)")
    assert result.timed_out is False
    assert result.exception is None
    assert result.stdout.strip() == "5"


def test_timeout_returns_timed_out_true():
    runner = SandboxExecutor()
    cfg = SandboxConfig(timeout_seconds=0.25)
    code = "i = 0\nwhile True:\n    i = i + 1\n"
    result = runner.execute(code, cfg)
    assert result.timed_out is True
    assert result.exception is not None
    assert "Timeout" in result.exception


def test_oversized_code_raises_SandboxViolation():
    runner = SandboxExecutor()
    cfg = SandboxConfig(max_code_len=100)
    big = "x = 1\n" * 200
    with pytest.raises(SandboxViolation) as info:
        runner.execute(big, cfg)
    assert "max_code_len" in info.value.reason
    assert len(info.value.code_snippet) <= 200


def test_blocked_import_raises_or_errors():
    runner = SandboxExecutor()
    result = runner.execute("import os\nprint(os.getcwd())")
    assert result.exception is not None
    assert "getcwd" not in result.stdout


def test_blocked_dynamic_primitives_missing():
    runner = SandboxExecutor()
    snippets = [
        "handle = open('/etc/passwd')",
        "probe = __import__('os')",
        "fn = compile('1+1', 'x', 'eval')",
    ]
    for snippet in snippets:
        result = runner.execute(snippet)
        assert result.exception is not None, f"expected failure for {snippet!r}"


def test_stdout_captured():
    runner = SandboxExecutor()
    result = runner.execute("print('hello'); print('world')")
    assert result.exception is None
    assert "hello" in result.stdout
    assert "world" in result.stdout


def test_getattr_type_traversal_escape_blocked():
    runner = SandboxExecutor()
    traversal_snippets = [
        "getattr(object, '__class__')",
        "setattr(object, '__class__', int)",
        "type(object)",
        "hasattr(object, '__class__')",
        "[c for c in type.__subclasses__(type)]",
        "getattr(int, '__subclasses__')()",
    ]
    for snippet in traversal_snippets:
        result = runner.execute(snippet)
        assert result.exception is not None, (
            f"expected sandbox violation for {snippet!r} but got none"
        )


def test_default_allowed_builtins_excludes_traversal_primitives():
    from src.security.sandbox_executor import DEFAULT_ALLOWED_BUILTINS

    for name in ("getattr", "setattr", "type", "hasattr", "object", "vars", "__build_class__"):
        assert name not in DEFAULT_ALLOWED_BUILTINS, (
            f"{name} must not be in DEFAULT_ALLOWED_BUILTINS "
            f"(sandbox escape vector — AUR-SEC-2026-0027 / AUR-SEC-2026-0028)"
        )


# ── New hardened tests ─────────────────────────────────────────────────────────


def test_blocked_dunder_attrs_are_none_in_globs():
    """The OPSEC dunder attrs set to None in _build_globals must appear as
    explicit None values, not be absent from the namespace (which would yield
    NameError instead)."""
    cfg = SandboxConfig()
    snap = SandboxExecutor._build_globals(SandboxExecutor(), cfg)
    for attr in (
        "__class__", "__bases__", "__subclasses__", "__mro__",
        "__globals__", "__code__", "__closure__", "__dict__",
    ):
        assert attr in snap, f"{attr!r} must be present in sandbox globs"
        assert snap[attr] is None, f"{attr!r} must be None, got {snap[attr]!r}"


def test_bare_blocked_builtin_names_are_not_reachable():
    """`__import__`, `eval`, `exec`, and `compile` as bare identifier references
    must raise NameError — they must NOT be secretly injected into globals."""
    runner = SandboxExecutor()
    blocked_bare = ["__import__", "eval", "exec", "compile"]
    for name in blocked_bare:
        # Use a non-call expression so we don't get TypeError wrapping NameError
        snippet = f"x = {name}"
        result = runner.execute(snippet)
        assert result.exception is not None, (
            f"bare name {name!r} should be unreachable in sandbox"
        )
        assert "NameError" in result.exception or "name" in result.exception.lower(), (
            f"expected NameError for {name!r}, got: {result.exception}"
        )


def test_isinstance_allowed_and_works():
    """`isinstance` is in DEFAULT_ALLOWED_BUILTINS — verify it actually works."""
    runner = SandboxExecutor()
    result = runner.execute("print(isinstance(1, int))")
    assert result.exception is None
    assert "True" in result.stdout


def test_sandbox_code_type_must_be_str():
    """Supplying non-str code must raise SandboxViolation, not some generic error."""
    with pytest.raises(SandboxViolation, match="code must be str"):
        SandboxExecutor().execute(123)  # type: ignore[arg-type]


def test_sandbox_result_has_expected_defaults():
    """Fresh SandboxResult must have sensible defaults."""
    from src.security.sandbox_executor import SandboxResult
    r = SandboxResult()
    assert r.stdout == ""
    assert r.stderr == ""
    assert r.exception is None
    assert r.timed_out is False
