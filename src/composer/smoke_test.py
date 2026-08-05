# ruff: noqa: S101
"""
Smoke test for Aurelius Composer modules.

Verifies:
  1. CodebaseIndexer builds and searches against the Aurelius repo itself
  2. DiffEngine generates and applies diffs correctly
  3. TerminalSandbox executes commands safely
  4. ContextAssembler selects compact context
  5. MultiFileEditOrchestrator plans dependency-aware edits
  6. EditVerifier catches syntax failures and passes valid files
  7. CheckpointRollback restores file-scoped edits
  8. ComposerAgent initializes and can run a fake-model edit loop

Run:
  python3 -m pytest src/composer/smoke_test.py -v
"""

from __future__ import annotations

import tempfile
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Test 1: CodebaseIndexer
# ---------------------------------------------------------------------------


def test_indexer_builds_and_searches():
    """Index the aurelius repo and search for known symbols."""
    from src.composer.codebase_indexer import CodebaseIndexer, _python_extract_symbols

    repo = Path(__file__).resolve().parents[2]  # aurelius root
    indexer = CodebaseIndexer(repo)

    stats = indexer.build(force=True)
    assert stats.total_files > 0, f"Expected files in {repo}, got {stats.total_files}"
    assert stats.total_symbols > 0, f"Expected symbols, got {stats.total_symbols}"

    hits = indexer.search("CodebaseIndexer")
    assert len(hits) > 0, "Expected to find CodebaseIndexer symbol"
    found = any("codebase_indexer.py" in h.file or "smoke_test" in h.file for h in hits)
    assert found, f"Expected CodebaseIndexer in source files, got: {[h.file for h in hits[:5]]}"

    importers = indexer.files_importing("dataclasses")
    assert len(importers) > 0, "Expected some files importing dataclasses"

    sample = textwrap.dedent(
        """
        class A:
            def m(self):
                pass

        def f():
            pass
        """
    )
    symbols, _, _, _ = _python_extract_symbols(sample)
    names = [(s.kind, s.name, s.parent) for s in symbols]
    assert names.count(("method", "m", "A")) == 1
    assert names.count(("function", "m", "")) == 0, (
        "method must not be duplicated as top-level function"
    )
    assert names.count(("function", "f", "")) == 1


# ---------------------------------------------------------------------------
# Test 2: DiffEngine
# ---------------------------------------------------------------------------


def test_diff_generator_produces_valid_diff():
    """Generate a diff and verify it applies correctly."""
    from src.composer.diff_engine import DiffApplier, DiffGenerator

    original = textwrap.dedent("""\
        def hello():
            print("hello world")
            return True

        def goodbye():
            print("goodbye")
            return False
    """)

    modified = textwrap.dedent("""\
        def hello(name: str = "world"):
            print(f"hello {name}")
            return True

        def goodbye():
            print("goodbye")
            return False
    """)

    file_edit = DiffGenerator.diff_content("test.py", original, modified)
    assert len(file_edit.hunks) > 0, "Expected at least one hunk"

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.py"
        test_file.write_text(original)

        applier = DiffApplier(Path(tmpdir))
        ok = applier.apply_file_edit(file_edit)
        assert ok, "Diff application failed"

        result = test_file.read_text()
        assert "hello(name" in result, f"Expected parameterized hello, got: {result[:100]}"
        assert 'f"hello {name}"' in result, f"Expected f-string, got: {result[:100]}"
        assert "goodbye" in result, "Expected goodbye function preserved"


def test_diff_parser_roundtrip():
    """Parse a unified diff and verify it can be generated back."""
    from src.composer.diff_engine import DiffGenerator, DiffParser

    original = textwrap.dedent("""\
        import os

        def main():
            x = 1
            y = 2
            return x + y
    """)

    modified = textwrap.dedent("""\
        import os
        import sys

        def main():
            x = 1
            y = 2
            z = 3
            return x + y + z
    """)

    file_edit = DiffGenerator.diff_content("main.py", original, modified)
    diff_text = file_edit.as_unified_diff()

    parsed = DiffParser.parse(diff_text)
    assert len(parsed) > 0, "Failed to parse generated diff"
    assert len(parsed[0].hunks) > 0, "Expected hunks in parsed diff"


# ---------------------------------------------------------------------------
# Test 3: TerminalSandbox
# ---------------------------------------------------------------------------


def test_sandbox_runs_basic_command():
    """Run a simple echo command."""
    from src.composer.terminal_sandbox import TerminalSandbox

    sandbox = TerminalSandbox()
    result = sandbox.run("echo hello world")
    assert result.success, f"Expected success, got exit {result.exit_code}"
    assert "hello world" in result.stdout, f"Expected 'hello world' in output, got: {result.stdout}"


def test_sandbox_timeout():
    """Verify timeout kills long-running commands."""
    from src.composer.terminal_sandbox import SandboxConfig, TerminalSandbox

    sandbox = TerminalSandbox()
    config = SandboxConfig(timeout_seconds=1)
    result = sandbox.run("sleep 10", config=config)
    assert result.was_timeout, f"Expected timeout, got exit {result.exit_code}"


def test_sandbox_deny_list():
    """Verify deny list blocks dangerous commands."""
    from src.composer.terminal_sandbox import SandboxConfig, TerminalSandbox

    sandbox = TerminalSandbox()
    config = SandboxConfig(deny_commands=["rm -rf /"])
    result = sandbox.run("rm -rf /tmp/test", config=config)
    assert result.exit_code == -1, f"Expected denial, got {result.exit_code}"
    assert "denied" in result.stderr.lower(), f"Expected denial message, got: {result.stderr}"


# ---------------------------------------------------------------------------
# Test 4: Context, planning, verification, checkpoint
# ---------------------------------------------------------------------------


def test_context_assembler_budgeted_context():
    from src.composer.codebase_indexer import CodebaseIndexer
    from src.composer.context_assembler import ContextAssembler, ContextBudget, ContextRequest

    repo = Path(__file__).resolve().parents[2]
    indexer = CodebaseIndexer(repo)
    indexer.build(force=True)
    assembler = ContextAssembler(repo, indexer)
    packet = assembler.assemble(
        ContextRequest(
            task="ComposerAgent diff edit verification",
            explicit_files=["src/composer/composer_agent.py"],
        ),
        ContextBudget(max_tokens=8_000, reserve_tokens=1_000, max_file_tokens=2_000),
    )
    assert packet.total_estimated_tokens <= 7_000
    assert any(item.file == "src/composer/composer_agent.py" for item in packet.items)
    rendered = packet.render()
    assert "Aurelius Composer Context" in rendered
    assert "composer_agent.py" in rendered


def test_multifile_orchestrator_orders_and_suggests_verifiers():
    from src.composer.codebase_indexer import CodebaseIndexer
    from src.composer.multifile_orchestrator import MultiFileEditOrchestrator

    repo = Path(__file__).resolve().parents[2]
    indexer = CodebaseIndexer(repo)
    indexer.build(force=True)
    planner = MultiFileEditOrchestrator(repo, indexer)
    plan = planner.plan(
        "harden composer",
        [
            "tests/composer/test_missing.py",
            "src/composer/composer_agent.py",
            "src/composer/context_assembler.py",
        ],
    )
    assert plan.ordered_files[0].startswith("src/"), plan.render()
    assert any("py_compile" in cmd for cmd in plan.verification_commands), (
        plan.verification_commands
    )
    assert sum(plan.risk_summary.values()) == len(plan.targets)


def test_edit_verifier_passes_and_fails_python_compile():
    from src.composer.edit_verifier import EditVerifier

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        good = root / "good.py"
        bad = root / "bad.py"
        good.write_text("def ok():\n    return 1\n")
        bad.write_text("def broken(:\n    pass\n")

        verifier = EditVerifier(root)
        good_result = verifier.verify(["good.py"])
        bad_result = verifier.verify(["bad.py"])
        assert good_result.success, good_result.render()
        assert not bad_result.success, bad_result.render()
        assert "py_compile" in bad_result.render()


def test_checkpoint_rollback_restores_file_and_deletes_new_file():
    from src.composer.checkpoint_rollback import CheckpointRollback

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        tracked = root / "tracked.py"
        created = root / "created.py"
        tracked.write_text("x = 1\n")

        checkpoints = CheckpointRollback(root, checkpoint_dir=root / ".checkpoints")
        cp = checkpoints.create(["tracked.py", "created.py"], description="test")
        tracked.write_text("x = 2\n")
        created.write_text("new = True\n")

        checkpoints.restore(cp.checkpoint_id)
        assert tracked.read_text() == "x = 1\n"
        assert not created.exists()


# ---------------------------------------------------------------------------
# Test 5: ComposerAgent
# ---------------------------------------------------------------------------


def test_composer_agent_initialization():
    """Verify ComposerAgent initializes with all subsystems."""
    from src.composer.composer_agent import AgentConfig, ComposerAgent

    repo = Path(__file__).resolve().parents[2]
    config = AgentConfig(max_turns=3, auto_verify=False, auto_lint=False, auto_test=False)
    agent = ComposerAgent(repo_root=repo, config=config)

    assert agent.repo_root == repo
    assert agent.indexer is not None
    assert agent.sandbox is not None
    assert agent.config.max_turns == 3


def test_composer_indexes_own_repo():
    """Build index on the Aurelius repo and verify basic search."""
    from src.composer.composer_agent import AgentConfig, ComposerAgent

    repo = Path(__file__).resolve().parents[2]
    config = AgentConfig(max_turns=1, auto_verify=False, auto_lint=False, auto_test=False)
    agent = ComposerAgent(repo_root=repo, config=config)

    hits = agent.indexer.search("transformer")
    assert len(hits) > 0, f"Expected to find transformer references, got {len(hits)}"


def test_composer_agent_fake_model_edit_loop():
    """Run a full agent loop with a deterministic fake model on a temp repo."""
    from src.composer.composer_agent import AgentConfig, ComposerAgent

    with tempfile.TemporaryDirectory() as tmpdir:
        repo = Path(tmpdir)
        (repo / "src").mkdir()
        target = repo / "src" / "demo.py"
        target.write_text("def value():\n    return 1\n")

        responses = iter(
            [
                "<action>explore</action><query>value function</query>",
                "<action>plan</action><plan>Change src/demo.py return value.</plan>",
                textwrap.dedent(
                    """
                    <action>edit</action>
                    <diffs>
                    --- a/src/demo.py
                    +++ b/src/demo.py
                    @@ -1,2 +1,2 @@
                     def value():
                    -    return 1
                    +    return 2
                    </diffs>
                    """
                ),
                "<action>complete</action><summary>Updated value.</summary>",
            ]
        )

        def fake_model(_: str) -> str:
            return next(responses)

        config = AgentConfig(max_turns=5, auto_lint=False, auto_test=False, model_fn=fake_model)
        agent = ComposerAgent(repo_root=repo, model_fn=fake_model, config=config)
        result = agent.run("change value to 2")
        assert result.success, result.final_state
        assert "return 2" in target.read_text()
        assert "src/demo.py" in result.files_changed


# ---------------------------------------------------------------------------
# Test 6: Integration — diff + apply + verify against real repo copy
# ---------------------------------------------------------------------------


def test_diff_apply_roundtrip_on_real_file():
    """Generate a diff against a real file, apply it, verify, and rollback."""
    from src.composer.diff_engine import DiffApplier, DiffGenerator

    repo = Path(__file__).resolve().parents[2]
    source_file = repo / "src" / "composer" / "__init__.py"
    original = source_file.read_text()

    with tempfile.TemporaryDirectory() as tmpdir:
        work = Path(tmpdir) / "work"
        work.mkdir()
        test_file = work / "__init__.py"
        test_file.write_text(original)

        modified = original.replace("Aurelius Composer", "Aurelius Composer v1")

        file_edit = DiffGenerator.diff_content("__init__.py", original, modified)
        assert len(file_edit.hunks) > 0, "Expected diff to contain changes"

        applier = DiffApplier(work)
        ok = applier.apply_file_edit(file_edit)
        assert ok, "Diff application to real file failed"

        result = test_file.read_text()
        assert "Aurelius Composer v1" in result, "Expected changed content"
        assert "CodebaseIndexer" in result, "Expected preserved content"

        applier.rollback()
        result = test_file.read_text()
        assert result == original, "Rollback should restore original content"
