#!/usr/bin/env python3
"""Broad-surface security audit runner (H-13).

Extends the audit beyond AMC memory only. This script aggregates:

1. The existing AMCSecurityAudit (in-tree adversarial memory tests)
2. Bandit static analysis (via subprocess — already baselined in bandit-baseline.json)
3. A directory-wide grep for common vulnerability patterns
   (secret-like strings, eval/exec, subprocess.shell=True, etc.)

Output: ``docs/reproducibility/results/broad_security_audit.json``

Note: Bandit is optional — if unavailable the scan is recorded as SKIPPED.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.security.amc_security_audit import AMCSecurityAudit  # noqa: E402

SCAN_DIRS = [_REPO_ROOT / "src", _REPO_ROOT / "gateway", _REPO_ROOT / "tools", _REPO_ROOT / "agent"]
IGNORE_DIRS = {"__pycache__", ".git", "node_modules", ".venv", "venv", "build", "dist"}

_SUSPICIOUS_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("hardcoded_secret_assignment", re.compile(r"""(?i)(?:api_key|secret|password|token)\s*=\s*['"][A-Za-z0-9_\-]{20,}['"]""")),
    ("eval_usage", re.compile(r"\beval\s*\(")),
    ("exec_usage", re.compile(r"\bexec\s*\(")),
    ("subprocess_shell_true", re.compile(r"subprocess\.(?:Popen|run|call)\s*\([^\)]*shell\s*=\s*True")),
    ("pickle_load", re.compile(r"\bpickle\.load\b")),
    ("yaml_unsafe_load", re.compile(r"\byaml\.load\s*\((?!.*Loader\s*=)")),
    ("torch_load_no_weights_only", re.compile(r"torch\.load\s*\([^)]*\)\s*(?!,\s*weights_only)")),
    ("os_system", re.compile(r"\bos\.system\s*\(")),
    ("md5_usage", re.compile(r"\bhashlib\.md5\b")),
]


@dataclass
class PatternFinding:
    pattern: str
    file: str
    line: int
    snippet: str


@dataclass
class BroadAuditResult:
    amc_audit: dict[str, object] = field(default_factory=dict)
    bandit_status: str = "SKIPPED"
    bandit_output: str = ""
    patterns_found: list[PatternFinding] = field(default_factory=list)
    scanned_files: int = 0
    skipped_dirs: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, object]:
        return {
            "amc_audit": self.amc_audit,
            "bandit_status": self.bandit_status,
            "bandit_output_excerpt": self.bandit_output[:1000],
            "patterns_found": [asdict(p) for p in self.patterns_found],
            "patterns_summary": self.summary_by_pattern(),
            "scanned_files": self.scanned_files,
            "skipped_dirs": self.skipped_dirs,
        }

    def summary_by_pattern(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for p in self.patterns_found:
            counts[p.pattern] = counts.get(p.pattern, 0) + 1
        return counts


def run_bandit() -> tuple[str, str]:
    """Return (status, stdout). SKIPPED if bandit not installed."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "bandit", "-r", "src", "gateway", "tools", "agent", "-q", "-f", "json"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            timeout=180,
        )
        return ("PASS" if result.returncode == 0 else "FAIL"), result.stdout + result.stderr
    except FileNotFoundError:
        return ("SKIPPED", "bandit not installed")
    except subprocess.TimeoutExpired:
        return ("TIMEOUT", "bandit >180s")
    except Exception as e:
        return ("ERROR", str(e))


def scan_patterns() -> tuple[list[PatternFinding], int, list[str]]:
    findings: list[PatternFinding] = []
    scanned = 0
    skipped: list[str] = []
    for target in SCAN_DIRS:
        if not target.exists():
            skipped.append(str(target))
            continue
        for path in target.rglob("*.py"):
            if any(part in IGNORE_DIRS for part in path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                skipped.append(str(path))
                continue
            scanned += 1
            for line_no, line in enumerate(text.splitlines(), start=1):
                # skip comments
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue
                for pattern_name, regex in _SUSPICIOUS_PATTERNS:
                    if regex.search(line):
                        findings.append(
                            PatternFinding(
                                pattern=pattern_name,
                                file=str(path.relative_to(_REPO_ROOT)),
                                line=line_no,
                                snippet=line.strip()[:200],
                            )
                        )
                        break  # one finding per line
    return findings, scanned, skipped


def main() -> int:
    result = BroadAuditResult()

    # 1. AMC audit
    amc = AMCSecurityAudit()
    amc.run_all()
    result.amc_audit = amc.to_json()

    # 2. Bandit
    result.bandit_status, result.bandit_output = run_bandit()

    # 3. Pattern scan
    result.patterns_found, result.scanned_files, result.skipped_dirs = scan_patterns()

    output = _REPO_ROOT / "docs/reproducibility/results/broad_security_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\n=== BROAD SECURITY AUDIT ===")
    print(f"  AMC audit:           {len([r for r in amc.results if r.status == 'PASS'])}/{len(amc.results)} PASS")
    print(f"  Bandit:              {result.bandit_status}")
    print(f"  Files scanned:       {result.scanned_files}")
    print(f"  Pattern findings:    {len(result.patterns_found)}")
    for k, v in result.summary_by_pattern().items():
        print(f"    - {k}: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
