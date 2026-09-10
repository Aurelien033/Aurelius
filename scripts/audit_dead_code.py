#!/usr/bin/env python3
"""Static dead-code reachability audit for a set of target directories.

Method:
  1. Parse imports (``ast``) from every tracked ``.py`` file.
  2. Resolve each import to a tracked file, including:
       - absolute imports (``src.model.x``) and ``from X import sub`` edges;
       - legacy namespace aliases registered via ``src.namespace_aliases``
         (``aurelius.model`` / ``model`` -> ``src.model``; same for
         ``alignment`` and ``serving``);
       - relative imports.
  3. BFS from every tracked ``.py`` outside the target dirs; anything in the
     target dirs that is not reached has no static importer.
  4. Exonerate candidates by a word-boundary scan across code/config files
     (``.py``, ``.yml``, ``.yaml``, ``.toml``, ``.cfg``, ``.ini``, ``.sh``,
     ``.ipynb``, ``.sql``, ``.json``, ``Makefile``, ``Dockerfile``) to catch
     dynamic references (``pytest.importorskip``, ``importlib``, string refs).
     Prose docs and scan artifacts are excluded on purpose.

Output: human-readable report; ``--json PATH`` also writes the full lists.

Usage:
    python scripts/audit_dead_code.py --targets src/model src/training
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from collections import deque
from pathlib import Path

# canonical prefix -> legacy aliases (see register_namespace_aliases calls)
ALIASES: list[tuple[str, str]] = [
    ("aurelius.model", "src.model"),
    ("model", "src.model"),
    ("aurelius.alignment", "src.alignment"),
    ("alignment", "src.alignment"),
    ("aurelius.serving", "src.serving"),
    ("serving", "src.serving"),
]

EXO_EXTS = (".py", ".yml", ".yaml", ".toml", ".cfg", ".ini", ".sh", ".ipynb", ".sql", ".json")
EXO_NAMES = ("Makefile", "Dockerfile")
EXO_EXCLUDE_PREFIXES = ("docs/", ".hermes/", "archive/")
EXO_EXCLUDE_SUBSTRS = ("bandit-baseline", "code_review", "CODE_REVIEW")


def git_ls(root: Path, pattern: str) -> list[str]:
    return subprocess.run(
        ["git", "ls-files", pattern], cwd=root, capture_output=True, text=True, check=True
    ).stdout.splitlines()


def module_candidates(root: Path, parts: list[str]) -> list[Path]:
    base = root.joinpath(*parts)
    return [base.with_suffix(".py"), base / "__init__.py"]


def parse_imports(text: str) -> list[tuple[int, str, str | None]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    out: list[tuple[int, str, str | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append((0, alias.name, None))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                out.append((node.level or 0, node.module or "", alias.name))
    return out


def resolve(root: Path, importer: Path, level: int, module: str, name: str | None) -> list[str]:
    cands: list[Path] = []

    def add(parts: list[str]) -> None:
        cands.extend(module_candidates(root, parts))

    if level == 0:
        parts = module.split(".") if module else []
        if parts:
            add(parts)
            for alias, canon in ALIASES:
                if module == alias:
                    add(canon.split("."))
                elif module.startswith(alias + "."):
                    rest = module[len(alias) + 1 :]
                    add(canon.split(".") + rest.split("."))
            if parts[0] not in ("src", "aurelius", "agent", "plugins", "tools", "gateway", "cron"):
                add(["src", *parts])
            if name and name != "*":
                add(parts + [name])
                for alias, canon in ALIASES:
                    if module == alias:
                        add(canon.split(".") + [name])
                    elif module.startswith(alias + "."):
                        rest = module[len(alias) + 1 :]
                        add(canon.split(".") + rest.split(".") + [name])
    else:
        base = importer.parent
        for _ in range(level - 1):
            base = base.parent
        rel = list(base.relative_to(root).parts)
        if module:
            add(rel + module.split("."))
        else:
            cands.append(base / "__init__.py")
        if name and name != "*":
            add(rel + (module.split(".") if module else []) + [name])

    out: list[str] = []
    for cand in cands:
        try:
            out.append(cand.relative_to(root).as_posix())
        except ValueError:
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", nargs="+", default=["src/model", "src/training"])
    ap.add_argument("--root", default=".")
    ap.add_argument("--json", dest="json_path", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    targets = [t.rstrip("/") for t in args.targets]

    def in_target(f: str) -> bool:
        return any(f.startswith(t + "/") for t in targets)

    fnames = git_ls(root, "*.py")
    nodes = set(fnames)
    edges: dict[str, set[str]] = {f: set() for f in fnames}
    for f in fnames:
        text = (root / f).read_text(encoding="utf-8", errors="replace")
        for level, module, name in parse_imports(text):
            for rel in resolve(root, root / f, level, module, name):
                if rel in nodes and rel != f:
                    edges[f].add(rel)

    seeds = [f for f in nodes if not in_target(f)]
    seen = set(seeds)
    q: deque[str] = deque(seeds)
    while q:
        cur = q.popleft()
        for nxt in edges[cur]:
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    candidates = sorted(f for f in nodes if f not in seen and in_target(f))

    exo: set[str] = set()
    for ext in EXO_EXTS:
        exo.update(git_ls(root, f"*{ext}"))
    for nm in EXO_NAMES:
        exo.update(git_ls(root, nm))
    exo = {
        f
        for f in exo
        if not f.startswith(EXO_EXCLUDE_PREFIXES)
        and not any(s in f for s in EXO_EXCLUDE_SUBSTRS)
        and f not in candidates
    }
    cache: dict[str, str] = {}
    for f in sorted(exo):
        p = root / f
        try:
            if p.stat().st_size > 2_000_000:
                continue
            cache[f] = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

    exempt: dict[str, list[str]] = {}
    for c in candidates:
        pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(Path(c).stem) + r"(?![A-Za-z0-9_])")
        hits = [f for f, text in cache.items() if pat.search(text)]
        if hits:
            exempt[c] = hits[:6]

    survivors = [c for c in candidates if c not in exempt]

    print(f"targets: {', '.join(targets)}")
    print(f"tracked .py: {len(fnames)} | in targets: {sum(1 for f in fnames if in_target(f))}")
    print(f"no static importer: {len(candidates)}")
    print(f"dynamically referenced (kept): {len(exempt)}")
    print(f"zero-reference (delete candidates): {len(survivors)}")
    print("\n-- zero-reference --")
    for s in survivors:
        print(f"  {s}")
    print("\n-- dynamically referenced --")
    for k, refs in exempt.items():
        print(f"  {k}  <- {', '.join(refs)}")

    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(
                {
                    "counts": {
                        "tracked": len(fnames),
                        "no_static_importer": len(candidates),
                        "dynamic": len(exempt),
                        "zero_ref": len(survivors),
                    },
                    "zero_reference": survivors,
                    "dynamically_referenced": exempt,
                },
                indent=1,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
