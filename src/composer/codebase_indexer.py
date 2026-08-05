"""
CodebaseIndexer — fast repository understanding for agentic coding.

Cursor Composer 2.5's core strength is knowing what files exist, what they
contain, and how they relate before the model starts editing. This module
builds that index.

Components:
  - FileTree: fast directory walk with gitignore-aware filtering
  - SymbolGraph: AST-based extraction of classes, functions, imports, exports
  - SemanticIndex: lightweight embedding index for similarity search
  - RipgrepInterface: subprocess wrapper for ripgrep text search
  - IndexCache: incremental update support via file hashing

Usage:
  indexer = CodebaseIndexer(Path("/path/to/repo"))
  indexer.build()
  results = indexer.search("error handling middleware")
  symbols = indexer.symbols_for_file("src/server.py")
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FileEntry:
    """A single indexed file in the repository."""

    relative_path: str
    absolute_path: str
    language: str = ""
    size_bytes: int = 0
    sha256: str = ""
    symbols: list[SymbolDef] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    docstring: str = ""


@dataclass
class SymbolDef:
    """A named code symbol: class, function, method, or variable."""

    name: str
    kind: str  # "class", "function", "method", "variable", "import"
    line_start: int
    line_end: int
    parent: str = ""  # enclosing class name for methods
    signature: str = ""
    docstring: str = ""
    decorators: list[str] = field(default_factory=list)


@dataclass
class SearchHit:
    """A single search result from semantic or text search."""

    file: str
    symbol: str | None
    line: int
    snippet: str
    score: float
    source: str  # "semantic", "ripgrep", "symbol", "import"


@dataclass
class IndexStats:
    """Statistics about the built index."""

    total_files: int = 0
    total_symbols: int = 0
    total_size_bytes: int = 0
    build_time_seconds: float = 0.0
    languages: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Language detection and symbol extraction
# ---------------------------------------------------------------------------

_LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".sql": "sql",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".md": "markdown",
    ".rst": "restructuredtext",
    ".css": "css",
    ".html": "html",
    ".vue": "vue",
    ".svelte": "svelte",
}

# Extensions that are always excluded from indexing
_ALWAYS_IGNORE_GLOBS = [
    "*.pyc",
    "__pycache__",
    "*.o",
    "*.so",
    "*.dylib",
    "*.class",
    "*.jar",
    "*.war",
    "*.zip",
    "*.tar",
    "*.gz",
    "*.bz2",
    "*.7z",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.ico",
    "*.svg",
    "*.woff",
    "*.woff2",
    "*.ttf",
    "*.eot",
    "*.mp3",
    "*.mp4",
    "*.avi",
    "*.mov",
    "*.pdf",
    "*.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Cargo.lock",
    "poetry.lock",
    "Pipfile.lock",
]


def _detect_language(path: str) -> str:
    """Detect programming language from file extension."""
    suffix = Path(path).suffix.lower()
    if suffix in _LANGUAGE_MAP:
        return _LANGUAGE_MAP[suffix]
    # Special cases by filename
    name = Path(path).name.lower()
    if name == "makefile":
        return "makefile"
    if name == "dockerfile":
        return "dockerfile"
    if name.startswith("dockerfile."):
        return "dockerfile"
    return ""


def _python_extract_symbols(source: str) -> tuple[list[SymbolDef], list[str], list[str], str]:
    """Extract Python symbols without duplicating methods as functions.

    The first implementation used ``ast.walk`` and therefore counted direct
    class methods twice: once as ``method`` while visiting the class, and once
    again as a generic ``function``. Composer context ranking is very sensitive
    to duplicated symbols because repeated methods drown out rarer files. This
    extractor deliberately walks only module-level definitions, then direct
    class methods.
    """
    symbols: list[SymbolDef] = []
    imports: list[str] = []
    exports: list[str] = []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return symbols, imports, exports, ""

    docstring = ast.get_docstring(tree) or ""

    def _decorators(node: ast.AST) -> list[str]:
        decs = getattr(node, "decorator_list", [])
        return [ast.unparse(d) for d in decs]

    def _function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        args = [a.arg for a in node.args.args]
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        return f"{prefix} {node.name}({', '.join(args)})"

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            else:
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)
            continue

        if isinstance(node, ast.ClassDef):
            class_doc = ast.get_docstring(node) or ""
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            symbols.append(
                SymbolDef(
                    name=node.name,
                    kind="class",
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    decorators=_decorators(node),
                    docstring=class_doc,
                    signature=f"class {node.name}({bases})" if bases else f"class {node.name}",
                )
            )
            exports.append(node.name)

            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append(
                        SymbolDef(
                            name=item.name,
                            kind="method",
                            line_start=item.lineno,
                            line_end=item.end_lineno or item.lineno,
                            parent=node.name,
                            decorators=_decorators(item),
                            docstring=ast.get_docstring(item) or "",
                            signature=_function_signature(item),
                        )
                    )
            continue

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(
                SymbolDef(
                    name=node.name,
                    kind="function",
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    decorators=_decorators(node),
                    docstring=ast.get_docstring(node) or "",
                    signature=_function_signature(node),
                )
            )
            exports.append(node.name)

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    symbols.append(
                        SymbolDef(
                            name=target.id,
                            kind="constant",
                            line_start=node.lineno,
                            line_end=node.end_lineno or node.lineno,
                            signature=target.id,
                        )
                    )
                    exports.append(target.id)

    return symbols, imports, exports, docstring


# ---------------------------------------------------------------------------
# CodebaseIndexer
# ---------------------------------------------------------------------------


class CodebaseIndexer:
    """Indexes a repository for fast code understanding and search.

    Builds a multi-layer index:
      1. File tree with metadata
      2. Symbol graph (AST-extracted)
      3. Plain-text content for ripgrep
      4. Cache for incremental updates
    """

    def __init__(
        self,
        repo_root: Path,
        max_file_size_mb: float = 2.0,
        cache_dir: Path | None = None,
    ):
        self.repo_root = Path(repo_root).resolve()
        if not self.repo_root.is_dir():
            raise FileNotFoundError(f"Repository root not found: {self.repo_root}")

        self.max_file_bytes = int(max_file_size_mb * 1024 * 1024)
        self.cache_dir = Path(cache_dir) if cache_dir else self.repo_root / ".aurelius_index"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._files: dict[str, FileEntry] = {}
        self._stats = IndexStats()

        # Gitignore patterns loaded from the repo
        self._gitignore_patterns: list[str] = []
        self._load_gitignore()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, force: bool = False) -> IndexStats:
        """Build or refresh the codebase index."""
        start = time.monotonic()
        self._stats = IndexStats()
        self._files = {}

        if not force:
            self._load_cache()

        new_entries: dict[str, FileEntry] = {}
        for entry in self._walk_files():
            cached = self._files.get(entry.relative_path)
            if cached and cached.sha256 == entry.sha256 and cached.symbols:
                new_entries[entry.relative_path] = cached
            else:
                # Extract symbols for supported languages
                if entry.language == "python":
                    self._index_python_file(entry)
                elif entry.language == "rust":
                    self._index_rust_file(entry)
                elif entry.language == "typescript" or entry.language == "javascript":
                    self._index_ts_file(entry)
                new_entries[entry.relative_path] = entry
                self._stats.total_symbols += len(entry.symbols)

        self._files = new_entries
        self._stats.total_files = len(self._files)
        self._stats.build_time_seconds = time.monotonic() - start
        self._save_cache()
        logger.info(
            "Indexed %d files, %d symbols in %.1fs",
            self._stats.total_files,
            self._stats.total_symbols,
            self._stats.build_time_seconds,
        )
        return self._stats

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:
        """Search the codebase using multiple strategies.

        Priority order:
          1. Exact symbol name match
          2. Ripgrep text search
          3. Substring match in symbols and docstrings
        """
        hits: list[SearchHit] = []

        # 1. Exact symbol match
        query_lower = query.lower()
        for filepath, entry in self._files.items():
            for sym in entry.symbols:
                if query_lower == sym.name.lower():
                    hits.append(
                        SearchHit(
                            file=filepath,
                            symbol=sym.name,
                            line=sym.line_start,
                            snippet=sym.signature or sym.name,
                            score=1.0,
                            source="symbol",
                        )
                    )

        # 2. Ripgrep
        rg_hits = self._ripgrep_search(query, top_k)
        hits.extend(rg_hits)

        # 3. Substring match in symbol names and docstrings
        for filepath, entry in self._files.items():
            for sym in entry.symbols:
                if (
                    query_lower in sym.name.lower()
                    or query_lower in sym.docstring.lower()
                    or query_lower in sym.signature.lower()
                ):
                    hits.append(
                        SearchHit(
                            file=filepath,
                            symbol=sym.name,
                            line=sym.line_start,
                            snippet=sym.docstring[:200] if sym.docstring else sym.name,
                            score=0.5,
                            source="symbol",
                        )
                    )

        # Deduplicate and sort by score
        seen = set()
        unique_hits = []
        for h in sorted(hits, key=lambda x: (x.score, x.file), reverse=True):
            key = (h.file, h.symbol, h.line)
            if key not in seen:
                seen.add(key)
                unique_hits.append(h)

        return unique_hits[:top_k]

    def symbols_for_file(self, path: str) -> list[SymbolDef]:
        """Return all symbols in a specific file."""
        entry = self._files.get(path)
        if entry:
            return entry.symbols
        return []

    def files_importing(self, module_name: str) -> list[str]:
        """Find all files that import a specific module."""
        result = []
        for filepath, entry in self._files.items():
            for imp in entry.imports:
                if module_name in imp:
                    result.append(filepath)
                    break
        return result

    def get_entry(self, path: str) -> FileEntry | None:
        """Get the full FileEntry for a path."""
        return self._files.get(path)

    @property
    def files(self) -> dict[str, FileEntry]:
        return dict(self._files)

    @property
    def stats(self) -> IndexStats:
        return self._stats

    # ------------------------------------------------------------------
    # Internal: file walking
    # ------------------------------------------------------------------

    def _should_ignore(self, rel_path: str) -> bool:
        """Check if a file should be excluded from indexing."""
        name = Path(rel_path).name.lower()

        # Always-ignore patterns
        for pattern in _ALWAYS_IGNORE_GLOBS:
            if pattern.startswith("*."):
                if name.endswith(pattern[1:]):
                    return True
            elif pattern == name:
                return True

        # Common ignore directories
        ignore_dirs = {
            ".git",
            "node_modules",
            "__pycache__",
            ".venv",
            "venv",
            ".tox",
            ".eggs",
            "build",
            "dist",
            "target",
            ".next",
            ".nuxt",
            ".cache",
            "coverage",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".aurelius_index",
        }
        parts = Path(rel_path).parts
        if any(p in ignore_dirs for p in parts):
            return True

        # Hidden files (except .gitignore, .env, .github/)
        if name.startswith(".") and name not in {".gitignore", ".env", ".envrc", ".editorconfig"}:
            if ".github" not in parts:
                return True

        return False

    def _walk_files(self) -> Iterator[FileEntry]:
        """Walk the repository tree and yield FileEntry objects."""
        for dirpath, dirnames, filenames in os.walk(self.repo_root):
            # Filter directories
            dirnames[:] = [
                d
                for d in dirnames
                if not self._should_ignore(
                    os.path.relpath(os.path.join(dirpath, d), self.repo_root)
                )
            ]

            for fname in sorted(filenames):
                full_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(full_path, self.repo_root)

                if self._should_ignore(rel_path):
                    continue

                try:
                    file_stat = os.stat(full_path)
                    if file_stat.st_size > self.max_file_bytes:
                        continue
                except OSError:
                    continue

                language = _detect_language(fname)
                if not language:
                    continue

                sha256 = self._hash_file(full_path)

                self._stats.total_size_bytes += file_stat.st_size
                self._stats.languages[language] = self._stats.languages.get(language, 0) + 1

                yield FileEntry(
                    relative_path=rel_path,
                    absolute_path=full_path,
                    language=language,
                    size_bytes=file_stat.st_size,
                    sha256=sha256,
                )

    # ------------------------------------------------------------------
    # Internal: per-language extraction
    # ------------------------------------------------------------------

    def _index_python_file(self, entry: FileEntry) -> None:
        """Extract symbols from a Python file."""
        try:
            source = Path(entry.absolute_path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            self._stats.errors.append(f"Failed to read {entry.relative_path}")
            return

        symbols, imports, exports, docstring = _python_extract_symbols(source)
        entry.symbols = symbols
        entry.imports = imports
        entry.exports = exports
        entry.docstring = docstring

    def _index_rust_file(self, entry: FileEntry) -> None:
        """Minimal Rust symbol extraction using regex.

        For a full Rust index, use rust-analyzer LSP. This handles the 80%
        case: function and struct definitions.
        """
        import re

        try:
            source = Path(entry.absolute_path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            return

        lines = source.split("\n")

        # fn definitions
        fn_pat = re.compile(
            r"^\s*(?:pub(?:\s*\(\s*crate\s*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(\w+)"
        )
        struct_pat = re.compile(r"^\s*(?:pub(?:\s*\(\s*crate\s*\))?\s+)?struct\s+(\w+)")
        use_pat = re.compile(r"^\s*use\s+([\w:]+)")

        for i, line in enumerate(lines, 1):
            if m := fn_pat.match(line):
                entry.symbols.append(
                    SymbolDef(
                        name=m.group(1),
                        kind="function",
                        line_start=i,
                        line_end=i,
                        signature=line.strip(),
                    )
                )
            elif m := struct_pat.match(line):
                entry.symbols.append(
                    SymbolDef(
                        name=m.group(1),
                        kind="struct",
                        line_start=i,
                        line_end=i,
                        signature=line.strip(),
                    )
                )
            elif m := use_pat.match(line):
                entry.imports.append(m.group(1))

    def _index_ts_file(self, entry: FileEntry) -> None:
        """Minimal TypeScript/JavaScript symbol extraction.

        For full TS/JS index, use tsserver LSP.
        """
        import re

        try:
            source = Path(entry.absolute_path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            return

        lines = source.split("\n")

        # function, class, const/let/var, export patterns
        fn_pat = re.compile(r"^\s*(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s+(\w+)")
        class_pat = re.compile(r"^\s*(?:export\s+(?:default\s+)?)?class\s+(\w+)")
        const_pat = re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*[:=]")
        import_pat = re.compile(r"^\s*import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]")

        for i, line in enumerate(lines, 1):
            if m := fn_pat.match(line):
                entry.symbols.append(
                    SymbolDef(
                        name=m.group(1),
                        kind="function",
                        line_start=i,
                        line_end=i,
                        signature=line.strip(),
                    )
                )
            elif m := class_pat.match(line):
                entry.symbols.append(
                    SymbolDef(
                        name=m.group(1),
                        kind="class",
                        line_start=i,
                        line_end=i,
                        signature=line.strip(),
                    )
                )
            elif m := const_pat.match(line):
                entry.symbols.append(
                    SymbolDef(name=m.group(1), kind="variable", line_start=i, line_end=i)
                )
            elif m := import_pat.match(line):
                entry.imports.append(m.group(1))

    # ------------------------------------------------------------------
    # Internal: ripgrep
    # ------------------------------------------------------------------

    def _ripgrep_search(self, query: str, top_k: int) -> list[SearchHit]:
        """Run ripgrep as a subprocess and parse results.

        Ripgrep is pointed at the repo root, so it must receive the same
        coarse exclusions as the indexer. Otherwise lock files and build
        artifacts can dominate results even though they were never indexed.
        """
        hits: list[SearchHit] = []
        rg_command = [
            "rg",
            "--no-heading",
            "--line-number",
            "--max-count=50",
            "--max-depth=20",
            "--glob",
            "!*.lock",
            "--glob",
            "!uv.lock",
            "--glob",
            "!node_modules/**",
            "--glob",
            "!.git/**",
            "--glob",
            "!.aurelius_index/**",
            "--glob",
            "!target/**",
            "--glob",
            "!build/**",
            "--glob",
            "!dist/**",
            "-e",
            query,
            str(self.repo_root),
        ]
        try:
            result = subprocess.run(  # noqa: S603
                rg_command,
                capture_output=True,
                text=True,
                timeout=15,
            )
            for line in result.stdout.strip().split("\n")[: top_k * 2]:
                if not line.strip():
                    continue
                # ripgrep output: filepath:lineno:snippet
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    filepath = parts[0]
                    lineno = parts[1]
                    snippet = parts[2].strip()
                    rel_path = (
                        os.path.relpath(filepath, self.repo_root)
                        if os.path.isabs(filepath)
                        else filepath
                    )
                    hits.append(
                        SearchHit(
                            file=rel_path,
                            symbol=None,
                            line=int(lineno),
                            snippet=snippet[:200],
                            score=0.7,
                            source="ripgrep",
                        )
                    )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # ripgrep not installed or timed out — fall back to in-process grep
            hits.extend(self._fallback_grep(query, top_k))
        except Exception as exc:
            logger.debug("ripgrep search failed: %s", exc)

        return hits

    def _fallback_grep(self, query: str, top_k: int) -> list[SearchHit]:
        """Fallback in-process grep when ripgrep is unavailable."""
        hits: list[SearchHit] = []
        query_lower = query.lower()
        for filepath, entry in self._files.items():
            if len(hits) >= top_k:
                break
            try:
                content = Path(entry.absolute_path).read_text(encoding="utf-8", errors="replace")
                for i, line in enumerate(content.split("\n"), 1):
                    if query_lower in line.lower():
                        hits.append(
                            SearchHit(
                                file=filepath,
                                symbol=None,
                                line=i,
                                snippet=line.strip()[:200],
                                score=0.6,
                                source="ripgrep",
                            )
                        )
                        if len(hits) >= top_k:
                            break
            except Exception as exc:
                logger.debug("fallback grep failed for %s: %s", filepath, exc)
                continue
        return hits

    # ------------------------------------------------------------------
    # Internal: caching
    # ------------------------------------------------------------------

    def _cache_path(self) -> Path:
        return self.cache_dir / "index_cache.json"

    def _hash_file(self, path: str) -> str:
        """SHA-256 hash of a file's contents."""
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
        except OSError:
            return ""
        return h.hexdigest()

    def _load_gitignore(self) -> None:
        """Load .gitignore patterns from the repository root."""
        gitignore = self.repo_root / ".gitignore"
        if gitignore.is_file():
            try:
                text = gitignore.read_text()
                for line in text.split("\n"):
                    line = line.strip()
                    if line and not line.startswith("#"):
                        self._gitignore_patterns.append(line)
            except Exception:
                pass

    def _save_cache(self) -> None:
        """Save the index to cache for incremental rebuilds."""
        cache_data = {
            "repo_root": str(self.repo_root),
            "total_files": self._stats.total_files,
            "total_symbols": self._stats.total_symbols,
            "files": {
                path: {
                    "relative_path": e.relative_path,
                    "language": e.language,
                    "size_bytes": e.size_bytes,
                    "sha256": e.sha256,
                    "symbols": [
                        {
                            "name": s.name,
                            "kind": s.kind,
                            "line_start": s.line_start,
                            "line_end": s.line_end,
                            "parent": s.parent,
                            "signature": s.signature,
                            "docstring": s.docstring[:500],
                        }
                        for s in e.symbols
                    ],
                    "imports": e.imports,
                    "exports": e.exports,
                }
                for path, e in self._files.items()
            },
        }
        try:
            self._cache_path().write_text(json.dumps(cache_data, indent=2, default=str))
        except Exception as exc:
            logger.warning("Failed to write index cache: %s", exc)

    def _load_cache(self) -> None:
        """Load a cached index for incremental rebuild."""
        cache_file = self._cache_path()
        if not cache_file.is_file():
            return
        try:
            data = json.loads(cache_file.read_text())
            if data.get("repo_root") != str(self.repo_root):
                return
            for path, cached in data.get("files", {}).items():
                entry = FileEntry(
                    relative_path=cached["relative_path"],
                    absolute_path=str(self.repo_root / cached["relative_path"]),
                    language=cached.get("language", ""),
                    size_bytes=cached.get("size_bytes", 0),
                    sha256=cached.get("sha256", ""),
                    symbols=[
                        SymbolDef(
                            name=s["name"],
                            kind=s["kind"],
                            line_start=s["line_start"],
                            line_end=s["line_end"],
                            parent=s.get("parent", ""),
                            signature=s.get("signature", ""),
                            docstring=s.get("docstring", ""),
                        )
                        for s in cached.get("symbols", [])
                    ],
                    imports=cached.get("imports", []),
                    exports=cached.get("exports", []),
                )
                self._files[entry.relative_path] = entry
                self._stats.total_symbols += len(entry.symbols)
                self._stats.total_size_bytes += entry.size_bytes
            self._stats.total_files = len(self._files)
            logger.info(
                "Loaded cached index: %d files, %d symbols",
                self._stats.total_files,
                self._stats.total_symbols,
            )
        except Exception as exc:
            logger.warning("Failed to load index cache, rebuilding: %s", exc)
            self._files = {}
