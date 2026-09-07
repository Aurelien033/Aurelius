#!/usr/bin/env python3
"""Analyze local Aurelius research files for reusable external resources.

This is intentionally local-first. It does not claim a resource is current or
correct on the web; it reports what the existing Aurelius research corpus cites,
then ranks candidates by how useful they look for the AGI-track Aurelius roadmap.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

TODAY = date.today().isoformat()
HOME = Path.home()
REPO = HOME / "aurelius"
OUTPUT_DIR = HOME / "Desktop" / "AI Plans" / f"aurelius-research-resource-audit-{TODAY}"

ROOTS = [
    REPO / "docs",
    REPO / "research",
    HOME / "Desktop" / "AI:ML Research",
    HOME / "Desktop" / "AI:ML Research II",
    HOME / "Desktop" / "AI Plans",
    HOME / "Desktop" / "Aurelius Research Deliverables",
]

TEXT_EXTS = {
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".csv",
    ".rst",
    ".tex",
}

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".aurelius_index",
    ".aurelius_checkpoints",
    "target",
    "build",
    "dist",
}

URL_RE = re.compile(r"https?://[^\s)\]}>\"']+")
ARXIV_RE = re.compile(
    r"(?:arxiv(?:\.org)?[:/ ]*(?:abs/|pdf/)?)?\b(\d{4}\.\d{4,5})(?:v\d+)?\b",
    re.IGNORECASE,
)
GITHUB_RE = re.compile(r"github\.com/([^/\s)\]}>\"']+)/([^/\s)\]}>\"']+)", re.IGNORECASE)
HF_RE = re.compile(
    r"huggingface\.co/(?:datasets/|spaces/)?([^/\s)\]}>\"']+/[^/\s)\]}>\"']+)",
    re.IGNORECASE,
)
LOCAL_PATH_RE = re.compile(r"/(?:Users|Volumes)/[^\s)\]}>\"']+")

CATEGORY_KEYWORDS = {
    "agi_cognition": [
        "agi",
        "cognitive",
        "world model",
        "metacognition",
        "reasoning",
        "thinking",
        "neuroscience",
        "cortex",
        "executive",
        "planning",
        "self-model",
        "global workspace",
    ],
    "composer_agent": [
        "composer",
        "cursor",
        "agent",
        "tool",
        "terminal",
        "diff",
        "edit",
        "swe",
        "repair",
        "codebase",
        "sandbox",
        "rollback",
        "workflow",
    ],
    "rlvr_verifier": [
        "rlvr",
        "grpo",
        "verifier",
        "verification",
        "reward",
        "process reward",
        "prm",
        "test pass",
        "execution",
        "fractional reward",
        "repair trace",
    ],
    "memory_amc": [
        "amc",
        "memory",
        "dreambank",
        "sdb",
        "cache",
        "ledger",
        "long-term",
        "skillnode",
        "semantic",
        "prefix",
        "event-sourced",
    ],
    "inference_efficiency": [
        "vllm",
        "sglang",
        "flashinfer",
        "flashattention",
        "kv cache",
        "quant",
        "cuda",
        "tensorrt",
        "mlx",
        "speculative",
        "paged attention",
        "latency",
        "throughput",
    ],
    "eval_safety": [
        "eval",
        "benchmark",
        "safety",
        "red team",
        "guard",
        "contamination",
        "audit",
        "falsifier",
        "kill gate",
        "model card",
    ],
    "data_training": [
        "dataset",
        "data",
        "synthetic",
        "sft",
        "dpo",
        "distill",
        "teacher",
        "curriculum",
        "flywheel",
        "pretrain",
        "huggingface",
    ],
}

POSITIVE_HINTS = [
    "recommend",
    "should",
    "must",
    "use",
    "adopt",
    "implement",
    "next",
    "highest",
    "priority",
    "promote",
    "assist",
    "useful",
    "winner",
    "proven",
    "verified",
    "evidence",
]
NEGATIVE_HINTS = [
    "kill",
    "killed",
    "falsified",
    "dead",
    "do not",
    "skip",
    "unsafe",
    "stale",
    "not work",
    "failed",
    "worse",
    "harm",
    "blocked",
]
RESOURCE_LINE_HINTS = [
    "http",
    "arxiv",
    "github",
    "huggingface",
    "paper",
    "repo",
    "dataset",
    "benchmark",
    "library",
    "framework",
    "tool",
    "reference",
    "model",
]


@dataclass
class FileRecord:
    path: str
    root: str
    size_bytes: int
    sha256_headtail: str
    line_count: int
    include_reason: str
    categories: dict[str, int] = field(default_factory=dict)
    resource_lines: int = 0


@dataclass
class ResourceRecord:
    key: str
    kind: str
    value: str
    score: float = 0.0
    files: list[str] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    categories: dict[str, int] = field(default_factory=dict)
    positive_hits: int = 0
    negative_hits: int = 0


def iter_candidate_files() -> Iterable[Path]:
    for root in ROOTS:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d
                for d in dirnames
                if d not in SKIP_DIRS and not d.startswith("aurelius-research-resource-audit-")
            ]
            base = Path(dirpath)
            for name in filenames:
                path = base / name
                if path.suffix.lower() not in TEXT_EXTS:
                    continue
                if path.name.startswith("."):
                    continue
                yield path


def read_text_window(path: Path, max_chars: int = 900_000) -> str:
    data = path.read_bytes()
    if len(data) <= max_chars:
        window = data
    else:
        half = max_chars // 2
        window = data[:half] + b"\n\n... [middle omitted by scanner] ...\n\n" + data[-half:]
    return window.replace(b"\x00", b"").decode("utf-8", errors="replace")


def headtail_sha(path: Path, window: int = 128_000) -> str:
    data = path.read_bytes()
    if len(data) <= window * 2:
        payload = data
    else:
        payload = data[:window] + data[-window:]
    return hashlib.sha256(payload).hexdigest()


def include_file(path: Path, text: str) -> tuple[bool, str]:
    lowered_path = str(path).lower()
    lowered_head = text[:80_000].lower()
    if "/aurelius/docs/" in lowered_path or "/aurelius/research/" in lowered_path:
        return True, "repo-doc-or-research"
    if "aurelius research deliverables" in lowered_path:
        return True, "desktop-aurelius-deliverables"
    if "aurelius" in lowered_path:
        return True, "path-contains-aurelius"
    if "aurelius" in lowered_head:
        return True, "content-contains-aurelius"
    return False, "not-aurelius-scoped"


def category_counts(text: str) -> Counter[str]:
    lower = text.lower()
    counts: Counter[str] = Counter()
    for cat, kws in CATEGORY_KEYWORDS.items():
        for kw in kws:
            counts[cat] += lower.count(kw)
    return counts


def clean_url(url: str) -> str:
    value = url.strip().strip("`'\"<>[](){}")
    value = value.rstrip(".,;:")
    value = value.replace("/pdf/", "/abs/")
    value = re.sub(r"(https?://arxiv\.org/abs/)(\d{4}\.\d{4,5})(v\d+)?", r"\1\2", value)
    value = re.sub(r"(https?://arxiv\.org/abs/)(\d{4}\.\d{4,5})\.pdf", r"\1\2", value)
    return value


def is_valid_resource(value: str) -> bool:
    lower = value.lower().strip()
    if not lower:
        return False
    if lower in {
        "https://github",
        "http://github",
        "https://github.com",
        "https://github.com/",
        "http://github.com/",
    }:
        return False
    if lower in {"https://arxiv.org/abs", "https://arxiv.org/abs/"}:
        return False
    if lower.startswith("http://127.0.0.1") or lower.startswith("http://localhost"):
        return False
    if "github.com/" in lower:
        match = GITHUB_RE.search(value)
        if not match:
            return False
        owner, repo = match.group(1), match.group(2).strip("`.,;:")
        if not owner or not repo or repo in {"blob", "tree", "issues", "pulls"}:
            return False
    if "arxiv.org/abs/" in lower and not re.search(r"arxiv\.org/abs/\d{4}\.\d{4,5}", lower):
        return False
    if (
        lower.startswith("/users/christienantonio/aurelius")
        and lower.rstrip("/`") == "/users/christienantonio/aurelius"
    ):
        return False
    if (
        lower.startswith("/users/christienantonio/desktop/aurelius")
        and lower.rstrip("/`") == "/users/christienantonio/desktop/aurelius"
    ):
        return False
    return True


def resource_kind(value: str) -> str:
    v = value.lower()
    if "github.com/" in v:
        return "github"
    if "huggingface.co/" in v:
        if "/datasets/" in v:
            return "hf_dataset"
        if "/spaces/" in v:
            return "hf_space"
        return "hf_model_or_repo"
    if "arxiv" in v or ARXIV_RE.fullmatch(v):
        return "arxiv"
    if "paperswithcode" in v:
        return "paperswithcode"
    if "openreview" in v:
        return "openreview"
    if "githubusercontent" in v:
        return "code_raw"
    if v.startswith("/users/") or v.startswith("/volumes/"):
        return "local_path"
    return "url"


def line_context(lines: list[str], idx: int) -> str:
    start = max(0, idx - 1)
    end = min(len(lines), idx + 2)
    joined = " / ".join(line.strip() for line in lines[start:end] if line.strip())
    return joined[:700]


def add_resource(
    resources: dict[str, ResourceRecord],
    raw: str,
    file_path: Path,
    context: str,
    cats: Counter[str],
) -> None:
    value = clean_url(raw)
    if not is_valid_resource(value):
        return
    kind = resource_kind(value)
    key = f"{kind}:{value.lower()}"
    rec = resources.get(key)
    if rec is None:
        rec = ResourceRecord(key=key, kind=kind, value=value)
        resources[key] = rec
    rel_file = str(file_path)
    if rel_file not in rec.files:
        rec.files.append(rel_file)
    if context and context not in rec.contexts and len(rec.contexts) < 8:
        rec.contexts.append(context)
    for cat, count in cats.items():
        if count:
            rec.categories[cat] = rec.categories.get(cat, 0) + min(count, 5)
    lower = context.lower()
    rec.positive_hits += sum(lower.count(h) for h in POSITIVE_HINTS)
    rec.negative_hits += sum(lower.count(h) for h in NEGATIVE_HINTS)


def score_resource(rec: ResourceRecord) -> float:
    kind_weight = {
        "github": 7.0,
        "hf_dataset": 6.5,
        "hf_model_or_repo": 6.0,
        "arxiv": 5.5,
        "openreview": 5.0,
        "paperswithcode": 5.0,
        "code_raw": 4.5,
        "local_path": 4.0,
        "url": 2.5,
    }.get(rec.kind, 2.0)
    file_weight = min(10.0, 1.5 * math.log2(1 + len(rec.files)))
    context_weight = min(8.0, 0.8 * len(rec.contexts))
    cat_weight = 0.0
    for cat, count in rec.categories.items():
        multiplier = {
            "agi_cognition": 1.4,
            "composer_agent": 1.3,
            "rlvr_verifier": 1.4,
            "memory_amc": 1.3,
            "inference_efficiency": 1.1,
            "eval_safety": 1.2,
            "data_training": 1.25,
        }.get(cat, 1.0)
        cat_weight += min(7.0, count * 0.12 * multiplier)
    polarity = min(8.0, rec.positive_hits * 0.35) - min(10.0, rec.negative_hits * 0.8)
    return round(kind_weight + file_weight + context_weight + cat_weight + polarity, 3)


def category_label(rec: ResourceRecord) -> str:
    if not rec.categories:
        return "unclassified"
    return max(rec.categories.items(), key=lambda kv: kv[1])[0]


def summarize_file(path: Path, text: str, root: Path, include_reason: str) -> FileRecord:
    cats = category_counts(text)
    line_count = text.count("\n") + 1 if text else 0
    return FileRecord(
        path=str(path),
        root=str(root),
        size_bytes=path.stat().st_size,
        sha256_headtail=headtail_sha(path),
        line_count=line_count,
        include_reason=include_reason,
        categories=dict(cats),
        resource_lines=0,
    )


def top_category_counts(files: list[FileRecord]) -> dict[str, int]:
    total: Counter[str] = Counter()
    for f in files:
        for cat, count in f.categories.items():
            total[cat] += count
    return dict(total.most_common())


def resource_to_row(rec: ResourceRecord) -> dict[str, object]:
    return {
        "score": rec.score,
        "kind": rec.kind,
        "value": rec.value,
        "primary_category": category_label(rec),
        "file_count": len(rec.files),
        "positive_hits": rec.positive_hits,
        "negative_hits": rec.negative_hits,
        "files": " | ".join(rec.files[:8]),
        "contexts": " || ".join(rec.contexts[:4]),
    }


def write_csv(path: Path, resources: list[ResourceRecord]) -> None:
    fields = [
        "score",
        "kind",
        "value",
        "primary_category",
        "file_count",
        "positive_hits",
        "negative_hits",
        "files",
        "contexts",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in resources:
            writer.writerow(resource_to_row(rec))


def write_report(
    path: Path,
    files: list[FileRecord],
    resources: list[ResourceRecord],
    skipped: list[dict[str, str]],
) -> None:
    by_cat: dict[str, list[ResourceRecord]] = defaultdict(list)
    for rec in resources:
        by_cat[category_label(rec)].append(rec)
    for items in by_cat.values():
        items.sort(key=lambda r: r.score, reverse=True)

    negative = [r for r in resources if r.negative_hits > r.positive_hits and r.score > 0]
    negative.sort(key=lambda r: (r.negative_hits, r.score), reverse=True)

    lines: list[str] = []
    lines.append("# Aurelius Research Resource Audit")
    lines.append("")
    lines.append(f"Generated: {TODAY}")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("Local roots scanned:")
    for root in ROOTS:
        lines.append(f"- `{root}`")
    lines.append("")
    lines.append(
        "This audit extracts resources already cited or implied inside local Aurelius research files. It does not claim web freshness; web validation should be a follow-up for the highest-priority resources."
    )
    lines.append("")
    lines.append("## Corpus summary")
    lines.append("")
    lines.append(f"- Candidate text files included: {len(files)}")
    lines.append(f"- Resources deduplicated: {len(resources)}")
    lines.append(f"- Files skipped/errors: {len(skipped)}")
    lines.append(f"- Total included bytes: {sum(f.size_bytes for f in files):,}")
    lines.append(f"- Total included lines: {sum(f.line_count for f in files):,}")
    lines.append("")
    lines.append("### Category density across included files")
    lines.append("")
    lines.append("| Category | Hits |")
    lines.append("|---|---:|")
    for cat, count in top_category_counts(files).items():
        lines.append(f"| {cat} | {count:,} |")
    lines.append("")

    lines.append("## Highest-priority resource clusters")
    lines.append("")
    lines.append(
        "These are the resources most repeatedly connected to Aurelius AGI-track work inside the existing corpus."
    )
    lines.append("")
    lines.append("| Rank | Score | Type | Category | Resource | Evidence files |")
    lines.append("|---:|---:|---|---|---|---:|")
    for i, rec in enumerate(resources[:40], 1):
        value = rec.value.replace("|", "%7C")
        lines.append(
            f"| {i} | {rec.score:.2f} | {rec.kind} | {category_label(rec)} | `{value}` | {len(rec.files)} |"
        )
    lines.append("")

    lines.append("## What this adds to Project Aurelius")
    lines.append("")
    lines.append("### 1. AGI/cognition resources")
    lines.append("")
    lines.append(
        "Use these to keep Aurelius from narrowing into only a coding assistant. They should feed the cognitive architecture, world-model, metacognition, and long-horizon planning tracks."
    )
    write_category(lines, by_cat, "agi_cognition")

    lines.append("### 2. Cursor Composer-like agent resources")
    lines.append("")
    lines.append(
        "Use these for the code-action environment: repo search, context assembly, diff application, verifier loop, rollback, repair traces, and SWE-style evaluation."
    )
    write_category(lines, by_cat, "composer_agent")

    lines.append("### 3. RLVR / verifier / repair resources")
    lines.append("")
    lines.append(
        "Use these for the learning flywheel. This is the highest-leverage path from code agent to AGI-track system because it creates grounded reward rather than imitation-only behavior."
    )
    write_category(lines, by_cat, "rlvr_verifier")

    lines.append("### 4. Memory / AMC / developmental state resources")
    lines.append("")
    lines.append(
        "Use these to make Aurelius persistent and developmental: event tape, memory admission, proof-carrying memory, DreamBank, SkillNode, and semantic addressability."
    )
    write_category(lines, by_cat, "memory_amc")

    lines.append("### 5. Inference / efficiency resources")
    lines.append("")
    lines.append(
        "Use these to make the system runnable locally and release-ready without pretending speculative mechanisms are proven before measurement."
    )
    write_category(lines, by_cat, "inference_efficiency")

    lines.append("### 6. Evaluation / safety resources")
    lines.append("")
    lines.append(
        "Use these for the evidence moat: contamination checks, model cards, red-team gates, falsifier contracts, and benchmark discipline."
    )
    write_category(lines, by_cat, "eval_safety")

    lines.append("### 7. Data / training resources")
    lines.append("")
    lines.append(
        "Use these to build the trace dataset and training curriculum: verified traces, repair pairs, teacher-filtered examples, and synthetic schema-only bootstraps."
    )
    write_category(lines, by_cat, "data_training")

    lines.append("## Caution list: resources mentioned in negative/falsified contexts")
    lines.append("")
    lines.append(
        "Do not treat these as immediate build targets without revalidation. The scanner found more negative than positive context around them."
    )
    lines.append("")
    lines.append("| Resource | Negative hits | Positive hits | Context |")
    lines.append("|---|---:|---:|---|")
    for rec in negative[:30]:
        context = (rec.contexts[0] if rec.contexts else "").replace("|", "/")
        lines.append(
            f"| `{rec.value}` | {rec.negative_hits} | {rec.positive_hits} | {context[:220]} |"
        )
    lines.append("")

    lines.append("## Recommended next-resource actions")
    lines.append("")
    lines.append(
        "1. Build a formal `Aurelius Resource Registry` from `resources.csv`, not just prose notes. Add status fields: `validated`, `implemented`, `research-gated`, `killed`, `needs-web-check`."
    )
    lines.append(
        "2. Web-validate the top 25 resources only. Do not spend time validating low-score or killed resources."
    )
    lines.append(
        "3. Promote resources into six active lanes: Composer/agent environment, verifier/RLVR, AGI cognition, AMC memory, inference efficiency, eval/safety."
    )
    lines.append(
        "4. For each promoted resource, require an ACDT cell: mechanism, integration hook, verifier, falsifier, cost, and kill gate."
    )
    lines.append(
        "5. Treat code/math tasks as the first AGI training arena, but reserve a separate cognition/memory lane so Aurelius does not collapse into a narrow coding model."
    )
    lines.append("")

    lines.append("## Files with densest useful-resource signal")
    lines.append("")
    dense_files = sorted(
        files,
        key=lambda f: (f.resource_lines, sum(f.categories.values()), f.size_bytes),
        reverse=True,
    )
    lines.append("| File | Resource lines | Top categories |")
    lines.append("|---|---:|---|")
    for f in dense_files[:60]:
        top = sorted(f.categories.items(), key=lambda kv: kv[1], reverse=True)[:3]
        cats = ", ".join(f"{k}:{v}" for k, v in top if v)
        lines.append(f"| `{f.path}` | {f.resource_lines} | {cats} |")
    lines.append("")

    lines.append("## Verification")
    lines.append("")
    lines.append("Machine-readable artifacts written alongside this report:")
    lines.append("- `resource_inventory.json` — full file/resource ledger")
    lines.append("- `resources.csv` — ranked resource table")
    lines.append("- `files.csv` — included file inventory")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_category(lines: list[str], by_cat: dict[str, list[ResourceRecord]], cat: str) -> None:
    items = by_cat.get(cat, [])[:12]
    if not items:
        lines.append("")
        lines.append("No high-confidence resources found in this category.")
        lines.append("")
        return
    lines.append("")
    lines.append("| Score | Type | Resource | Why it matters |")
    lines.append("|---:|---|---|---|")
    for rec in items:
        context = rec.contexts[0] if rec.contexts else ""
        context = context.replace("|", "/")[:260]
        value = rec.value.replace("|", "%7C")
        lines.append(f"| {rec.score:.2f} | {rec.kind} | `{value}` | {context} |")
    lines.append("")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files: list[FileRecord] = []
    resources: dict[str, ResourceRecord] = {}
    skipped: list[dict[str, str]] = []

    for path in sorted(set(iter_candidate_files())):
        try:
            text = read_text_window(path)
            include, reason = include_file(path, text)
            if not include:
                continue
            root = next((r for r in ROOTS if str(path).startswith(str(r))), path.parent)
            frec = summarize_file(path, text, root, reason)
            cats = category_counts(text)
            resource_lines = 0
            lines = text.splitlines()
            for idx, line in enumerate(lines):
                lower_line = line.lower()
                if any(h in lower_line for h in RESOURCE_LINE_HINTS):
                    resource_lines += 1
                context = line_context(lines, idx)
                for url in URL_RE.findall(line):
                    add_resource(resources, url, path, context, cats)
                for m in ARXIV_RE.finditer(line):
                    arxiv_id = m.group(1)
                    if "arxiv" in lower_line or "paper" in lower_line or "http" in lower_line:
                        add_resource(
                            resources, f"https://arxiv.org/abs/{arxiv_id}", path, context, cats
                        )
                for lp in LOCAL_PATH_RE.findall(line):
                    if "aurelius" in lp.lower() or "AI" in lp:
                        add_resource(resources, lp, path, context, cats)
            frec.resource_lines = resource_lines
            files.append(frec)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"path": str(path), "error": repr(exc)})

    resource_list = list(resources.values())
    for rec in resource_list:
        rec.score = score_resource(rec)
    resource_list.sort(key=lambda r: r.score, reverse=True)

    inventory = {
        "generated": TODAY,
        "roots": [str(r) for r in ROOTS],
        "summary": {
            "files_included": len(files),
            "resources_deduped": len(resource_list),
            "files_skipped_or_errors": len(skipped),
            "total_included_bytes": sum(f.size_bytes for f in files),
            "total_included_lines": sum(f.line_count for f in files),
            "category_counts": top_category_counts(files),
        },
        "files": [asdict(f) for f in files],
        "resources": [asdict(r) for r in resource_list],
        "skipped": skipped,
    }
    (OUTPUT_DIR / "resource_inventory.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(OUTPUT_DIR / "resources.csv", resource_list)

    with (OUTPUT_DIR / "files.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "path",
            "root",
            "size_bytes",
            "line_count",
            "include_reason",
            "resource_lines",
            "sha256_headtail",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in files:
            row = {field: getattr(rec, field) for field in fields}
            writer.writerow(row)

    write_report(OUTPUT_DIR / "AURELIUS_RESEARCH_RESOURCE_AUDIT.md", files, resource_list, skipped)

    print(json.dumps(inventory["summary"], indent=2))
    print(f"OUT={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
