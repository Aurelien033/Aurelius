#!/usr/bin/env python3
"""Resolve selected arXiv metadata and create a curated Aurelius resource report."""

from __future__ import annotations

import csv
import json
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path
from xml.etree import ElementTree as ET

TODAY = date.today().isoformat()
BASE = Path.home() / "Desktop" / "AI Plans" / f"aurelius-research-resource-audit-{TODAY}"
INV = BASE / "resource_inventory.json"
OUT = BASE / "AURELIUS_ADDITIONAL_RESOURCES_CURATED.md"
CURATED_JSON = BASE / "curated_resources.json"
CURATED_CSV = BASE / "curated_resources.csv"

ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}

RESOURCE_PLAN = [
    {
        "id": "fastcontext",
        "lane": "composer_agent",
        "name": "FastContext / microsoft/fastcontext",
        "resources": ["https://arxiv.org/abs/2606.14066", "https://github.com/microsoft/fastcontext"],
        "priority": "P0",
        "status": "immediate_build_reference",
        "why": "Dedicated repository-exploration subagent; local corpus says it improves SWE-style resolution up to 5.5% and cuts main-agent tokens up to 60%. This directly upgrades ContextAssembler beyond heuristic search.",
        "aurelius_action": "Build a FastContext-like ExplorerAgent: parallel search/read/glob, file-line citation output, no edits, trained/evaluated separately from the solver.",
        "falsifier": "If ExplorerAgent does not improve file localization recall or reduce solver context tokens versus current ContextAssembler on Aurelius repair-gym tasks, keep it as heuristic-only and do not train a separate model.",
    },
    {
        "id": "swe-agent",
        "lane": "composer_agent",
        "name": "SWE-agent",
        "resources": ["https://arxiv.org/abs/2405.15793", "https://github.com/SWE-agent/SWE-agent"],
        "priority": "P0",
        "status": "workflow_reference",
        "why": "Open agent-computer interface pattern for automated software engineering; useful as a behavioral reference for Aurelius Composer-like tool loop.",
        "aurelius_action": "Compare Aurelius Composer actions/tool protocol against SWE-agent loop: localization, edit, test, submit, failure handling.",
        "falsifier": "If SWE-agent-style loop adds token/tool overhead without improving solved tasks over current ComposerAgent, keep only its eval harness ideas.",
    },
    {
        "id": "swe-bench-livecodebench",
        "lane": "eval_safety",
        "name": "SWE-bench + LiveCodeBench + repair gym",
        "resources": ["https://arxiv.org/abs/2310.06770", "https://arxiv.org/abs/2403.07974"],
        "priority": "P0",
        "status": "eval_reference",
        "why": "The corpus repeatedly points to realistic software-engineering and contamination-resistant code evaluation. Aurelius needs evolving held-out evaluation, not HumanEval-only claims.",
        "aurelius_action": "Create a pass@1/oracle@K/repair-success evaluation battery: Aurelius repair gym, SWE-bench-lite subset, LiveCodeBench monthly sample, MBPP+/HumanEval+.",
        "falsifier": "If Composer improvements do not transfer beyond handpicked Aurelius tests, they are workflow-specialized, not AGI-track progress.",
    },
    {
        "id": "react-toolformer-reflexion",
        "lane": "composer_agent",
        "name": "ReAct + Toolformer + Reflexion",
        "resources": ["https://arxiv.org/abs/2210.03629", "https://arxiv.org/abs/2302.04761", "https://arxiv.org/abs/2303.11366"],
        "priority": "P0",
        "status": "agent_reasoning_reference",
        "why": "The local corpus repeatedly uses this trio as the agent/tool/memory backbone: reason-act traces, tool-use learning, and verbal reinforcement repair.",
        "aurelius_action": "Store Composer traces as typed ReAct trajectories with tool calls, observations, diffs, verifier outputs, reflection, and next repair action.",
        "falsifier": "If reflection text does not improve next-edit success under controlled verifier feedback, do not train reflection as free-form prose; train structured error-to-fix actions instead.",
    },
    {
        "id": "process-supervision",
        "lane": "rlvr_verifier",
        "name": "Process supervision / Let’s Verify Step by Step / verifier training",
        "resources": ["https://arxiv.org/abs/2305.20050", "https://arxiv.org/abs/2110.14168"],
        "priority": "P0",
        "status": "training_reference",
        "why": "Direct support for PRM-style step labels and verifier-based learning. This is the bridge from Composer traces to trainable reasoning behavior.",
        "aurelius_action": "Add step labels to Composer trajectories: search relevance, context relevance, diff minimality, test choice, repair correctness, final solve.",
        "falsifier": "If PRM scores improve process-looking metrics but not final solve rate, kill PRM as reward and use it only for debugging dashboards.",
    },
    {
        "id": "rlvr-stack",
        "lane": "rlvr_verifier",
        "name": "DeepSeekMath / DAPO / BPPO / RLVR-over-SFT sources",
        "resources": ["https://arxiv.org/abs/2402.03300", "https://arxiv.org/abs/2503.14476", "https://arxiv.org/abs/2605.28028", "https://arxiv.org/abs/2606.22938"],
        "priority": "P0/P1",
        "status": "training_recipe_pool",
        "why": "The corpus repeatedly frames RLVR as the only lever that moved held-out code/math scores versus SFT-only specialization.",
        "aurelius_action": "Use Composer/verifier traces as RLVR tasks with fractional reward, repair reward, no-unrelated-file reward, and failure taxonomy logging.",
        "falsifier": "If RLVR improves training tasks but not held-out code/math/repo tasks across seeds, stop scaling that mix and inspect oracle@K/selection gap first.",
    },
    {
        "id": "pagedattention-vllm-sglang-lmcache",
        "lane": "inference_efficiency",
        "name": "PagedAttention / vLLM / SGLang / LMCache",
        "resources": ["https://arxiv.org/abs/2309.06180", "https://github.com/vllm-project/vllm", "https://github.com/sgl-project/sglang", "https://github.com/LMCache/LMCache"],
        "priority": "P0 for serving, not raw intelligence",
        "status": "release_infra_reference",
        "why": "High-frequency corpus cluster for serving and KV-cache management. Makes Aurelius runnable and cheaper, but not by itself more intelligent.",
        "aurelius_action": "Harden serving path with measured vLLM/SGLang profiles, prefix/KV reuse, batch endpoint, and request-cost ledger.",
        "falsifier": "If local CUDA/Azure benchmark does not improve TTFT/TPOT/memory at equal quality, do not claim serving improvement.",
    },
    {
        "id": "structured-generation",
        "lane": "composer_agent",
        "name": "XGrammar / Outlines / constrained structured generation",
        "resources": ["https://arxiv.org/abs/2411.15100", "https://github.com/mlc-ai/xgrammar", "https://github.com/dottxt-ai/outlines"],
        "priority": "P0",
        "status": "implementation_reference",
        "why": "Composer-like agents need reliable JSON/tool/diff protocol. Structured generation reduces malformed actions and Apply Model training burden.",
        "aurelius_action": "Use grammar-constrained action envelopes for ComposerAgent: explore/read/plan/edit/run/verify/complete; leave diff body verifier-checked.",
        "falsifier": "If grammar constraints reduce malformed actions but lower solved tasks due to overconstraint, keep for action envelope only, not code diff content.",
    },
    {
        "id": "memory-agent-stack",
        "lane": "memory_amc",
        "name": "MemGPT + Titans + RMT / recurrent memory lines",
        "resources": ["https://arxiv.org/abs/2310.08560", "https://arxiv.org/abs/2501.00663", "https://arxiv.org/abs/2207.06881"],
        "priority": "P1",
        "status": "agi_memory_reference",
        "why": "These support Aurelius as a persistent developmental system rather than a stateless code assistant.",
        "aurelius_action": "Map memory papers into AMC/SDB: memory admission, retrieval provenance, event tape, retirement, and proof-carrying memory entries.",
        "falsifier": "If memory write/retrieve harms exact-task performance or injects stale state, keep memory shadow-only until verifier precision improves.",
    },
    {
        "id": "tapered-language-models",
        "lane": "agi_cognition",
        "name": "Tapered Language Models",
        "resources": ["https://arxiv.org/abs/2606.23670"],
        "priority": "P1 research spine",
        "status": "architecture_reference",
        "why": "Aligns with Aurelius V3 memory: Tapered FFN as a first from-scratch mechanism candidate, but still needs falsifiers and code-level ablations.",
        "aurelius_action": "Keep as V3 scratch architecture candidate; run small-scale ablations versus same-param dense FFN on reasoning/repair tasks.",
        "falsifier": "If same-param dense or standard FFN matches quality/latency on repair and reasoning tasks, kill tapering as architectural claim.",
    },
    {
        "id": "fast-kv-quant",
        "lane": "inference_efficiency",
        "name": "KIVI / KVQuant / AWQ / GPTQ / SmoothQuant",
        "resources": ["https://arxiv.org/abs/2402.02750", "https://arxiv.org/abs/2401.18079", "https://arxiv.org/abs/2306.00978", "https://arxiv.org/abs/2210.17323", "https://arxiv.org/abs/2211.10438"],
        "priority": "P1/P2 gated",
        "status": "research_gated_efficiency",
        "why": "Important for local deployment, long context, and memory cost, but the corpus warns not to claim savings until packed storage and quality are measured.",
        "aurelius_action": "Build a KV/weight quantization ladder with measured bytes/token, long-context retention, code/schema accuracy, and fallback to Q8/dense.",
        "falsifier": "Any quant method that regresses repair/code/schema or does not physically reduce allocated memory is demoted to research-only.",
    },
    {
        "id": "speculative-decoding-stack",
        "lane": "inference_efficiency",
        "name": "Speculative decoding / Medusa / EAGLE / SpecInfer",
        "resources": ["https://arxiv.org/abs/2203.16487", "https://arxiv.org/abs/2302.01318", "https://arxiv.org/abs/2401.10774", "https://arxiv.org/abs/2401.15077", "https://arxiv.org/abs/2406.16858", "https://arxiv.org/abs/2305.09781"],
        "priority": "P1 serving",
        "status": "acceptance_rate_gated",
        "why": "Can speed decoding with exact or near-exact verification. Useful after baseline serving is measured, not before.",
        "aurelius_action": "Add acceptance-rate ledger and auto-disable speculation when overhead exceeds benefit; use existing serving backends first.",
        "falsifier": "If acceptance rate/cost ledger shows slower TPOT or quality risk, disable speculation for that model/profile.",
    },
    {
        "id": "agentic-abstention",
        "lane": "eval_safety",
        "name": "Agentic abstention / CONVOLVE",
        "resources": ["https://lhannnn.github.io/agentic-abstention"],
        "priority": "P1 safety/eval",
        "status": "safety_reference",
        "why": "Aurelius AGI-track agents need to know when not to act, not just how to act. Especially relevant for tool calls, repo edits, and uncertain memory writes.",
        "aurelius_action": "Add abstention metrics to ComposerAgent: insufficient context, unsafe edit, uncertain verifier, ambiguous task, missing tests.",
        "falsifier": "If abstention rises without reducing harmful/bad edits or improves safety only by refusing useful tasks, tune or demote.",
    },
    {
        "id": "skillopt-openclaw-skill",
        "lane": "composer_agent",
        "name": "SkillOpt / OpenClaw-Skill / trace-derived agent skills",
        "resources": ["https://aka.ms/SkillOpt", "https://arxiv.org/abs/2606.16774", "https://arxiv.org/abs/2601.22607"],
        "priority": "P1/P2",
        "status": "skill_learning_reference",
        "why": "Supports moving from raw trajectories to reusable skills. This is central to AGI-track continual learning but needs careful provenance and eval gates.",
        "aurelius_action": "Convert successful Composer traces into skill candidates with preconditions, verifier contract, and expiry/retirement metadata.",
        "falsifier": "If extracted skills do not transfer across repos/tasks, keep them as retrieval examples, not procedural skills.",
    },
    {
        "id": "datacomp-fineweb",
        "lane": "data_training",
        "name": "DataComp-LM / FineWeb / quality-aware data pool",
        "resources": ["https://arxiv.org/abs/2406.11794", "https://arxiv.org/abs/2406.17557"],
        "priority": "P1 data",
        "status": "pretrain_data_reference",
        "why": "Useful for corpus quality and data filtering philosophy, but lower immediate value than verified Aurelius traces for Composer/AGI flywheel.",
        "aurelius_action": "Use as data-quality methodology reference; prioritize local verified traces and cleaned Aurelius dataset before broad web pretrain changes.",
        "falsifier": "If quality filtering does not improve held-out repair/reasoning per token, do not expand generic data volume.",
    },
    {
        "id": "frontend-future-swe",
        "lane": "data_training",
        "name": "SWE-Future / forecast-conditioned software data synthesis",
        "resources": ["https://arxiv.org/abs/2606.18733"],
        "priority": "P2 needs_reading",
        "status": "candidate_source_pool",
        "why": "Local corpus flags it as candidate for future-oriented software-agent training, but also says to read the full paper before implementation claims.",
        "aurelius_action": "Park in source pool; validate by reading full paper and checking whether it adds beyond verified repair trace generation.",
        "falsifier": "If synthetic future tasks do not transfer to held-out real repo repairs, keep as augmentation only.",
    },
    {
        "id": "avoid-layer-skip-mod",
        "lane": "avoid_for_now",
        "name": "LayerSkip / Mixture-of-Depths / frozen-base routing path",
        "resources": ["https://arxiv.org/abs/2404.16710", "https://arxiv.org/abs/2404.02258", "https://github.com/facebookresearch/LayerSkip"],
        "priority": "DO_NOT_PRIORITIZE",
        "status": "caution_falsified_context",
        "why": "The local corpus mentions these in negative/falsified contexts. They may still be useful as literature, but not as immediate Aurelius build direction.",
        "aurelius_action": "Do not spend next sprint on route-conditioned SFT/layer-skip claims. Only revisit after P0 truth-surface shows a real selection signal.",
        "falsifier": "Already flagged: if frozen-base routing cannot beat dense at equal cost/quality, it remains research-gated.",
    },
]


def load_inventory() -> dict:
    return json.loads(INV.read_text())


def lookup_resource(resources: list[dict], value: str) -> dict | None:
    value_l = value.lower().rstrip("/")
    best = None
    for rec in resources:
        rec_v = str(rec.get("value", "")).lower().rstrip("/")
        if rec_v == value_l or rec_v.startswith(value_l + "/"):
            if best is None or rec.get("score", 0) > best.get("score", 0):
                best = rec
    return best


def resolve_arxiv(ids: list[str]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    ids = sorted(set(ids))
    for i in range(0, len(ids), 25):
        batch = ids[i : i + 25]
        query = ",".join(batch)
        url = "https://export.arxiv.org/api/query?id_list=" + urllib.parse.quote(query)
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
                xml = resp.read()
            root = ET.fromstring(xml)  # noqa: S314
            for entry in root.findall("atom:entry", ARXIV_NS):
                id_url = entry.findtext("atom:id", default="", namespaces=ARXIV_NS)
                match = re.search(r"(\d{4}\.\d{4,5})(?:v\d+)?", id_url)
                if not match:
                    continue
                aid = match.group(1)
                title = " ".join(entry.findtext("atom:title", default="", namespaces=ARXIV_NS).split())
                summary = " ".join(entry.findtext("atom:summary", default="", namespaces=ARXIV_NS).split())
                published = entry.findtext("atom:published", default="", namespaces=ARXIV_NS)
                out[aid] = {"title": title, "summary": summary[:600], "published": published}
        except Exception as exc:  # noqa: BLE001
            for aid in batch:
                out[aid] = {"title": "UNRESOLVED", "summary": repr(exc), "published": ""}
        time.sleep(0.5)
    return out


def arxiv_ids_from_plan() -> list[str]:
    ids: list[str] = []
    for item in RESOURCE_PLAN:
        for r in item["resources"]:
            m = re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", r)
            if m:
                ids.append(m.group(1))
    return ids


def decorate_plan(inv: dict, arxiv_meta: dict[str, dict[str, str]]) -> list[dict]:
    resources = inv["resources"]
    decorated = []
    for item in RESOURCE_PLAN:
        evidence = []
        for res in item["resources"]:
            rec = lookup_resource(resources, res)
            aid_match = re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", res)
            meta = arxiv_meta.get(aid_match.group(1), {}) if aid_match else {}
            evidence.append(
                {
                    "resource": res,
                    "local_score": rec.get("score") if rec else None,
                    "local_files": len(rec.get("files", [])) if rec else 0,
                    "sample_context": (rec.get("contexts") or [""])[0] if rec else "not found in scan",
                    "arxiv_title": meta.get("title", "") if meta else "",
                    "arxiv_published": meta.get("published", "") if meta else "",
                }
            )
        row = dict(item)
        row["evidence"] = evidence
        decorated.append(row)
    return decorated


def write_outputs(inv: dict, decorated: list[dict]) -> None:
    CURATED_JSON.write_text(json.dumps(decorated, indent=2, ensure_ascii=False), encoding="utf-8")
    with CURATED_CSV.open("w", newline="", encoding="utf-8") as f:
        fields = ["id", "priority", "lane", "status", "name", "resources", "why", "aurelius_action", "falsifier"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for item in decorated:
            w.writerow({k: (" | ".join(item[k]) if k == "resources" else item[k]) for k in fields})

    by_lane: dict[str, list[dict]] = defaultdict(list)
    for item in decorated:
        by_lane[item["lane"]].append(item)

    lines = []
    lines.append("# Curated Additional Resources for Project Aurelius")
    lines.append("")
    lines.append(f"Generated: {TODAY}")
    lines.append("")
    lines.append("## Grounding")
    lines.append("")
    s = inv["summary"]
    lines.append(f"- Local research files analyzed: {s['files_included']:,}")
    lines.append(f"- Included corpus size: {s['total_included_bytes']:,} bytes / {s['total_included_lines']:,} lines")
    lines.append(f"- Deduplicated resources extracted: {s['resources_deduped']:,}")
    lines.append("- This curated report filters the raw regex inventory into resources that can actually assist Aurelius.")
    lines.append("- ArXiv titles are resolved through the arXiv API when available; unresolved/future IDs remain local-corpus candidates.")
    lines.append("")
    lines.append("## Executive conclusion")
    lines.append("")
    lines.append("The biggest additional resource signal is not another generic LLM paper. It is a stack:")
    lines.append("")
    lines.append("1. FastContext-style repository exploration subagent for Cursor Composer-like coding agency.")
    lines.append("2. SWE-agent/SWE-bench/LiveCodeBench-style evaluation for realistic software tasks.")
    lines.append("3. Process supervision + RLVR for verifier-grounded learning from those traces.")
    lines.append("4. PagedAttention/vLLM/SGLang/LMCache for scalable serving and trace generation.")
    lines.append("5. AMC/SDB/DreamBank-style memory plus MemGPT/Titans/RMT references for developmental continuity.")
    lines.append("6. Structured generation (XGrammar/Outlines) so Aurelius emits reliable actions, diffs, and schemas.")
    lines.append("")
    lines.append("That stack supports the AGI-track goal because it creates a closed loop: explore -> act -> verify -> repair -> remember -> train.")
    lines.append("")

    lane_order = [
        "composer_agent",
        "rlvr_verifier",
        "eval_safety",
        "memory_amc",
        "inference_efficiency",
        "agi_cognition",
        "data_training",
        "avoid_for_now",
    ]
    lane_names = {
        "composer_agent": "Cursor Composer-like agent resources",
        "rlvr_verifier": "Verifier/RLVR learning resources",
        "eval_safety": "Evaluation and safety resources",
        "memory_amc": "Memory / AMC / developmental-state resources",
        "inference_efficiency": "Inference and serving resources",
        "agi_cognition": "AGI/cognition architecture resources",
        "data_training": "Data and training resources",
        "avoid_for_now": "Do-not-prioritize / caution resources",
    }
    for lane in lane_order:
        items = by_lane.get(lane, [])
        if not items:
            continue
        lines.append(f"## {lane_names[lane]}")
        lines.append("")
        for item in items:
            lines.append(f"### {item['priority']} — {item['name']}")
            lines.append("")
            lines.append(f"Status: `{item['status']}`")
            lines.append("")
            lines.append(f"Why it matters: {item['why']}")
            lines.append("")
            lines.append(f"Aurelius action: {item['aurelius_action']}")
            lines.append("")
            lines.append(f"Falsifier: {item['falsifier']}")
            lines.append("")
            lines.append("Evidence/resources:")
            for ev in item["evidence"]:
                title = ev.get("arxiv_title") or ""
                title_part = f" — {title}" if title and title != "UNRESOLVED" else ""
                lines.append(
                    f"- `{ev['resource']}`{title_part}; local_score={ev['local_score']}; local_files={ev['local_files']}"
                )
            lines.append("")

    lines.append("## Recommended immediate work order")
    lines.append("")
    lines.append("1. P0: Build `ExplorerAgent` / FastContext analogue on top of the new `ContextAssembler`.")
    lines.append("2. P0: Add trajectory recorder for ComposerAgent: task, search/read, selected context, diff, verifier output, repair, final result.")
    lines.append("3. P0: Add PRM-ready step labels and a repair dataset export format.")
    lines.append("4. P0: Stand up the truth surface: pass@1, oracle@K, selection gap, repair success, verifier precision/recall, cost per solved task.")
    lines.append("5. P1: Harden serving with vLLM/SGLang/LMCache only after benchmark ledger exists.")
    lines.append("6. P1: Promote memory resources into AMC/SDB contracts, but keep memory writes shadow-only until verifier admission is measured.")
    lines.append("7. P2: Revisit architecture papers like Tapered LM only after the Composer/RLVR trace engine is producing reliable evidence.")
    lines.append("")
    lines.append("## Files emitted")
    lines.append("")
    lines.append("- `curated_resources.json`")
    lines.append("- `curated_resources.csv`")
    lines.append("- `AURELIUS_ADDITIONAL_RESOURCES_CURATED.md`")
    lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    inv = load_inventory()
    meta = resolve_arxiv(arxiv_ids_from_plan())
    decorated = decorate_plan(inv, meta)
    write_outputs(inv, decorated)
    print(json.dumps({"curated": len(decorated), "out": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
