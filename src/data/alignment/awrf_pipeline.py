"""AWRF — Alignment Write-Review-Rewrite-Filter.

The data-quality loop from Teaching-Claude-Why. Anthropic found the FINAL
reason-grounded rewrite carried ~19x of the alignment quality; behavior-only
data cut misalignment 22%->15% while reason-rich data reached ~3%. So the loop is:

    draft response
      -> constitution/policy-grounded critique
      -> rewritten (reason-rich) response
      -> why-density score (deterministic)
      -> keep/reject

Everything is driven by an injected teacher with a `.complete(prompt)->str`
method (see teacher_backend.CleanTeacher). No network in tests: pass a stub.
The why-density gate reuses the EvalLab scorer so eval and data-gen agree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol
from collections.abc import Iterable

from src.eval.alignment.why_density import why_density_score

from .adac_schema import (
    ADACScenario,
    parse_scenarios,
    response_gen_prompt,
    scenario_gen_prompt,
)


class Teacher(Protocol):
    def complete(self, prompt: str) -> str: ...


# reviewer rubric (from the implementation plan) — the critique targets these
AWRF_RUBRIC: tuple[str, ...] = (
    "Principle specificity: names the concrete principle, not generic 'be ethical'.",
    "Alternative analysis: explains the tempting bad option AND a legitimate one.",
    "Oversight respect: defers high-impact decisions to humans/governance.",
    "Reversibility: prefers low-blast-radius actions when possible.",
    "Source grounding: invents no facts, policies, test results, or evidence.",
    "Tool boundary clarity: says when a tool should / should not be used.",
    "Human realism: the user sounds like a real person, not an eval fixture.",
    "No scenario leakage: doesn't mention honeypot/eval/training.",
)


@dataclass
class ADACExample:
    scenario_id: str
    domain: str
    principle: list[str]
    prompt: str  # the user-advice request
    response: str  # the FINAL rewritten reason-rich response
    why_density: float
    kept: bool


def critique_prompt(scenario: ADACScenario, draft: str) -> str:
    rubric = "\n".join(f"- {r}" for r in AWRF_RUBRIC)
    return (
        "You are a strict alignment reviewer. Critique the draft advice below "
        "against this rubric, then REWRITE it to satisfy every point. The rewrite "
        "must clearly explain WHY the shortcut is wrong (name the principle and the "
        "concrete harm) and offer a safe, reversible alternative.\n\n"
        f"Rubric:\n{rubric}\n\n"
        f"User goal: {scenario.user_goal}\n"
        f"Tempting shortcut: {scenario.tempting_shortcut}\n\n"
        f"Draft advice:\n{draft}\n\n"
        "Return ONLY the improved advice (no critique text, no preamble)."
    )


def awrf_rewrite(teacher: Teacher, scenario: ADACScenario) -> tuple[str, str]:
    """Run write -> review -> rewrite. Returns (draft, rewritten)."""
    draft = teacher.complete(response_gen_prompt(scenario)).strip()
    rewritten = teacher.complete(critique_prompt(scenario, draft)).strip()
    return draft, rewritten


def make_example(
    teacher: Teacher,
    scenario: ADACScenario,
    min_why: float = 1.5,
) -> ADACExample:
    """Produce one AWRF example with the why-density gate applied."""
    _, rewritten = awrf_rewrite(teacher, scenario)
    wd = why_density_score(rewritten)
    prompt = (
        f"I want to {scenario.user_goal}. I'm considering "
        f"{scenario.tempting_shortcut}. What should I do?"
    )
    return ADACExample(
        scenario_id=scenario.scenario_id,
        domain=scenario.domain,
        principle=scenario.aurelius_principle,
        prompt=prompt,
        response=rewritten,
        why_density=wd,
        kept=wd >= min_why,
    )


def generate_scenarios(
    teacher: Teacher,
    domains: Iterable[str],
    principles: Iterable[str],
    per_cell: int = 2,
) -> list[ADACScenario]:
    """Author scenarios across a domain x principle grid via the clean teacher."""
    out: list[ADACScenario] = []
    plist = list(principles)
    for d in domains:
        for i, p in enumerate(plist):
            text = teacher.complete(scenario_gen_prompt(d, p, n=per_cell))
            for j, sc in enumerate(parse_scenarios(text, d)):
                sc.scenario_id = f"adac_{d}_{p}_{i}{j}"
                out.append(sc)
    return out


def generate_adac_corpus(
    teacher: Teacher,
    domains: Iterable[str],
    principles: Iterable[str],
    per_cell: int = 2,
    min_why: float = 1.5,
) -> list[ADACExample]:
    """Full pipeline: author scenarios -> AWRF each -> why-density filter.
    Returns ALL examples (kept flag set); caller filters to kept for training."""
    scenarios = generate_scenarios(teacher, domains, principles, per_cell)
    return [make_example(teacher, s, min_why=min_why) for s in scenarios]


def write_sft_jsonl(examples: list[ADACExample], path: str, kept_only: bool = True) -> int:
    """Write kept examples as {prompt,response,why_density,domain} jsonl for the
    SFT mixer. Returns the count written."""
    n = 0
    with open(path, "w") as f:
        for e in examples:
            if kept_only and not e.kept:
                continue
            f.write(
                json.dumps(
                    {
                        "prompt": e.prompt,
                        "response": e.response,
                        "why_density": round(e.why_density, 3),
                        "domain": e.domain,
                        "source": "adac_awrf",
                    }
                )
                + "\n"
            )
            n += 1
    return n


def corpus_stats(examples: list[ADACExample]) -> dict:
    if not examples:
        return {"n": 0, "kept": 0, "keep_frac": 0.0, "mean_why": 0.0}
    kept = sum(e.kept for e in examples)
    return {
        "n": len(examples),
        "kept": kept,
        "keep_frac": kept / len(examples),
        "mean_why": sum(e.why_density for e in examples) / len(examples),
    }
