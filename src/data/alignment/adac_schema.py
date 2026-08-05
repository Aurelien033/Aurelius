"""ADAC — Aurelius Difficult Advice Corpus: schema + scenario generation.

Teaching-Claude-Why's key efficiency result: training on ethically-ambiguous
USER-ADVICE dilemmas (not direct agent honeypots) matched much larger direct
honeypot training at ~28x token efficiency, and generalized better OOD. This
module defines the scenario schema (from the implementation plan) and turns a
clean teacher into a scenario generator + validator.

An ADAC example is a {scenario, reason-rich response} pair. The scenario is a
difficult-advice situation; the response models constitution-grounded advice
that names the tempting shortcut, explains WHY it's wrong (the principle), and
offers a safe reversible alternative. The AWRF pipeline (awrf_pipeline.py) does
the draft->critique->rewrite->why-filter loop on top of this.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

# Aurelius principle vocabulary (from the plan; also the why-density frame)
PRINCIPLES: tuple[str, ...] = (
    "honesty", "oversight", "non-coercion", "reversibility",
    "source-grounding", "privacy", "test-integrity",
)
DOMAINS: tuple[str, ...] = (
    "coding", "it_ops", "security", "research", "data",
    "tool_use", "memory", "professional",
)


@dataclass
class ADACScenario:
    scenario_id: str
    domain: str
    aurelius_principle: list[str]
    user_goal: str
    tempting_shortcut: str
    why_shortcut_is_instrumental: str
    violated_norm_or_policy: str
    safe_alternative: str
    human_checkpoint_needed: bool
    expected_response_properties: list[str] = field(default_factory=lambda: [
        "acknowledges legitimate goal",
        "names tempting shortcut",
        "explains why shortcut is wrong",
        "offers safe alternative",
        "states uncertainty / need for human approval when needed",
    ])

    def to_dict(self) -> dict:
        return asdict(self)


def validate_scenario(d: dict) -> list[str]:
    """Return a list of problems (empty = valid). Deterministic schema check."""
    problems: list[str] = []
    # scenario_id is infra-assigned by the parser, not teacher-authored -> not required here
    required = ("domain", "aurelius_principle", "user_goal",
                "tempting_shortcut", "why_shortcut_is_instrumental",
                "violated_norm_or_policy", "safe_alternative", "human_checkpoint_needed")
    for k in required:
        if k not in d or d[k] in (None, "", []):
            problems.append(f"missing/empty: {k}")
    if d.get("domain") and d["domain"] not in DOMAINS:
        problems.append(f"unknown domain: {d['domain']}")
    princ = d.get("aurelius_principle") or []
    if isinstance(princ, str):
        princ = [princ]
    if not any(p in PRINCIPLES for p in princ):
        problems.append("no recognized aurelius_principle")
    if not isinstance(d.get("human_checkpoint_needed"), bool):
        problems.append("human_checkpoint_needed must be bool")
    return problems


# ---- generation prompts (clean-teacher authored; the model writes scenarios) ----
def scenario_gen_prompt(domain: str, principle: str, n: int = 1) -> str:
    """Prompt a clean teacher to author difficult-ADVICE scenarios as JSON."""
    return (
        "You are helping build an AI-safety training corpus of *difficult advice* "
        "situations. Write realistic scenarios where a USER (a person, not an AI) "
        "faces an ethically ambiguous choice and can reach a legitimate goal by "
        "cutting a corner that violates a norm.\n\n"
        f"Domain: {domain}. Principle at stake: {principle}.\n"
        f"Return a JSON array of exactly {n} object(s), each with keys: "
        "domain, aurelius_principle (list), user_goal, tempting_shortcut, "
        "why_shortcut_is_instrumental, violated_norm_or_policy, safe_alternative, "
        "human_checkpoint_needed (true/false).\n"
        "Make the user sound like a real person. Do NOT mention that this is an "
        "eval, test, or training data. Output ONLY the JSON array."
    )


def response_gen_prompt(scenario: ADACScenario) -> str:
    """Prompt a clean teacher to DRAFT reason-rich advice for one scenario."""
    return (
        "A user asks you for advice in the situation below. Give thoughtful, "
        "honest advice. You MUST: acknowledge their legitimate goal; name the "
        "tempting shortcut; explain clearly WHY the shortcut is wrong (name the "
        "principle and the harm); offer a concrete safe, reversible alternative; "
        "and note when a human/manager sign-off is needed.\n\n"
        f"User goal: {scenario.user_goal}\n"
        f"They are considering: {scenario.tempting_shortcut}\n\n"
        "Write only the advice, in a natural voice."
    )


_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def parse_scenarios(text: str, domain: str) -> list[ADACScenario]:
    """Parse a teacher's JSON-array output into validated ADACScenario objects.
    Skips malformed entries; assigns stable ids. Robust to prose around the JSON."""
    m = _JSON_ARRAY_RE.search(text or "")
    if not m:
        return []
    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out: list[ADACScenario] = []
    for i, d in enumerate(raw if isinstance(raw, list) else []):
        if not isinstance(d, dict):
            continue
        d.setdefault("domain", domain)
        princ = d.get("aurelius_principle") or []
        if isinstance(princ, str):
            princ = [princ]
        d["aurelius_principle"] = [p for p in princ if p in PRINCIPLES] or ["honesty"]
        d.setdefault("scenario_id", f"adac_{domain}_{i:03d}")
        d["human_checkpoint_needed"] = bool(d.get("human_checkpoint_needed", False))
        if validate_scenario(d):
            continue
        out.append(ADACScenario(
            scenario_id=d["scenario_id"], domain=d["domain"],
            aurelius_principle=d["aurelius_principle"], user_goal=d["user_goal"],
            tempting_shortcut=d["tempting_shortcut"],
            why_shortcut_is_instrumental=d["why_shortcut_is_instrumental"],
            violated_norm_or_policy=d["violated_norm_or_policy"],
            safe_alternative=d["safe_alternative"],
            human_checkpoint_needed=d["human_checkpoint_needed"],
        ))
    return out
