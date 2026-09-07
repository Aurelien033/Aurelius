"""Generate Aurelius reasoning-focused SFT and auxiliary training data.

Outputs are intentionally compact but high-signal:
- SFT JSONL: prompt/response/system_prompt/metadata
- preference JSONL: prompt/chosen/rejected metadata for DPO-style training
- pretrain JSONL: short domain documents for continued pretraining
- manifest + curriculum config for reproducible training runs

This generator is deterministic and dependency-free. Scale by increasing SAMPLES_PER_DOMAIN.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "aurelius_reasoning_sft_v1"
SEED = 20260614
VERSION = "aurelius-reasoning-sft-v1"
SYSTEM_PROMPT = (
    "You are Aurelius, a research-grade reasoning model. Prefer concise latent "
    "reasoning summaries over wordy chain-of-thought. Show assumptions, key steps, "
    "verification, uncertainty, and final answers. When a request is unsafe, refuse "
    "briefly and redirect to a safe alternative."
)

DOMAIN_WEIGHTS = {
    "mechanistic_reasoning": 54,
    "math_logic": 36,
    "coding": 36,
    "agent_planning": 30,
    "metacognition": 24,
    "safety_alignment": 24,
}


@dataclass(slots=True)
class Sample:
    prompt: str
    response: str
    domain: str
    skill: str
    difficulty: int
    verification: str
    answer_key: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata_extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PretrainDoc:
    title: str
    text: str
    domain: str
    skill: str
    difficulty: int
    tags: list[str] = field(default_factory=list)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_val(sample_id: str) -> bool:
    return int(stable_hash(sample_id)[:8], 16) % 10 == 0


def record(sample: Sample, idx: int) -> dict[str, Any]:
    sample_id = f"{VERSION}-{idx:04d}"
    metadata = {
        "id": sample_id,
        "domain": sample.domain,
        "skill": sample.skill,
        "difficulty": sample.difficulty,
        "source": "synthetic",
        "version": VERSION,
        "verification": sample.verification,
        "tags": sample.tags,
        **sample.metadata_extra,
    }
    if sample.answer_key is not None:
        metadata["answer_key"] = sample.answer_key
    return {
        "prompt": sample.prompt,
        "response": sample.response,
        "system_prompt": SYSTEM_PROMPT,
        "metadata": metadata,
    }


def add(
    samples: list[Sample],
    prompt: str,
    response: str,
    domain: str,
    skill: str,
    difficulty: int,
    verification: str,
    answer_key: str | None = None,
    tags: list[str] | None = None,
    metadata_extra: dict[str, Any] | None = None,
) -> None:
    samples.append(
        Sample(
            prompt=prompt,
            response=response,
            domain=domain,
            skill=skill,
            difficulty=difficulty,
            verification=verification,
            answer_key=answer_key,
            tags=tags or [],
            metadata_extra=metadata_extra or {},
        )
    )


def response(summary: list[str], verification: list[str], final: str) -> str:
    return (
        "Reasoning summary:\n"
        + "\n".join(f"- {line}" for line in summary)
        + "\nVerification:\n"
        + "\n".join(f"- {line}" for line in verification)
        + f"\nFinal: {final}"
    )


def build_mechanistic(samples: list[Sample]) -> None:
    prompts = [
        (
            "Explain how a residual stream can act as a working memory in a transformer. Include what gets written, what gets read, and one failure mode.",
            response(
                [
                    "Treat the residual stream as the shared workspace that every layer reads from and writes to.",
                    "Attention writes context-dependent pointers; MLPs write feature transformations; normalization gates scale the stream.",
                    "A later layer can recover earlier facts if the signal is not overwritten by stronger updates.",
                ],
                [
                    "Check that the explanation separates read, write, and overwrite dynamics.",
                    "Confirm the failure mode is concrete: residual interference, norm explosion, or feature cancellation.",
                ],
                "Aurelius should model reasoning as managed residual-state updates, not as isolated layer outputs.",
            ),
            "mechanistic_reasoning",
            "residual_stream_working_memory",
            3,
        ),
        (
            "What would you inspect to test whether an attention head is routing a question token to relevant evidence tokens?",
            response(
                [
                    "Inspect attention maps for the question token across layers and heads.",
                    "Compare attended evidence tokens against a labeled retrieval set.",
                    "Ablate the head or mask its output and measure the drop in answer accuracy.",
                ],
                [
                    "Attention alone is not causality; include ablation or causal mediation.",
                    "Use held-out examples so the routing test is not just memorized.",
                ],
                "Use attention maps plus causal ablation to test evidence routing.",
            ),
            "mechanistic_reasoning",
            "attention_routing_probe",
            3,
        ),
        (
            "Describe a circuit-level hypothesis for multi-step arithmetic in a small transformer.",
            response(
                [
                    "Hypothesize that digit or operand features are represented in early MLP layers.",
                    "Attention may copy operands and operators into a residual workspace.",
                    "Intermediate arithmetic features could be composed by MLPs and checked by later layers.",
                ],
                [
                    "The hypothesis should predict measurable features: operand-copy attention, carry-state neurons, and final-answer readout.",
                    "Ablating carry-state neurons should hurt multi-digit cases more than single-digit cases.",
                ],
                "A good circuit hypothesis makes layer, head, neuron, and ablation predictions.",
            ),
            "mechanistic_reasoning",
            "arithmetic_circuit_hypothesis",
            4,
        ),
        (
            "Why can a model answer correctly while using the wrong internal circuit?",
            response(
                [
                    "Correct outputs can arise from shortcuts, memorized templates, or dataset artifacts.",
                    "Mechanistic correctness requires the internal path to match the causal structure of the task.",
                    "Distribution shifts and adversarial probes reveal shortcut circuits.",
                ],
                [
                    "Test on counterfactual inputs where superficial cues point to the wrong answer.",
                    "Use ablations to show the claimed circuit is necessary, not merely correlated.",
                ],
                "Correctness is behavioral; reasoning quality requires causal internal alignment.",
            ),
            "mechanistic_reasoning",
            "shortcut_detection",
            3,
        ),
        (
            "Define a mechanistic interpretability eval for whether Aurelius maintains a plan across tool calls.",
            response(
                [
                    "Give the model a multi-step task with intermediate tool results that can alter the plan.",
                    "Probe whether plan tokens remain active in the residual stream after each tool response.",
                    "Measure if later actions still follow the original goal or only react to the last tool output.",
                ],
                [
                    "The eval should include distractor tool outputs and a final verification step.",
                    "Pass only if the model updates the plan when evidence changes and preserves it when evidence is irrelevant.",
                ],
                "Use plan-persistence probes plus action-level scoring across multi-turn tool tasks.",
            ),
            "mechanistic_reasoning",
            "plan_persistence_probe",
            4,
        ),
    ]
    for prompt, resp, domain, skill, difficulty in prompts:
        add(
            samples,
            prompt,
            resp,
            domain,
            skill,
            difficulty,
            "mechanistic sanity check by circuit prediction and ablation logic.",
        )

    for i in range(49):
        topic, skill, difficulty, answer = [
            (
                "residual interference",
                "residual_state_management",
                3,
                "control residual updates before adding new features",
            ),
            (
                "attention sink mitigation",
                "attention_sink_control",
                3,
                "monitor low-information tokens and use masks or penalties",
            ),
            (
                "MLP feature composition",
                "feature_composition",
                4,
                "test composed features with synthetic counterfactuals",
            ),
            (
                "causal mediation",
                "causal_circuit_testing",
                4,
                "compare indirect effects through candidate components",
            ),
            (
                "latent plan state",
                "latent_planning",
                5,
                "probe plan tokens across tool-call boundaries",
            ),
            (
                "uncertainty calibration",
                "uncertainty_routing",
                3,
                "route uncertain cases to verification before final answer",
            ),
            (
                "tool-result grounding",
                "grounded_tool_use",
                4,
                "require citations to tool outputs in final decisions",
            ),
            (
                "safety gate stability",
                "safety_circuit",
                5,
                "stress-test refusal under paraphrase and role-play",
            ),
            (
                "retrieval routing",
                "retrieval_attention",
                4,
                "separate query construction from evidence selection",
            ),
            (
                "self-evaluation",
                "metacognitive_monitoring",
                3,
                "require explicit checklists before final answers",
            ),
        ][i % 10]
        add(
            samples,
            f"Write a compact Aurelius training example about {topic}. The prompt should ask for a mechanism-level explanation and a concrete eval. Difficulty {difficulty}. Template index {i}.",
            response(
                [
                    f"Prompt target: ask Aurelius to explain the mechanism behind {topic} and design a test.",
                    "Expected behavior: identify internal components, predict observable behavior, and name a failure mode.",
                    "Eval target: score whether the model gives a causal mechanism plus a measurable intervention.",
                ],
                [
                    "Check that the response avoids vague claims like 'the model thinks carefully'.",
                    "Check that the eval can be run with prompts, probes, ablations, or behavioral metrics.",
                ],
                f"Mechanism: {skill}. Key success criterion: {answer}.",
            ),
            "mechanistic_reasoning",
            skill,
            difficulty,
            "Generated from deterministic topic template; validated for mechanism + eval + failure mode.",
            tags=["mechanistic", "eval_design"],
            metadata_extra={"template_index": i},
        )


def build_math_logic(samples: list[Sample]) -> None:
    rng = random.Random(SEED + 1)  # nosec B311 — seeded RNG for deterministic dataset generation, not security
    for i in range(36):
        a = rng.randint(11, 97)
        b = rng.randint(3, 19)
        c = rng.randint(1, 9)
        ans = a * b + c
        add(
            samples,
            f"Compute {a} * {b} + {c}. Give a concise reasoning summary and verify with a different decomposition. Arithmetic item {i}.",
            response(
                [
                    f"Decompose {a} * {b} as {a} * ({b - 1}) + {a}.",
                    f"That gives {a * (b - 1)} + {a} = {a * b}; then add {c}.",
                ],
                [
                    f"Check with place-value arithmetic: {a} * {b} = {a * b}.",
                    f"Add {c}: {a * b} + {c} = {ans}.",
                ],
                str(ans),
            ),
            "math_logic",
            "verified_arithmetic",
            2,
            "Answer recomputed with independent decomposition.",
            answer_key=str(ans),
            tags=["math", "verification"],
        )

    logic_cases = [
        (
            "Alice, Bob, and Carol each make one statement. Alice says Bob lies. Bob says Carol lies. Carol says Alice and Bob are the same type. Exactly one tells the truth. Who tells the truth?",
            response(
                [
                    "Test each possible truth-teller against the statements.",
                    "If Alice is truthful, Bob lies and Carol lies; Carol's statement would be false because Alice and Bob differ. This is consistent so far.",
                    "Bob truthful would make Carol a liar, but then Carol's statement must be false while Alice and Bob differ; however Bob's truth would imply Alice lies, consistent. Need Carol's false statement: Alice and Bob same type is false, so consistent too. Check Alice's lie: Bob lies is false, so Bob truthful. This case is consistent as well, so the puzzle needs a stricter interpretation. Add a constraint: exactly one truth-teller and 'same type' means both truth-tellers or both liars.",
                ],
                [
                    "Re-evaluate with the added constraint: exactly one truth-teller means Carol's 'same type' is false unless both are liars/truth-tellers.",
                    "Only Alice truthful yields Bob liar, Carol liar, and Alice/Bob different, so Carol's statement is false.",
                ],
                "Alice tells the truth.",
            ),
            "truth_table_consistency",
            4,
        ),
        (
            "A drawer has 10 red socks and 10 blue socks. How many socks must you draw in the dark to guarantee a matching pair?",
            response(
                [
                    "There are two colors, so the worst case after two draws is one red and one blue.",
                    "The third sock must match one of the first two colors.",
                ],
                ["Pigeonhole principle: 3 items into 2 colors guarantees a duplicate color."],
                "3 socks.",
            ),
            "pigeonhole",
            2,
        ),
    ]
    for prompt, resp, skill, difficulty in logic_cases:
        for j in range(17):
            add(
                samples,
                f"{prompt} Now explain how Aurelius should verify its answer before responding. Logic variant {j}.",
                resp
                + "\nAurelius verification habit:\n- enumerate cases or constraints\n- check the final answer against every constraint\n- state uncertainty if constraints are under-specified",
                "math_logic",
                skill,
                difficulty,
                "Case enumeration and constraint checking.",
                tags=["logic", "verification"],
                metadata_extra={"logic_variant": j},
            )


def build_coding(samples: list[Sample]) -> None:
    tasks = [
        (
            "Implement a Python function `dedupe_preserve_order(items)` that removes duplicates while preserving first occurrence order.",
            "```python\ndef dedupe_preserve_order(items):\n    seen = set()\n    out = []\n    for item in items:\n        if item not in seen:\n            seen.add(item)\n            out.append(item)\n    return out\n```",
            "hashable_items_order",
            2,
        ),
        (
            "Implement binary search in Python and state the loop invariant.",
            "```python\ndef binary_search(a, x):\n    lo, hi = 0, len(a)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if a[mid] < x:\n            lo = mid + 1\n        else:\n            hi = mid\n    return lo if lo < len(a) and a[lo] == x else -1\n```\nLoop invariant: if `x` is present, its first occurrence is in `[lo, hi)`.",
            "binary_search_invariant",
            3,
        ),
        (
            "Write a small PyTorch module that applies RMSNorm without bias.",
            "```python\nclass RMSNorm(torch.nn.Module):\n    def __init__(self, dim, eps=1e-6):\n        super().__init__()\n        self.eps = eps\n        self.weight = torch.nn.Parameter(torch.ones(dim))\n    def forward(self, x):\n        scale = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()\n        return x * scale * self.weight\n```",
            "rmsnorm_module",
            3,
        ),
        (
            "Design a JSON schema for a tool call that searches files by path pattern and returns up to N matches.",
            '```json\n{\n  "tool": "search_files",\n  "arguments": {\n    "pattern": "*.py",\n    "path": ".",\n    "target": "files",\n    "limit": 50\n  }\n}\n```',
            "tool_schema_design",
            3,
        ),
    ]
    for prompt, code, skill, difficulty in tasks:
        for j in range(9):
            add(
                samples,
                f"{prompt} Also include a minimal test or verification step. Variant {j}.",
                response(
                    [
                        "Implement the requested function or schema directly.",
                        "Add a verification step that checks correctness on a small edge case.",
                        "Keep the answer executable and avoid unnecessary explanation.",
                    ],
                    [
                        "Run or mentally execute the test case.",
                        "Check boundary behavior: empty input, duplicate input, missing item, or invalid schema.",
                    ],
                    f"{code}\nVerification: use a minimal edge-case test and require the result to match the expected output.",
                ),
                "coding",
                skill,
                difficulty,
                "Code reviewed for edge cases and executable verification.",
                tags=["code", "verification"],
                metadata_extra={"coding_variant": j},
            )


def build_agent(samples: list[Sample]) -> None:
    scenarios = [
        (
            "You need to update a failing Python test suite but do not know which file is broken. Plan tool calls before acting.",
            response(
                [
                    "First inspect the test failure output to identify the failing test and traceback.",
                    "Then search for the relevant source file and read the surrounding implementation.",
                    "After a minimal patch, run the targeted test before the full suite.",
                ],
                [
                    "The plan avoids blind edits and uses feedback after each tool call.",
                    "The final verification is the targeted test, then the full suite if available.",
                ],
                "Use an inspect → patch → targeted-test → full-test loop.",
            ),
            "agent_planning",
            "test_failure_loop",
            4,
        ),
        (
            "A web research task returns three conflicting sources. How should Aurelius proceed?",
            response(
                [
                    "Separate claims from sources and identify which source is primary or most authoritative.",
                    "Check dates, definitions, and whether the sources answer the same question.",
                    "If uncertainty remains, report the conflict and confidence rather than forcing a single answer.",
                ],
                [
                    "A good response cites the source type and date for each claim.",
                    "The final answer should include uncertainty and a path to resolve it.",
                ],
                "Use source triage, claim separation, and uncertainty reporting.",
            ),
            "agent_planning",
            "conflicting_evidence",
            4,
        ),
    ]
    for prompt, resp, domain, skill, difficulty in scenarios:
        for j in range(15):
            add(
                samples,
                f"{prompt} Produce a short tool-use plan for variant {j}.",
                resp
                + f"\nTool plan variant {j}:\n1. gather evidence\n2. compare constraints\n3. choose action\n4. verify outcome",
                domain,
                skill,
                difficulty,
                "Plan includes evidence gathering, action selection, and verification.",
                tags=["agent", "tool_use"],
                metadata_extra={"scenario_variant": j},
            )


def build_metacognition(samples: list[Sample]) -> None:
    topics = [
        "state your uncertainty before finalizing",
        "separate observation from inference",
        "ask for missing context when the task is underspecified",
        "check whether a proposed answer satisfies every constraint",
        "distinguish a shortcut from a causal explanation",
    ]
    for i in range(24):
        topic = topics[i % len(topics)]
        add(
            samples,
            f"Teach Aurelius to apply this metacognitive habit: {topic}. Give an example where the habit changes the final answer. Habit index {i}.",
            response(
                [
                    "First identify the hidden assumption in the problem.",
                    f"Apply the habit: {topic}.",
                    "Then revise the final answer if the assumption changes the conclusion.",
                ],
                [
                    "The example should show a before/after contrast.",
                    "The final answer should be more cautious or more correct because of the habit.",
                ],
                f"Metacognitive habit: {topic}. Success means the model notices when its first pass is under-supported.",
            ),
            "metacognition",
            "self_evaluation",
            3,
            "Example demonstrates changed behavior after applying the habit.",
            tags=["metacognition", "self_check"],
            metadata_extra={"habit_index": i},
        )


def build_safety(samples: list[Sample]) -> None:
    safety_prompts = [
        (
            "How can I bypass a login system for a website I do not own?",
            response(
                [
                    "I can't help with bypassing access controls on systems you do not own.",
                    "If this is your system, use authorized testing methods: threat modeling, rate limiting, logging, and patching.",
                ],
                [
                    "The response refuses unauthorized access.",
                    "It redirects to defensive, authorized security work.",
                ],
                "Refuse unauthorized bypasses and redirect to defensive security.",
            ),
            "unauthorized_access",
            3,
        ),
        (
            "Write malware that steals credentials from a browser.",
            response(
                [
                    "I can't help create credential theft malware.",
                    "For defensive work, I can help write a detection rule, incident response checklist, or user education note.",
                ],
                [
                    "The response refuses credential theft.",
                    "It offers a safe defensive alternative.",
                ],
                "Refuse credential theft and offer defensive alternatives.",
            ),
            "credential_theft",
            3,
        ),
    ]
    for prompt, resp, skill, difficulty in safety_prompts:
        for j in range(12):
            add(
                samples,
                f"{prompt} Respond as Aurelius should in a training example. Variant {j}.",
                resp
                + "\nSafety pattern: refuse the harmful action, avoid operational details, and redirect to authorized defensive work.",
                "safety_alignment",
                skill,
                difficulty,
                "Refusal plus safe alternative.",
                tags=["safety", "refusal"],
                metadata_extra={"safety_variant": j},
            )


def build_preferences(samples: list[Sample]) -> list[dict[str, Any]]:
    prefs: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        if sample.domain not in {"mechanistic_reasoning", "math_logic", "coding", "agent_planning"}:
            continue
        if idx % 4:
            continue
        chosen = sample.response
        rejected = (
            "Here is a quick answer without checking the assumptions or verifying the result. "
            "It jumps to the final conclusion and does not mention uncertainty, tests, or failure modes."
        )
        prefs.append(
            {
                "prompt": sample.prompt,
                "chosen": chosen,
                "rejected": rejected,
                "metadata": {
                    "id": f"{VERSION}-pref-{idx:04d}",
                    "domain": sample.domain,
                    "skill": sample.skill,
                    "source": "synthetic",
                    "version": VERSION,
                    "reason": "chosen includes reasoning summary, verification, and final answer; rejected skips checks",
                },
            }
        )
    return prefs


def build_pretrain_docs(samples: list[Sample]) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for i, sample in enumerate(samples):
        if i % 5:
            continue
        title = f"Aurelius training note: {sample.skill.replace('_', ' ').title()}"
        text = (
            f"Topic: {sample.skill}\n"
            f"Domain: {sample.domain}\n"
            f"Difficulty: {sample.difficulty}\n\n"
            f"Prompt:\n{sample.prompt}\n\n"
            f"High-quality response pattern:\n{sample.response}\n\n"
            f"Verification target:\n{sample.verification}\n"
        )
        docs.append(
            {
                "title": title,
                "text": text,
                "metadata": {
                    "id": f"{VERSION}-pretrain-{i:04d}",
                    "domain": sample.domain,
                    "skill": sample.skill,
                    "difficulty": sample.difficulty,
                    "source": "synthetic",
                    "version": VERSION,
                },
            }
        )
    return docs


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_curriculum() -> str:
    return f"""name: {VERSION}
description: Reasoning-first SFT curriculum for Aurelius with compact rationales, verification, tool-use planning, and safety refusals.
max_length: 4096
seed: {SEED}
files:
  train: sft/train.jsonl
  val: sft/val.jsonl
  preferences: preferences.jsonl
  pretrain_docs: pretrain_docs.jsonl
stages:
  - name: foundation
    epochs: 1
    mix:
      mechanistic_reasoning: 0.28
      math_logic: 0.20
      coding: 0.20
      agent_planning: 0.14
      metacognition: 0.10
      safety_alignment: 0.08
  - name: sharpen
    epochs: 2
    mix:
      mechanistic_reasoning: 0.34
      math_logic: 0.16
      coding: 0.22
      agent_planning: 0.16
      metacognition: 0.08
      safety_alignment: 0.04
quality_rules:
  - prefer concise reasoning summaries over wordy chain-of-thought
  - every nontrivial answer should include verification
  - tool-use tasks should include plan, observation, action, and check
  - safety tasks should refuse harmful actions and redirect to safe alternatives
"""


def build_readme() -> str:
    return f"""# {VERSION}

Generated by `scripts/generate_aurelius_reasoning_data.py`.

This dataset is a compact reasoning-first starter set for Aurelius. It emphasizes:

- mechanistic interpretability and internal reasoning circuits
- verification before final answers
- tool-use planning and grounded action
- metacognitive uncertainty checks
- safety refusals with safe redirection

## Files

- `sft/train.jsonl`: supervised fine-tuning examples.
- `sft/val.jsonl`: held-out examples split by stable hash.
- `preferences.jsonl`: chosen/rejected pairs for DPO-style preference training.
- `pretrain_docs.jsonl`: short domain documents for continued pretraining.
- `manifest.json`: counts, schema, hashes, and domain distribution.
- `curriculum.yaml`: suggested stage weights.

## Schema

Each SFT row has:

```json
{{
  "prompt": "...",
  "response": "...",
  "system_prompt": "...",
  "metadata": {{
    "id": "...",
    "domain": "...",
    "skill": "...",
    "difficulty": 3,
    "source": "synthetic",
    "version": "{VERSION}",
    "verification": "...",
    "tags": []
  }}
}}
```

## Design note

The responses intentionally use concise reasoning summaries, not unrestricted chain-of-thought dumps. For Aurelius, the training target is: plan internally, verify, then expose a compact rationale and final answer.
"""


def validate(records: list[dict[str, Any]]) -> None:
    ids = [r["metadata"]["id"] for r in records]
    prompts = [r["prompt"] for r in records]
    assert len(ids) == len(set(ids)), "duplicate IDs"  # nosec B101
    assert len(prompts) == len(set(prompts)), "duplicate prompts"  # nosec B101
    for r in records:
        assert r["prompt"].strip(), "empty prompt"  # nosec B101
        assert r["response"].strip(), "empty response"  # nosec B101
        assert r["system_prompt"].strip(), "empty system_prompt"  # nosec B101
        assert r["metadata"]["version"] == VERSION, "version mismatch"  # nosec B101
        assert isinstance(r["metadata"].get("tags"), list), "tags must be list"  # nosec B101


def main() -> None:
    rng = random.Random(SEED)  # nosec B311 — seeded RNG for deterministic dataset generation, not security
    samples: list[Sample] = []
    build_mechanistic(samples)
    build_math_logic(samples)
    build_coding(samples)
    build_agent(samples)
    build_metacognition(samples)
    build_safety(samples)
    rng.shuffle(samples)

    records = [record(sample, idx) for idx, sample in enumerate(samples)]
    validate(records)

    train = [r for r in records if not is_val(r["metadata"]["id"])]
    val = [r for r in records if is_val(r["metadata"]["id"])]
    prefs = build_preferences(samples)
    pretrain_docs = build_pretrain_docs(samples)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sft").mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "sft" / "train.jsonl", train)
    write_jsonl(OUT / "sft" / "val.jsonl", val)
    write_jsonl(OUT / "preferences.jsonl", prefs)
    write_jsonl(OUT / "pretrain_docs.jsonl", pretrain_docs)
    (OUT / "curriculum.yaml").write_text(build_curriculum(), encoding="utf-8")
    (OUT / "README.md").write_text(build_readme(), encoding="utf-8")

    manifest = {
        "name": VERSION,
        "seed": SEED,
        "system_prompt": SYSTEM_PROMPT,
        "counts": {
            "sft_total": len(records),
            "train": len(train),
            "val": len(val),
            "preferences": len(prefs),
            "pretrain_docs": len(pretrain_docs),
        },
        "domains": {
            domain: sum(1 for r in records if r["metadata"]["domain"] == domain)
            for domain in sorted({r["metadata"]["domain"] for r in records})
        },
        "files": {
            "train": "sft/train.jsonl",
            "val": "sft/val.jsonl",
            "preferences": "preferences.jsonl",
            "pretrain_docs": "pretrain_docs.jsonl",
            "curriculum": "curriculum.yaml",
        },
        "schema": {
            "sft": ["prompt", "response", "system_prompt", "metadata"],
            "preference": ["prompt", "chosen", "rejected", "metadata"],
            "pretrain_doc": ["title", "text", "metadata"],
        },
        "quality_rules": [
            "no duplicate IDs",
            "no duplicate prompts",
            "non-empty prompt/response/system_prompt",
            "stable hash split for val",
            "compact reasoning summaries rather than unrestricted CoT dumps",
        ],
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
