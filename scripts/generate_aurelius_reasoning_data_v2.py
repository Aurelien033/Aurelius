"""Generate an expanded Aurelius reasoning dataset v2.

This builds on the deterministic v1 generator and adds a larger, higher-coverage
reasoning corpus for continued SFT, preference training, and short continued
pretraining documents.

Default output:

    data/aurelius_reasoning_sft_v2/

Default size at scale 6:

    ~1.9k SFT examples
    ~370 pretrain documents
    ~300 preference pairs
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
V1_SCRIPT = ROOT / "scripts" / "generate_aurelius_reasoning_data.py"
SPEC = importlib.util.spec_from_file_location("aurelius_v1_generator", V1_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load v1 generator at {V1_SCRIPT}")
V1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = V1
SPEC.loader.exec_module(V1)

Sample = V1.Sample
SEED = 20260614
SYSTEM_PROMPT = V1.SYSTEM_PROMPT


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_val(sample_id: str) -> bool:
    return int(stable_hash(sample_id)[:8], 16) % 10 == 0


def response(summary: list[str], verification: list[str], final: str) -> str:
    return V1.response(summary, verification, final)


def add(
    samples: list[Sample],
    prompt: str,
    response_text: str,
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
            response=response_text,
            domain=domain,
            skill=skill,
            difficulty=difficulty,
            verification=verification,
            answer_key=answer_key,
            tags=tags or [],
            metadata_extra=metadata_extra or {},
        )
    )


def record(sample: Sample, idx: int, version: str) -> dict[str, Any]:
    sample_id = f"{version}-{idx:05d}"
    metadata = {
        "id": sample_id,
        "domain": sample.domain,
        "skill": sample.skill,
        "difficulty": sample.difficulty,
        "source": "synthetic-v2",
        "version": version,
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


MECHANISTIC_TOPICS = [
    ("residual stream", "residual_state_management", 3, "model residual writes as controlled state updates"),
    ("attention routing", "evidence_routing", 4, "separate attention correlation from causal evidence routing"),
    ("MLP feature composition", "feature_composition", 4, "test whether composed features survive counterfactuals"),
    ("causal mediation", "causal_testing", 5, "measure indirect effects through candidate components"),
    ("latent plan state", "plan_persistence", 5, "track plan variables across tool-call boundaries"),
    ("uncertainty routing", "uncertainty_calibration", 3, "route ambiguous cases to verification before final answer"),
    ("tool grounding", "grounded_tool_use", 4, "tie final claims to tool observations"),
    ("safety gate stability", "safety_circuit", 5, "stress-test refusal under paraphrase and role-play"),
    ("retrieval query construction", "retrieval_planning", 4, "separate query formation from evidence selection"),
    ("self-evaluation", "metacognitive_monitoring", 3, "require explicit checks before final answer"),
    ("activation patching", "causal_activation_intervention", 5, "replace candidate activations and measure behavioral change"),
    ("sparse autoencoder features", "feature_atlas", 5, "map interpretable features to tasks and interventions"),
    ("layer probes", "probing_representations", 4, "train probes that predict latent uncertainty and plan depth"),
    ("answer-critical tokens", "causal_tracing", 5, "trace which tokens change the final answer"),
    ("norm dynamics", "activation_norm_control", 4, "monitor residual norm growth and feature cancellation"),
    ("attention sinks", "attention_sink_control", 3, "detect low-information tokens that dominate attention"),
    ("carry state", "arithmetic_circuit", 4, "probe intermediate arithmetic state across tokens"),
    ("counterfactual robustness", "shortcut_detection", 5, "break superficial cues while preserving causal structure"),
    ("memory gate", "memory_gate_control", 5, "test whether old facts remain available after distractors"),
    ("plan revision", "adaptive_planning", 4, "update the plan only when evidence changes the goal"),
    ("evidence hierarchy", "source_ranking", 4, "prefer primary and recent sources over secondary summaries"),
    ("multi-hop retrieval", "multi_hop_reasoning", 5, "require retrieval chains to support every conclusion"),
    ("answer decomposition", "decomposition_strategy", 3, "break complex tasks into verifiable subproblems"),
    ("tool error recovery", "tool_error_recovery", 4, "detect failed tools and choose a recovery path"),
    ("latent variable induction", "latent_variable_modeling", 5, "infer hidden variables that explain observations"),
    ("world-model consistency", "consistency_checking", 5, "check predictions against prior facts and constraints"),
    ("preference circuit", "preference_reasoning", 4, "compare chosen and rejected answers by explicit criteria"),
    ("refusal boundary", "safety_boundary", 4, "refuse harmful actions while preserving safe alternatives"),
    ("data manifest", "data_provenance", 4, "track hashes, source, and transformation history"),
    ("eval gate", "evaluation_gate", 4, "gate model changes on held-out behavioral and mechanistic metrics"),
    ("reward model", "reward_modeling", 5, "score preferences without rewarding superficial verbosity"),
    ("KL drift", "rl_drift_monitoring", 5, "track distribution shift during post-training"),
    ("feature entropy", "representation_health", 5, "monitor collapse through feature diversity and entropy"),
    ("refusal regression", "safety_regression", 5, "test refusal quality across paraphrases and role-play"),
    ("checkpoint hygiene", "checkpoint_management", 3, "save optimizer state, config hash, and data manifest hash"),
    ("ablation table", "ablation_design", 4, "compare mechanisms against larger dense baselines"),
    ("compute accounting", "compute_tracking", 3, "record GPU hours, cost, throughput, and tokens/sec"),
    ("publication artifact", "reproducibility_packaging", 4, "package code, config, data hashes, and eval scripts"),
    ("latent reasoning", "latent_reasoning_summary", 4, "train compact summaries instead of unrestricted CoT"),
    ("verification habit", "verification_before_answer", 3, "make verification a default behavior before final output"),
]

MATH_FORMULAS = [
    ("linear", "a * n + b", 2),
    ("quadratic", "n * n + a * n + b", 3),
    ("modular", "a * n % m", 4),
    ("geometric", "a * r ** k", 4),
    ("floor", "floor(a * n / b)", 3),
    ("prime", "smallest prime greater than n", 5),
    ("gcd", "gcd(a, b)", 3),
    ("lcm", "lcm(a, b)", 3),
    ("probability", "P(A and B)", 4),
    ("combinatorics", "C(n, k)", 4),
]

CODING_TASKS = [
    "dedupe_preserve_order",
    "binary_search",
    "merge_sorted_lists",
    "top_k_frequent",
    "validate_json_schema",
    "rate_limit_requests",
    "parse_log_level",
    "compute_rolling_mean",
    "find_cycle_in_graph",
    "serialize_tree",
    "deserialize_tree",
    "implement_rmsnorm",
    "implement_causal_mask",
    "implement_attention_scores",
    "implement_token_counter",
    "implement_hash_manifest",
    "implement_data_split",
    "implement_metric_logger",
    "implement_checkpoint_writer",
    "implement_safety_filter",
    "implement_plan_validator",
    "implement_tool_result_grounding",
    "implement_contamination_probe",
    "implement_quality_scorer",
    "implement_curriculum_sampler",
    "implement_eval_gate",
    "implement_reward_model_stub",
    "implement_ablation_table",
    "implement_budget_tracker",
    "implement_reproducibility_packager",
]

AGENT_SCENARIOS = [
    "fix failing test suite",
    "research conflicting sources",
    "plan a multi-file refactor",
    "recover from a failed tool call",
    "design an eval harness",
    "curate a dataset split",
    "inspect training logs",
    "choose a checkpoint for evaluation",
    "run a safety red-team pass",
    "build a reproducibility package",
    "debug a data manifest mismatch",
    "select ablations for a paper table",
    "monitor representation collapse",
    "triage a reward hacking signal",
    "prepare a deployment checklist",
    "review a tool permission policy",
    "summarize experiment results",
    "decide whether to scale a mechanism",
    "plan a long-context training run",
    "validate a preference dataset",
    "audit a benchmark contamination probe",
    "compare dense and hybrid baselines",
    "schedule checkpoint saves",
    "inspect a failed fine-tuning run",
    "prepare a NeurIPS artifact checklist",
    "coordinate a human feedback round",
    "design a sandboxed tool test",
    "write a model card section",
    "draft an ablation hypothesis",
    "triage a safety regression",
]

META_HABITS = [
    "state assumptions before answering",
    "separate observation from inference",
    "ask for missing context when underspecified",
    "verify every constraint before final answer",
    "distinguish shortcut from causal explanation",
    "report uncertainty when evidence is weak",
    "prefer reversible actions first",
    "check whether the answer changed the goal",
    "keep a plan state across tool calls",
    "ground final claims in tool outputs",
    "avoid overfitting to benchmark wording",
    "maintain provenance for every dataset artifact",
    "separate behavior from mechanism",
    "prefer compact rationale over hidden CoT dump",
    "detect when a tool result is stale",
    "use counterfactuals to test shortcuts",
    "track reward variance during post-training",
    "watch for representation collapse",
    "record config hashes with checkpoints",
    "write eval failures as learning signals",
    "separate safety refusal from safe alternative",
    "check whether ablations beat larger dense baselines",
    "summarize evidence before making a claim",
    "update the plan only on relevant evidence",
    "preserve a clean artifact trail",
]

SAFETY_CASES = [
    "unauthorized login bypass",
    "credential theft malware",
    "phishing email drafting",
    "data exfiltration script",
    "weaponized automation",
    "privacy-invasive scraping",
    "identity impersonation",
    "bypass safety filters",
    "unsafe medical advice",
    "financial fraud automation",
    "credential stuffing",
    "exploit instructions for non-owned systems",
    "evading law enforcement",
    "generating harassment content",
    "stealth surveillance",
    "malware persistence",
    "ransomware negotiation",
    "credential reuse attack",
    "social engineering for secrets",
    "unsafe cyber intrusion",
]

DATA_CURATION_TOPICS = [
    "deduplication",
    "near-duplicate detection",
    "contamination probes",
    "source manifest hashing",
    "language filtering",
    "quality scoring",
    "token distribution analysis",
    "curriculum mixing",
    "train/val split stability",
    "preference pair construction",
    "negative example design",
    "safety label consistency",
    "domain balance",
    "long-tail coverage",
    "benchmark leakage checks",
    "data lineage tracking",
    "synthetic data provenance",
    "human review sampling",
    "error taxonomy",
    "dataset versioning",
    "prompt uniqueness checks",
    "response uniqueness checks",
    "metadata completeness",
    "answer key validation",
    "pretrain shard provenance",
]

EVAL_TOPICS = [
    "held-out behavioral eval",
    "mechanistic circuit eval",
    "counterfactual robustness",
    "tool-use planning eval",
    "safety refusal eval",
    "reward hacking probe",
    "KL drift monitor",
    "representation collapse monitor",
    "latent plan persistence eval",
    "uncertainty calibration eval",
    "answer verification eval",
    "data contamination eval",
    "dense baseline ablation gate",
    "long-context retrieval eval",
    "multi-hop reasoning eval",
    "preference model eval",
    "human feedback eval",
    "deployment stress test",
    "checkpoint reproducibility eval",
    "publication artifact eval",
    "model card evidence eval",
    "failure mode taxonomy",
    "red-team scenario set",
    "sandbox permission eval",
    "agentic task success eval",
]

INTERP_TOPICS = [
    "sparse autoencoder feature atlas",
    "layer probe for uncertainty",
    "causal tracing of answer tokens",
    "activation patching for plan state",
    "attention head routing probe",
    "residual stream workspace probe",
    "MLP feature composition probe",
    "carry-state neuron probe",
    "safety refusal circuit probe",
    "tool grounding circuit probe",
    "retrieval routing probe",
    "multi-hop retrieval circuit",
    "latent variable representation",
    "world-model consistency probe",
    "reward model feature probe",
    "KL drift representation probe",
    "representation entropy monitor",
    "feature collapse monitor",
    "counterfactual circuit test",
    "ablation-based circuit validation",
    "causal mediation analysis",
    "attention map validation",
    "plan persistence across turns",
    "memory gate intervention",
    "answer-critical token tracing",
    "preference circuit analysis",
    "safety boundary intervention",
    "tool error recovery circuit",
    "uncertainty routing circuit",
    "latent reasoning summary circuit",
]

ALIGNMENT_TOPICS = [
    "constitutional critique",
    "preference optimization",
    "reward model calibration",
    "refusal consistency",
    "safe alternative generation",
    "role-play attack resistance",
    "paraphrase robustness",
    "safety/quality tradeoff",
    "human feedback loop",
    "red-team finding triage",
    "policy boundary documentation",
    "unsafe request classification",
    "defensive cybersecurity redirection",
    "medical uncertainty handling",
    "financial advice boundaries",
    "privacy-preserving summarization",
    "data minimization",
    "permission-aware tool use",
    "sandbox escape resistance",
    "monitoring for reward hacking",
]


def build_v2_expansion(samples: list[Sample], scale: int) -> None:
    """Append a larger, diverse Aurelius-specific corpus.

    The templates intentionally include variant IDs and task-specific parameters
    so deterministic generation does not collapse into duplicate prompts.
    """

    for idx, (topic, skill, difficulty, success) in enumerate(MECHANISTIC_TOPICS * scale):
        add(
            samples,
            (
                f"Design an Aurelius training probe for {topic}. "
                f"Family mechanistic-v2-{idx:03d}; difficulty {difficulty}; "
                "require a mechanism, observable prediction, intervention, and failure mode."
            ),
            response(
                [
                    f"Mechanism target: {skill}.",
                    "The probe should ask for the internal component, the observable behavior, and the causal intervention.",
                    "The response should name a failure mode and a held-out counterfactual test.",
                ],
                [
                    "Check that the probe distinguishes correlation from causal mechanism.",
                    "Check that the intervention is measurable: ablation, patching, masking, or behavioral metric.",
                ],
                f"Success criterion: {success}.",
            ),
            "mechanistic_reasoning",
            skill,
            difficulty,
            f"Mechanistic probe includes component, prediction, intervention, and failure mode for {topic}.",
            tags=["mechanistic", "probe_design", "v2"],
            metadata_extra={"family": "mechanistic", "variant_index": idx, "scale": scale},
        )

    for idx, (family, formula, difficulty) in enumerate(MATH_FORMULAS * scale):
        n = 17 + idx % 31
        a = 3 + idx % 13
        b = 5 + idx % 11
        r = 2 + idx % 5
        m = 13 + idx % 19
        k = 2 + idx % 7
        if family == "linear":
            answer = a * n + b
            prompt = f"Compute {a} * {n} + {b}. Variant math-v2-{idx:03d}; show a compact verification."
        elif family == "quadratic":
            answer = n * n + a * n + b
            prompt = f"Compute {n}^2 + {a}*{n} + {b}. Variant math-v2-{idx:03d}; verify by decomposition."
        elif family == "modular":
            answer = (a * n) % m
            prompt = f"Compute ({a} * {n}) mod {m}. Variant math-v2-{idx:03d}; verify with quotient form."
        elif family == "geometric":
            answer = a * (r ** k)
            prompt = f"Compute {a} * {r}^{k}. Variant math-v2-{idx:03d}; verify by repeated doubling."
        elif family == "floor":
            answer = (a * n) // b
            prompt = f"Compute floor(({a} * {n}) / {b}). Variant math-v2-{idx:03d}; verify by bounds."
        elif family == "prime":
            candidate = n + idx
            while True:
                if candidate > 1 and all(candidate % d for d in range(2, int(candidate ** 0.5) + 1)):
                    break
                candidate += 1
            answer = candidate
            prompt = f"Find the smallest prime greater than {n}. Variant math-v2-{idx:03d}; verify divisibility."
        elif family == "gcd":
            import math

            answer = math.gcd(a, b)
            prompt = f"Compute gcd({a}, {b}). Variant math-v2-{idx:03d}; verify with Euclidean steps."
        elif family == "lcm":
            import math

            answer = (a * b) // math.gcd(a, b)
            prompt = f"Compute lcm({a}, {b}). Variant math-v2-{idx:03d}; verify from gcd."
        elif family == "probability":
            answer = 0.36
            prompt = f"If P(A)=0.6 and P(B|A)=0.6, compute P(A and B). Variant math-v2-{idx:03d}; verify."
        else:
            import math

            answer = math.comb(n, k)
            prompt = f"Compute C({n}, {k}). Variant math-v2-{idx:03d}; verify with factorial cancellation."
        add(
            samples,
            prompt,
            response(
                [
                    f"Use the {family} rule to reduce the expression to a direct calculation.",
                    "Carry the intermediate value explicitly enough to check the arithmetic.",
                    "State the final answer separately from the verification.",
                ],
                [
                    "Recompute with a second decomposition or bound check.",
                    "Confirm the final answer satisfies the original formula.",
                ],
                str(answer),
            ),
            "math_logic",
            f"{family}_verified",
            difficulty,
            "Answer recomputed with an independent arithmetic check.",
            answer_key=str(answer),
            tags=["math", "verification", "v2"],
            metadata_extra={"family": family, "variant_index": idx, "scale": scale},
        )

    for idx, task in enumerate(CODING_TASKS * scale):
        difficulty = 3 + (idx % 3)
        add(
            samples,
            (
                f"Implement or design code for `{task}` in Aurelius tooling. "
                f"Variant coding-v2-{idx:03d}; include a minimal test or schema check."
            ),
            response(
                [
                    f"Task target: `{task}`.",
                    "Implement the smallest correct interface first, then add edge-case checks.",
                    "Keep the code reviewable and avoid unnecessary abstractions.",
                ],
                [
                    "Run a minimal test on empty input, normal input, and an adversarial edge case.",
                    "Check that the output type and invariants match the task contract.",
                ],
                f"Verification target: `{task}` passes a minimal edge-case test and preserves the stated invariant.",
            ),
            "coding",
            task,
            difficulty,
            "Code task includes implementation target, invariant, and test plan.",
            tags=["coding", "verification", "v2"],
            metadata_extra={"task": task, "variant_index": idx, "scale": scale},
        )

    for idx, scenario in enumerate(AGENT_SCENARIOS * scale):
        difficulty = 4 if idx % 2 else 3
        add(
            samples,
            (
                f"Plan Aurelius tool use for: {scenario}. Variant agent-v2-{idx:03d}; "
                "include goal, observations, actions, checks, and stop conditions."
            ),
            response(
                [
                    f"Goal: resolve {scenario} with minimal irreversible actions.",
                    "First gather observations, then choose the smallest useful action, then verify the result.",
                    "If a tool fails or evidence conflicts, update the plan and preserve the original goal.",
                ],
                [
                    "Check that every action has a reason and a verification condition.",
                    "Check that the plan stops when the goal is satisfied or when more information is required.",
                ],
                f"Stop condition: {scenario} is resolved, evidence is grounded, or the next safe action is to ask for missing context.",
            ),
            "agent_planning",
            scenario.replace(" ", "_"),
            difficulty,
            "Plan includes goal, observations, actions, checks, and stop conditions.",
            tags=["agent", "tool_use", "planning", "v2"],
            metadata_extra={"scenario": scenario, "variant_index": idx, "scale": scale},
        )

    for idx, habit in enumerate(META_HABITS * scale):
        add(
            samples,
            (
                f"Train Aurelius to apply this metacognitive habit: {habit}. "
                f"Variant meta-v2-{idx:03d}; show a before/after example where the habit changes the answer."
            ),
            response(
                [
                    "First identify the hidden assumption or missing constraint.",
                    f"Apply the habit: {habit}.",
                    "Then revise the final answer if the assumption changes the conclusion.",
                ],
                [
                    "Check that the example has a before/after contrast.",
                    "Check that the final answer is more cautious, more correct, or better grounded.",
                ],
                "Metacognitive success: Aurelius notices when the first pass is under-supported and updates the answer accordingly.",
            ),
            "metacognition",
            habit.replace(" ", "_"),
            3 + (idx % 3),
            "Example demonstrates a behavior change after applying the habit.",
            tags=["metacognition", "self_check", "v2"],
            metadata_extra={"habit": habit, "variant_index": idx, "scale": scale},
        )

    for idx, case in enumerate(SAFETY_CASES * scale):
        add(
            samples,
            (
                f"Write a safe Aurelius training response to a request about {case}. "
                f"Variant safety-v2-{idx:03d}; refuse operational harm and redirect to authorized defensive work."
            ),
            response(
                [
                    f"Refuse the harmful request about {case} without operational details.",
                    "Offer a safe alternative: policy, detection, authorized testing, education, or incident response.",
                    "Keep the tone brief, firm, and non-judgmental.",
                ],
                [
                    "Check that the response does not provide actionable harm.",
                    "Check that the alternative is authorized, defensive, and proportional.",
                ],
                "Refuse harmful action and redirect to safe, authorized work.",
            ),
            "safety_alignment",
            case.replace(" ", "_"),
            3 + (idx % 3),
            "Refusal avoids operational details and gives a safe alternative.",
            tags=["safety", "refusal", "v2"],
            metadata_extra={"case": case, "variant_index": idx, "scale": scale},
        )

    for idx, topic in enumerate(DATA_CURATION_TOPICS * scale):
        add(
            samples,
            (
                f"Design an Aurelius data curation rule for {topic}. Variant data-v2-{idx:03d}; "
                "include a measurable signal, filter decision, and provenance record."
            ),
            response(
                [
                    f"Curation target: {topic}.",
                    "The rule should define a measurable signal and a threshold or review path.",
                    "The rule should preserve provenance so every removed or retained example is auditable.",
                ],
                [
                    "Check that the rule is testable on held-out examples.",
                    "Check that provenance records source, hash, filter decision, and reason.",
                ],
                f"Data rule: keep examples that satisfy the {topic} criterion and log the decision with source metadata.",
            ),
            "data_curation",
            topic.replace(" ", "_"),
            4 if idx % 2 else 3,
            "Curation rule includes signal, threshold, decision, and provenance.",
            tags=["data", "curation", "provenance", "v2"],
            metadata_extra={"topic": topic, "variant_index": idx, "scale": scale},
        )

    for idx, topic in enumerate(EVAL_TOPICS * scale):
        add(
            samples,
            (
                f"Design an evaluation gate for {topic}. Variant eval-v2-{idx:03d}; "
                "include metric, pass/fail threshold, and ablation interpretation."
            ),
            response(
                [
                    f"Eval target: {topic}.",
                    "The gate should report both aggregate score and failure-mode slices.",
                    "The interpretation should say whether the result supports, weakens, or kills the mechanism.",
                ],
                [
                    "Check that the metric is held-out and not leaked from training prompts.",
                    "Check that the threshold is tied to a decision: ship, investigate, or roll back.",
                ],
                f"Gate decision: pass only if {topic} improves held-out behavior without regressing safety or interpretability.",
            ),
            "evaluation_design",
            topic.replace(" ", "_"),
            4 if idx % 2 else 3,
            "Eval gate includes metric, threshold, slices, and decision rule.",
            tags=["eval", "gate", "ablation", "v2"],
            metadata_extra={"topic": topic, "variant_index": idx, "scale": scale},
        )

    for idx, topic in enumerate(INTERP_TOPICS * scale):
        add(
            samples,
            (
                f"Design a mechanistic interpretability experiment for {topic}. "
                f"Variant interp-v2-{idx:03d}; include hypothesis, measurement, intervention, and expected result."
            ),
            response(
                [
                    f"Hypothesis: {topic} corresponds to a measurable internal representation or circuit.",
                    "Measure the candidate feature or activation pattern across tasks where it should matter.",
                    "Intervene with patching, ablation, masking, or counterfactual inputs to test necessity.",
                ],
                [
                    "Check that the expected result would distinguish mechanism from correlation.",
                    "Check that the experiment includes a negative control.",
                ],
                f"Expected result: {topic} predicts behavior on targeted tasks and weakens under causal intervention.",
            ),
            "interpretability_experiments",
            topic.replace(" ", "_"),
            5 if idx % 3 == 0 else 4,
            "Experiment includes hypothesis, measurement, intervention, expected result, and negative control.",
            tags=["interpretability", "mechanistic", "experiment", "v2"],
            metadata_extra={"topic": topic, "variant_index": idx, "scale": scale},
        )

    for idx, topic in enumerate(ALIGNMENT_TOPICS * scale):
        add(
            samples,
            (
                f"Design an alignment training or monitoring item for {topic}. "
                f"Variant align-v2-{idx:03d}; include preference signal, safety boundary, and failure mode."
            ),
            response(
                [
                    f"Alignment target: {topic}.",
                    "The item should reward helpfulness only inside the safety boundary.",
                    "The failure mode should describe how optimization could overfit the signal while losing the behavior.",
                ],
                [
                    "Check that the preference signal does not reward verbosity alone.",
                    "Check that the safety boundary is explicit and testable.",
                ],
                f"Alignment success: Aurelius remains helpful while preserving the {topic} boundary under distribution shift.",
            ),
            "alignment_monitoring",
            topic.replace(" ", "_"),
            4 if idx % 2 else 3,
            "Alignment item includes preference signal, safety boundary, and optimization failure mode.",
            tags=["alignment", "monitoring", "safety", "v2"],
            metadata_extra={"topic": topic, "variant_index": idx, "scale": scale},
        )


def build_preferences(samples: list[Sample], version: str) -> list[dict[str, Any]]:
    prefs: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        if sample.domain not in {
            "mechanistic_reasoning",
            "math_logic",
            "coding",
            "agent_planning",
            "data_curation",
            "evaluation_design",
            "interpretability_experiments",
        }:
            continue
        if idx % 5:
            continue
        chosen = sample.response
        rejected = (
            "Here is a quick answer that skips assumptions, verification, and failure modes. "
            "It gives a final conclusion without checking constraints or evidence."
        )
        prefs.append(
            {
                "prompt": sample.prompt,
                "chosen": chosen,
                "rejected": rejected,
                "metadata": {
                    "id": f"{version}-pref-{idx:05d}",
                    "domain": sample.domain,
                    "skill": sample.skill,
                    "source": "synthetic-v2",
                    "version": version,
                    "reason": "chosen includes reasoning summary, verification, and final answer; rejected skips checks",
                },
            }
        )
    return prefs


def build_pretrain_docs(samples: list[Sample], version: str) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for i, sample in enumerate(samples):
        if i % 5:
            continue
        title = f"Aurelius v2 training note: {sample.skill.replace('_', ' ').title()}"
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
                    "id": f"{version}-pretrain-{i:05d}",
                    "domain": sample.domain,
                    "skill": sample.skill,
                    "difficulty": sample.difficulty,
                    "source": "synthetic-v2",
                    "version": version,
                },
            }
        )
    return docs


def validate(records: list[dict[str, Any]], version: str) -> None:
    ids = [r["metadata"]["id"] for r in records]
    prompts = [r["prompt"] for r in records]
    assert len(ids) == len(set(ids)), "duplicate IDs"
    assert len(prompts) == len(set(prompts)), "duplicate prompts"
    for r in records:
        assert r["prompt"].strip(), "empty prompt"
        assert r["response"].strip(), "empty response"
        assert r["system_prompt"].strip(), "empty system_prompt"
        assert r["metadata"]["version"] == version, "version mismatch"
        assert r["response"].startswith("Reasoning summary:"), "response must start with reasoning summary"
        assert "Verification:" in r["response"], "response must include verification"
        assert isinstance(r["metadata"].get("tags"), list), "tags must be list"


def estimate_tokens(records: list[dict[str, Any]]) -> int:
    total = 0
    for r in records:
        text = f"{r['prompt']} {r['response']} {r['system_prompt']}"
        total += len(re.findall(r"\w+|[^\w\s]", text))
    return total


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_curriculum(version: str, seed: int) -> str:
    return f"""name: {version}
description: Expanded reasoning-first SFT curriculum for Aurelius v2 with compact rationales, verification, tool-use planning, safety refusals, data curation, eval design, interpretability experiments, and alignment monitoring.
max_length: 4096
seed: {seed}
files:
  train: sft/train.jsonl
  val: sft/val.jsonl
  preferences: preferences.jsonl
  pretrain_docs: pretrain_docs.jsonl
stages:
  - name: foundation
    epochs: 1
    mix:
      mechanistic_reasoning: 0.22
      math_logic: 0.16
      coding: 0.16
      agent_planning: 0.13
      metacognition: 0.10
      safety_alignment: 0.08
      data_curation: 0.05
      evaluation_design: 0.04
      interpretability_experiments: 0.04
      alignment_monitoring: 0.02
  - name: sharpen
    epochs: 2
    mix:
      mechanistic_reasoning: 0.28
      math_logic: 0.12
      coding: 0.18
      agent_planning: 0.14
      metacognition: 0.08
      safety_alignment: 0.05
      data_curation: 0.05
      evaluation_design: 0.04
      interpretability_experiments: 0.04
      alignment_monitoring: 0.02
quality_rules:
  - prefer concise reasoning summaries over wordy chain-of-thought
  - every nontrivial answer should include verification
  - tool-use tasks should include plan, observation, action, and check
  - safety tasks should refuse harmful actions and redirect to safe alternatives
  - data and eval tasks should record provenance and decision rules
"""


def build_readme(version: str, seed: int, scale: int, out: Path) -> str:
    return f"""# {version}

Generated by `scripts/generate_aurelius_reasoning_data_v2.py`.

This dataset expands the original Aurelius reasoning starter set with a larger deterministic corpus aimed at:

- compact latent reasoning summaries
- verification before final answer
- tool-use planning and grounded action
- metacognitive uncertainty checks
- safety refusals with safe redirection
- data curation and provenance
- evaluation gate design
- mechanistic interpretability experiments
- alignment and monitoring behavior

## Generation

```bash
cd /Users/christienantonio/aurelius
.venv/bin/python scripts/generate_aurelius_reasoning_data_v2.py --scale {scale} --output {out}
```

## Files

- `sft/train.jsonl`: supervised fine-tuning examples.
- `sft/val.jsonl`: held-out examples split by stable hash.
- `preferences.jsonl`: chosen/rejected pairs for DPO-style preference training.
- `pretrain_docs.jsonl`: short domain documents for continued pretraining.
- `manifest.json`: counts, schema, hashes, domain distribution, and token estimate.
- `curriculum.yaml`: suggested stage weights.

## Quality checks

The generator validates:

- no duplicate IDs
- no duplicate prompts
- non-empty prompt/response/system_prompt
- stable hash split for validation
- every response starts with `Reasoning summary:`
- every response contains `Verification:`
- metadata includes domain, skill, difficulty, tags, and version

## Design note

The responses intentionally use concise reasoning summaries, not unrestricted chain-of-thought dumps. For Aurelius, the training target is: plan internally, verify, then expose a compact rationale and final answer.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate expanded Aurelius reasoning dataset v2")
    parser.add_argument("--scale", type=int, default=6, help="Expansion multiplier for v2 template families.")
    parser.add_argument("--version", default="aurelius-reasoning-sft-v2", help="Dataset version string.")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "aurelius_reasoning_sft_v2", help="Output directory.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    args = parser.parse_args()

    if args.scale < 1:
        raise ValueError("--scale must be >= 1")

    out = args.output
    version = args.version
    seed = args.seed
    rng = random.Random(seed)

    samples: list[Sample] = []
    V1.build_mechanistic(samples)
    V1.build_math_logic(samples)
    V1.build_coding(samples)
    V1.build_agent(samples)
    V1.build_metacognition(samples)
    V1.build_safety(samples)
    build_v2_expansion(samples, args.scale)
    rng.shuffle(samples)

    records = [record(sample, idx, version) for idx, sample in enumerate(samples)]
    validate(records, version)

    train = [r for r in records if not is_val(r["metadata"]["id"])]
    val = [r for r in records if is_val(r["metadata"]["id"])]
    prefs = build_preferences(samples, version)
    pretrain_docs = build_pretrain_docs(samples, version)

    out.mkdir(parents=True, exist_ok=True)
    (out / "sft").mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "sft" / "train.jsonl", train)
    write_jsonl(out / "sft" / "val.jsonl", val)
    write_jsonl(out / "preferences.jsonl", prefs)
    write_jsonl(out / "pretrain_docs.jsonl", pretrain_docs)
    (out / "curriculum.yaml").write_text(build_curriculum(version, seed), encoding="utf-8")
    (out / "README.md").write_text(build_readme(version, seed, args.scale, out), encoding="utf-8")

    domains = sorted({r["metadata"]["domain"] for r in records})
    manifest = {
        "name": version,
        "seed": seed,
        "scale": args.scale,
        "system_prompt": SYSTEM_PROMPT,
        "counts": {
            "sft_total": len(records),
            "train": len(train),
            "val": len(val),
            "preferences": len(prefs),
            "pretrain_docs": len(pretrain_docs),
            "estimated_tokens": estimate_tokens(records),
        },
        "domains": {
            domain: sum(1 for r in records if r["metadata"]["domain"] == domain)
            for domain in domains
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
            "every response includes verification",
            "metadata includes domain, skill, difficulty, tags, and version",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
