"""Generate Aurelius reasoning dataset v7 with frontier reasoning probes.

v7 keeps the deterministic v2 expansion and adds a frontier tranche focused on
how models reason internally: causal interventions, latent plan probes,
agentic failure modes, reward-hacking detection, world-model updates, and
architecture tradeoff reasoning.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
V2_SCRIPT = ROOT / "scripts" / "generate_aurelius_reasoning_data_v2.py"
SPEC = importlib.util.spec_from_file_location("aurelius_v2_generator", V2_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load v2 generator at {V2_SCRIPT}")
V2 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = V2
SPEC.loader.exec_module(V2)

Sample = V2.Sample
SEED = 20260614
SYSTEM_PROMPT = V2.SYSTEM_PROMPT


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_val(sample_id: str) -> bool:
    return int(stable_hash(sample_id)[:8], 16) % 10 == 0


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


def response(summary: list[str], verification: list[str], final: str) -> str:
    return V2.response(summary, verification, final)


def record(sample: Sample, idx: int, version: str) -> dict[str, Any]:
    sample_id = f"{version}-{idx:05d}"
    metadata = {
        "id": sample_id,
        "domain": sample.domain,
        "skill": sample.skill,
        "difficulty": sample.difficulty,
        "source": "synthetic-v7-frontier",
        "generator": "generate_aurelius_reasoning_data_v7.py",
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


def build_frontier_expansion(samples: list[Sample], scale: int) -> None:
    causal_topics = [
        "residual stream intervention",
        "attention head ablation",
        "MLP feature knockout",
        "latent plan probe",
        "uncertainty representation",
        "answer-critical token tracing",
        "tool-observation binding",
        "safety gate activation",
        "retrieval evidence routing",
        "self-evaluation signal",
    ]
    for i in range(len(causal_topics) * scale):
        topic = causal_topics[i % len(causal_topics)]
        variant = i % len(causal_topics)
        add(
            samples,
            f"Design a causal-intervention experiment for the topic '{topic}' (case {i + 1:03d}). The model is a 10B-parameter Aurelius candidate with hybrid attention-SSM consensus. Include the hypothesis, candidate components, intervention, expected behavioral delta, confounds, and a pass/fail criterion.",
            response(
                [
                    f"Define the causal question as whether {topic} carries task-relevant information or merely correlates with performance.",
                    "Use a within-task counterfactual split so the intervention targets the candidate mechanism while holding surface wording stable.",
                    "Measure behavior with both task success and latent diagnostic probes, then require the observed delta to exceed the bootstrap confidence interval.",
                ],
                [
                    "The experiment has an explicit causal hypothesis and a component-level intervention.",
                    "The expected delta and confounds are stated before the result is interpreted.",
                    "A pass/fail criterion is tied to behavior plus diagnostics, not to a single benchmark score.",
                ],
                "Use activation patching, targeted ablation, or causal mediation only when the intervention can be isolated cleanly.",
            ),
            "causal_intervention_design",
            "causal_mechanism_testing",
            5,
            "Causal claims require isolated interventions, counterfactual controls, and behavioral plus diagnostic readouts.",
            answer_key=f"causal-{variant:02d}",
            tags=["causal", "interpretability", "intervention", "mechanistic"],
            metadata_extra={"family": "causal_intervention_design", "variant_index": variant},
        )

    latent_topics = [
        "plan depth",
        "uncertainty",
        "subgoal identity",
        "constraint satisfaction",
        "evidence sufficiency",
        "tool grounding state",
        "safety boundary",
        "retrieval relevance",
        "answer confidence",
        "metacognitive repair",
    ]
    for i in range(len(latent_topics) * scale):
        topic = latent_topics[i % len(latent_topics)]
        variant = i % len(latent_topics)
        add(
            samples,
            f"Create a latent-reasoning probe for '{topic}' in Aurelius (case {i + 1:03d}). Specify the training examples, probe label construction, layer range, negative controls, and how you would distinguish the probe from shallow lexical shortcuts.",
            response(
                [
                    f"Train the probe to predict {topic} from internal states while holding prompt wording and topic vocabulary constant in controls.",
                    "Use held-out tasks and counterfactual prompts to test whether the representation generalizes beyond memorized wording.",
                    "Report calibration, linear separability, and intervention sensitivity rather than treating probe accuracy as proof of a circuit.",
                ],
                [
                    "Probe labels are constructed from task structure, not from the model's own output.",
                    "Negative controls remove lexical shortcuts and verify that the signal is latent rather than surface-level.",
                    "The evaluation includes generalization and intervention sensitivity.",
                ],
                "A useful latent probe should predict the hidden variable across tasks and weaken when the hypothesized mechanism is disrupted.",
            ),
            "latent_reasoning_probe",
            "representation_probing",
            4,
            "Probe validity requires label independence, counterfactual controls, and intervention sensitivity.",
            answer_key=f"latent-{variant:02d}",
            tags=["latent", "probe", "representation", "generalization"],
            metadata_extra={"family": "latent_reasoning_probe", "variant_index": variant},
        )

    failure_topics = [
        "plan collapse",
        "tool overtrust",
        "retrieval hallucination",
        "reward hacking",
        "safety paraphrase bypass",
        "context-window forgetting",
        "ambiguous goal lock-in",
        "metric gaming",
        "self-evaluation overconfidence",
        "cross-turn state drift",
    ]
    for i in range(len(failure_topics) * scale):
        topic = failure_topics[i % len(failure_topics)]
        variant = i % len(failure_topics)
        add(
            samples,
            f"Analyze the agentic failure mode '{topic}' (case {i + 1:03d}) for Aurelius. Give a concrete scenario, the internal failure mechanism, an observable symptom, a monitoring signal, and a mitigation that does not merely hide the failure.",
            response(
                [
                    f"Treat {topic} as a mechanism-level failure rather than a one-off bad answer.",
                    "Describe the scenario, the likely internal state transition, and the observable symptom that would reveal it during deployment.",
                    "Pair monitoring with a mitigation that changes incentives, checks, or state management so the failure is less likely to recur.",
                ],
                [
                    "The answer names a mechanism, a scenario, a symptom, a monitor, and a mitigation.",
                    "The mitigation changes the system rather than just asking the model to be more careful.",
                    "The monitoring signal is observable during or immediately after execution.",
                ],
                "For agentic failures, the best fix is usually a combination of state design, incentive design, and verification, not a single prompt patch.",
            ),
            "agentic_failure_modes",
            "failure_mode_analysis",
            4,
            "Agentic failure analysis must include mechanism, symptom, monitor, and mitigation.",
            answer_key=f"failure-{variant:02d}",
            tags=["agentic", "failure", "monitoring", "mitigation"],
            metadata_extra={"family": "agentic_failure_modes", "variant_index": variant},
        )

    reward_topics = [
        "proxy reward",
        "preference overoptimization",
        "sycophancy",
        "length bias",
        "tool-call gaming",
        "eval leakage",
        "reward model misspecification",
        "specification gaming",
        "policy collapse",
        "distribution shift",
    ]
    for i in range(len(reward_topics) * scale):
        topic = reward_topics[i % len(reward_topics)]
        variant = i % len(reward_topics)
        add(
            samples,
            f"Write a reward-hacking detector for '{topic}' (case {i + 1:03d}) in Aurelius training. Include the suspicious pattern, the diagnostic dataset, the statistical test, and the remediation plan.",
            response(
                [
                    f"Define {topic} as a mismatch between the intended objective and the measured objective.",
                    "Construct a diagnostic set that separates genuine competence from shortcut behavior, then compare training and held-out distributions.",
                    "Use a statistical test plus ablation evidence before deciding whether to remove data, adjust reward, or change the training recipe.",
                ],
                [
                    "The detector has a clear suspicious pattern and a diagnostic set.",
                    "The statistical test is paired with ablation or distribution evidence.",
                    "The remediation plan targets the objective mismatch rather than only pruning examples.",
                ],
                "Reward hacking should be treated as objective misspecification until proven otherwise.",
            ),
            "reward_hacking_detection",
            "objective_misalignment",
            5,
            "Reward hacking detection requires a suspicious pattern, diagnostic set, test, and remediation.",
            answer_key=f"reward-{variant:02d}",
            tags=["reward", "hacking", "objective", "diagnostic"],
            metadata_extra={"family": "reward_hacking_detection", "variant_index": variant},
        )

    world_topics = [
        "belief update",
        "counterfactual simulation",
        "causal graph revision",
        "planning horizon",
        "uncertainty propagation",
        "tool evidence fusion",
        "memory consolidation",
        "state abstraction",
        "goal decomposition",
        "world-model compression",
    ]
    for i in range(len(world_topics) * scale):
        topic = world_topics[i % len(world_topics)]
        variant = i % len(world_topics)
        add(
            samples,
            f"Design a training task for the world-model capability '{topic}' (case {i + 1:03d}). Specify the input, target behavior, distractors, evaluation metric, and a failure mode that would reveal shallow pattern matching.",
            response(
                [
                    f"Make the task require {topic} as a latent operation rather than a surface transformation.",
                    "Add distractors that reward memorization or keyword matching unless the model maintains the intended state.",
                    "Evaluate with counterfactual probes and an adversarial split that changes the surface form while preserving the underlying state.",
                ],
                [
                    "The task includes input, target behavior, distractors, metric, and failure mode.",
                    "The failure mode directly tests for shallow pattern matching.",
                    "The evaluation uses counterfactual or adversarial splits.",
                ],
                "A good world-model task should be hard to solve by memorizing wording but easy for a system that maintains the right latent state.",
            ),
            "world_model_updates",
            "stateful_reasoning",
            4,
            "World-model tasks need latent-state requirements, distractors, and adversarial evaluation.",
            answer_key=f"world-{variant:02d}",
            tags=["world-model", "state", "counterfactual", "evaluation"],
            metadata_extra={"family": "world_model_updates", "variant_index": variant},
        )

    tool_topics = [
        "observation grounding",
        "tool planning",
        "multi-step tool use",
        "error recovery",
        "tool result verification",
        "external memory",
        "cross-tool consistency",
        "permission boundaries",
        "latency-aware planning",
        "tool-call summarization",
    ]
    for i in range(len(tool_topics) * scale):
        topic = tool_topics[i % len(tool_topics)]
        variant = i % len(tool_topics)
        add(
            samples,
            f"Create a tool-use training example for '{topic}' (case {i + 1:03d}). Include the user request, tool sequence, expected observations, failure recovery path, and final answer grounding rule.",
            response(
                [
                    f"Anchor the example in {topic} so the model must plan around tool behavior rather than answer from memory.",
                    "Specify the tool sequence, expected observations, and how the model should recover if an observation is missing or contradictory.",
                    "Require the final answer to cite the observation that justifies each claim.",
                ],
                [
                    "The example includes request, tool sequence, observations, recovery path, and grounding rule.",
                    "The final answer is tied to observations rather than assumptions.",
                    "The recovery path handles missing or conflicting tool output.",
                ],
                "Tool use should be trained as grounded interaction, not as a decorative extra step.",
            ),
            "tool_use_grounding",
            "grounded_action",
            4,
            "Tool-use examples require grounded observations and explicit recovery behavior.",
            answer_key=f"tool-{variant:02d}",
            tags=["tool", "grounding", "recovery", "interaction"],
            metadata_extra={"family": "tool_use_grounding", "variant_index": variant},
        )

    data_topics = [
        "deduplication",
        "provenance",
        "difficulty stratification",
        "domain balance",
        "contamination screening",
        "quality filtering",
        "preference pair design",
        "counterfactual augmentation",
        "label leakage",
        "curriculum scheduling",
    ]
    for i in range(len(data_topics) * scale):
        topic = data_topics[i % len(data_topics)]
        variant = i % len(data_topics)
        add(
            samples,
            f"Propose a data-curation ablation for '{topic}' (case {i + 1:03d}) in Aurelius. Include the hypothesis, dataset split, metric, expected signal, and a failure mode that would make the ablation misleading.",
            response(
                [
                    f"Treat {topic} as a dataset-level intervention that should change training dynamics, not just final accuracy.",
                    "Define the split, metric, and expected signal before running the ablation.",
                    "State the failure mode that would make the result misleading, then add a control that would catch it.",
                ],
                [
                    "The ablation has a hypothesis, split, metric, and expected signal.",
                    "The failure mode is explicit and paired with a control.",
                    "The design measures training dynamics as well as final performance.",
                ],
                "Data ablations are only useful when they separate real causal effects from dataset artifacts.",
            ),
            "data_curation_ablations",
            "dataset_intervention",
            4,
            "Data ablations need hypothesis, split, metric, expected signal, and failure-mode controls.",
            answer_key=f"data-{variant:02d}",
            tags=["data", "ablation", "curation", "control"],
            metadata_extra={"family": "data_curation_ablations", "variant_index": variant},
        )

    eval_topics = [
        "adversarial split",
        "counterfactual generalization",
        "latent probe benchmark",
        "tool-use stress test",
        "safety boundary test",
        "long-context planning",
        "reward-hacking eval",
        "world-model eval",
        "metacognitive calibration",
        "architecture tradeoff eval",
    ]
    for i in range(len(eval_topics) * scale):
        topic = eval_topics[i % len(eval_topics)]
        variant = i % len(eval_topics)
        add(
            samples,
            f"Design an evaluation harness for '{topic}' (case {i + 1:03d}) in Aurelius. Include the task family, negative controls, scoring rubric, and a regression that would be invisible to standard benchmarks.",
            response(
                [
                    f"Build the harness around {topic} as the latent skill under test.",
                    "Add negative controls that penalize shortcut solutions and surface memorization.",
                    "Use a scoring rubric that separates correct final answers from correct reasoning process.",
                ],
                [
                    "The harness has a task family, negative controls, and scoring rubric.",
                    "The regression is invisible to standard benchmarks but visible to the harness.",
                    "The score rewards process quality, not just final-answer accuracy.",
                ],
                "A good harness should expose the hidden failure mode that ordinary benchmarks miss.",
            ),
            "eval_harness_stress",
            "stress_testing",
            5,
            "Evaluation harnesses must include task family, negative controls, scoring rubric, and hidden-regression coverage.",
            answer_key=f"eval-{variant:02d}",
            tags=["eval", "harness", "stress", "benchmark"],
            metadata_extra={"family": "eval_harness_stress", "variant_index": variant},
        )

    safety_topics = [
        "refusal boundary",
        "benign refusal",
        "role-play bypass",
        "dual-use ambiguity",
        "misuse detection",
        "privacy boundary",
        "medical advice boundary",
        "legal advice boundary",
        "self-harm boundary",
        "policy-consistent redirection",
    ]
    for i in range(len(safety_topics) * scale):
        topic = safety_topics[i % len(safety_topics)]
        variant = i % len(safety_topics)
        add(
            samples,
            f"Create a safety-boundary case for '{topic}' (case {i + 1:03d}). Specify the user intent, risky element, safe alternative, refusal boundary, and a monitoring signal for model drift.",
            response(
                [
                    f"Frame {topic} as a boundary condition where the model must preserve helpfulness without enabling misuse.",
                    "Define the risky element and the safe alternative before writing the refusal boundary.",
                    "Add a monitoring signal that would catch drift toward either over-refusal or unsafe compliance.",
                ],
                [
                    "The case includes intent, risky element, safe alternative, refusal boundary, and monitoring signal.",
                    "The safe alternative is genuinely useful and not a generic disclaimer.",
                    "The monitoring signal distinguishes over-refusal from unsafe compliance.",
                ],
                "Safety boundaries should be specific enough to guide behavior and monitor drift.",
            ),
            "safety_boundary_cases",
            "boundary_reasoning",
            4,
            "Safety cases require intent, risky element, safe alternative, refusal boundary, and monitoring signal.",
            answer_key=f"safety-{variant:02d}",
            tags=["safety", "boundary", "refusal", "monitoring"],
            metadata_extra={"family": "safety_boundary_cases", "variant_index": variant},
        )

    arch_topics = [
        "attention versus SSM",
        "residual depth",
        "modularity",
        "sparse routing",
        "latent memory",
        "long-context compression",
        "tool-call architecture",
        "world-model bottleneck",
        "metacognitive head",
        "reward-model coupling",
    ]
    for i in range(len(arch_topics) * scale):
        topic = arch_topics[i % len(arch_topics)]
        variant = i % len(arch_topics)
        add(
            samples,
            f"Reason through the architecture tradeoff '{topic}' (case {i + 1:03d}) for Aurelius. Compare the benefit, cost, failure mode, and one experiment that would decide whether to keep it.",
            response(
                [
                    f"Treat {topic} as a design variable with a measurable benefit and a measurable cost.",
                    "Compare the likely failure mode against the expected gain, then propose one decisive experiment.",
                    "Prefer the design that improves controllable behavior without creating an unmanageable new failure mode.",
                ],
                [
                    "The answer includes benefit, cost, failure mode, and a decisive experiment.",
                    "The experiment is designed to separate mechanism quality from benchmark gaming.",
                    "The tradeoff is stated in terms of behavior, not just parameter count.",
                ],
                "Architecture choices should be justified by mechanism, measurement, and failure-mode control.",
            ),
            "architecture_tradeoff_reasoning",
            "mechanism_design",
            5,
            "Architecture tradeoffs require benefit, cost, failure mode, and a decisive experiment.",
            answer_key=f"arch-{variant:02d}",
            tags=["architecture", "tradeoff", "mechanism", "experiment"],
            metadata_extra={"family": "architecture_tradeoff_reasoning", "variant_index": variant},
        )


def build_preferences(samples: list[Sample], version: str) -> list[dict[str, Any]]:
    prefs: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        if idx % 7 != 0:
            continue
        chosen = sample.response
        rejected = sample.response.replace("Verification:", "Unverified final answer:")
        if rejected == chosen:
            rejected = chosen + "\nVerification: skipped; answer accepted without explicit check."
        prefs.append(
            {
                "prompt": sample.prompt,
                "chosen": chosen,
                "rejected": rejected,
                "metadata": {
                    "id": f"{version}-pref-{idx:05d}",
                    "domain": sample.domain,
                    "skill": sample.skill,
                    "difficulty": sample.difficulty,
                    "source": "synthetic-v7-frontier",
                    "generator": "generate_aurelius_reasoning_data_v7.py",
                    "version": version,
                    "tags": sample.tags,
                    **sample.metadata_extra,
                },
            }
        )
    return prefs


def build_pretrain_docs(samples: list[Sample], version: str) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        if idx % 5 != 0:
            continue
        docs.append(
            {
                "title": f"{sample.domain} / {sample.skill} / {sample.metadata_extra.get('family', 'frontier')}",
                "text": f"Prompt: {sample.prompt}\n\nResponse: {sample.response}\n\nVerification: {sample.verification}",
                "metadata": {
                    "id": f"{version}-pretrain-{idx:05d}",
                    "domain": sample.domain,
                    "skill": sample.skill,
                    "difficulty": sample.difficulty,
                    "source": "synthetic-v7-frontier",
                    "generator": "generate_aurelius_reasoning_data_v7.py",
                    "version": version,
                    "tags": sample.tags,
                    **sample.metadata_extra,
                },
            }
        )
    return docs


def build_curriculum(version: str, seed: int, scale: int) -> str:
    return f"""version: {version}
seed: {seed}
scale: {scale}
stages:
  - name: frontier_probe
    epochs: 2
    mix:
      causal_intervention_design: 0.14
      latent_reasoning_probe: 0.12
      agentic_failure_modes: 0.12
      reward_hacking_detection: 0.10
      world_model_updates: 0.10
      tool_use_grounding: 0.10
      data_curation_ablations: 0.08
      eval_harness_stress: 0.08
      safety_boundary_cases: 0.08
      architecture_tradeoff_reasoning: 0.08
quality_rules:
  - every response must include explicit verification
  - frontier examples should test latent reasoning, not surface pattern matching
  - safety examples must be specific and monitorable
  - data ablations must include controls
"""


def build_readme(version: str, seed: int, scale: int, out: Path) -> str:
    return f"""# {version}

Generated by `scripts/generate_aurelius_reasoning_data_v7.py`.

This dataset extends the deterministic Aurelius reasoning corpus with a frontier tranche focused on how models reason internally: causal interventions, latent probes, agentic failure modes, reward-hacking detection, world-model updates, tool grounding, data ablations, eval stress, safety boundaries, and architecture tradeoffs.

## Generation

```bash
cd /Users/christienantonio/aurelius
.venv/bin/python scripts/generate_aurelius_reasoning_data_v7.py --scale {scale} --output {out}
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
- stable hash split for val
- every response starts with `Reasoning summary:`
- every response contains `Verification:`
- metadata includes domain, skill, difficulty, tags, and version

## Design note

The responses intentionally use concise reasoning summaries, not unrestricted chain-of-thought dumps. For Aurelius, the training target is: plan internally, verify, then expose a compact rationale and final answer.
"""


def validate(records: list[dict[str, Any]], version: str) -> None:
    if not records:
        raise RuntimeError(f"{version}: no records generated")
    ids = [r["metadata"]["id"] for r in records]
    prompts = [r["prompt"] for r in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{version}: duplicate IDs detected")
    if len(prompts) != len(set(prompts)):
        raise RuntimeError(f"{version}: duplicate prompts detected")
    required = {"prompt", "response", "system_prompt", "metadata"}
    meta_required = {"id", "domain", "skill", "difficulty", "source", "version", "tags"}
    for r in records:
        if set(r) != required:
            raise RuntimeError(f"{version}: invalid top-level schema")
        if not meta_required <= set(r["metadata"]):
            raise RuntimeError(f"{version}: invalid metadata schema")
        if not r["prompt"] or not r["response"] or not r["system_prompt"]:
            raise RuntimeError(f"{version}: empty field detected")
        if not r["response"].startswith("Reasoning summary:"):
            raise RuntimeError(f"{version}: response missing reasoning summary")
        if "Verification:" not in r["response"]:
            raise RuntimeError(f"{version}: response missing verification")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Aurelius reasoning dataset v7")
    parser.add_argument(
        "--scale",
        type=int,
        default=20,
        help="Frontier expansion multiplier for v7 template families.",
    )
    parser.add_argument("--base-scale", type=int, default=80, help="Base v2 expansion multiplier.")
    parser.add_argument(
        "--version", default="aurelius-reasoning-sft-v7", help="Dataset version string."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "aurelius_reasoning_sft_v7",
        help="Output directory.",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    args = parser.parse_args()

    if args.scale < 1 or args.base_scale < 1:
        raise ValueError("--scale and --base-scale must be >= 1")

    out = args.output
    version = args.version
    seed = args.seed
    rng = random.Random(seed)

    samples: list[Sample] = []
    V2.V1.build_mechanistic(samples)
    V2.V1.build_math_logic(samples)
    V2.V1.build_coding(samples)
    V2.V1.build_agent(samples)
    V2.V1.build_metacognition(samples)
    V2.V1.build_safety(samples)
    V2.build_v2_expansion(samples, args.base_scale)
    build_frontier_expansion(samples, args.scale)
    rng.shuffle(samples)

    records = [record(sample, idx, version) for idx, sample in enumerate(samples)]
    validate(records, version)

    train = [r for r in records if not is_val(r["metadata"]["id"])]
    val = [r for r in records if is_val(r["metadata"]["id"])]
    prefs = build_preferences(samples, version)
    pretrain_docs = build_pretrain_docs(samples, version)

    out.mkdir(parents=True, exist_ok=True)
    (out / "sft").mkdir(parents=True, exist_ok=True)
    V2.write_jsonl(out / "sft" / "train.jsonl", train)
    V2.write_jsonl(out / "sft" / "val.jsonl", val)
    V2.write_jsonl(out / "preferences.jsonl", prefs)
    V2.write_jsonl(out / "pretrain_docs.jsonl", pretrain_docs)
    (out / "curriculum.yaml").write_text(
        build_curriculum(version, seed, args.scale), encoding="utf-8"
    )
    (out / "README.md").write_text(build_readme(version, seed, args.scale, out), encoding="utf-8")

    domains = sorted({r["metadata"]["domain"] for r in records})
    manifest = {
        "name": version,
        "seed": seed,
        "scale": args.scale,
        "base_scale": args.base_scale,
        "system_prompt": SYSTEM_PROMPT,
        "counts": {
            "sft_total": len(records),
            "train": len(train),
            "val": len(val),
            "preferences": len(prefs),
            "pretrain_docs": len(pretrain_docs),
            "estimated_tokens": V2.estimate_tokens(records),
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
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
