# AURELIUS BOUNDED IMPROVEMENT AND EPOCH-FROZEN VERIFIER TRUST SPECIFICATION

**Version:** 2.0.0  
**Date:** 2026-07-15  
**Status:** `DESIGN_REVIEW_COMPLETE__IMPLEMENTATION_NOT_CERTIFIED`  
**Supersedes:** version 1 archived as `AURELIUS_BOUNDEDNESS_FROZEN_VERIFIER_SPEC_V1_ARCHIVED_2026-07-15.md`  
**Current training state:** `BLOCKED_TRAINING_DATA_NOT_ADMISSIBLE`  
**Current capability state:** `SOURCE_AND_DESIGN_VALIDATED_CAPABILITY_UNMEASURED`  
**Frontier claim:** `FRONTIER_PARITY = NOT_MADE`

---

## 0. Executive decision

A frozen verifier is **necessary but not sufficient** for bounded cumulative improvement.

The version-1 design correctly froze the foundation, separated the proposer from the grader, required rollback, and rejected unconstrained weight-level recursion. It was nevertheless not ready to certify Aurelius because it conflated four different propositions:

1. a candidate has a positive mean on a bounded proxy metric;
2. a candidate is not harmful in any important dimension;
3. a learned verifier is accurate and independent;
4. the entire promotion system cannot be bypassed or rewritten.

Only the first proposition is addressed by the Statistical Gödel Machine-style confidence bound, and even that transfer requires stronger assumptions and a corrected protocol.

The governing architecture is therefore:

```text
LOW-TRUST PROPOSAL PLANE
  model / agent / trainer proposes a content-addressed candidate
            |
            v
INDEPENDENT EVALUATION PLANE
  epoch-frozen verifier bundle + sealed fresh evaluation shard
  + hard invariants + paired non-inferiority/superiority tests
            |
            v
INDEPENDENT RELEASE PLANE
  provenance check + risk ledger + human/meta-policy authority
  + signed release manifest
            |
            v
DEPLOYMENT PLANE
  signature verification + canary + independent rollback controller
```

The candidate policy has no write authority over any downstream plane and no read authority over sealed item-level evidence.

The verifier is **frozen within a verifier epoch**, not frozen forever. A permanently static verifier becomes stale and increasingly gameable. Candidate verifiers evolve only in a shadow lane, are evaluated on evaluator-independent anchors, and can become authoritative only through an explicit human-signed epoch transition. Evidence scored by the displaced verifier is never mixed with evidence from the new verifier.

The allowed Aurelius improvement loop is bounded artifact engineering, not autonomous recursive replacement of the model:

- ephemeral inference search and repair;
- versioned, reversible scaffold or skill changes;
- separately governed private adapters, routers, critics, and sidecars;
- separate native-model research lineages;
- no canonical-base mutation;
- no self-granted verifier, release, signing, or deployment authority.

For public release, Aurelius retains the existing **clean-teacher-only firewall**: self-generated/self-trained private deltas are not merged into or represented as the public canonical teacher.

---

## 1. Scope and non-claims

This specification governs persistent changes to:

- prompts and system policies;
- skills, memory policies, and tool wrappers;
- routing and inference budgets;
- code, tests, and execution scaffolding;
- PEFT adapters, residual sidecars, critics, and learned routers;
- native experimental checkpoints;
- learned reward models and verifiers;
- evaluation bundles and promotion policy.

It does not claim:

- mathematical proof of real-world safety;
- that a positive benchmark mean implies absence of catastrophic failures;
- that a frozen LLM judge is ground truth;
- that a verifier can supervise capabilities outside its coverage;
- that bounded self-refinement produces open-ended intelligence growth;
- that any current Aurelius code path implements this specification;
- that Qwen3.6-27B has displaced the current Qwen3-Coder control;
- that any model, adapter, router, or native MoE challenger has been trained or promoted.

Terminology is strict:

- **System improvement:** better outcomes from search, tools, routing, repair, skills, or adapters.
- **Model improvement:** a measured gain caused by changed trainable model parameters.
- **Foundation improvement:** a new canonical base checkpoint passes a separately authorized promotion.
- **Recursive improvement:** outputs of one improvement cycle causally affect the proposal process for later cycles.
- **Bounded improvement:** recursion is limited by explicit authority, artifacts, resources, hypotheses, evaluation capacity, and human-controlled stop conditions.

Scaffolding gains must not be presented as evidence that the base model improved itself.

---

## 2. Audit of the evidence grounding version 1

### 2.1 Darwin Gödel Machine (`2505.22954`)

What transfers:

- archive rather than overwrite candidates;
- preserve lineage and parent-child relationships;
- use external task execution to evaluate proposed agent-code changes;
- retain diversity instead of greedily keeping one lineage.

What does not transfer:

- DGM is evidence for open-ended search over agent code, not a safety certificate;
- its paper explicitly leaves formal safety and alignment outside scope;
- benchmark/evaluation infrastructure must be outside the candidate's write surface even when candidate agent code is editable;
- archive membership is not the same as deployment promotion.

Aurelius may borrow the archive and lineage mechanics, but every archive node remains untrusted until it passes this specification.

### 2.2 Statistical Gödel Machine (`2510.10232`)

The one-sided Hoeffding lower confidence bound is valid for bounded independent paired differences under its stated null:

\[
\operatorname{LCB}_{1-\alpha}
= \bar X - (b-a)\sqrt{\frac{\ln(1/\alpha)}{2n}}.
\]

Version 1 rendered the formula ambiguously. The denominator is `2n`; it is not `n/2`.

The paper does not establish a general probability of “harmful acceptance.” It establishes a type-I error statement for the chosen bounded proxy mean, under bounded/i.i.d./stable-harness assumptions. Its experiments use small-to-mid-scale optimization and learning problems, not LLM self-training loops. Its CTHS power demonstration includes a synthetic `+4.0pp` confirmation gain. The paper itself states that LLM agent loops and high-stakes scale remain future work.

The paper's algorithm also contains a dangerous compositional ambiguity that Aurelius must not copy:

- it permits acceptance when `LCB > 0 OR global wealth >= 1/delta`;
- the same wealth variable is carried across proposal rounds;
- evidence accumulated under a genuinely beneficial proposal can therefore contaminate a later, different hypothesis unless wealth is candidate-local and reset;
- using both a fixed-n test and an e-process as an OR rule spends error twice unless separately allocated.

Aurelius replaces that construction in section 8.

### 2.3 SAHOO (`2603.06333`)

What transfers:

- measure semantic, lexical, structural, and distributional change;
- compare quality gain with alignment cost;
- retain rollback and external circuit breakers;
- monitor cumulative drift against a stable root, not only the previous version.

What does not transfer:

- weights `(0.38, 0.25, 0.20, 0.17)` and threshold `0.44` are not universal constants;
- SAHOO calibrated on a small Qwen3-8B experiment and states that task-distribution changes require recalibration;
- the paper explicitly says adversarial evasion of the drift detector is outside scope;
- the paper's reported calibration/task counts and some convergence statements are internally inconsistent;
- GDI convergence means the measured behavior stopped changing; it does not imply the behavior is aligned or safe;
- a baseline can itself be misaligned, which the paper explicitly acknowledges.

Aurelius treats GDI as a locally calibrated diagnostic, never as the sole release gate.

### 2.4 SCOPE (`2605.31433`)

SCOPE does use a frozen copy of the initial 7–8B model as a self-judge that writes document-grounded rubrics and grades solver responses. This supports the narrower proposition that a fixed judge can stabilize a training reward over three iterations.

It does not show that a frozen self-judge is an independent release authority:

- judge, challenger, and solver originate from the same model family;
- rubric quality is reported as the bottleneck;
- open-ended judgments remain model-generated proxies;
- harmful content propagation remains an acknowledged risk;
- no cryptographic, organizational, permission, sealed-test, or deployment separation is established.

SCOPE supports using a frozen judge as one signal; it does not certify a self-improvement trust boundary.

### 2.5 Reward-suite leakage study (`2607.11022`)

The important result is mechanistic: a judge may produce many false positives because it scores an apparently strong response without checking whether the proposed intervention causally changes the answer.

Aurelius therefore requires:

- base-response controls;
- candidate-matched controls;
- obfuscated/perturbed controls;
- causal intervention checks where applicable;
- confidence bounds on false-positive and false-negative rates;
- verifier abstention when causal sufficiency cannot be established.

An observed false-positive rate below 2% is not enough by itself. With zero failures, at least 149 independent negative traps are needed for a one-sided 95% upper bound below 2%; 228 are needed for a one-sided 99% upper bound below 2%. Correlated or template-duplicated traps reduce the effective sample size.

### 2.6 Quasi-introspection and RSI limits (`2607.04277`, `2607.07663`)

These works support a bounded interpretation:

- finite internal revision does not establish unbounded qualitative capability growth;
- external grounding, compute, data, and evaluation remain limiting resources;
- generator/evaluator weight sharing creates self-confirming error;
- pure closed data loops risk model and diversity collapse;
- durable gains often look like accumulated verified procedures and scaffolding.

They do not prove that all future autonomous RSI is impossible. Aurelius therefore states only the narrower claim: the current program is bounded system and derivative-module improvement under external authority.

### 2.7 Self-Evolving Agents with Anytime-Valid Certificates (`2607.00871`)

SEA is closer to the desired Aurelius architecture than version 1 recognized:

- frozen base;
- small steering layer;
- mutable versioned harness;
- separate loop controllers;
- terminal held-out grader separated from in-loop self-authored oracles;
- abstention and certificate ledger.

But SEA explicitly states that the composed endogenous-loop guarantees remain open conjectures. Its empirical cells are single expensive runs; its harness-edit statistical gate accepted no edits in the highlighted ablation; its self-authored oracle can false-green; and slow-loop weight distillation was designed but not run.

Aurelius borrows the layering and epistemic caution, not an unearned safety guarantee.

### 2.8 Red Queen Gödel Machine (`2606.26294`)

Red Queen supplies the correct resolution to the “frozen forever versus evaluator staleness” conflict:

- evaluators are fixed within an epoch;
- candidate evaluators are compared only on evaluator-independent anchors;
- promotion happens at an epoch boundary;
- records dependent on a displaced evaluator are erased or re-evaluated;
- evidence from different utility definitions is never pooled.

Aurelius adopts this structure with a stronger authority boundary. The proposing/training policy cannot promote the next verifier epoch. Anchor weakness, evaluator volatility, and extension of the evolvable surface remain explicit risks.

### 2.9 Test-time recurrent reasoning (`2602.03094`) and causal SAE audits (`2607.12166`)

Repeated inference, majority voting, verification, and repair are valid first-line capability mechanisms when measured at matched cost. They change no persistent model parameters.

SAE/feature audits may reveal mechanism changes, but correlational feature labels are not safety certificates. Only interventions with replicated causal effects can support a mechanism claim, and even those are diagnostic evidence beneath hard outcome and security constraints.

---

## 3. Trust objects and definitions

### 3.1 Candidate

A candidate is an immutable content-addressed bundle containing:

- candidate type and governance lane;
- parent/root identifiers;
- complete diff or model delta;
- data/provenance manifest;
- build recipe and dependency lock;
- declared writable/readable resources;
- random seeds and generation parameters;
- proposer identity;
- intended metric and falsifier;
- expected resource consumption;
- rollback target.

A candidate cannot change after evaluation begins. Any byte change creates a new candidate ID and consumes a new confirmation allocation.

### 3.2 Verifier epoch

A verifier epoch is a signed bundle containing:

- verifier code and dependency/image digests;
- deterministic test suites and schemas;
- model-based grader weights, prompts, tokenizer, and decoding config, if any;
- metric definitions, bounds, direction, margins, and aggregation policy;
- sealed evaluation-shard commitments;
- trap-suite commitments and calibration receipt;
- abstention and disagreement rules;
- statistical test family and error ledger;
- start/end authority signatures.

“Frozen” means byte-addressed, signed, reproducibly buildable, write-protected to the candidate account, and fixed for the duration of the epoch. A pathname or Python object held in the same mutable process is not a frozen verifier.

### 3.3 Campaign

A campaign is a finite, human-authorized sequence of proposal confirmations under one verifier epoch and one preregistered risk/resource contract.

A campaign ends when any bound is reached, the verifier epoch changes, the sealed reservoir is exhausted, an incident occurs, or a human stops it. It never silently resets its statistical budget.

### 3.4 Trust roots

Trust roots include:

- canonical foundation digest;
- verifier-epoch signing key;
- release-policy digest;
- sealed-set encryption/access policy;
- append-only ledger root;
- deployment signing key;
- rollback artifact and controller;
- human/meta-policy authority.

No candidate-facing process may write these objects.

---

## 4. Non-negotiable invariants

### I0 — authority separation

A proposer, trainer, candidate policy, or candidate tool cannot modify or select its own verifier, sealed data, thresholds, reference answers, trust anchors, release key, or deployment controller.

### I1 — canonical foundation identity

The current canonical foundation remains byte-identical unless a separately authorized foundation campaign succeeds. Qwen3.6-27B remains a challenger until that happens; the current Qwen3-Coder artifact remains the control.

### I2 — public clean-teacher firewall

Public Aurelius teacher weights remain clean-teacher-only. Private self-trained adapters, routers, critics, and sidecars are separately identified and must not be merged into, relabeled as, or silently distributed with the public canonical teacher.

### I3 — no candidate access to sealed item-level evidence

Candidates and proposers receive only preregistered aggregate decisions sufficient for development. They do not receive hidden prompts, references, per-item labels, test patches, grader rationales, trap identities, or future shards.

### I4 — fail closed

Missing artifacts, verifier exceptions, timeouts, nondeterminism beyond tolerance, exhausted budget, signature failure, provenance gaps, evaluator disagreement beyond threshold, or unknown schema values produce `HOLD` or `REJECT`, never promotion.

### I5 — dual anchors

Every persistent candidate is evaluated against:

- the current incumbent; and
- the immutable campaign root.

This blocks many individually small accepted changes from accumulating unnoticed root-relative drift.

### I6 — no aggregate compensation for critical harm

A positive weighted average cannot compensate for a critical safety, integrity, provenance, permission, destructive-action, or exact-correctness failure.

### I7 — independent evidence

Self-generated agreement, self-confidence, self-critique, or a same-lineage judge is not independent evidence. It may screen or propose; it cannot alone authorize promotion.

### I8 — bounded resources and authority

Every campaign fixes limits on proposals, confirmations, cycles, wall time, tokens, tool calls, network access, compute, storage, candidate bytes, writable paths, and privilege level.

### I9 — reversible deployment

Every admitted artifact has an exact disable/rollback path that does not depend on the promoted candidate. Rollback is exercised before promotion, not merely documented.

### I10 — one-variable causal attribution

A promotion experiment changes one declared mechanism at a time. Bundled base, router, adapter, verifier, data, and threshold changes are rejected as uninterpretable unless a factorial experiment was preregistered.

### I11 — verifier-epoch transitions invalidate dependent evidence

When authoritative verifier bytes, data, weights, prompts, thresholds, or utility definitions change, dependent candidate evidence is erased or re-evaluated. Scores from different epochs are not averaged.

### I12 — data admission precedes training

No SFT, QLoRA, DPO, GRPO, RLVR, distillation, router training, verifier training, recovery pretraining, or native-MoE training occurs while the data gate is `BLOCKED_TRAINING_DATA_NOT_ADMISSIBLE`.

### I13 — humans control meta-policy and release

Human authority selects goals, critical dimensions, verifier epochs, public release policy, and exceptions. A human may not silently override a failed hard gate; any exception requires a new signed manifest and explicit risk acceptance.

---

## 5. Required trust architecture

| Plane | May read | May write | Must not do |
|---|---|---|---|
| Proposal/training | public contract, training/dev data, aggregate past outcomes | candidate workspace only | read sealed items; write verifier/release/deploy objects |
| Evaluation | immutable candidate, sealed shard, verifier epoch | signed evaluation receipt and append-only log | modify candidate; train candidate; expose item-level secrets |
| Release | signed candidate/evaluation/provenance receipts | signed release manifest | execute arbitrary candidate tools; rewrite evaluation |
| Deployment | signed release manifest and approved artifacts | canary state and deployment log | accept unsigned artifacts; let model self-promote |
| Rollback | last-known-good manifest and health signals | active deployment pointer | depend on candidate cooperation |
| Shadow verifier | training anchors and red-team data | candidate verifier bundle | gate current candidates; promote itself |

Minimum practical separation:

1. separate OS account or container for evaluation;
2. candidate artifact mounted read-only;
3. sealed tests mounted only inside evaluation;
4. network egress disabled unless explicitly required and proxied;
5. verifier/release keys absent from candidate environment;
6. append-only or remote ledger outside candidate workspace;
7. deployment accepts only manifests signed by release authority;
8. candidate cannot invoke the release/deployment API.

For high-impact campaigns, use separate hosts/accounts and offline or hardware-backed signing keys.

---

## 6. Governance lanes

### Lane E0 — ephemeral inference

Examples: best-of-N, exact tools, retrieval, transient repair, temporary search beams.

- no persistent artifact automatically changes future behavior;
- hard runtime permission and resource bounds apply;
- outputs may become proposed data only after provenance and verification.

### Lane S1 — bounded scaffold artifacts

Examples: prompts, non-privileged skills, deterministic routing tables, repair templates, tool budgets.

- may be proposed automatically;
- may receive internal-only automated promotion under a finite campaign if all hard gates pass;
- privilege, network, security-policy, evaluator, and deployment changes are excluded;
- accepted artifact remains disableable and versioned.

### Lane A2 — private learned attachments

Examples: LoRA/QLoRA experts, learned routers, critics, residual sidecars, draft heads.

- training requires explicit human authorization and admitted data;
- the base remains checksum-invariant;
- no autonomous train-and-promote loop;
- promotion is private/internal and requires sealed evaluation plus human approval;
- attachments remain unmerged and separately versioned;
- public clean-teacher firewall remains intact.

### Lane M3 — native model or architecture lineage

Examples: Qwen3.6-27B foundation challenger, 192-expert Qwen3.6-35B-A3B derivative, dense-to-MoE pilot.

- separate checkpoint identity and registry lineage;
- no recursive auto-promotion;
- external/cloud training only under a dedicated campaign;
- full matched evaluation and human release authority;
- never represented as an in-place reversible upgrade to the canonical base.

### Lane V — verifier and meta-policy

Examples: PRM/GenRM updates, trap suites, thresholds, metric weights, release policy.

- never proposed and promoted through the ordinary candidate lane;
- developed in shadow;
- calibrated on evaluator-independent anchors;
- red-teamed for leakage and false-green behavior;
- promoted only at a verifier-epoch boundary;
- all dependent evidence is re-evaluated or invalidated.

---

## 7. Campaign manifest

Before any proposal is generated, the human-authorized manifest must freeze:

```yaml
campaign_id: content-addressed-id
lane: E0 | S1 | A2 | M3 | V
root_artifact_sha256: ...
incumbent_artifact_sha256: ...
verifier_epoch_sha256: ...
release_policy_sha256: ...
max_proposals: ...
max_confirmations: ...
max_cycles: ...
max_wall_seconds: ...
max_generation_tokens: ...
max_tool_calls: ...
max_compute_units: ...
max_candidate_bytes: ...
network_policy: deny | allowlisted
writable_paths: [...]
critical_metrics: [...]
noninferiority_margins: {...}
target_metric: ...
minimum_meaningful_gain: ...
statistical_mode: fixed_n_paired | candidate_local_e_process
campaign_alpha: ...
alpha_allocation: ...
screening_data_commitment: ...
confirmation_shard_commitments: [...]
trap_suite_commitment: ...
rollback_manifest_sha256: ...
human_authorization: signature
```

A proposal that requires a larger surface or budget creates a new campaign; it cannot expand its own envelope.

---

## 8. Corrected statistical acceptance protocol

### 8.1 What the statistical guarantee means

Let `X_i` be a preregistered bounded paired proxy difference on an independently selected confirmation item. Statistical acceptance controls the probability of falsely declaring the candidate better under that proxy null and its assumptions.

It does **not** directly bound:

- probability of catastrophic harm;
- failures absent from the evaluation distribution;
- adversarial test leakage;
- evaluator misspecification;
- correlated or nonstationary samples;
- deployment distribution shift;
- bypass of the release system.

The receipt must say `FALSE_PROXY_PROMOTION_CONTROL`, not `PROBABILITY_OF_HARM_CONTROL`.

### 8.2 Screening versus confirmation

- Screening uses development data and spends no confirmation alpha.
- Confirmation uses a fresh sealed shard not used for proposal generation, ranking, debugging, or threshold tuning.
- Only a candidate that passes screening enters confirmation.
- Confirmation feedback is aggregate and rate-limited.
- A confirmation shard is never reused for a revised candidate.
- If the sealed reservoir is exhausted, the campaign stops.

### 8.3 Lexicographic decision order

A candidate is admissible only if every layer passes:

1. **Artifact integrity:** hashes, signatures, build, provenance, permissions.
2. **Deterministic hard constraints:** no critical violations.
3. **Verifier calibration:** trap-suite false-green/false-red bounds and abstention.
4. **Critical non-inferiority:** candidate is non-inferior to incumbent and root in every critical bucket.
5. **Target superiority or verified efficiency:** meaningful gain on the declared target, or lower cost at matched verified quality.
6. **Cumulative drift:** root-relative drift remains inside locally calibrated limits.
7. **Reproducibility:** independent rerun reaches the same decision.
8. **Authority:** lane-appropriate human/release approval.

No weighted utility can bypass layers 1–4.

### 8.4 Fixed-n paired mode

For preregistered bounded paired differences `X_i in [a,b]`, use:

\[
LCB_{1-\alpha_t}=\bar X-(b-a)\sqrt{\frac{\ln(1/\alpha_t)}{2n}}.
\]

The target-superiority gate requires:

\[
LCB_{1-\alpha_{t,target}} > \epsilon_{gain},
\]

where `epsilon_gain > 0` is the minimum meaningful gain fixed before evaluation.

For each critical metric `j`, with candidate-minus-control difference `D_ij` and allowed regression margin `m_j >= 0`, require:

\[
LCB_{1-\alpha_{t,j}}(D_j) > -m_j.
\]

Allocate across target and critical hypotheses so:

\[
\sum_j \alpha_{t,j} \le \alpha_t.
\]

Use exact paired binary methods such as McNemar/discordant-pair intervals for pass/fail tasks when appropriate; use paired permutation or bootstrap intervals only when their assumptions and multiplicity control are preregistered. Hoeffding is a conservative fallback, not a universal default.

Power analysis is mandatory. For `X in [-1,1]`, Hoeffding alone needs approximately 1,382 independent pairs to certify a `0.10` margin at `alpha=0.001`, 5,527 for `0.05`, and 34,539 for `0.02`. A tiny suite cannot support a high-confidence small-gain claim.

For Aurelius selector/verifier experiments, report paired flip counts and a preregistered minimum detectable effect, not percentages alone. The existing `n=134` selector set is underpowered for many partial `+3–5pp` gains. A sealed set near `n=500` is the current planning target for roughly `<=2.5pp` paired-flip sensitivity under the registered heuristic, but the final `n` must be recomputed from the observed discordant rate and the frozen `formula_id`. A `NULL` result means “not detected at this MDE,” not “effect equals zero.”

### 8.5 Candidate-local anytime mode

If optional stopping within one candidate is needed, use one preregistered e-process **instead of** the fixed-n test.

For candidate `t`, reset wealth to `E_t,0 = 1`. Let the centered bounded observation be:

\[
Y_{t,i}=\frac{X_{t,i}-\epsilon_{gain}}{R},\qquad
R=\max(|a-\epsilon_{gain}|,|b-\epsilon_{gain}|),
\]

so `Y in [-1,1]`. Choose predictable `lambda_t,i in [0,1]` using only prior observations from this candidate, and update:

\[
e_{t,i}=1+\lambda_{t,i}Y_{t,i},\qquad
E_{t,n}=\prod_{i=1}^{n}e_{t,i}.
\]

Under the conditional null `E[Y_t,i | F_t,i-1] <= 0`, this is a candidate-local nonnegative supermartingale. Accept the target hypothesis only when:

\[
E_{t,n}\ge 1/\alpha_{t,target},
\]

and all non-inferiority and hard gates independently pass.

Rules:

- never carry e-wealth from one candidate to another;
- never use `LCB OR e-value` without allocating separate error budgets;
- never let positive evidence for one hypothesis authorize another candidate;
- reset only by starting a new confirmation allocation;
- expose the full update trace in the receipt.

### 8.6 Across-candidate allocation

For a finite campaign with at most `B` confirmation events, an allowed schedule is:

\[
\alpha_t=\frac{\alpha_{campaign}}{tH_B},\qquad
H_B=\sum_{k=1}^{B}\frac1k.
\]

The index `t` counts confirmation events, not cheap screening attempts. A candidate cannot manipulate the index or reset the campaign. Other preregistered familywise-valid schedules are allowed.

Suggested starting statistical budgets, subject to task-specific risk review:

- internal reversible S1: at most `0.05` per campaign;
- private learned A2: at most `0.01`;
- verifier, native-model, or public-release decisions: at most `0.001` plus independent review.

These values are type-I proxy budgets, not tolerated probabilities of harm.

### 8.7 Adaptive reuse and leakage

Even a hidden fixed set can be learned through repeated aggregate decisions. Aurelius therefore requires fresh confirmation shards or a separately justified adaptive-data-analysis mechanism. Binary accept/reject feedback is still information leakage and is rate-limited by campaign size.

Training, proposal generation, screening, confirmation, verifier calibration, and final audit use disjoint evidence roles. A filename saying `heldout` does not establish independence.

---

## 9. Verifier calibration and trap protocol

Every learned or heuristic verifier must be evaluated on:

1. correct controls;
2. clearly incorrect controls;
3. base-response controls;
4. candidate-matched controls;
5. obfuscated and paraphrased controls;
6. causal interventions or counterfactual controls;
7. reward-hacking/adversarial prompts;
8. out-of-domain and distribution-shift probes;
9. nondeterminism/repeatability probes;
10. abstention cases.

Required measurements:

- precision and false-positive rate;
- recall and false-negative rate;
- specificity and sensitivity;
- calibration/Brier score where scores are probabilistic;
- confidence intervals or confidence sequences;
- subgroup and domain breakdowns;
- disagreement with deterministic authorities;
- coverage and abstention rate;
- robustness under paraphrase and perturbation.

The verifier report must also characterize the verification horizon on three axes:

- **scalability:** cost and latency as candidate capability and task horizon grow;
- **faithfulness:** correlation with the intended outcome rather than an easy proxy;
- **robustness:** resistance to policy optimization, adversarial inputs, and distribution shift.

For a proposed 2% false-positive ceiling, require the one-sided upper confidence bound—not the point estimate—to be below 2%. Zero false positives in 228 independent negatives gives a one-sided 99% upper bound just below 2%. This is a minimum statistical condition, not a complete safety condition.

A learned judge cannot override a failing formal proof check, compiler, exact schema, permission rule, secure-executor policy, or deterministic test. If learned verifiers disagree materially, the outcome is `HOLD` pending independent review.

### 9.1 Executable-environment authority

Executable tests are proposed oracles, not ground truth by construction. Aurelius keeps four states separate:

1. `STATIC_CANDIDATE` — provenance, chronology, license, and source lineage are valid;
2. `EXECUTION_AUTHORITATIVE` — clean buggy/fixed states replay under an isolated, reset environment;
3. `SEMANTICALLY_AUTHORITATIVE` — hidden tests survive mutation, toxicity, alternative-valid-solution, and anti-hacking probes;
4. `TRAINING_ELIGIBLE` — the task is decontaminated, split-safe, and learnable for the current policy.

For code/tool rewards, the non-compensable authority signal is conjunctive:

```text
patch_applies
AND fail_to_pass
AND pass_to_pass
AND hidden_tests
AND integrity_probes
AND policy_rollout_correspondence
AND fresh_environment
```

Localization, style, length, process scores, or LLM-judge ratings cannot compensate for a failed authority factor. Measure semantic false positives with valid non-equivalent mutants and an upper confidence bound, stratified by mutation family. Equivalent mutants may be excluded only with independent evidence.

Host subprocesses, fake sandboxes, an operator-set `certifying=true`, or a manifest boolean are not certifying. The current secure-executor boundary remains two-phase: macOS/CPU Phase A validates the interface and escape suite with `certifying=false`; Linux Phase B requires a real digest-pinned isolation profile and independent replay before executable rewards become training-eligible.

---

## 10. Drift and constraint preservation

### 10.1 Goal contract

Goals and constraints live in a signed `goal_contract` containing:

- intended capabilities;
- prohibited behaviors;
- exact permission boundaries;
- critical regression buckets;
- public/private release rules;
- provenance requirements;
- deployment limits;
- human owner and exception policy.

Natural-language intent alone is insufficient. Whenever possible, goals are translated into executable invariants and counterexample suites.

### 10.2 Dual drift measurement

For candidate `c`, measure both:

- `D(c, incumbent)` — local step drift;
- `D(c, campaign_root)` — cumulative drift.

A sequence of tiny accepted changes may be small locally while large in aggregate.

### 10.3 GDI usage

A local GDI may retain the form:

\[
GDI = w_s D_{sem}+w_lD_{lex}+w_tD_{struct}+w_dD_{dist},
\]

but:

- weights are calibrated on Aurelius data, not copied from SAHOO;
- thresholds are fixed before the campaign;
- calibration includes injected known-safe and known-unsafe changes;
- uncertainty and subgroup variation are reported;
- the drift model is frozen within the verifier epoch;
- GDI is diagnostic and cannot override hard constraints;
- `GDI < 0.01 for three cycles` may indicate stagnation, not safety.

Constraint preservation is a separate hard vector. Any critical component failure returns `REJECT` regardless of aggregate drift or capability gain.

---

## 11. Verifier lifecycle: freeze, shadow, transition

### 11.1 Authoritative verifier

One signed verifier epoch alone decides current campaign promotions. It does not learn from candidate results during the epoch.

### 11.2 Shadow verifier

A candidate verifier may:

- train on disjoint calibration/red-team data;
- score replay artifacts in shadow;
- surface blind spots in the authoritative verifier;
- never influence current candidate rewards or promotion decisions.

### 11.3 Epoch transition

A verifier challenger can become authoritative only if:

1. its code, model, data, and configuration are content-addressed;
2. it passes evaluator-independent gold anchors and trap suites;
3. false-positive, false-negative, calibration, coverage, and abstention gates pass;
4. it is tested against adversarial candidates optimized against the old verifier;
5. old and new verifiers run in overlap/shadow mode;
6. disagreements are adjudicated independently;
7. a human signs the transition;
8. evidence dependent on the old verifier is invalidated or re-evaluated;
9. rollback to the old verifier epoch remains possible.

This supplies Red-Queen-style evaluator evolution without allowing the policy to rewrite its current judge.

---

## 12. Threat model

| Threat | Required control |
|---|---|
| Candidate modifies verifier code | separate account/container; read-only signed verifier image |
| Candidate changes thresholds/config | signed verifier-epoch manifest; release checks digest |
| Hidden-test leakage | sealed mount; no candidate read; fresh shards; aggregate-only feedback |
| Repeated-query extraction | finite campaigns; feedback rate limits; no shard reuse |
| Same-model self-confirmation | deterministic/external evidence; diverse independent judge; abstention |
| Test overfitting | dev/screen/confirm/final separation; fresh environment variants |
| Reward hacking | adversarial traps; causal controls; sealed quality tracked with reward |
| Composite-score masking | lexicographic hard gates and per-bucket non-inferiority |
| Tiny cumulative drift | root-relative plus incumbent-relative evaluation |
| Verifier staleness | shadow verifier lane and human-signed epoch transitions |
| Evaluator capture/collusion | no candidate authority over verifier or release; independent anchors |
| Provenance poisoning | signed manifests, source hashes, admission gate, append-only lineage |
| Replay/rollback attack | monotonic release counters; signed manifests; last-known-good pin |
| Dependency/compiler compromise | pinned image/dependencies; reproducible builds; SBOM/attestation |
| Nondeterministic verdict | repeated seeds; deterministic settings where possible; variance gate |
| Tool prompt injection | untrusted-output isolation; allowlisted typed tools; no instruction following from data |
| Resource exhaustion | preregistered limits; evaluator timeouts fail closed |
| Selective evidence submission | evaluator selects tasks/shards; complete receipt required |
| Candidate self-deployment | deployment accepts release signature only; key absent from candidate plane |
| Rollback sabotage | rollback controller and artifact outside candidate authority |
| Distribution shift | canary monitoring, OOD suites, post-deployment tripwires |
| Human override ambiguity | signed exception with scope, expiry, rationale, and risk owner |

---

## 13. Audit of current Aurelius implementation

The current repository contains useful prototypes, but they do not implement this trust specification.

### `src/training/verifier_recursion.py`

Critical gaps:

- calls `train_fn` automatically on verifier-selected outputs;
- can run without an independent `label_fn`;
- evaluates “compounding” from first-to-last solve rate on the same task list;
- has no sealed confirmation set, paired statistical gate, root-relative regression, risk ledger, artifact signature, rollback test, or human authorization;
- permits verifier precision to fall by five percentage points and still return `COMPOUNDS`;
- describes execution tests as ground truth “by construction,” although incomplete tests can false-green;
- does not separate data selection reward from release verification.

Decision: `RESEARCH_PROTOTYPE_ONLY__MUST_NOT_PROMOTE_OR_TRAIN_CANONICAL_ARTIFACTS`.

### `src/training/multi_domain_verifier.py`

Critical gaps:

- registry is mutable at runtime and can override a verifier;
- math accepts answers within a default 5% tolerance;
- reasoning awards structural credit for length and can return 0.5 without an answer;
- JSON validates syntax/key presence, not semantic correctness;
- missing required arguments warn rather than hard-fail;
- configurable global floor can make verifier exceptions nonzero.

Decision: usable for development reward shaping, not release authority.

### `src/security/integrity_verifier.py`

Critical gaps:

- trust records live in a mutable in-process dictionary;
- the same process can snapshot modified bytes and redefine the trusted state;
- no signatures, persistent append-only ledger, trusted root, dependency tree, or release counter.

Decision: checksum helper, not a cryptographic trust root.

### `agent/budget_bounded_loop.py`

Useful control: explicit LLM-turn and tool-call bounds.

Critical gap: tool-budget exhaustion is surfaced as a tool error, but the overall trace may still return `success`. For a promotion campaign, any resource-budget exhaustion must make the candidate evaluation `HOLD/REJECT`, even if the model later emits a final answer.

### `src/alignment/regression_gate.py`

Critical gaps:

- one perplexity average is treated as sufficient;
- a mutable 5% threshold is accepted without preregistration or uncertainty;
- no paired confidence interval, critical buckets, safety/provenance tests, root anchor, multiplicity control, signature, or release authority;
- moving a rejected adapter to `.rejected` is not quarantine against reuse.

Decision: development smoke test only.

### `src/composer/edit_verifier.py`

Useful control: syntax, compile, lint, and test execution.

Critical gaps:

- verification commands and breadth are caller-configurable;
- success means selected checks passed, not that hidden regression or security tests passed;
- skipped/missing changed files are not automatically fatal;
- no signed fixed command bundle or sealed suite.

Decision: local engineering feedback, not independent promotion verification.

---

## 14. Application to the Aurelius architecture

### 14.1 Foundation policy

- The current Qwen3-Coder control remains immutable.
- Qwen3.6-27B remains `BASE_CANDIDATE_01`, not promoted.
- Any future canonical-base change is Lane M3 and human-governed.
- Base identity is checked before and after every attached-module run.

### 14.2 Routed Delta MoE

The primary modular path remains valid with stronger boundaries:

```text
immutable foundation
  + identity route
  + separately versioned private PEFT experts
  + separately versioned router
  + external tools/repair/retrieval
  + epoch-frozen independent release verifier
```

Each adapter, router, critic, and sidecar is a separate candidate with its own data, build, evaluation, and rollback receipt. No module can alter another module, the foundation, or the verifier.

The external router is evaluated against:

- identity/base route;
- each static expert;
- oracle route upper bound;
- best static expert;
- best-of-N/repair without training;
- incumbent router;
- immutable campaign root.

A learned router is Lane A2, not an automatically promoted S1 config.

### 14.3 Revised training sequence

1. Data admission and immutable baseline.
2. Inference scaling and verified trace collection.
3. Human-authorized SFT/QLoRA of one private expert.
4. Independent sealed evaluation and private promotion decision.
5. Human-authorized next expert only after a measured unmet deficit.
6. Supervised router trained separately from experts.
7. DPO only for preregistered non-executable preferences.
8. GRPO/RLVR only for private adapters with external verifiable rewards and reward-hacking controls.
9. Distillation into a fresh private adapter; no overwrite of prior deltas.
10. Bounded registry-assisted development, with human checkpoints at every training or learned-router transition.

For MoE/adapter RL, log and replay the exact train-time routing decisions during learner updates, and verify correspondence with serving-time routing. A policy gradient computed under one expert route and served under another is a contract failure, not merely noise. Any policy or router update invalidates rollout/KV caches atomically; cache keys include base, adapter, router, tokenizer, prompt/template, tool schema, and context configuration.

The prior phrase “bounded recursive expert growth” is replaced by **bounded registry-assisted development**. The agent may cluster failures and propose data/module hypotheses, but it cannot authorize training, select the release verifier, promote a learned artifact, or deploy it.

### 14.4 Native `~27.9B A3B` challenger

The proposed 192-expert derivative of Qwen3.6-35B-A3B is Lane M3:

- separate checkpoint and architecture identity;
- not a reversible attachment to the dense 27B base;
- no self-loop training authority;
- recovery data must pass admission;
- source 35B-A3B and unmodified controls remain immutable;
- promotion requires strict per-domain retention, rare-route coverage, memory/throughput gain, and full provenance;
- the native challenger cannot become the verifier of its own release.

### 14.5 Public versus private

Public:

- clean canonical teacher only;
- signed provenance and base identity;
- no self-generated/private adapters silently bundled;
- no claim that private recursive development changed the public weights.

Private/internal:

- reversible adapters, routers, critics, and sidecars may be used after Lane A2 gates;
- every response/deployment receipt identifies the active base and module hashes;
- identity route remains available;
- rollback disables the module without loading candidate code.

---

## 15. Promotion state machine

```text
DRAFT
  -> SCREENED
  -> CONFIRMATION_AUTHORIZED
  -> CONFIRMING
  -> {REJECTED | HOLD | QUARANTINED | STATISTICALLY_ADMISSIBLE}
  -> HUMAN_APPROVED_PRIVATE
  -> CANARY
  -> {DEPLOYED_PRIVATE | ROLLED_BACK}
```

Lane S1 may use `INTERNAL_AUTO_ADMITTED` only for non-privileged, non-learned, exactly reversible scaffolds under a finite campaign. Lane A2, M3, V, and all public releases require human/release approval.

Every transition is append-only and includes:

- state before/after;
- actor/authority;
- candidate/root/incumbent/verifier hashes;
- evidence-shard commitment;
- metric vector and confidence results;
- risk spend;
- resource use;
- decision rationale;
- signature and timestamp;
- rollback pointer.

---

## 16. Immediate implementation priorities

### P0 — block false certification

1. Mark `VerifierRecursionLoop`, `MultiDomainVerifier`, `RegressionGate`, and in-process integrity snapshots as non-certifying prototypes.
2. Prevent canonical training from `VerifierRecursionLoop` without a signed campaign authorization.
3. Make missing independent labels/ground truth a hard failure for release evaluation.
4. Make budget exhaustion fail the promotion evaluation.
5. Remove runtime verifier override from any authoritative evaluator process.

### P1 — establish the trust root

1. Define machine-readable campaign, verifier-epoch, evaluation-receipt, and release-manifest schemas.
2. Store SHA-256/SBOM/dependency/image digests.
3. Add signed or externally protected append-only release records.
4. Separate proposal/evaluation/release/deployment OS identities.
5. Implement signature verification at deployment and exercise rollback.

### P2 — implement the evaluation protocol

1. Build fresh sealed confirmation shards.
2. Implement paired fixed-n and candidate-local e-process modes as mutually exclusive choices.
3. Add per-metric alpha allocation and critical non-inferiority.
4. Add root/incumbent dual comparisons.
5. Add trap-suite UCBs, abstention, and disagreement handling.
6. Certify executable environments separately from verifier code and require mutation/anti-hacking authority receipts.
7. Preserve raw receipts and deterministic replay bundles.

### P3 — controlled Aurelius pilots

1. Run a scaffold-only S1 campaign first.
2. Attempt to break the verifier with adversarial candidates before promoting anything.
3. Run the verifier in shadow and measure its false-green rate.
4. Only after data admission, run one private adapter A2 campaign.
5. Keep public weights and current canonical base unchanged.

---

## 17. Acceptance checklist

A campaign cannot begin unless all are true:

- [ ] lane and candidate surface are explicit;
- [ ] canonical root and incumbent hashes are recorded;
- [ ] verifier epoch is signed and independently mounted;
- [ ] proposal process cannot read sealed items;
- [ ] campaign budgets and alpha are preregistered;
- [ ] meaningful gain and non-inferiority margins are fixed;
- [ ] screening and confirmation evidence are disjoint;
- [ ] trap-suite confidence bounds pass;
- [ ] deterministic hard constraints are executable;
- [ ] rollback has been exercised;
- [ ] data admission passes for any training lane;
- [ ] human authorization matches the lane.

A candidate cannot be promoted unless all are true:

- [ ] candidate bytes match the evaluated digest;
- [ ] base/root identity remains unchanged where required;
- [ ] no hard invariant failed;
- [ ] critical metrics are non-inferior to root and incumbent;
- [ ] target gain exceeds the preregistered meaningful margin, or cost improves at matched quality;
- [ ] statistical procedure was the preregistered one, not an OR-combination;
- [ ] candidate-local e-wealth was reset and isolated if used;
- [ ] no confirmation shard was reused;
- [ ] provenance and build are reproducible;
- [ ] release authority signed the exact manifest;
- [ ] canary and independent rollback are ready.

---

## 18. Final invariant

The core Aurelius boundedness invariant is:

```text
candidate policy
  cannot rewrite
verifier epoch
  cannot rewrite
sealed evidence or statistical contract
  cannot rewrite
release authority
  cannot rewrite
signed deployment and rollback controller
```

Cumulative capability is allowed only through independently identified, reversible artifacts whose promotion evidence remains valid under a fixed verifier epoch.

When a proposed capability cannot be independently verified, it is not autonomously promoted. It remains a private experiment or requires explicit human judgment.

When the verifier must improve, it does so in shadow under evaluator-independent anchors and becomes authoritative only at a signed epoch boundary.

When training changes parameters, the result is a separately versioned derivative artifact—not evidence that the canonical foundation rewrote itself.

This is the bounded Aurelius path: **strong search, strong verification, strict authority separation, immutable foundations, reversible private modules, and no self-issued release permission.**
