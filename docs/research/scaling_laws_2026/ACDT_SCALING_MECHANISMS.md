# ACDT Formal Spec Sheet: Scaling-Law Improvements for Aurelius

This appendix turns the research dossier into falsifiable mechanisms.

## ACDT-1: Cosine Tapered FFN

Assumptions:
- Later MLPs mostly refine residual state rather than writing novel features.
- Tapering `d_ff`, while preserving average `d_ff`, changes capacity allocation without changing residual shape.

Contract:
- Keep `d_model`, attention, RoPE, and residual stream unchanged.
- Use `d_ff(l)=d_end+(d_start-d_end)*(1+cos(pi*l/(L-1)))/2`.
- Default: `d_start=1.5*d_ff`, `d_end=0.5*d_ff`, rounded to 256.

Dynamics:
- Early layers receive more memory-feature capacity.
- Later layers receive less FFN capacity but keep same residual channel width.

Tests:
- Matched-parameter proxy eval must beat uniform or lose <1% while improving downstream rare-task or long-context metrics.
- Schedule monotone and average-preserving; verified in `tests/model/test_tapered_transformer.py`.

## ACDT-2: Rare-Task MoE Capacity

Assumptions:
- Small dense models lose rare/complex task features due to utility-ranked capacity competition.
- Sparse experts can store rare features without increasing active compute linearly.

Contract:
- Route tokens to top-k experts with monitored load balance.
- Add rare-task replay buckets and per-domain evals.

Dynamics:
- Common tasks occupy shared/general experts.
- Rare task clusters get expert-specific capacity, reducing interference.

Tests:
- Expert load max/min <2.0, token drop <0.1%, rare-task suite improves at matched active FLOPs.

## ACDT-3: Neural Garbage Collection for Reasoning KV

Assumptions:
- Long reasoning traces contain disposable KV entries.
- Eviction can be trained from outcome reward, not only attention proxies.

Contract:
- Add resource action `evict(indices|blocks)` beside token actions.
- Reward = task_success - lambda * peak_kv_fraction.

Dynamics:
- Model alternates grow-then-evict cycles during reasoning.
- Remaining KV becomes a learned working memory state.

Tests:
- 2x cache reduction target with accuracy within tolerance of full-cache upper bound.
- Reject if eviction harms constraint recall, code identifiers, numbers, or tool arguments.

## ACDT-4: SPIRAL Candidate-Set RL

Assumptions:
- Reasoning quality depends on sequential, parallel, and aggregative compute.
- Candidates should receive credit for set-level usefulness, not only individual correctness.

Contract:
- For prompt x, sample N traces; evaluate subsets G; train marginal set advantage.
- Aggregator gets final reward only.

Dynamics:
- Generator learns diversity useful to aggregation.
- Aggregator learns verify/filter/synthesize rather than majority-vote blindly.

Tests:
- pass@k and aggregate@k improve at equal inference FLOPs.
- Diversity must correlate with verifier gain, not just lexical variance.

## ACDT-5: Fara-Style CUA Data Flywheel

Assumptions:
- CUA performance is data/verifier limited more than parameter limited at 4B-9B scale.
- Synthetic stateful websites allow safe tasks that live web cannot support.

Contract:
- Environments + solver + user simulator + three verifiers.
- Verifiers: correctness, efficiency, critical-point adherence.

Dynamics:
- Teacher rollouts produce candidate traces.
- Verifiers filter and label traces; deficiency-targeted data is regenerated iteratively.

Tests:
- No trajectory enters SFT unless all three verifiers pass.
- CUA eval tracks task success, redundant actions, unsafe critical-point crossing, and recovery after ambiguity.
