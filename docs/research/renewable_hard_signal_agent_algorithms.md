# Renewable-Hard-Signal Algorithms for Aurelius Code Agents

**Status:** design portfolio; six candidates to prototype, one explicitly killed.  
**Objective:** improve code-agent capability from repeatable execution evidence, not generic RAG, free-form reflection, or LLM-as-judge scores.

## 0. Design contract

### Operational interpretation of the source mechanisms

This document uses the named mechanisms as computational primitives, not as labels to stack together:

- **AVEF loop:** generate or mutate an executable environment/task, validate its verifier, register it, execute agents, and feed verified failures back into the next generation. The essential property is a renewable supply of *new, machine-checkable* tasks.
- **TTHE:** evolve the executable harness at test time while model weights remain frozen.
- **AgentTether:** represent runs as dependency-linked transition units, localize a failure-critical subtrajectory, and intervene during replay.
- **STRACE:** select diverse representative failures across a batch, then prune each trajectory to a causal dependency slice.
- **Remember When It Matters:** maintain execution state separately and selectively intervene only when it should affect the next action.
- **Hierarchical Memory / AMC:** Tier 1 is immediate working state, Tier 2 is episodic session state, and Tier 3 is durable state with trust, quarantine, revocation, and decay.
- **MILES:** store reusable instructions as modular subgoal/instruction pairs and learn selection from outcome correctness rather than semantic similarity alone.
- **Tool-Making:** compile repeated procedures into validated, versioned executable tools instead of regenerating code every time.
- **WebSwarm:** recursively instantiate nodes with `(local objective, solving mode)` pairs; let decomposition and collaboration evolve as evidence arrives.
- **TerminalSandbox / EnvironmentRegistry:** execute and meter commands; instantiate versioned environments and their verifiers.

### What counts as hard signal

Primary rewards may come only from:

1. test exit status plus parsed test identities;
2. deterministic environment-state predicates;
3. build/type/lint results when the task contract names them;
4. differential behavior between original and patched states;
5. reproducible performance measurements with correctness held constant;
6. state hashes, file diffs, and explicit tool outputs.

LLM critique, cosine similarity, self-confidence, and textual “looks correct” judgments may route work, but cannot promote an environment, harness, memory, or tool by themselves.

### Existing repo seams and gaps

- `src/simulation/environment_registry.py` has a usable registry seam but its default factories are mocks with zero reward. It needs code-environment manifests, lifecycle methods, verifier contracts, and lineage.
- `src/composer/terminal_sandbox.py` is a real subprocess execution wrapper with time/output/resource limits. It is **not strong hostile-code isolation**: it defaults to `shell=True`, and command policy is substring based. Public/untrusted environment generation must add a container or VM backend before execution.
- `src/agent/react_loop.py` already materializes every model/tool failure in `AgentTrace` and has Tier-2/Tier-3 hooks. It lacks repository-state hashes, checkpoint IDs, transition dependencies, verifier events, and intervention provenance.
- `src/memory/amc_tier2.py` uses surprise-gated writes, lexical search, and recent fallback. This is a good baseline but not MILES-style learned selection.
- `src/memory/amc_tier3.py` already has trust, quarantine, revocation, and time decay. Promotion is currently tied mainly to Tier-2 importance and budget exhaustion rather than repeated executable verification.
- `src/memory/amc_tensor_api.py` already defines the five required memory ablations: `no_memory`, `tier2_context`, `tier1_only`, `tier1_tier2`, and `full`.
- `src/composer/checkpoint_rollback.py` provides safe file-scoped snapshots without disturbing unrelated work. It is the correct base for counterfactual replay.

## 1. Causal Failure Forge (CFF)

**Decision:** BUILD FIRST.  
**Combines:** AVEF + STRACE + AgentTether + TerminalSandbox + EnvironmentRegistry + AMC Tier 2/3.

### Thesis

The most valuable new code-agent environments are not arbitrary synthetic tasks. They are *minimal executable counterexamples grown from causally localized real failures*.

### Exact mechanism

1. Run a fixed agent on registered code environments and record typed transition units:
   `(state_hash_before, action, observation, touched_files, tests_run, state_hash_after)`.
2. Across failures, STRACE clusters by verifier signature and transition-graph shape; retain medoids so repeated copies of the same failure do not dominate.
3. AgentTether builds a critical-transition graph and identifies the smallest ancestor-closed slice reaching the failed verifier.
4. A forge mutates only dimensions represented in that slice: requirement edge case, fixture, dependency version, file layout, API schema, or initial repository state.
5. A candidate environment is admitted only if:
   - its clean/reference solution passes;
   - the parent failing patch or behavior fails;
   - reset is deterministic under the recorded seed;
   - the verifier cannot be satisfied by no-op, deleting tests, or emitting expected strings;
   - two fresh rebuilds agree.
6. Register the child with parent lineage and verifier hashes. Store its causal signature in Tier 2. Promote a failure family to Tier 3 only after it appears in multiple repository or mutation lineages.
7. Sample future tasks by a competence frontier, not uniformly: keep tasks whose recent success probability is neither saturated nor impossible.

### Pseudocode

```python
def forge_generation(runs, registry, frontier):
    failures = [r for r in runs if not r.verifier.passed]
    representatives = strace_diverse_medoids(failures, key=("verifier_sig", "graph_sig"))

    for run in representatives:
        graph = build_critical_transition_graph(run.transitions)
        root_slice = tether_localize(graph, failed_predicate=run.verifier.failed_predicate)

        for mutation in mutate_from_slice(root_slice):
            child = materialize_child_env(run.env_id, mutation)
            checks = [
                child.reset_twice_same_hash(),
                child.reference_solution_passes(),
                child.parent_behavior_fails(),
                child.anti_hack_suite_passes(),
                child.rebuilds_agree(count=2),
            ]
            if all(checks):
                registry.register(child.spec, child.factory)
                frontier.observe(child.env_id, initial_status="unrated")
                amc_tier2.observe("failure_family", child.causal_signature,
                                  surprise=1.0, importance=1.0)
```

### Repo attachment

- Extend `src/simulation/environment_registry.py::EnvSpec` with `manifest_hash`, `setup`, `reset`, `verify`, `snapshot`, `parent_env_id`, `mutation`, and `lineage`.
- Add `src/simulation/verified_code_env.py` for lifecycle + deterministic state hashing.
- Add `src/eval/avef_flywheel.py` for candidate generation/admission/frontier sampling.
- Add `src/agent/causal_trace.py` for transition units and dependency edges.
- Extend `AgentStep` / `AgentTrace` in `src/agent/react_loop.py` with checkpoint, state, verifier, and intervention fields.
- Use `TerminalSandbox` for trusted local fixtures initially; add a container backend before importing public code.
- Tests: `tests/simulation/test_verified_code_env.py`, `tests/eval/test_avef_flywheel.py`.

### Overhead

- **Environment creation:** budget 2–6 sandbox runs per candidate plus one clean rebuild.
- **Execution:** no extra model call for an already admitted task; state hashing adds filesystem I/O.
- **Storage:** content-addressed base snapshot plus patch/fixture deltas, not full repository copies.
- **Human cost:** review only admitted high-novelty families, not every generated instance.

### Baseline

A static EnvironmentRegistry populated with human-curated tasks, sampled uniformly; same agent, model, token budget, and verifier.

### Falsifier

CFF’s causal thesis is false if random valid mutations matched for difficulty and compute transfer equally well to untouched repositories, or if STRACE/AgentTether-selected mutations are no more likely than random mutations to reproduce the parent failure mechanism.

### Kill gate

Stop after the first 200 candidate attempts if any condition holds:

- fewer than 30% yield valid deterministic environments;
- verifier disagreement across clean rebuilds exceeds 5%;
- more than 2% of admitted tasks accept an anti-hack patch;
- training or harness adaptation on admitted children gives less than +3 percentage points on immutable, lineage-disjoint held-out tasks at matched compute.

### Anti-thesis

Human-curated tasks may have much higher construct validity. A self-generated environment distribution can teach the agent the forge’s mutation grammar and verifier quirks rather than software engineering.

---

## 2. Tethered Counterfactual Replay (TCR)

**Decision:** BUILD FIRST.  
**Combines:** AgentTether + STRACE + Remember When It Matters + AMC Tier 1/2/3 + TerminalSandbox + checkpoint rollback.

### Thesis

A repair memory deserves trust only when inserting it at a localized decision point changes the executable outcome under controlled replay. Correlation with successful traces is not enough.

### Exact mechanism

1. On failure, localize a critical transition `v*` from the dependency graph.
2. Restore the file-scoped checkpoint immediately before `v*`; preserve external tool fixtures and seed.
3. Generate bounded intervention candidates:
   - concise reminder of a verified invariant;
   - “do not repeat” failed action pattern;
   - required diagnostic command;
   - MILES-style subinstruction selected for the local subgoal.
4. Replay paired branches from the same checkpoint: control (no intervention) and treatment (one intervention). Hold model seed and remaining budget fixed when the backend permits it.
5. Credit an intervention only if treatment passes and control fails, then repeat on at least two perturbations of irrelevant state.
6. Store the immediate facts in Tier 1, the episode/intervention in Tier 2, and promote to Tier 3 only after cross-variant recovery. Tier-3 confidence is updated by future uses and decays when stale; failures revoke or quarantine it.
7. During a new run, the proactive memory controller remains silent unless the current transition signature matches a validated intervention trigger.

### Pseudocode

```python
def counterfactual_repair(failed_run):
    graph = build_critical_transition_graph(failed_run.transitions)
    pivot = tether_localize(graph, failed_run.verifier.failed_predicate)
    cp = failed_run.checkpoint_before(pivot)

    for reminder in bounded_interventions(pivot, amc_tier2):
        control = replay(cp, intervention=None, seed=failed_run.seed)
        treatment = replay(cp, intervention=reminder, seed=failed_run.seed)
        if (not control.pass_) and treatment.pass_:
            robust = all(
                replay(cp.perturb_irrelevant(k), reminder, failed_run.seed).pass_
                for k in ("path_order", "log_noise")
            )
            if robust:
                amc_tier3.promote(
                    key=causal_key(pivot, reminder),
                    value=reminder,
                    confidence=0.8,
                    tags=frozenset({"counterfactual", "execution_verified"}),
                )
                return reminder
    return None
```

### Repo attachment

- Add `src/agent/tethered_replay.py`.
- Extend `src/composer/checkpoint_rollback.py` manifests with environment seed, fixture hash, and verifier hash.
- Add intervention events and branch IDs to `AgentTrace`.
- Add evidence counters to `Tier3Entry`: `control_failures`, `treatment_successes`, `variant_successes`, `last_failed_at`.
- Replace promotion-on-importance for repair memories with evidence-gated promotion in `src/memory/amc_tier3.py`.
- Tests: deterministic paired replay, irrelevant-perturbation robustness, and revocation after a failed reuse.

### Overhead

- Paid only on failures: one control replay plus one treatment replay per candidate, capped at three candidates.
- Online successful-path overhead is a trigger check and, only on match, one short reminder.
- Checkpoint storage is file-scoped and delta-friendly.

### Baseline

Blind retry, full-trace self-reflection, and AgentTether diagnosis without paired replay; all get the same total replay budget.

### Falsifier

The causal-memory claim is false if reminders inserted at random earlier nodes recover at the same rate, or if treatment gains disappear when the replay seed and irrelevant state are perturbed.

### Kill gate

Kill if fewer than 15% of failed episodes yield a robust intervention, if paired replay exceeds 2.5× average failed-episode compute without at least +5 percentage points recovery, or if more than 10% of Tier-3 repair memories later cause a regression.

### Anti-thesis

Long-horizon code runs are not perfectly replayable: package downloads, clocks, concurrent tests, and stochastic models can make the apparent counterfactual effect spurious. A cheaper plain retry may capture most of the gain.

---

## 3. Regression-Escrow Harness Evolution (REHE)

**Decision:** PROTOTYPE AFTER CFF.  
**Combines:** TTHE + STRACE + AVEF + AMC Tier 3 trust/decay + TerminalSandbox + EnvironmentRegistry.

### Thesis

Harness evolution is useful only if every mutation earns the right to persist through execution-based, lineage-disjoint regression escrow. Proxy-only TTHE risks making a permanently worse harness.

### Exact mechanism

1. Treat the harness as typed modules: context policy, tool schema, verification schedule, retry policy, memory trigger, and swarm policy.
2. STRACE summarizes recent representative failures into module-specific edit evidence.
3. A proposer mutates exactly one module per candidate. Examples: insert targeted-test-before-broad-test, alter memory trigger, add rollback on verifier regression, or change context pruning.
4. Evaluate each candidate on three matched sets:
   - recent failures (plasticity);
   - a frozen replay bank (retention);
   - CFF-generated lineage-disjoint variants (generalization).
5. Score with hard outcomes and resource penalties. A candidate enters **escrow**, not production, only if it beats the incumbent with no critical regression.
6. In escrow, route a fixed fraction of subsequent tasks to the candidate in shadow or paired mode. Promote only after the lower confidence bound of paired utility is positive and no protected task regresses.
7. Store each module version in Tier 3 with trust, verifier provenance, task strata, and expiry. Roll back automatically on drift.

### Pseudocode

```python
def evolve_harness(incumbent, recent_runs, registry):
    causes = strace_module_causes(recent_runs)
    population = [mutate_one_module(incumbent, c) for c in causes]

    suites = {
        "plasticity": recent_failure_suite(recent_runs),
        "retention": frozen_replay_suite(),
        "transfer": registry.sample(tags=["lineage_disjoint", "verified"]),
    }
    reports = {h.id: paired_execute(h, incumbent, suites) for h in population}
    eligible = [
        h for h in population
        if reports[h.id].critical_regressions == 0
        and reports[h.id].matched_utility_delta > 0
        and reports[h.id].cost_ratio <= 1.25
    ]
    if not eligible:
        return incumbent

    candidate = max(eligible, key=lambda h: reports[h.id].matched_utility_delta)
    escrow = shadow_route(candidate, incumbent, next_tasks=50)
    if escrow.lower_bound_utility > 0 and escrow.critical_regressions == 0:
        tier3_store_harness(candidate, trust="unverified", evidence=escrow)
        return candidate
    return incumbent
```

### Repo attachment

- Add `src/agent/harness_spec.py` with typed, serializable modules and a stable hash.
- Add `src/agent/harness_evolution.py` for proposer/population/escrow/rollback.
- Add `src/eval/harness_regression_suite.py`.
- Add harness hash and module versions to `AgentTrace` and AMC benchmark metadata.
- Use all five `AMCMemoryModes` when the mutation touches memory, so a harness gain is not misreported as a memory gain.
- Register benchmark strata and immutable suite hashes through `EnvironmentRegistry`.

### Overhead

- Offline evaluation budget: one incumbent and up to four single-module candidates per evolution epoch.
- Escrow: shadow only on a bounded task slice; no permanent double execution.
- Production: zero extra model calls after promotion unless the promoted harness itself adds one.

### Baseline

Fixed ReAct harness; proxy-only TTHE with the same population and model-call budget; and random single-module mutation.

### Falsifier

The escrow mechanism is unnecessary if proxy-only selection predicts held-out executable utility equally well, or if gains vanish after normalizing for extra tool/model calls.

### Kill gate

Kill if no candidate survives escrow in three consecutive epochs, promotion churn exceeds one rollback per 100 tasks, retained-suite performance drops by more than 1 percentage point, or utility gain is non-positive at a 1.25× cost cap.

### Anti-thesis

The harness search space is too non-stationary and interaction-heavy for one-module mutations. A small hand-engineered harness plus better model weights may dominate while being easier to reason about.

---

## 4. Verified Procedural Crystallization (VPC)

**Decision:** BUILD FIRST.  
**Combines:** MILES + Tool-Making + AMC Tier 2/3 + TerminalSandbox + EnvironmentRegistry + TTHE.

### Thesis

Repeated successful reasoning should be compiled into a typed tool only when a live environment can prove its preconditions, postconditions, and cross-case behavior. This turns experience into deterministic capability rather than retrieved prose.

### Exact mechanism

1. Mine recurring successful trace segments bounded by a subgoal and a hard postcondition.
2. Create a MILES-like unit:
   `(subgoal_embedding, precondition_predicate, instruction, postcondition_predicate, provenance)`.
3. The selector gathers supervision from confident cases: a unit is positive only when its use precedes a verified pass and the postcondition is attributable to that segment.
4. When a unit reaches minimum support and low outcome variance, a tool-maker synthesizes a typed function or script plus schema and declared side effects.
5. Validate by differential replay on:
   - original positive cases;
   - negative precondition cases;
   - CFF variants;
   - malformed outputs and timeout cases.
6. Register the tool version only if it rejects invalid preconditions, matches the reference behavior, and preserves unrelated state.
7. Runtime selection is coarse-to-fine: retrieve by subgoal, rerank by learned outcome head, check preconditions, invoke the tool, verify the postcondition, otherwise fall back to normal code generation.
8. Drift or a failed postcondition quarantines the tool and reopens its source instruction as Tier-2 memory. Repeated cross-environment passes promote the tool recipe and selection head to Tier 3.

### Pseudocode

```python
def crystallize(trace_segments):
    units = build_modular_instruction_units(trace_segments)
    for unit in units:
        if unit.verified_support < 5 or unit.outcome_variance > 0.1:
            continue
        candidate = tool_maker.compile(
            instruction=unit.instruction,
            input_schema=infer_schema(unit.cases),
            precondition=unit.precondition,
            postcondition=unit.postcondition,
        )
        suite = original_cases(unit) + negative_cases(unit) + avef_variants(unit)
        report = differential_validate(candidate, suite, sandbox=TerminalSandbox())
        if report.all_positive_match and report.all_negative_reject and report.state_leaks == 0:
            tool_registry.register(versioned(candidate, report))
            amc_tier3.promote(key=candidate.hash, value=unit,
                              confidence=report.confidence,
                              tags=frozenset({"tool_recipe", "execution_verified"}))


def invoke(subgoal, state):
    candidates = coarse_retrieve(subgoal)
    for tool in learned_rerank(candidates, state):
        if tool.precondition(state):
            result = tool.run(state)
            if tool.postcondition(result, state):
                return result
            quarantine(tool, reason="postcondition_failed")
    return generative_fallback(subgoal, state)
```

### Repo attachment

- Add `src/tools/instruction_memory.py`, `src/tools/tool_maker.py`, `src/tools/versioned_registry.py`.
- Expose registered tools to `ReActLoop` through its existing `tool_registry` mapping.
- Use `TerminalSandbox` for compilation and differential tests.
- Store tool manifests as registered environment artifacts so exact backends/schemas are replayable.
- Add `tool_version`, `precondition_result`, and `postcondition_result` to `AgentStep`.
- Tests: schema rejection, side-effect boundary, version rollback, drift quarantine, and fallback behavior.

### Overhead

- Offline compilation only after at least five verified supports.
- Validation budget is a bounded case matrix; no tool ships after a single anecdote.
- Runtime should replace multiple code-generation turns with one direct call plus postcondition check.

### Baseline

Tier-2 retrieval of natural-language procedures; regenerating procedural code per request; static human-authored tool library.

### Falsifier

The compilation thesis is false if a cached natural-language instruction with the same selection policy achieves equal correctness and latency, or if gains come only from memoizing outputs rather than procedural abstraction.

### Kill gate

Kill or narrow to a specific domain if fewer than 20% of supported units compile into tools that pass cross-case validation, if direct tools fail to reduce p50 task latency by at least 20%, or if tool-enabled correctness is not non-inferior within 1 percentage point of generative fallback.

### Anti-thesis

The long tail dominates software work. Maintaining, securing, and versioning generated tools may cost more than regenerating short code, and rigid tools can conceal upstream schema drift until they fail catastrophically.

---

## 5. Obligation-Triggered Memory Interrupts (OTMI)

**Decision:** PROTOTYPE.  
**Combines:** Remember When It Matters + AMC Tier 1/2/3 + MILES + AgentTether + TerminalSandbox.

### Thesis

For code agents, memory should interrupt on an impending *obligation violation*, not merely on semantic relevance or a fixed step interval.

### Exact mechanism

1. Extract an obligation ledger from machine-groundable sources:
   - user requirements tied to tests or explicit file/API constraints;
   - current failing tests;
   - environment facts observed from commands;
   - unresolved subgoals;
   - prohibited repeated actions with verified failure evidence.
2. Each obligation is a predicate with scope, evidence, expiry, and an optional diagnostic action.
3. Tier 1 holds current ledger status and recent state hashes. Tier 2 stores attempts and local diagnoses. Tier 3 stores only obligations/instructions verified across runs and applies trust/decay.
4. Before an irreversible or high-impact action (patch application, dependency change, broad delete, final answer), estimate whether the proposed action conflicts with any active predicate. This may be deterministic; uncertain cases use a small selector.
5. If no conflict is likely, remain silent. If conflict is likely, inject one concise evidence-grounded reminder or force the diagnostic action. Do not emit broad strategy.
6. AgentTether paired replay credits or debits the interrupt. MILES selection heads learn which instruction is useful for which obligation signature.

### Pseudocode

```python
@dataclass
class Obligation:
    oid: str
    predicate: Callable[[RepoState, ProposedAction], TriState]
    evidence_ids: tuple[str, ...]
    scope: frozenset[str]
    expires_when: Callable[[RepoState], bool]
    diagnostic: str | None = None


def pre_action_gate(state, proposed_action, ledger):
    active = [o for o in ledger if not o.expires_when(state)]
    risks = [(o, o.predicate(state, proposed_action)) for o in active]
    violated = [o for o, r in risks if r is TRUE]
    unknown = [o for o, r in risks if r is UNKNOWN]

    if violated:
        o = learned_instruction_selector(violated, state, proposed_action)
        return INTERRUPT(render_grounded_reminder(o))
    if unknown and proposed_action.is_irreversible:
        return REQUIRE_DIAGNOSTIC(select_diagnostic(unknown))
    return SILENT
```

### Repo attachment

- Add `src/agent/obligation_ledger.py` and `src/agent/memory_intervention.py`.
- Insert the gate in `ReActLoop` immediately before `_dispatch_tool` and final-answer acceptance.
- Add `proposed_action`, `obligations_checked`, `decision`, and `evidence_ids` to `AgentStep`.
- Use `EditVerifier` and `TerminalSandbox` outputs as obligation evidence.
- Add an obligation-aware Tier-3 retrieval method keyed by predicate/action signatures rather than task text alone.

### Overhead

- Deterministic predicate checks on every high-impact action.
- At most one short intervention; diagnostic execution is paid only for unknown predicates on irreversible actions.
- No fixed periodic memory-model call.

### Baseline

Existing task-query Tier-2 injection in `_build_messages`; fixed-interval proactive memory; always-on memory bank exposure; no-memory ReAct.

### Falsifier

The obligation mechanism is false if random or always-on reminders produce the same recovery, if interrupts do not occur near causal pivots, or if deterministic guards alone capture all gains without memory selection.

### Kill gate

Kill if interrupt precision is below 60%, if more than 5% of otherwise successful episodes are derailed, if median step count grows by more than 15% without at least +3 percentage points success, or if the same performance is obtained by a pure deterministic pre-commit test gate.

### Anti-thesis

A code agent may need creative plan changes that look inconsistent with current obligations. Early predicates can encode a mistaken diagnosis and become an anchor. For many tasks, simply running tests before finalization is safer and cheaper.

---

## 6. Evidence-Contract Recursive Code Swarm (ERCS)

**Decision:** BUILD ONLY FOR MULTI-SUBSYSTEM TASKS.  
**Combines:** WebSwarm + TerminalSandbox + checkpoint rollback + STRACE + MILES + AMC Tier 2/3 + EnvironmentRegistry.

### Thesis

Recursive delegation helps coding only when every child returns an executable evidence contract and parents merge isolated state, not prose opinions.

### Exact mechanism

1. Probe repository structure first: dependency graph, ownership boundaries, failing tests, build graph, and likely integration cuts.
2. The root creates nodes as `(objective, mode, contract)`:
   - `probe`: inspect and return state facts;
   - `patch`: modify a bounded file set;
   - `verify`: run a named command and return artifacts;
   - `review`: adversarially generate a counterexample tied to code;
   - `integrate`: merge child checkpoints and verify the union.
3. A node may solve locally or recurse when new evidence reveals a deeper dependency. Parallelize only independent file/test scopes.
4. Every child returns:
   `(checkpoint_id, diff_hash, files_touched, verifier_commands, outcomes, state_hash, unresolved_obligations)`.
   Natural-language explanation is optional and never sufficient.
5. Homogeneous siblings share only verified MILES instructions (for example, the discovered command to test one package), not raw chain-of-thought.
6. The parent restores child checkpoints into an integration checkpoint, detects overlapping diffs, and verifies the merged state. A child pass never implies parent success.
7. Stop recursion when expected marginal hard-signal gain per token or per sandbox second falls below a threshold.
8. Store successful decomposition motifs in Tier 2. Promote to Tier 3 only if the motif transfers across lineage-disjoint environments.

### Pseudocode

```python
def solve_node(node, repo_state):
    evidence = probe(repo_state, node.objective)
    if should_recurse(node, evidence):
        children = propose_objective_mode_contracts(node, evidence)
        groups = partition_by_independent_scope(children)
        child_results = parallel_map(solve_node, groups.parallel) + serial_map(
            solve_node, groups.conflicting
        )
        assert all(r.contract_is_complete() for r in child_results)
        merged = integrate_checkpoints(repo_state, child_results)
        report = run_parent_verifiers(merged, node.contract)
        return EvidenceContract.from_report(merged, report)

    result = execute_mode(node.mode, repo_state, node.contract)
    return require_machine_evidence(result)
```

### Repo attachment

- Extend `src/composer/multifile_orchestrator.py` with recursive nodes and explicit scopes.
- Add `src/composer/evidence_contract.py` and `src/composer/recursive_swarm.py`.
- Use `CheckpointRollback` per node and `EditVerifier` at both child and parent levels.
- Add environment tags such as `multi_subsystem`, `merge_conflict`, and `recursive_dependency` to `EnvironmentRegistry`.
- Add swarm node/parent IDs and evidence hashes to `AgentTrace`.

### Overhead

- Multiple model contexts and isolated checkpoints; cap depth at three and live children at four initially.
- Parallelism reduces wall time only when scopes are independent.
- Parent integration always pays an additional verifier pass.

### Baseline

Single ReAct agent; fixed root-only parallel decomposition; equal-budget best-of-N single-agent trajectories.

### Falsifier

The recursive-collaboration thesis is false if equal-budget best-of-N matches performance, if removing evidence contracts does not hurt, or if recursive depth contributes no benefit beyond root-level parallelism on tasks pre-stratified as multi-subsystem.

### Kill gate

Disable outside the tagged task stratum. Kill the general mechanism if it gives less than +5 percentage points on multi-subsystem tasks, if coordination consumes more than 35% of total tokens, if merge-conflict rollback exceeds 20% of episodes, or if depth greater than one has non-positive marginal utility.

### Anti-thesis

Most repository tasks are tightly coupled. Splitting them destroys global coherence, and a strong single agent with one consistent workspace may beat a swarm once communication, redundant exploration, and merge conflicts are counted.

---

## 7. Self-Adversarial Verifier Memory Swarm (SAVMS)

**Decision:** **KILL NOW — do not implement as proposed.**  
**Combines:** AVEF + WebSwarm + TTHE + AMC Tier 3 + Tool-Making + EnvironmentRegistry.

### Proposed mechanism

A swarm continually writes new hidden tests against successful patches; tests that break the current agent become durable Tier-3 verifier memories; TTHE then evolves the harness against this growing suite, and frequently reused tests compile into verifier tools.

### Pseudocode

```python
def unsafe_self_adversarial_loop(patch, agents):
    tests = parallel_map(lambda a: a.propose_breaking_test(patch), agents)
    for test in tests:
        if test.fails_on(patch):
            tier3.store(test)              # fatal flaw: no independent validity proof
            registry.register(test.env)
    return tthe_optimize_against(registry.latest())
```

### Repo attachment if it were built

It would attach to `EnvironmentRegistry`, `TerminalSandbox`, Tier 3, and a new verifier-tool registry. That attachment is technically easy, which makes the idea especially dangerous.

### Overhead

At least one test-generation call and sandbox run per swarm member per accepted patch, followed by repeated harness evaluation. The suite grows without a principled bound.

### Baseline

Human tests plus standard mutation testing; CFF’s two-sided admission with reference-solution and anti-hack checks.

### Falsifier

Its premise fails if agent-generated breaking tests have low agreement with human specifications, if they reject valid alternate implementations, or if performance gains disappear on external test suites.

### Kill gate

Already triggered conceptually: the proposal has no independent oracle proving that a generated test encodes user intent rather than implementation preference. Do not start until a domain supplies formal specs, trusted reference implementations, or differential oracles with demonstrated validity.

### Anti-thesis / narrow salvage

The narrow antithesis is credible: in formal methods, pure functions with property specifications, or protocol conformance, adversarial test generation can be excellent. Salvage only there, with two-sided or metamorphic validity checks. Do **not** make unconstrained self-written tests durable memory for general repository coding.

### Why this idea is explicitly killed

It confuses a renewable signal with a renewable *truth source*. A test is hard to execute but not necessarily valid. Because solver, test author, memory, and harness co-adapt, the loop can drift toward a private specification, reward-hack its own suite, and report impressive but meaningless gains. CFF is the stronger replacement because admission requires parent failure, reference success, determinism, anti-hack checks, and lineage-disjoint evaluation.

---

## 8. Portfolio order and common experiment

| Priority | Algorithm | First implementation milestone | Primary hard metric |
|---|---|---|---|
| P0 | CFF | One real Python repository environment with lineage + two-sided verifier admission | valid-child yield and held-out transfer |
| P0 | TCR | Deterministic checkpoint replay on existing Composer tasks | paired failure recovery |
| P0 | VPC | Compile one repeated diagnostic procedure into a versioned tool | correctness-preserving latency reduction |
| P1 | REHE | Evolve one typed harness module under frozen replay escrow | paired utility without regression |
| P1 | OTMI | Interrupt before final answer and patch application | interrupt precision and success lift |
| P2 | ERCS | Two-level recursive swarm on multi-subsystem tasks | success at matched token/tool budget |
| KILL | SAVMS | none | invalid without independent oracle |

### Shared evaluation protocol

1. Freeze model, decoding settings, tool permissions, and total token/tool budget per comparison.
2. Stratify environments by repository, lineage, language, task family, and observed baseline difficulty.
3. Keep immutable lineage-disjoint test environments that never enter memory, harness evolution, tool validation, or environment mutation.
4. Use paired runs where possible; report success, verifier calls, model tokens, wall time, sandbox time, and regression count.
5. Run the AMC five-mode matrix for every memory-bearing candidate:
   `no_memory`, `tier2_context`, `tier1_only`, `tier1_tier2`, `full`.
6. Add negative controls:
   random causal pivot, shuffled memory, random valid environment mutation, proxy-only harness selection, and best-of-N at matched compute.
7. Promote only mechanisms whose improvement survives:
   - lineage-disjoint tasks;
   - irrelevant-state perturbations;
   - at least three seeds or deterministic paired replay;
   - a cost-normalized comparison.
8. Preserve negative results in Tier 3 as revoked/quarantined mechanism records so failed ideas are not rediscovered and retried indefinitely.

## 9. Source anchors

- STRACE: https://arxiv.org/abs/2607.07702
- AgentTether: https://arxiv.org/abs/2607.06273
- TTHE: https://arxiv.org/abs/2607.08124
- MILES: https://arxiv.org/abs/2607.06974
- Tool-Making: https://arxiv.org/abs/2607.08010
- WebSwarm: https://arxiv.org/abs/2607.08662
- Remember When It Matters: https://arxiv.org/abs/2607.08716
- Hierarchical organize/retrieve memory (HORMA): https://arxiv.org/abs/2606.11680

The algorithms above are syntheses. They are not claims that the cited papers already implement these combinations.