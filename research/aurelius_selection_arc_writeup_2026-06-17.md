# Aurelius — The Selection Arc: First Light → E87 → k=2 → k=4
### Results and honest status, 2026-06-17 (k=4 verdict 2026-06-18)

## Abstract

A recurring claim in efficient-inference research is that a fixed language model need not run all of its
layers on every token: if one can *cheaply, per input* decide which layers to skip, one can match a denser
model at lower compute. We test this claim end-to-end on a frozen Qwen2.5-1.5B over a functionally-verified
code/JSON repair gym, with strict re-scoring discipline. We report a **powered negative**. (1) The cheapest
estimators are *anti-informative*: an entropy heuristic skips *worse* than random. (2) A learned per-layer
marginal-gain estimator is **falsified** — no informed single-layer policy beats random — because skip-set
cost is strongly **non-additive** (a layer's individual importance anti-predicts its value inside a joint
skip set). (3) Exhaustive enumeration shows a good skip set *exists* for almost every input (true k=2
ceiling 96.7% > dense 75%), with a clean structural mechanism (specific "poison" layers + adjacency, *not*
"early layers"). (4) But a **cheap per-task, prompt-conditioned selector adds nothing** over a fixed global
policy — a result we replicate at k=2 and k=4 (three powered nulls; the key mechanistic feature has ~0
correlation with success, model-free). (5) A fixed global skip set (a structural depth-prune) is ≥ dense at
k=2 by a modest, pool-dependent margin, but **−12pp at k=4** — the lever does not scale. (6) Co-training the
base to make all skips cheap produces a uniformly mediocre model. The honest, defensible result is a
**reliable falsification of cheap single-pass layer-skip routing on a frozen LM**, a *small* non-additive
structured-pruning finding, and a verification methodology that caught six would-be-wrong results. (7) A
**scoped positive**: *replacing* (rather than deleting) layers with cheap low-rank surrogates distilled to
dense scales far *within distribution* — it matches the full model substituting ~10 of 14 routable layers,
where deletion is dead by ~6 — but it **does not generalize out-of-family** (surrogates distilled on
JSON-repair collapse on type-repair, below even deletion). So those layers' contributions are low-rank-
approximable *for a given task distribution*: LayerDelta is a **within-distribution structured-substitution
(task-compression)** method, not a general model improvement, and still *static* (not the dynamic per-input
routing of the original thesis). The one path that could revive the *strong* claim — a model *built to
route* (skip-native, pre-registered) — has, after this arc, a low prior.

> **Scope.** This document synthesizes the experimental arc that tests Aurelius's core thesis on one frozen
> base (Qwen2.5-1.5B). It is written honest-first: the negatives and the corrections are in the body, not
> buried. Every headline number below was produced by a validated harness and, where stated, independently
> re-scored. Where a result was overstated and later corrected, both are shown. The k=4 experiment was
> running on Kaggle as this was written; its slot is marked PENDING.

---

## 0. One-paragraph summary

The strong original thesis — *a cheap per-token/per-layer estimator of "verified gain" can route compute
(SKIP/NORMAL/AMPLIFY) to beat a denser model* — is **not supported** in the form first stated. Two
independent attempts to realize it are **powered nulls**: (i) a per-layer marginal-gain estimator
anti-predicts the right layers to skip (E87), and (ii) a per-task, prompt-conditioned selector built from
cheap prefill activations adds **nothing** over a fixed global policy (the k=2 selector). What *does*
survive is weaker but real and replicated: **a good skip set exists for almost every input** (k=2 true
ceiling 96.7%), the cost of a skip set is **strongly non-additive** (specific "poison" layers + adjacency,
not "early layers"), and a **fixed global skip set — i.e. a permanent structural depth-prune — matches or
modestly beats the full model at less compute** on the representative task pool (k=2: 68.3% vs dense 62%).
The orthogonal route of *training the base to make all skips cheap* (LoRA co-train) is **killed** (it
produces a uniformly mediocre model). And the more valuable scaling question is now answered: at **k=4 the
permanent prune LOSES (−12pp)** — the base tolerates removing ~2 layers but not 4 — and the per-task selector
is **null a third time**. The **single-pass layer-skip lever is exhausted on this frozen base**: the only
policy that beats dense is verified multi-try (more compute, not less). The honest net result is a
**powered, airtight falsification of cheap single-pass layer-skip routing on a frozen LM**, alongside a
*small* structured-pruning result and a verification methodology that caught six would-be-wrong papers.

---

## 1. The thesis and what is at stake

**Claim under test (CLM-001).** A fixed network need not run all layers on every token. If one can cheaply
estimate each layer's *marginal verified gain* on the current input and SKIP the low-gain layers, one can
match or beat a larger/denser model at lower average compute. The crux is the word **cheaply**: an oracle
that tries everything is uninteresting; the thesis needs a *cheap selector*.

**The estimator ladder** (cheapest → richest): `g_entropy` (a layer's effect on next-token entropy) →
`g_gain` (a learned per-layer marginal-importance head) → `g_VBMCA` (a richer joint estimator, never built).
The arc below is, in effect, a walk up this ladder, asking at each rung: *does a selector this cheap beat
random skipping at matched compute?*

---

## 1.5 Related work

**Depth is prunable — statically.** A line of work shows transformer layers are redundant and can be
dropped: LayerDrop (Fan, Grave & Joulin, 2019, *Reducing Transformer Depth on Demand*, arXiv:1909.11556)
trains with structured layer dropout so sub-networks of varying depth can be pruned at inference; recent
LLM depth-pruning (e.g. ShortGPT; LaCo; Gromov et al., 2024, *The Unreasonable Ineffectiveness of the Deeper
Layers*, arXiv:2403.17887) removes whole layers post-hoc with modest loss. **Our static-prune result agrees
and quantifies it**: the frozen base tolerates ~2 removed layers (k=2 ≥ dense, modest) but not 4 (k=4 −12pp)
— and we add the *non-additivity* mechanism that static pruning ignores.

**Dynamic / per-input depth — the claim we test.** Depth-Adaptive Transformer (Elbayad et al., 2019,
arXiv:1910.10073), early-exit methods (DeeBERT, arXiv:2004.12993; FastBERT, arXiv:2004.02178), CALM
(Schuster et al., 2022, arXiv:2207.07061), LayerSkip (Elhoushi et al., 2024, arXiv:2404.16710), and
Mixture-of-Depths (Raposo et al., 2024, arXiv:2404.02258) all route *per input/token* through variable
compute. Most either (a) **train** the model for it (LayerSkip, MoD train the router/early-exit jointly), or
(b) restrict to **prefix/early-exit** structure (a contiguous "run the first m layers"), which sidesteps the
*combinatorial, non-contiguous* skip-set selection we study. **Our negative is specifically about the
*cheap, post-hoc, per-input selection of arbitrary skip sets on a frozen base*** — and we show it fails
(three powered nulls), while the *existence* of good sets does not (96.7% oracle). This locates the
difficulty precisely: not in the absence of good skip sets, but in *cheaply selecting* them per input
without training the base to be routable — which motivates our pre-registered skip-native follow-up (cf.
MoD/LayerSkip, which *do* train for it).

**Train-time robustness vs. mediocrity.** Once-for-All (Cai et al., 2020, arXiv:1908.09791) and slimmable
networks use *sandwich* training + distillation to make sub-networks robust without collapsing the full
model. Our co-train negative is exactly the failure these methods were designed to prevent (uniform
"averaged-compromise" mediocrity from naive multi-width/-depth training) — and our skip-native prereg adopts
their sandwich+distillation fix.

**Positioning.** Relative to this literature our contribution is *not* another routing method but a
**rigorous, powered negative** on the cheapest and most-assumed version of the idea, a precise mechanistic
account (non-additivity; poison layers; adjacency), and a reusable verification methodology.

## 2. Methods (shared across all experiments)

- **Base:** `Qwen/Qwen2.5-1.5B` **base** (not instruct), pinned to revision `8faed761`. 28 layers;
  **routable band = layers 7–20** (14 layers); the first/last few are never skipped (known-catastrophic).
- **SKIP = identity pass** (`h → h`): a skipped layer returns its input unchanged. Gradient-transparent
  (so it is also trainable), and verified to truly bypass attention+MLP.
- **Evaluation = a functional "repair gym":** **F2** (fix malformed JSON to satisfy a schema) and **F3**
  (fix a Python function to pass hidden type/logic tests). "Functional" = the verifier *runs* the
  candidate; **pass** requires it to actually work, not merely resemble the answer.
- **Decoding:** greedy (deterministic) unless noted, `max_new=256`.
- **Verification discipline (the project's spine):** the model that *produces* a result never *scores* it.
  A separate harness regenerates and re-scores completions; headline numbers are reproduced; runner code is
  read before any receipt is trusted. This caught, in this single program: a **fabricated** run, a
  **confounded** verifier, a verifier **under-count** bug, a hardcoded throughput placeholder, an
  auto-generated **wrong verdict**, and an **"exhaustive-vs-sampled" overstatement** (see §5). Treat this
  discipline as a *result*, not overhead.

---

## 3. First Light — baselines separate; the cheapest estimator is *anti*-informative

With no clever routing, on the clean gym (verified by re-scoring **2,700/2,700** completions):

| policy | pass |
|---|---|
| dense (k=0) | **76%** |
| random skip | 60% |
| entropy-guided skip | **25%** |

The cheapest rung, `g_entropy` (skip the layers whose next-token entropy changes least), is not merely weak
— it is **worse than random**. It actively selects bad layers to skip. *First negative: the cheapest
estimator is anti-informative.*

---

## 4. E87 — the per-layer-marginal estimator is falsified; layer cost is non-additive

The central test, two framings.

**Framing 1 — per-layer marginal gain (`g_gain`).** Label each layer by leave-one-out importance (skip
exactly that layer, measure the drop); train `g_gain` to predict it; skip the lowest-gain layers at matched
k. On 60 held-out tasks, 3 seeds, independently re-scored **900/900**:

| policy | pass |
|---|---|
| dense | 73.3% |
| **random** | **30%** |
| learned `g_gain` | 20% |
| "oracle" (greedy single-layer) | 8.3% |
| entropy | 1.7% |

**No informed policy beat random.** The supposedly-smart per-layer importance signal *anti-predicts* the
right joint skip set.

**Why — layer non-additivity (the central obstacle, likely general).** A layer's *individual* leave-one-out
importance **anti-correlates** with its value inside a *joint* skip set. The informed policies concentrate
their skips in early routable layers (7–10), each individually cheap to skip, but skipping several of them
*together* is catastrophic. **The cost of a skip set is not the sum of its parts.** Per-layer saliency
optimizes the wrong objective. (The runner's auto-verdict "NO SIGNAL / thesis dead" was itself *corrected*:
it had used a greedy single-layer oracle, not a true joint search — see §5.)

---

## 5. The selection reframe — and the exhaustive-vs-sampled correction

Reframe: stop scoring per-layer estimators; ask *does a good skip set EXIST per input?* A sweep over k
suggested the achievable ceiling stays near-dense at every budget while random collapses — i.e. the signal
is there, the failure is **selection**, not absence of signal.

**The correction (caught by an external reviewer, re-verified by us).** That sweep's k=2/k=4 "ceiling" was
**best-of-~20 sampled** subsets, *not* an exhaustive oracle (only k=1, with 14 single skips, was
exhaustive). Confirmed against the raw data (k=2: median 18 of 91 sets/task). Consequences, stated
honestly: (i) the reported ceiling is a **lower bound** on the true one, so "a good set exists" *survives
and strengthens*; (ii) but best-of-20 is **verifier-assisted search**, not a cheap selector — "a good set
exists" must never be conflated with "a cheap selector finds it." This correction reset the language across
the canonical records and motivated the exact experiments below.

---

## 6. k=2 exact pair matrix — a good set exists for almost every input

We enumerated **all C(14,2)=91 layer pairs × 60 held-out tasks** (the true k=2 matrix), retiring the
sampled numbers.

- **True ceiling 96.7%** (a passing pair exists for **58/60** tasks) — *above* dense (75%): the
  best-chosen 2-layer skip **beats running all 28 layers**.
- **Avg-random 38.9%**: random skipping is bad; the ceiling-vs-random gap (~58pp) is the selection signal.
- **Good pairs are common** (median ~40 of 91 pairs pass per task): a *dense compatibility basin*, not a
  needle in a haystack.

**Mechanism (replicated on the larger pool, §7).** Poison to skip: **L14 (−18.7pp), L11 (−12.3), L15
(−7.7)**. Safe to skip: **L18 (+8.9), L9 (+7.7), L16 (+7.6), L10 (+7.3)**. **Adjacency is bad** (adjacent
pairs 29% vs non-adjacent 40%); the worst pair is **14-15** (poison + adjacent, 0%). Critically this
**refutes E87's "early layers are the problem"** — early layers 8/9/10 are among the *safest*; the real
structure is *specific* layers (11/14/15) and *adjacency* — exactly the non-additive interaction.

---

## 7. The per-task selector — global routing ≥ dense (modest); cheap per-task selection is a POWERED NULL

We then asked the thesis's real question with task-level cross-validation (train on some tasks, test on
*unseen* tasks). Two pools: the easy 60, and a **representative, powered N=300 pool** (150 F2 + 150 F3).

**The powered (N=300) verdict** — the representative pool is harder (dense **62%**, ceiling **89.7%**,
random **36%**):

| policy (held-out 5-fold CV) | pass |
|---|---|
| random | 36.0% |
| dense (full model) | 62.0% |
| **global best fixed pair (B)** | **68.3%** |
| C_struct (learned, no prompt) | 68.3% |
| **C_full (+ prompt features)** | **68.3%** |
| per-task oracle | 89.7% |

- **Per-task prompt-conditioned selector: NULL (H-SEL-KILL).** `C_full = C_struct = B` exactly — adding
  prompt features changes nothing (H-SEL-1 = +0.0pp). And the *model-free* check is decisive: the
  mechanistic feature I bet on (logit-space alignment of the two skipped layers' residual contributions)
  has **corr(feature, pass) = +0.020 ≈ 0**. The ~21pp per-task headroom (oracle 89.7 vs global 68.3) is
  **not cheaply capturable** from these prefill activations. This is the honest negative the
  pre-registration pre-stated — one rung up from E87's per-layer kill.
- **The real positive: global routing ≥ dense.** The single global best pair (a permanent 2-layer prune)
  reaches **68.3% held-out, above dense's 62%** and far above random's 36%. *Caveat (important): this
  flipped with the pool.* On the easy 60, the global pair was **56.7% < dense 75%**; on the representative
  300 it is **68.3% > dense 62%**. So "global prune ≥ dense" is **pool-dependent and modest** — precisely
  why the k=4 experiment computes a paired bootstrap CI, not just point estimates.
- **The verified-shortlist regime works but costs compute.** On the easy-60 matrix, taking the top-M
  global pairs and accepting any that passes: top-3 = 75% (= dense), top-5 = 88%, top-10 = 95% (≈ ceiling).
  Real, but it is *M× compute + a verifier* (best-of-N), not a single-pass speedup.

---

## 8. The orthogonal route — skip-robust co-train is killed

Instead of *finding* good skips, try to *make all skips cheap* by LoRA-fine-tuning the frozen base under a
skip distribution.

- **v1:** catastrophic forgetting (dense 73% → 21.7%) — the corpus had silently collapsed to web text (no
  code). Untestable, not refuted.
- **v2 (forgetting fixed: code in the corpus, gentler training, drift monitor):** a **clean negative**.
  Dense 73% → 58.3% (a real tax, not collapse), but **Pareto-dominated by the frozen base at every
  operating point** (k=1 40<54, k=2 32<39, k=4 13≈15). The one signal — the co-trained achievable ceiling
  is *flat* across k (75/77/77) while the frozen ceiling drops (92/92/72) — is the **"averaged compromise"
  failure**: training for *arbitrary* skip-robustness produces a *uniformly mediocre* model (the
  Once-for-All / slimmable-network pathology).

**Lesson that reshaped the program:** "make selection unnecessary" is the wrong goal — it levels *down*.
**Selection is essential and cannot be cheaply trained away.** This is what redirected us to test selection
directly (§6–§7), and now pruning at k=4 (§9).

---

## 9. k=4 routing (PENDING) — does a permanent prune pay off?

At k=2 the win is marginal (~0.93× compute, identity-skip). At **k=4** a *fixed* good 4-set is not
identity-skip — it is a **permanent 4-layer prune = a genuinely smaller, faster 24-layer model**. The
combinatorics force sampling: C(14,4)=1001, so we use a **pinned sample of 300 sets (30% coverage)** across
**N=200** tasks; the achievable ceiling is then an honest *lower bound*.

**Primary endpoint (pre-registered):** `global_best_set − dense`, paired per-task bootstrap CI.

**Result (2026-06-18, N=200 × 300 sampled sets).** Matrix: ceiling(sampled) 92%, avg-random **14.8%**
(random collapses far harder than k=2's 36%), dense **64%**. Held-out CV:

| policy | pass |
|---|---|
| random | 14.8% |
| **global_best_set (permanent 4-prune)** | **52.0%** |
| C_struct / C_full | 52.0% / 48.5% |
| **dense** | **64.0%** |
| verified_top3 / top5 | 68.0% / 69.0% |
| sampled_ceiling | 92.0% |

- **H-K4-0 = −12.0pp — the prune LOSES.** Removing the best 4 layers drops 64% → 52%. The k=2 "free ~2
  layers" **does not extend to 4**; by four layers you cut into capability. The single-pass compute win does
  not materialize at k=4.
- **H-K4-2 = −3.5pp — per-task selector null again** (corr(set-interference, pass) = +0.025, model-free).
  The **third** powered null; more headroom did not make cheap features informative.
- **Only verified multi-try beats dense** (top-3/5 = 68–69%) — i.e. *more* compute + a verifier, not less.

**Verdict: H-K4-KILL.** The single-pass layer-skip lever is **exhausted on FROZEN-BASE-v1.**

---

## 9.5 LayerDelta — cheap *substitution* recovers what *pruning* loses (preliminary positive)

The k=4 KILL is about *deletion* (identity-skip, Δ=0). A natural follow-up: replace each skipped layer not
with zero but with a **cheap learned surrogate** Δ̂_l(h) = U_l(V_l h) (rank-32, ≈100K params/layer, ≈99% of
the layer's matmul removed), trained jointly by distilling the substituted model toward dense (logit-KL on
dense's own greedy completions, on a train split disjoint from eval). On the k=4 best set {8,15,16,19}
(held-out 200, bf16): **identity-skip 44% → LayerDelta 62.5% ≈ dense 60%** (H-LD-0 = +18.5pp over zeros;
H-LD-1 = +2.5pp vs dense, within noise → "matches"). The surrogates **generalize** (train/eval disjoint),
which means those four layers' residual contributions are **low-rank-approximable**: *deleting* them is the
wrong move, *substituting* them cheaply is not.

**Two follow-ups then scoped it precisely (and tempered it).**

*Substitution frontier* (substitute the top-k safest layers, k=2…12, dense 59.3%): LayerDelta stays ≈ or
above dense **all the way to k≈10** (k=2 71%, k=4 70%, k=6 63%, k=10 66%), where *identity-skip is already
dead by k=6* (0%); it only breaks at k=12 (30%). So *deleting* layers fails past ~2, but *substituting* them
holds to ~10 — substituting ~10 of 14 routable layers and matching the full model. (Per-k values are
seed-noisy — k=8 dips below k=6/k=10 — so the *trend* is the signal, not the exact numbers.)

*Out-of-family generalization* (2×2, distill-family × eval-family, k=4) — **the decisive test, and it is a
negative.** Same-family works (F2→F2 LayerDelta 75% > dense 65%; F3→F3 57% ≈ dense 60%), but **cross-family
collapses**: surrogates distilled on JSON-repair score **9%** on type-repair (dense 60%, *below even
deletion's* 37%); F3→F2 is 23% vs dense 65%. The pattern is clean and symmetric, so it is real: **the
surrogates learn a task-family-specific adaptation, not the layers' general function.**

**Corrected scope:** the strong frontier result holds only *within the distillation distribution*. LayerDelta
is therefore best read as a **within-distribution structured-substitution / task-compression** method — you
can replace many layers with cheap distilled surrogates *for a given task distribution* (like task-specific
distillation), but it is **not** a general model improvement and does not transfer off-distribution. It is
also *static* substitution — it does **not** revive the *dynamic per-input routing* thesis (still three
nulls).

## 10. Honest synthesis — what survives, what is falsified

**Falsified (powered):**
- The **cheap per-layer-marginal estimator** (`g_gain`, `g_entropy`) as a skip selector (E87; First Light).
- The **cheap per-task prompt-conditioned selector** from prefill activations — null at **both k=2 and
  k=4** (three powered nulls total).
- The **co-train route** to make all skips cheap (v2 clean negative).
- **A single-pass layer-skip compute win that scales:** the permanent prune is ≥ dense at k=2 (~2 layers,
  modest) but **−12pp at k=4** — the lever does not scale, and is exhausted on this frozen base.

**Survives (weaker, real, replicated):**
- **Existence:** a good skip set exists for almost every input (k=2 true ceiling 96.7% > dense).
- **Mechanism:** skip-set cost is strongly **non-additive** — specific poison layers (11/14/15) + adjacency,
  *not* "early layers." Clean and replicated across pools.
- **Global structural pruning:** a fixed global skip set ≥ dense on the representative pool (modest,
  pool-dependent). This is best read not as Aurelius's original *dynamic per-token routing* thesis but as a
  **structured depth-pruning** result: the frozen base is depth-over-parameterized on these tasks, and
  removing the right layers permanently does not hurt (can help). This aligns with the LayerDrop /
  depth-pruning literature, and is the honest, defensible core of what we have.
- **Cheap low-rank *substitution* (LayerDelta, §9.5) — but only within-distribution:** rank-32 surrogates
  distilled to dense replace *many* layers and match the full model *in-family* (to ~10 of 14 routable,
  where deletion dies by ~6), but **do not generalize out-of-family** (JSON-distilled surrogates collapse on
  type-repair). So it is a real **within-distribution structured-substitution / task-compression** result —
  not a general model improvement, and *static* (not dynamic routing). The arc's one positive, correctly
  scoped down by its own generalization test.

**The gap that defines "Aurelius vs ordinary pruning":** the per-task oracle headroom (89.7% vs global
68.3% at k=2; presumably larger at k=4) is large and real, but **no cheap method we have found captures
it**. Aurelius's distinctive claim — *cheap, input-conditioned* compute organization — currently rests
entirely on closing that gap, and every cheap method tried has failed to. That is the central open problem,
stated without flinching.

---

## 11. Mechanism, consolidated

- Layer importance is **non-additive**: individual leave-one-out gain anti-predicts joint skip value (E87),
  because the residual stream is robust to removing *one* contribution but not to removing several that
  *constructively interfere* in task-relevant directions.
- The interference is **structural, not per-input** (so far): the same poison layers (11/14/15) and the
  same adjacency penalty hold across tasks — which is *why* a global selector works and a per-task one adds
  nothing. The hoped-for per-input modulation of this structure is what the cheap features failed to find.
- Entropy "works" only at k=1 (it can flag an individually-safe single skip) and fails at higher k because
  it has no notion of *compatibility* between skips — keep it as a node feature, never as a selector.

---

## 12. Limitations and non-claims

- **One base** (Qwen2.5-1.5B), **one gym** (F2+F3), **identity-skip**. Existence proofs, not laws.
- **Identity-skip is ~no wall-clock speedup** (a depth-skip with the kernel still resident); results are
  **accuracy-at-matched-FLOPs**. The exception — and the reason k=4 matters — is a **fixed global prune**,
  which *is* a genuinely smaller, faster model.
- **Sampling** where stated (E87 k≥2 ceiling; k=4 sets). Sampled ceilings are lower bounds.
- The per-task null is for **these cheap prefill features + a linear model** — it does not prove no per-task
  signal exists (richer features / nonlinear models / larger k untested in full).
- `--no_completions` runs (k=2, k=4) traded away completion-level re-scoring; the aggregate re-derivation
  holds, but a re-scorable completion *sample* should be reinstated (planned).

---

## 13. The verification discipline as a meta-result

The most reusable output of this program may be the *process*. Independent re-scoring and headline
reproduction caught, here: a fabricated experiment; a verifier that rewarded terseness (a false entropy
"win"); a byte-identical-correct answer scored FAIL; a hardcoded tok/s; an auto "NO SIGNAL" verdict from a
greedy oracle masquerading as combinatorial; and an "exhaustive" claim that was best-of-20 sampled. Each
would have produced a confidently-wrong paper. The operating rules — pre-register; the trainer never
scores; reproduce the headline; read the runner; no claim without artifact+command+seed; refuse the
certainty register — are not bureaucracy; they are the reason the negatives above are trustworthy.

---

## 14. What is next

The k=4 KILL resolves the branch: **the frozen-base single-pass layer-skip lever is exhausted.** The earned
options, in order:
1. **Consolidate + publish this negative.** It is complete: a powered, airtight falsification of cheap
   single-pass layer-skip routing on a frozen LM, a small structured-pruning result, and a verification
   methodology — a genuine contribution.
2. **From-scratch skip-native** (the only thesis-revival path): every null was on a base *never built to
   route*; a model trained with the skip distribution from step 0 (LayerDrop + sandwich + distillation)
   *might* develop the per-input routability a frozen base lacks. A real commitment, honestly uncertain prior.
3. **LayerDelta** (cheap frozen-base probe): replace a skipped set's layers with cheap learned surrogates —
   can it *recover* the k=4 prune's 12pp loss? Low prior, but cheap and fast on the existing harness.
4. **Width-sparsity / AMPLIFY** (the untested third routing action); reinstate completion-sample re-scoring.
**Not warranted:** any further whole-layer-skip variant on this frozen base.

> **Bottom line.** Aurelius's strong original thesis — *cheap, input-conditioned compute organization beats
> scale* — is **not supported on a frozen base**: three powered nulls on cheap per-input routing, a static
> prune that is small (~2 layers) and does not scale (k=4 −12pp), and a co-train that produces mediocrity.
> The honest, defensible outputs are a **reliable powered negative** on a much-hyped mechanism, a *small*
> non-additive structured-pruning result, and a verification discipline that caught six would-be-wrong
> papers. The one path that could still revive the strong thesis is to test it on a base *built to route*
> (skip-native) — but the prior, after this arc, is that it confirms the negative rather than overturning it.
> **A reliable negative on a plausible mechanism, this rigorously established, is a real contribution.**
