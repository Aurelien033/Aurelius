# Aurelius — External Brief for Brainstorming

**Purpose of this document.** You are a strong reasoning model being asked to study an ongoing ML
research program and brainstorm improvements — to the *methodology/process* and to the *open technical
problem*. This brief is deliberately self-contained and written in plain language (the project's own
notes use heavy internal shorthand that would waste your reasoning). Everything below is stated as
honestly as we can: what is *proven* (independently re-scored), what is *conjectured*, and what is
*unknown*. Please be equally rigorous back — we value a sharp disconfirming idea over encouragement.

Read time ~15 min. There is a glossary at the end. The single most important section for you is
**§7 (What we want from you)**; everything before it is the context you need to answer it well.

---

## 1. The one-sentence thesis

> **Organizing compute beats adding compute.** A fixed neural network does not need to run all of its
> layers on every token. If you can cheaply decide, per token and per layer, whether to **SKIP** a layer
> (don't run it), run it **NORMAL**, or **AMPLIFY** it, you can match or beat a bigger/denser model at
> lower average compute — *provided you can identify which layers to skip*.

The whole program lives or dies on that last clause: **can you cheaply identify which layers to skip
without hurting the output?** That is a *selection* problem, and it has turned out to be the crux.

## 2. The concrete setup (so the numbers below are interpretable)

- **Base model:** `Qwen/Qwen2.5-1.5B` **base** (not instruct), pinned to git revision `8faed761`.
  28 transformer layers. We call this the **frozen base**.
- **Routable band:** layers **7–20** (14 layers). Layers 0–6 and 21–27 are left alone (skipping the very
  first/last layers is known to be catastrophic, so they're off the table by construction).
- **SKIP = identity pass:** a "skipped" layer returns its input hidden state unchanged (`x -> x`). This
  passes gradients through cleanly, so it's also trainable. We verified this identity-skip is wired
  correctly (it really bypasses the layer's attention+MLP, confirmed by activation checks).
- **Evaluation = a "repair gym":** small, *functionally graded* code/JSON repair tasks. Two clean
  families: **F2** (fix malformed JSON to satisfy a schema) and **F3** (fix a Python function so it
  passes hidden type/logic tests). "Clean" means the verifier runs the candidate and checks it actually
  *works* — not a fuzzy text-similarity score. A task is **passed** only if the generated fix executes
  and satisfies the checker. We hold out **60 test instances** with a frozen split hash, and we
  **independently re-score every completion** (see §5 — this matters).
- **Compute knob `k`:** how many of the 14 routable layers are skipped per generation. `k=0` is the full
  dense model. We study `k=1, 2, 4`. (Note: with identity-skip there is little wall-clock speedup on our
  hardware — this is an **accuracy-at-matched-compute** study, i.e. "how much capability survives at
  budget k," not a latency claim. A real speedup would require fused kernels / actually-shorter compute.)

## 3. What we have *actually established* (independently verified)

### 3a. Baselines separate cleanly ("First Light")
On the repair gym, with no clever routing:
- **Dense (k=0): 76%** pass.
- **Random skip: 60%** (averaged over many random skip sets).
- **Entropy-guided skip: 25%** — *worse than random.*

The entropy heuristic (skip the layers whose next-token-distribution entropy changes least — a cheap
"this layer didn't do much" proxy) is **anti-informative**: it actively picks bad layers to skip. This
result was reproduced by re-scoring all 2,700 completions (2,700/2,700 agree). So the *cheapest* estimator
we tried is not just weak — it's negative.

### 3b. The thesis test ("E87") — falsified at the obvious rung, then revived at a deeper one
We ran the central experiment two ways.

**First framing (per-layer marginal gain).** We labeled each layer by its *individual* importance using
leave-one-out: skip exactly that one layer, measure the drop. Then we trained a small predictor
(`g_gain`) to estimate this per-layer importance, and used it to choose which layers to skip at each k.
Result, at matched compute:
- dense 73% > **random 30%** > learned-gain 20% > "oracle" 8% > entropy 2%.
- **No informed policy beat random.** The supposedly-smart per-layer importance signal *anti-predicted*
  the right skip set. We initially read this as "the thesis is dead."

**Why it failed — layer non-additivity (important, possibly general).** The single-layer leave-one-out
importance of a layer **anti-correlates** with its value as part of a *joint* skip set. The informed
policies concentrated their skips in early routable layers (7–10), each of which is individually cheap to
skip — but skipping *several of them together* is catastrophic. **The cost of a skip set is not the sum of
its parts.** Per-layer saliency is the wrong objective.

**Second framing (sampled achievable ceiling).** We stopped scoring per-layer estimators and instead
asked: *for each task and each k, does there EXIST a good skip set?* We measured a **best-of-~20 sampled**
achievable performance — k=1 is exhaustive (only 14 single-layer skips), but **k=2 and k=4 sampled only
~20 random subsets per task** (of C(14,2)=91 and C(14,4)=1001), so these are a **lower bound** on the true
exhaustive ceiling, *not* a true combinatorial oracle:
- **Sampled achievable ceiling (lower bound) stays near-dense at every budget:** k=1 → 92% (exhaustive),
  k=2 → 92%, k=4 → 72% (≈ dense 73%). The *true* exhaustive ceiling can only be ≥ these.
- Meanwhile **random collapses**: 54% / 39% / 15%.

**This is the key finding — read it carefully.** A near-dense skip set *exists at every compute budget*
(20 random tries found a passing set for 55/60 tasks at k=2). The thesis is **alive**. But two honest
caveats: (1) finding a good set via best-of-20 is *verifier-assisted search* (20 verifier calls), **not** a
cheap deployable selector — "a good set exists" ≠ "a cheap selector can find it"; (2) what's dead is only
the *per-layer-marginal estimator*. **The game is SELECTION under non-additivity:** good sets appear to be
*common* (a dense compatibility basin, not a needle-in-haystack), but whether a *cheap* method can locate
them is exactly the open, untested question. Best operating point is **k=2** (sampled ceiling 92% vs random
39%; and at k=2 the exhaustive matrix is only 91 pairs — a tractable microscope for non-additivity).

### 3c. So the program's real open problem, stated precisely
> A near-optimal skip set exists at every k (shown by best-of-~20 sampling — a lower bound; the true
> exhaustive ceiling is ≥ this). Per-layer importance anti-predicts it (proven). **Find a cheap selector —
> or a training scheme — that recovers most of the achievable ceiling without verifier-assisted search.**
> Equivalently: beat random skip at matched compute with a method whose cost is small relative to just
> running the dense model. (Note: best-of-20 is itself verifier-assisted search, not a cheap selector.)

## 4. The current active experiment — "skip-robust co-train" (where we are *right now*)

Two ways to attack §3c: **(A)** get smarter at *selecting* skip sets on the fixed model, or **(B)** change
the *model* so that *most* skip sets become cheap (then selection matters less). We are currently testing
(B), the cheaper-to-try option.

**Method.** Train **LoRA adapters** (low-rank, ~37 MB, r=16) on the frozen base, *under the skip
distribution* — every training step samples a random skip set (k∈{0,1,2,4} with weights .4/.2/.2/.2) and
computes the loss *with those layers skipped*. The idea: the adapters learn to compensate for absent
layers, so skipping becomes cheap. Only the adapters train; the base stays frozen. Eval is the **same
held-out 60 tasks, same independent re-scoring.**

Three pre-registered hypotheses (written *before* the run):
- **H-CT-0 (skipping gets cheaper):** co-trained avg-random pass-rate at k beats the frozen base's.
- **H-CT-1 (routing becomes learnable):** on the co-trained model, a cheap controller (e.g. entropy)
  beats random. (E87's negative was on a model that had *never* been trained to be skippable.)
- **H-CT-2 (no capability loss):** co-trained dense pass-rate ≈ frozen dense (don't break the model).
- **KILL clause (pre-stated):** if skipping does NOT get cheaper AND routing still can't beat random,
  then layer-skip is the wrong compute lever for this base → pivot to a *from-scratch skip-native* model.

**Run history (each independently verified locally):**

| Attempt | Dense (vs 73% frozen) | What happened | Root cause |
|---|---|---|---|
| **v1** | **21.7%** (−51pp) | **Catastrophic forgetting** — every hypothesis failed; skipping got *worse* | The two code datasets were access-gated, so the training corpus silently collapsed to general web text (no code). 2,000 steps of strong LoRA on off-distribution text overwrote the model's code ability. The skip hypothesis was *untestable* on a globally-broken model, not refuted. |
| **v2** | **58.3%** (−15pp) | **No collapse** — the forgetting fix worked, recovering ~37 of v1's lost points. A *moderate* capability tax remains. | Fixes: real code in the corpus (60% MBPP), gentler training (lower LR, smaller adapter, fewer steps), and a live drift monitor. |

**v2 diagnostic detail (for your reasoning).** The adapter's weight change is **uniform across all 14
routable layers** (~7 norm each, mild U-shape at the edges) and **MLP-heavy** (the feed-forward
projections changed more than attention). It did *not* blow up in one place — it's a broad, even rewrite.
That looks like the **intrinsic cost** of asking a frozen base to become skip-robust everywhere, not a
localized bug. The frozen-vs-cotrained logit drift on a probe is ~6 (large).

**The decisive number is being measured as you read this:** does the co-trained model *skip more cheaply*
than the frozen base? Frozen base at k=2 was 39% (avg-random) / 92% (achievable ceiling). If co-trained
k=2 avg-random comes in well above 39%, the 15pp dense tax *bought* cheaper skipping → first positive for
the lever. If it's ≈/below 39%, the tax bought nothing → KILL clause → pivot to from-scratch skip-native.

> **[PENDING — k-sweep result to be appended here when the local eval finishes.
>  As of writing: dense 58.3% confirmed; k=1/2/4 co-trained avg-random + entropy not yet in.]**

## 5. The verification discipline (this is half of "our process" — please critique it)

Every result in this program is **independently re-derived**, never trusted from the runner that produced
it. This is not paranoia for its own sake — in this single project we have caught, by re-scoring:
- a **fabricated** experiment run (no routing was actually applied; verdicts were hardcoded);
- a **confounded** benchmark (the verifier rewarded terse outputs, which made the entropy heuristic look
  good for the wrong reason) — fixed by switching to a *functional* verifier that runs the code;
- a **verifier under-count bug** (a byte-identical-correct answer scored as FAIL) — retracted a headline;
- a **hardcoded throughput placeholder** masquerading as a measurement;
- an **auto-generated wrong verdict** ("no signal, thesis dead") that came from a *greedy single-layer*
  oracle masquerading as a true combinatorial one — corrected by actually searching skip sets jointly.

The operating rules we now follow: (1) **pre-register** hypotheses and the eval before running;
(2) the model that *trains* never *scores* — a separate harness regenerates and re-scores every
completion; (3) **reproduce the headline number** and **read the runner's code** before believing it;
(4) no claim without an artifact path + the exact command + the seed; (5) refuse the "certainty register"
— state confidence honestly. The local verification runs on a single M1 laptop; training runs on a free
cloud GPU. The split is deliberate: *they train, we measure.*

## 6. Constraints we operate under (so your suggestions are actionable)

- **Compute:** one consumer GPU on a free cloud tier for training (hours, not days); one M1 laptop (no
  CUDA) for verification. No large clusters. A from-scratch ~92M-param model is affordable (~2B tokens,
  hours, ~$20–50 spot); a from-scratch 1B is a real commitment.
- **Base is fixed** for the routing work: Qwen2.5-1.5B, 28 layers, routable 7–20.
- **A from-scratch path exists and is ready:** we have a validated ~92M-param trainer (`config_100m.yaml`,
  14 layers — matching the routable band) and have located the exact ~8-line change to make it train
  *skip-native* (sample a skip set each step from step 0, so there is *nothing to forget*). This is the
  KILL-clause destination.
- **Datasets:** must be openly accessible (we got burned by gated datasets — see v1).
- **We care about honest negatives.** A clean falsification is a publishable result here, not a failure.

## 7. What we want from you (please be concrete and disconfirming)

Pick whichever of these you can say something *non-obvious* about. Specifics beat breadth.

**A. The selection problem under non-additivity (the deepest one).**
A near-dense skip set provably exists at every k, but per-layer marginal importance anti-predicts it,
because layer costs are non-additive (early-layer skips are individually safe, jointly fatal). *What is a
cheap selector that respects non-additivity?* Concrete angles we'd want stress-tested: pairwise/low-order
interaction models over skip sets; a small learned set-scorer trained on combinatorial-oracle labels
(amortizing the search); RL or bandit over skip sets with the verifier as reward; gradient/Hessian-based
joint saliency rather than leave-one-out; submodular/anti-submodular structure (is the cost function
*super*modular, and does that itself suggest an algorithm?). Which of these is most likely to actually
beat random cheaply, and which are dead ends and why?

**B. Co-train vs. from-scratch skip-native — which bet, and how to de-risk it.**
v2 shows a frozen base *can* partly absorb skip-robustness (no collapse) but pays a ~15pp dense tax. Is
the tax fundamental to bolting robustness onto a frozen base (→ go from-scratch), or an artifact of the
recipe we can engineer away (curriculum that ramps skip probability; KL-anchor to the frozen base to
bound forgetting; only adapt MLPs; lower skip-k during training; distill the dense model into the
skipped forward)? If from-scratch skip-native (LayerDrop-from-step-0): what design choices most determine
whether routing becomes *learnable* afterward, vs. just producing a uniformly-mediocre model?

**C. Make skipping actually pay off.**
Identity-skip gives accuracy-at-matched-compute but little real speedup. If the science holds, what's the
cleanest path to a *real* compute win (fused depth-pruned kernels, early-exit with a shared head,
width-sparsity/AMPLIFY instead of whole-layer skip, speculative-style verification)? Which preserves the
"selection under non-additivity" insight rather than discarding it?

**D. The process itself.**
Critique §5. Where is our verification discipline still foolable? What cheap experiment would most quickly
*disconfirm* the thesis (we want the fastest kill shot, not confirmation)? Given the one-GPU/one-laptop
budget, how would you sequence the next 3 experiments for maximum information per dollar? Is there a
standard result in the depth-pruning / early-exit / LayerDrop / conditional-computation literature that
we are about to rediscover the hard way — i.e., what should we just *read* before running anything?

**Please flag:** anything above that you think is already-known/solved in the literature (with a pointer),
any place our reasoning is likely fooling itself, and the single highest-value next move if you had to pick one.

---

## 8. Glossary (decode our shorthand)

- **Frozen base / FROZEN-BASE-v1** — the untouched Qwen2.5-1.5B base model; the comparison anchor.
- **Routable layers / band [7–20]** — the 14 middle layers we're allowed to skip.
- **k** — number of layers skipped per generation (the compute budget; k=0 = dense).
- **SKIP / NORMAL / AMPLIFY** — the three per-layer routing actions (only SKIP is tested so far).
- **Identity-skip** — implementing SKIP as `hidden_state -> hidden_state` (bypass the layer).
- **First Light** — the baseline-separation experiment (dense vs random vs entropy). Verified.
- **E87** — the central thesis test (per-layer-gain framing, then combinatorial k-sweep framing).
- **g_entropy / g_gain / g_VBMCA** — the "estimator ladder": progressively more expensive estimates of a
  layer's marginal value. g_entropy = cheap entropy proxy; g_gain = learned per-layer importance;
  g_VBMCA = a richer estimator not yet built.
- **Achievable ceiling** — best pass-rate over *all* skip sets at a given k (combinatorial oracle).
- **avg-random** — pass-rate averaged over random skip sets at a given k (the baseline to beat).
- **Layer non-additivity** — the cost of skipping a *set* of layers ≠ sum of individual skip costs;
  the central obstacle.
- **Co-train / skip-robust co-train** — LoRA-fine-tuning the frozen base under the skip distribution so
  skipping becomes cheap. v1 (failed: forgetting), v2 (no collapse, 15pp tax, decisive number pending).
- **H-CT-0/1/2, KILL clause** — the co-train's pre-registered hypotheses (§4).
- **Skip-native / LayerDrop** — training a model *from scratch* with random layer-dropping, so
  skip-robustness is built in (nothing to forget); the KILL-clause destination.
- **Repair gym / F2 / F3** — the functional evaluation tasks (F2 = JSON-to-schema, F3 = Python type/logic
  repair); pass = the fix actually runs and satisfies the checker.
- **The certainty register** — overconfident phrasing; we deliberately avoid it.
