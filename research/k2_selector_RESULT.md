# Exact k=2 pair matrix — selector verdict (2026-06-17)

Source: k2_pair_matrix.py (all 91 pairs × 60 held-out tasks, frozen base, Kaggle T4 fp16, GREEDY),
analysis by k2_analysis.py (task-level 5-fold CV). Re-derivation from the matrix AGREES with the run
headline (ceiling 96.7%, avg-random 38.9%, dense 75.0%) — matrix sound.

## Held-out 5-fold task-level CV (evaluated on UNSEEN tasks)
| policy | held-out pass | note |
|---|---|---|
| random            | 38.9% (±7.8)  | the bar |
| dense (full model)| 75.0% (±13.9) | k=0 |
| global_best_pair  | **56.7%** (±9.7)  | ONE fixed pair (skip 12&18), zero features — beats random +18pp, BELOW dense |
| best_of_top3      | 75.0% (±15.8) | top-3 fixed pairs, any-pass (verifier-assisted) = DENSE |
| best_of_top5      | **88.3%** (±8.5)  | = dense +13pp |
| best_of_top10     | 95.0% (±6.7)  | ≈ ceiling |
| ceiling (oracle)  | 96.7% (±4.1)  | per-task best of 91 |

top-1 pair per fold: 12-18,12-18,12-16,12-18,12-16  (STABLE -> selection signal robust)
per-family avg-random: F2_json 46.3%, F3_type 31.4%

## Verdict
- SELECTION IS CHEAP + LEARNABLE: a fixed GLOBAL shortlist generalizes to unseen tasks. A few verified
  tries (top-3) = dense; top-5 EXCEEDS dense (+13pp); top-10 ≈ ceiling.
- CORRECTION: single-best-pair HELD-OUT = 56.7%, NOT the 72% full-data figure (that was selection-on-test
  inflation; the rigorous CV caught it). A single cheap skip set does NOT beat dense.
- HEADROOM: global single-pair 56.7% vs per-task ORACLE 96.7% = ~40pp gap that only a PER-TASK
  (prompt-conditioned / activation-feature) selector can capture -> the clean next experiment.
- COMPUTE caveat: best_of_topM is M× the k=2 compute + a verifier/acceptance check (self-verifying /
  best-of-N regime), NOT a single-pass speedup. The genuine single-pass-cheaper point is global_best_pair
  (56.7% at ~0.93× compute). The accuracy WIN above dense requires verified multi-try OR per-task selection.

## Mechanism (full-data)
POISON to skip: L14 -18.7pp, L11 -12.3, L15 -7.7, L17 -4.7.  SAFE: L18 +8.9, L9 +7.7, L16 +7.6, L10 +7.3,
L8 +4.9, L19 +4.6, L13 +3.4.  Adjacency BAD (adjacent 29% vs non-adjacent 40%). Worst pair 14-15=0%
(poison+adjacent). Best pairs = well-separated SAFE layers (12-18, 12-16, 14-18, 8-19). REFUTES E87's
"early layers bad" — early L8/9/10 are among the SAFEST; it's specific layers (11/14/15) + adjacency.
