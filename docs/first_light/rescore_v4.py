#!/usr/bin/env python3
"""Independent re-scoring of EVERY v4 completion — re-run the verifiers on the raw saved completions
and check agreement with the runner's recorded pass/fail. This is the gold-standard verification:
it does not trust results_main.json's verdicts, it recomputes them from completion text. No MPS."""
import json, sys
from pathlib import Path
sys.path.insert(0, "/Users/christienantonio/aurelius/docs/first_light")
import first_light_runner_v4 as R   # reuse the SAME verifiers the run used

RUN = Path("/Users/christienantonio/Desktop/AI:ML Research/first_light_receipt/first-light-v4-24515383")
idx = R.build_index()
recs = json.load(open(RUN / "results_main.json"))

agree = disagree = 0
mism = []
for r in recs:
    iid, pol, sd = r["instance_id"], r["policy"], r["seed"]
    cf = RUN / "completions" / f"main_{pol}" / f"{iid.replace('/','_')}_s{sd}.txt"
    if not cf.exists():
        mism.append((iid, pol, sd, "MISSING_COMPLETION")); disagree += 1; continue
    comp = cf.read_text()
    fam = r["family"]
    my = R.VERIFIERS[fam](idx[iid], comp)
    if bool(my) == bool(r["passed"]):
        agree += 1
    else:
        disagree += 1
        mism.append((iid, pol, sd, f"runner={r['passed']} me={my}"))

print(f"re-scored {len(recs)} completions: AGREE={agree} DISAGREE={disagree}")
for m in mism[:20]:
    print("  MISMATCH:", m)
# independent per-family/policy from MY re-scoring
from collections import defaultdict
bi = defaultdict(lambda: defaultdict(list))
for r in recs:
    cf = RUN / "completions" / f"main_{r['policy']}" / f"{r['instance_id'].replace('/','_')}_s{r['seed']}.txt"
    my = R.VERIFIERS[r["family"]](idx[r["instance_id"]], cf.read_text()) if cf.exists() else False
    bi[(r["family"], r["policy"])][r["instance_id"]].append(bool(my))
print("\nMY independent per-family (instance-level, any seed):")
for fam in ["F1_mbpp", "F2_json", "F3_type"]:
    cells = []
    for pol in ["dense_uniform", "random_matched_mix", "heuristic_entropy"]:
        d = bi[(fam, pol)]; n = sum(1 for i in d if any(d[i]))
        cells.append(f"{pol.split('_')[0]}={n}/{len(d)}")
    print(f"  {fam}: " + "  ".join(cells))
