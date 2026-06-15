#!/usr/bin/env python3
"""Independent verification of the v4 First Light run — reconstruct ALL verdicts from raw artifacts,
never trusting the runner's own receipt. Pure stdlib + jsonschema. No model, no MPS.

Usage: python verify_v4.py <run_dir>
"""
import json, statistics, sys
from pathlib import Path

R = Path("/Users/christienantonio/Desktop/AI:ML Research")
SCHEMA = json.load(open(R / "directive_trace_schema.json"))
POL = ["dense_uniform", "random_matched_mix", "heuristic_entropy"]


def jload(p): return json.loads(Path(p).read_text()) if Path(p).exists() else None
def jlines(p): return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def fam_of(iid, idx): return idx[iid]["metadata"]["family"]


def main():
    rd = Path(sys.argv[1])
    print(f"VERIFY v4: {rd}")
    recs = jload(rd / "results_main.json") or []
    if not recs:
        print("  results_main.json empty/absent — run not finished"); return

    # build instance index for family lookup
    idx = {}
    for root, fams in [(R/"gym-v0.1-FL", ["F1_mbpp","F2_json"]), (R/"gym-v0.2", ["F3_type"])]:
        for fam in fams:
            for sp in ["test","smoke"]:
                for f in (root/fam/sp).glob("*.json"):
                    d = json.loads(f.read_text()); idx[d["instance_id"]] = d

    bi = {}
    for r in recs:
        bi.setdefault(r["instance_id"], {}).setdefault(r["policy"], []).append(r["passed"])
    paired = [i for i in bi if all(p in bi[i] for p in POL)]
    print(f"  records={len(recs)} instances={len(bi)} fully-paired={len(paired)}")

    # P2 independent: same instance set per policy
    sets = {p: frozenset(i for i in bi if p in bi[i]) for p in POL}
    print(f"  P2 same-instance-set: {len(set(sets.values()))==1}")

    # H-FL-1: validate ALL main traces
    try:
        from jsonschema import Draft7Validator
        v = Draft7Validator(SCHEMA)
        tr = jlines(rd / "traces_main.jsonl")
        errs = sum(1 for r in tr for _ in v.iter_errors(r))
        rc = sum(1 for r in tr if not r.get("reason_code"))
        print(f"  H-FL-1: {len(tr)} traces, schema_errors={errs}, missing_rc={rc} -> {'PASS' if errs==0 and rc==0 and tr else 'FAIL'}")
    except ImportError:
        print("  H-FL-1: jsonschema unavailable")

    # per-policy + per-family (instance-level, any seed)
    def rate(ids, p):
        a=[any(bi[i][p]) for i in ids if p in bi[i]]; return (sum(a), len(a))
    print("  per-policy (paired):", {p: f"{rate(paired,p)[0]}/{rate(paired,p)[1]}" for p in POL})
    print("  per-family:")
    for fam in ["F1_mbpp","F2_json","F3_type"]:
        fi=[i for i in paired if fam_of(i,idx)==fam]
        print(f"    {fam}: " + "  ".join(f"{p.split('_')[0]}={rate(fi,p)[0]}/{rate(fi,p)[1]}" for p in POL))

    # paired deltas + bootstrap
    import random
    def pboot(pa, pb, B=10000):
        rng=random.Random(20260613)
        delt=[int(any(bi[i][pa]))-int(any(bi[i][pb])) for i in paired]
        n=len(delt); pt=statistics.fmean(delt)
        bs=sorted(statistics.fmean(delt[rng.randrange(n)] for _ in range(n)) for _ in range(B))
        return pt,(bs[int(.025*B)],bs[int(.975*B)])
    dr,drc=pboot("dense_uniform","random_matched_mix")
    er,erc=pboot("heuristic_entropy","random_matched_mix")
    print(f"  H-FL-2 Δ(dense-random)={dr*100:+.1f}pp CI[{drc[0]*100:+.1f},{drc[1]*100:+.1f}] -> {'PASS' if drc[0]>0 else 'FAIL(harness-bug protocol)'}")
    tie = abs(erc[0])<=0.08 and abs(erc[1])<=0.08
    print(f"  H-FL-3 Δ(entropy-random)={er*100:+.1f}pp CI[{erc[0]*100:+.1f},{erc[1]*100:+.1f}] -> {'ENTROPY WINS' if erc[0]>0 else ('TIE' if tie else 'ENTROPY BEHIND')}")

    # H-FL-4 from ledger
    led = jlines(rd / "ledger_main.jsonl")
    def tps(pred):
        vv=[l["cost"]["wall_clock_sustained_tps"]["p50"] for l in led if pred(l) and l["cost"]["wall_clock_sustained_tps"]["p50"]>0]
        return statistics.median(vv) if vv else 0.0
    dt=tps(lambda l: l["policy"]["baseline_type"]=="dense_uniform")
    st=tps(lambda l: l["policy"]["baseline_type"]!="dense_uniform")
    print(f"  H-FL-4: dense {dt:.1f} tok/s, skip {st:.1f} tok/s | vs v2-pred 27/28 (band 21-34) -> {'PASS' if 21<=dt<=34 else 'CHECK'}")

    # pilot distinctness (the v3 bug)
    pmain = sorted((rd/"completions"/"main_dense_uniform").glob("*.txt")) if (rd/"completions"/"main_dense_uniform").exists() else []
    ppil = sorted((rd/"completions"/"pilot_dense_uniform").glob("*.txt")) if (rd/"completions"/"pilot_dense_uniform").exists() else []
    if pmain and ppil:
        common={p.name for p in pmain} & {p.name for p in ppil}
        diff=sum(1 for n in list(common)[:50] if (rd/"completions"/"main_dense_uniform"/n).read_text()!=(rd/"completions"/"pilot_dense_uniform"/n).read_text())
        print(f"  PILOT distinct from greedy: {diff}/{min(50,len(common))} differ (temp 0.7 ran for real)" if common else "  PILOT: no overlap to compare")
    else:
        print(f"  PILOT: main={len(pmain)} pilot={len(ppil)} completion files")


if __name__ == "__main__":
    main()
