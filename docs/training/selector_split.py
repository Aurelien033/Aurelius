#!/usr/bin/env python3
"""Draw + hash-pin the per-task selector split (default 150 F2 + 150 F3 = 300), SUPERSET of the E87 60.
Reads the existing clean gym pools in repo/eval_data — NO generation. Deterministic (seeded).

Run:  python docs/training/selector_split.py            # writes eval_data/selector_split_v0.1.json
"""
import argparse, json, hashlib, random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "eval_data"
POOLS = [(EVAL / "gym-v0.1-FL" / "F2_json", "F2_json"), (EVAL / "gym-v0.3" / "F3_type", "F3_type")]
SPLITS = ["dev", "test", "smoke", "train"]


def pool_ids(root):
    ids = []
    for sp in SPLITS:
        for f in (root / sp).glob("*.json"):
            ids.append(json.loads(f.read_text())["instance_id"])
    return sorted(set(ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_family", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260617)
    ap.add_argument("--out", default=str(EVAL / "selector_split_v0.1.json"))
    a = ap.parse_args()
    rng = random.Random(a.seed)
    e87 = set(json.loads((EVAL / "e87_receipt" / "e87_split.json").read_text())["test"])

    families = {}
    for root, fam in POOLS:
        ids = pool_ids(root)
        keep = sorted(i for i in ids if i in e87)            # carry the E87 60 (superset / nested check)
        rest = [i for i in ids if i not in e87]; rng.shuffle(rest)
        need = max(0, a.n_per_family - len(keep))
        if a.n_per_family > len(ids):
            print(f"  WARN {fam}: only {len(ids)} available, wanted {a.n_per_family}")
        families[fam] = sorted(keep + rest[:need])

    allids = sorted(families["F2_json"] + families["F3_type"])
    h = hashlib.sha256("\n".join(allids).encode()).hexdigest()[:16]
    out = {"id": "selector_split_v0.1", "seed": a.seed, "n": len(allids),
           "by_family": {k: len(v) for k, v in families.items()},
           "includes_e87_60": e87.issubset(set(allids)),
           "split_hash": h, "all": allids, "families": families}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {a.out}\n  N={len(allids)} {out['by_family']}  includes_e87_60={out['includes_e87_60']}  hash={h}")


if __name__ == "__main__":
    main()
