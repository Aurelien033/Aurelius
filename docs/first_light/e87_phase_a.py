#!/usr/bin/env python3
"""E87 Phase A — leave-one-layer-out ablation → per-layer verified-gain ORACLE labels.

For each clean-family instance (F2 + gym-v0.3 F3): run dense, then skip exactly ONE routable layer
l at a time; verify each. Per-layer gold label = V(dense) - V(skip only l). Also records cheap
features (per-layer entropy H_l, |ΔH_l|) for training g_gain in Phase B. 1 greedy seed (labels are
deterministic). Reuses the v4 spike-verified IdentitySkip + verifiers. Checkpoint/resume. No cloud.
"""
import hashlib, json, sys, time
from pathlib import Path
import torch
sys.path.insert(0, "/Users/christienantonio/aurelius/docs/first_light")
import first_light_runner_v4 as R

RESEARCH = Path("/Users/christienantonio/Desktop/AI:ML Research")
OUT = RESEARCH / "e87_receipt"; OUT.mkdir(exist_ok=True)
ROUTABLE = list(range(7, 21))
SEED = 1337
MAX_NEW = 256          # short repairs; consistent for Phase A and the Phase B chessboard (prereg freeze)
SPLIT_SEED = 20260614
def sha16(s): return hashlib.sha256(s.encode()).hexdigest()[:16]


def build_clean_index():
    """F2 from gym-v0.1-FL, F3 from gym-v0.3 (harder). Clean families only for E87 primary signal."""
    idx = {}
    for root, fam in [(RESEARCH/"gym-v0.1-FL", "F2_json"), (RESEARCH/"gym-v0.3", "F3_type")]:
        for sp in ["test", "smoke"]:
            for f in (root/fam/sp).glob("*.json"):
                d = json.loads(f.read_text()); idx[d["instance_id"]] = d
    return idx


def pin_split(ids_by_fam):
    """70/30 train/test per family, seeded + hash-pinned."""
    import random
    rng = random.Random(SPLIT_SEED)
    split = {"train": [], "test": []}
    for fam, ids in ids_by_fam.items():
        ids = sorted(ids); rng.shuffle(ids)
        cut = int(0.7 * len(ids))
        split["train"] += ids[:cut]; split["test"] += ids[cut:]
    split["train"].sort(); split["test"].sort()
    split["split_hash"] = sha16(json.dumps(split["train"]) + json.dumps(split["test"]))
    (OUT/"e87_split.json").write_text(json.dumps(split, indent=2))
    return split


def gen_skip(tok, model, prompt, skip):
    return R.generate(tok, model, prompt, SEED, temperature=0.0, skip=skip)


def main():
    idx = build_clean_index()
    by_fam = {}
    for iid, d in idx.items():
        by_fam.setdefault(d["metadata"]["family"], []).append(iid)
    # use 100 per clean family (the full pinned sets)
    man02 = __import__("yaml").safe_load((RESEARCH/"gym-v0.1-FL"/"gym_manifest.yaml").read_text())
    man03 = __import__("yaml").safe_load((RESEARCH/"gym-v0.3"/"gym_manifest.yaml").read_text())
    f2_ids = [i for i in man02["fl_subset_ids"] if "/F2/" in i]
    f3_ids = [i for i in man03["fl_subset_ids"] if "/F3/" in i]
    inst_ids = [i for i in (f2_ids + f3_ids) if i in idx]
    split = pin_split({"F2_json": f2_ids, "F3_type": f3_ids})
    print(f"E87 Phase A: F2={len(f2_ids)} F3v0.3={len(f3_ids)} total={len(inst_ids)} "
          f"| split train={len(split['train'])} test={len(split['test'])} hash={split['split_hash']}")

    tok = R.AutoTokenizer.from_pretrained(R.MODEL_REPO, revision=R.MODEL_REVISION)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = R.AutoModelForCausalLM.from_pretrained(R.MODEL_REPO, revision=R.MODEL_REVISION,
        dtype=torch.bfloat16, low_cpu_mem_usage=True).to("mps").eval()
    R.MAX_NEW = MAX_NEW  # override module default for consistency

    ckpt = OUT / "phase_a_labels.jsonl"
    done = set()
    if ckpt.exists():
        for l in open(ckpt):
            r = json.loads(l); done.add(r["instance_id"])
    print(f"resuming: {len(done)} instances already labelled")

    t0 = time.time(); n = 0
    with open(ckpt, "a") as out:
        for iid in inst_ids:
            if iid in done: continue
            inst = idx[iid]; fam = inst["metadata"]["family"]; prompt = inst["prompt_context"]
            verify = R.VERIFIERS[fam]
            # dense + features
            skip_layers, H = R.entropy_skip(tok, model, prompt, SEED)  # gives per-layer entropy too
            dcomp, dtok, ddt = gen_skip(tok, model, prompt, None)
            dV = int(verify(inst, dcomp))
            per_layer = {}
            for l in ROUTABLE:
                scomp, stok, sdt = gen_skip(tok, model, prompt, [l])
                sV = int(verify(inst, scomp))
                per_layer[l] = {"skip_pass": sV, "gain": dV - sV,
                                "H": H.get(l), "dH": (abs(H[l]-H[l-1]) if (l-1) in H else 0.0)}
            row = {"instance_id": iid, "family": fam, "split": ("train" if iid in split["train"] else "test"),
                   "dense_pass": dV, "per_layer": per_layer, "dense_tps": dtok/ddt if ddt > 0 else 0.0}
            out.write(json.dumps(row) + "\n"); out.flush()
            n += 1
            if n % 10 == 0:
                el = time.time()-t0
                print(f"  [{time.strftime('%H:%M:%S')}] {n} done this run ({el/n:.1f}s/inst, {len(done)+n}/{len(inst_ids)} total)", flush=True)
    print(f"PHASE A DONE: {len(done)+n}/{len(inst_ids)} instances; labels -> {ckpt}")


if __name__ == "__main__":
    main()
