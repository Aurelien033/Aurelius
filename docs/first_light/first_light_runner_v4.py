#!/usr/bin/env python3
"""First Light runner v4 — gym-v0.2, all v3 defects fixed (authored by verification line 2026-06-13).

Fixes vs v3:
  - F3: functional verifier on gym-v0.2 (was mypy-on-completion → terseness confound).
  - F2: json.loads AND jsonschema against the embedded schema (was json.loads-only → '{}' passed).
  - H-FL-4: read measured tps from the ledger (was hardcoded 15.0/18.0 placeholder).
  - Pilot block: actually executes at temp 0.7 (v3 wrote a byte-identical duplicate of entropy).
Reuses v3's spike-verified IdentitySkip + completion-style generation verbatim.
"""
import hashlib, json, random, re, subprocess, sys, tempfile, time, types
from pathlib import Path
import numpy as np, torch, yaml
from jsonschema import Draft7Validator, validate as js_validate
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_REPO, MODEL_REVISION = "Qwen/Qwen2.5-1.5B", "8faed761d45a263340a0528343f099c05c9a4323"
CONFIG_HASH = "0e8c8aa86468aba0"
ROUTABLE = list(range(7, 21)); K_SKIP = 4; SEEDS = [1337, 2026, 7]; MAX_NEW = 512
import os
# resolution: $AURELIUS_RESEARCH > local Mac Research folder (full data) > bundled repo/eval_data (Lightning)
_mac = Path("/Users/christienantonio/Desktop/AI:ML Research")
_bundled = Path(__file__).resolve().parents[2] / "eval_data"
RESEARCH = Path(os.environ["AURELIUS_RESEARCH"]) if os.environ.get("AURELIUS_RESEARCH") else (_mac if _mac.exists() else _bundled)
GYM_V02, GYM_V01 = RESEARCH / "gym-v0.2", RESEARCH / "gym-v0.1-FL"
SCHEMA = json.load(open(RESEARCH / "directive_trace_schema.json"))
OUT = RESEARCH / "first_light_receipt"
RUN_ID = "first-light-v4-" + hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]


def now(): return time.strftime("%Y-%m-%dT%H:%M:%S")
def sha16(s): return hashlib.sha256(s.encode()).hexdigest()[:16]


# ── instance index: id → file, across both gym roots ──
def build_index():
    idx = {}
    for root, fams in [(GYM_V01, ["F1_mbpp", "F2_json"]), (GYM_V02, ["F3_type"])]:
        for fam in fams:
            for split in ["test", "smoke"]:
                for p in (root / fam / split).glob("*.json"):
                    d = json.loads(p.read_text()); idx[d["instance_id"]] = d
    return idx


def first_brace_block(s):
    i = s.find("{")
    if i < 0: return None
    depth = 0
    for j in range(i, len(s)):
        depth += (s[j] == "{") - (s[j] == "}")
        if depth == 0: return s[i:j + 1]
    return None


# ── verifiers ──
def verify_f1(inst, c):
    prog = c.rstrip() + "\n" + "\n".join(inst["hidden_tests"]) + "\n"
    return _run_py(prog)


def verify_f2(inst, c):
    block = first_brace_block(c)
    if not block: return False
    try:
        obj = json.loads(block)
    except Exception:
        return False
    schema_str = inst["hidden_tests"][1] if len(inst["hidden_tests"]) > 1 else ""
    sb = first_brace_block(schema_str)
    if sb:
        try:
            js_validate(obj, json.loads(sb))
        except Exception:
            return False
    return True


def verify_f3(inst, c):
    fn = inst["metadata"]["function_name"]
    # Completion-style: the prompt ends with the function header, so the model's completion is the
    # BODY only. If the def line is absent, prepend the header (mirrors the spike's sig-prepend).
    if f"def {fn}(" not in c:
        c = inst["function_header"] + "\n" + c
    lines = c.splitlines(); code = ""
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith(f"def {fn}("):
            body = [lines[i]]
            for nxt in lines[i + 1:]:
                if nxt.strip() == "" or nxt.startswith((" ", "\t")): body.append(nxt)
                else: break
            code = "\n".join(body); break
    if not code: return False
    return _run_py(code.rstrip() + "\n" + "\n".join(inst["hidden_tests"]) + "\n")


def _run_py(prog):
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(prog); p = f.name
        return subprocess.run([sys.executable, p], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False
    finally:
        Path(p).unlink(missing_ok=True)


VERIFIERS = {"F1_mbpp": verify_f1, "F2_json": verify_f2, "F3_type": verify_f3}


# ── IdentitySkip (spike-verified, from v3 verbatim) ──
class IdentitySkip:
    def __init__(self, model, idxs): self.layers = model.model.layers; self.idxs = idxs; self.orig = {}
    def __enter__(self):
        for i in self.idxs:
            self.orig[i] = self.layers[i].forward
            def ident(self_layer, hidden_states, *a, **k): return hidden_states
            self.layers[i].forward = types.MethodType(ident, self.layers[i])
    def __exit__(self, *a):
        for i, o in self.orig.items(): self.layers[i].forward = o
        self.orig.clear()


def generate(tok, model, prompt, seed, temperature=0.0, skip=None):
    torch.manual_seed(seed)
    inp = tok(prompt, return_tensors="pt").to(model.device)
    n_in = inp["input_ids"].shape[1]
    kw = dict(**inp, max_new_tokens=MAX_NEW, pad_token_id=tok.eos_token_id,
              do_sample=(temperature > 0))
    if temperature > 0: kw.update(temperature=temperature, top_p=0.95)
    t0 = time.time()
    with torch.no_grad():
        if skip:
            with IdentitySkip(model, skip): gen = model.generate(**kw)
        else: gen = model.generate(**kw)
    dt = time.time() - t0
    new = gen[0][n_in:]
    return tok.decode(new, skip_special_tokens=True), int(new.shape[0]), dt


def entropy_skip(tok, model, prompt, seed):
    """4 routable layers with smallest |H_l - H_{l-1}| from dense prefill."""
    torch.manual_seed(seed)
    inp = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    H = {}
    for li in ROUTABLE:
        logits = model.lm_head(out.hidden_states[li + 1][:, -1, :])
        p = torch.softmax(logits.float(), -1)
        H[li] = float(-(p * torch.log(p + 1e-10)).sum())
    d = {ROUTABLE[i]: (abs(H[ROUTABLE[i]] - H[ROUTABLE[i-1]]) if i else 0.0) for i in range(len(ROUTABLE))}
    return sorted(d, key=d.get)[:K_SKIP], H


def trace_rows(iid, policy, skip, run_id):
    rung = "g_entropy" if policy == "heuristic_entropy" else "none_forced"
    rows = []
    for li in ROUTABLE:
        d = "SKIP" if (skip and li in skip) else "NORMAL"
        rows.append({"schema_version": "1.1.0", "run_id": run_id, "config_id": "FROZEN-BASE-v1",
            "adr_branch": "ratified", "seq_id": iid, "token_pos": 0, "layer_group": f"L{li}",
            "phase": "decode", "execution_mode": "live" if policy == "dense_uniform" else "offline_replay",
            "cadence_path": "slow_full",
            "proposal": {"directive": d, "logits_q": [0.0, 0.0, 0.0]},
            "projected_directive": d, "final_directive": d, "reason_code": "FORCED_BASELINE",
            "estimator_rung": rung, "g_hat": None,
            "cost_estimate": {"flops_directive": 0.0, "controller_tax_flops": 0.0},
            "ring": "R0", "contaminated_for_paper1": True})
    return rows


def ledger_row(iid, seed, policy, passed, tps, run_id):
    rmode = {"dense_uniform": "none_dense"}.get(policy, "forced_offline_replay")
    return {"run_id": run_id, "instance_id": iid, "seed": seed,
        "policy": {"baseline_type": policy, "routing_mode": rmode, "contract_version": "acdt-minimal-v1"},
        "config_id": "FROZEN-BASE-v1",
        "cost": {"flops_total": 0.0, "controller_tax_flops": 0.0, "guard_tax_flops": 0.0,
                 "bytes_per_token": 0.0, "wall_clock_sustained_tps": {"p50": tps, "p95": tps}, "phase": "decode"},
        "outcome": {"verifier_class": "deterministic", "verified_pass": passed,
                    "counterfactual_kind": "observed_replay" if policy != "dense_uniform" else "observed_replay",
                    "estimator_rung": "g_entropy" if policy == "heuristic_entropy" else "none_forced"},
        "flags": {"pilot_or_claim_eligible": "pilot", "contaminated_for_paper1": True}}


def run_one(tok, model, inst, policy, seed, temperature=0.0):
    fam = inst["metadata"]["family"]; iid = inst["instance_id"]; prompt = inst["prompt_context"]
    skip = None; Hdata = None
    if policy == "random_matched_mix":
        skip = random.Random(sha16(iid + str(seed))).sample(ROUTABLE, K_SKIP)
    elif policy == "heuristic_entropy":
        skip, Hdata = entropy_skip(tok, model, prompt, seed)
    comp, ntok, dt = generate(tok, model, prompt, seed, temperature, skip)
    passed = VERIFIERS[fam](inst, comp)
    tps = ntok / dt if dt > 0 else 0.0
    return {"instance_id": iid, "family": fam, "policy": policy, "seed": seed,
            "passed": bool(passed), "realized_tokens": ntok, "tps": tps,
            "skip": skip, "completion": comp}


def selfproof_p1(tok, model):
    inp = tok("def f(x: int) -> int:", return_tensors="pt").to(model.device)
    with torch.no_grad():
        a = model(**inp).logits.float().cpu()
        with IdentitySkip(model, ROUTABLE[:K_SKIP]): b = model(**inp).logits.float().cpu()
        c = model(**inp).logits.float().cpu()
    return {"max_logit_diff": float((a - b).abs().max()), "restore_bit_identical": bool(torch.equal(a, c))}


def main():
    OUT.mkdir(exist_ok=True); rd = OUT / RUN_ID; rd.mkdir(exist_ok=True)
    print(f"[{now()}] {RUN_ID}  gym-v0.2")
    man = yaml.safe_load((GYM_V02 / "gym_manifest.yaml").read_text())
    idx = build_index()
    fl_ids = [i for i in man["fl_subset_ids"] if i in idx]
    smoke_ids = [i for i in man["smoke_set_ids"] if i in idx]
    print(f"  fl_subset={len(fl_ids)} smoke={len(smoke_ids)} (manifest fl={len(man['fl_subset_ids'])})")
    assert len(fl_ids) == len(man["fl_subset_ids"]), "missing instances in index"

    tok = AutoTokenizer.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_REPO, revision=MODEL_REVISION,
        dtype=torch.bfloat16, low_cpu_mem_usage=True).to("mps").eval()
    cfg = __import__("huggingface_hub").hf_hub_download(MODEL_REPO, "config.json", revision=MODEL_REVISION)
    assert hashlib.sha256(open(cfg, "rb").read()).hexdigest()[:16] == CONFIG_HASH
    sp = {"p1_routing": selfproof_p1(tok, model),
          "p2_same_instances": {"id_hash": sha16(json.dumps(sorted(fl_ids))), "n": len(fl_ids)},
          "p3_bench_gate": man.get("bench_gate", "see gym-v0.2 build")}
    print(f"  P1 {sp['p1_routing']}  P2 n={sp['p2_same_instances']['n']}")
    assert sp["p1_routing"]["max_logit_diff"] > 1.0 and sp["p1_routing"]["restore_bit_identical"]

    pol3 = ["dense_uniform", "random_matched_mix", "heuristic_entropy"]
    val = Draft7Validator(SCHEMA)

    def run_block(ids, policies, seeds, temp, tag):
        recs, traces, ledg = [], [], []
        for iid in ids:
            for pol in policies:
                for sd in seeds:
                    r = run_one(tok, model, idx[iid], pol, sd, temp)
                    recs.append({k: r[k] for k in ("instance_id", "family", "policy", "seed", "passed", "realized_tokens", "tps")})
                    (rd / "completions" / f"{tag}_{pol}").mkdir(parents=True, exist_ok=True)
                    (rd / "completions" / f"{tag}_{pol}" / f"{iid.replace('/','_')}_s{sd}.txt").write_text(r["completion"])
                    for tr in trace_rows(iid, pol, r["skip"], RUN_ID):
                        assert not next(val.iter_errors(tr), None)
                        traces.append(tr)
                    ledg.append(ledger_row(iid, sd, pol, r["passed"], r["tps"], RUN_ID))
            _dump(rd, tag, recs, traces, ledg)  # checkpoint per instance
        return recs, traces, ledg

    # ── SMOKE GATE ──
    print(f"[{now()}] SMOKE")
    srecs, straces, _ = run_block(smoke_ids, pol3, [SEEDS[0]], 0.0, "smoke")
    if any(next(val.iter_errors(t), None) for t in straces) or len(straces) == 0:
        print("SMOKE GATE FAIL: traces"); sys.exit(1)
    fr = {p: sum(r["passed"] for r in srecs if r["policy"] == p) for p in pol3}
    print(f"  smoke passes {fr} (traces={len(straces)}, 0 schema errors)")
    if fr["random_matched_mix"] == 0 and fr["heuristic_entropy"] == 0:
        print("  NOTE forced rows 0 on smoke — proceeding (fallback ladder not triggered at n=smoke)")

    # ── MAIN CHESSBOARD ──
    print(f"[{now()}] DENSE+FORCED  ({len(fl_ids)} x {len(pol3)} x {len(SEEDS)})")
    recs, traces, ledg = run_block(fl_ids, pol3, SEEDS, 0.0, "main")

    # ── PILOT (temp 0.7) — ACTUALLY runs (v3 bug: was a duplicate) ──
    print(f"[{now()}] PILOT temp=0.7")
    precs, ptraces, pledg = run_block(fl_ids, pol3, SEEDS, 0.7, "pilot")

    # ── ANALYSIS ──
    print(f"[{now()}] ANALYSIS")
    bi = {}
    for r in recs: bi.setdefault(r["instance_id"], {}).setdefault(r["policy"], []).append(r["passed"])
    paired = [i for i in fl_ids if all(p in bi[i] for p in pol3)]
    def rate(p): return float(np.mean([any(bi[i][p]) for i in paired]))
    def fam_split():
        out = {}
        for fam in ["F1_mbpp", "F2_json", "F3_type"]:
            fi = [i for i in paired if idx[i]["metadata"]["family"] == fam]
            out[fam] = {p: (sum(any(bi[i][p]) for i in fi), len(fi)) for p in pol3}
        return out
    def pboot(pa, pb, B=10000):
        rng = np.random.RandomState(20260613)
        deltas = np.array([int(any(bi[i][pa])) - int(any(bi[i][pb])) for i in paired])
        bs = [np.mean(deltas[rng.randint(0, len(deltas), len(deltas))]) for _ in range(B)]
        return float(deltas.mean()), (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    dr, dr_ci = pboot("dense_uniform", "random_matched_mix")
    re_pt, re_ci = pboot("heuristic_entropy", "random_matched_mix")
    # H-FL-4 from REAL ledger tps
    def tps_p(pred, q):
        v = [l["cost"]["wall_clock_sustained_tps"]["p50"] for l in ledg if pred(l) and l["cost"]["wall_clock_sustained_tps"]["p50"] > 0]
        return float(np.percentile(v, q)) if len(v) >= 2 else (v[0] if v else 0.0)
    dense_tps = tps_p(lambda l: l["policy"]["baseline_type"] == "dense_uniform", 50)
    bp = yaml.safe_load((RESEARCH / "first_light_byte_predictions.yaml").read_text())
    pred = bp.get("throughput_predictions", {}).get("utilization_prior_tps", 15.0)
    h4_off = abs(dense_tps - pred) / pred if pred else 1.0
    verdicts = {
        "H-FL-1": {"pass": len(traces) > 0, "detail": f"{len(traces)} traces, 0 schema errors (validated at emit)"},
        "H-FL-2": {"pass": dr_ci[0] > 0, "detail": f"Δ(dense-random)={dr*100:+.1f}pp CI[{dr_ci[0]*100:+.1f},{dr_ci[1]*100:+.1f}]"},
        "H-FL-3": {"detail": f"Δ(entropy-random)={re_pt*100:+.1f}pp CI[{re_ci[0]*100:+.1f},{re_ci[1]*100:+.1f}]",
                   "tie": abs(re_ci[0]) <= 0.08 and abs(re_ci[1]) <= 0.08, "entropy_wins": re_ci[0] > 0},
        "H-FL-4": {"pass": h4_off <= 0.20, "detail": f"dense {dense_tps:.1f} tok/s vs pred {pred} ({h4_off*100:.0f}% off)"},
    }
    analysis = {"per_policy": {p: {"rate": rate(p), "n": len(paired)} for p in pol3},
                "per_family": {f: {p: list(v) for p, v in d.items()} for f, d in fam_split().items()},
                "deltas": {"dense_vs_random": [dr, list(dr_ci)], "entropy_vs_random": [re_pt, list(re_ci)]},
                "pilot_ran": len(precs) > 0 and (rd / "completions" / "pilot_dense_uniform").exists(),
                "hypotheses": verdicts}
    receipt = {"run_id": RUN_ID, "date": now(), "status": "COMPLETE", "gym": "gym-v0.2",
               "gym_manifest_hash": man["manifest_hash"], "model": {"repo": MODEL_REPO, "rev": MODEL_REVISION},
               "self_proofs": sp, "analysis": analysis, "n_paired": len(paired),
               "non_claims": ["NO VB-MCA superiority", "NO speedup", "NO scaling"]}
    (rd / f"{RUN_ID}_receipt.yaml").write_text(yaml.safe_dump(receipt, sort_keys=False, allow_unicode=True))
    _dump(rd, "main", recs, traces, ledg); _dump(rd, "pilot", precs, ptraces, pledg)
    print(f"[{now()}] DONE")
    print(f"  per_policy: " + "  ".join(f"{p.split('_')[0]}={rate(p)*100:.1f}%" for p in pol3))
    print(f"  per_family: {json.dumps(analysis['per_family'])}")
    for h, v in verdicts.items(): print(f"  {h}: {v}")
    print(f"  receipt: {rd / (RUN_ID + '_receipt.yaml')}")


def _dump(rd, tag, recs, traces, ledg):
    (rd / f"results_{tag}.json").write_text(json.dumps(recs))
    with open(rd / f"traces_{tag}.jsonl", "w") as f:
        for t in traces: f.write(json.dumps(t) + "\n")
    with open(rd / f"ledger_{tag}.jsonl", "w") as f:
        for l in ledg: f.write(json.dumps(l) + "\n")


if __name__ == "__main__":
    main()
