#!/usr/bin/env python3
"""bestofn_repair.py — inference-time verifier-native wrapper: best-of-N + one-step repair (HumanEval/MBPP).

The cheap, no-training lever from the v2/parity plan. Per problem: sample N, EXECUTE each against the verifier,
return a passing solution; if none pass, feed the failing test output back and REPAIR (R rounds). Measures how
far the SYSTEM alone gets beyond greedy pass@1 — i.e. it cashes in any selection gap the pass@k map revealed,
with zero training. Also emits a verified-data FLYWHEEL: winners (SFT) + winner-vs-near-miss (DPO) to distill later.

  python docs/training/bestofn_repair.py --model /content/aurelius-rlvr --bench humaneval --n 8 --repair 1 \
     --flywheel /content/flywheel.jsonl
"""
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_code_bench import humaneval_items, mbpp_items, extract_code, run_with_feedback, build_msg


def gen(tok, model, prompt, dev, max_new, n=1, temp=0.8, think=0):
    enc = tok(build_msg(tok, prompt, think), return_tensors="pt", add_special_tokens=False).to(dev)
    kw = dict(max_new_tokens=max_new, pad_token_id=tok.eos_token_id, use_cache=True)
    kw.update(dict(num_return_sequences=n, do_sample=True, temperature=temp, top_p=0.95) if n > 1 else dict(do_sample=False))
    with torch.no_grad():
        out = model.generate(**enc, **kw)
    L = enc["input_ids"].shape[1]
    return [tok.decode(r[L:], skip_special_tokens=True) for r in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bench", choices=["humaneval", "mbpp"], default="humaneval")
    ap.add_argument("--n", type=int, default=8, help="best-of-N samples per problem")
    ap.add_argument("--repair", type=int, default=1, help="repair rounds if all N fail")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max_new", type=int, default=768)
    ap.add_argument("--think", type=int, default=0)
    ap.add_argument("--n_probs", type=int, default=0, help="cap #problems (0=all)")
    ap.add_argument("--timeout", type=int, default=12)
    ap.add_argument("--flywheel", default="", help="jsonl out: verified winners (SFT) + chosen/rejected (DPO)")
    ap.add_argument("--load_4bit", action="store_true", help="4-bit NF4 base -> fits Qwen3-14B on a 16GB T4 / 24GB L4 (keeps LoRA attached, no merge)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    items = (humaneval_items if a.bench == "humaneval" else mbpp_items)(a.n_probs)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    def _load(mid):                                                   # bf16 (.to dev) or 4-bit NF4 (device_map, no .to)
        if a.load_4bit:
            from transformers import BitsAndBytesConfig
            cdt = torch.bfloat16 if (dev == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=cdt, bnb_4bit_use_double_quant=True)
            return AutoModelForCausalLM.from_pretrained(mid, quantization_config=bnb, device_map={"": 0})
        return AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).to(dev)
    acfg = Path(a.model) / "adapter_config.json"
    if acfg.exists():                                                 # adapter dir -> base + adapter (merge only if not 4-bit)
        from peft import PeftModel
        base_id = json.loads(acfg.read_text())["base_model_name_or_path"]
        m = PeftModel.from_pretrained(_load(base_id), a.model)
        model = (m if a.load_4bit else m.merge_and_unload()).eval()   # can't merge into a 4-bit base -> keep adapter attached
    else:
        model = _load(a.model).eval()
    print(f"{a.bench} best-of-{a.n} + repair{a.repair}: {a.model} on {dev}, {len(items)} problems", flush=True)

    n_greedy = n_bon = n_final = 0
    fly = []
    for i, it in enumerate(items):
        g = extract_code(gen(tok, model, it["ask"], dev, a.max_new, 1, think=a.think)[0])   # greedy = pass@1 baseline
        gp, _ = run_with_feedback(it["check"](g), a.timeout); n_greedy += int(gp)
        winner = near_miss = None                                     # best-of-N
        for c in (extract_code(x) for x in gen(tok, model, it["ask"], dev, a.max_new, a.n, a.temp, a.think)):
            ok, _ = run_with_feedback(it["check"](c), a.timeout)
            if ok: winner = c; break
            elif near_miss is None and c.strip(): near_miss = c
        n_bon += int(winner is not None or gp)
        final_pass = winner is not None or gp; final = winner or (g if gp else None)
        if not final_pass and a.repair:                              # repair: show the failing test output, fix
            base_c = near_miss or g
            for _ in range(a.repair):
                _, fb = run_with_feedback(it["check"](base_c), a.timeout)
                rep_ask = (it["ask"] + f"\n\nYour previous attempt:\n```python\n{base_c}\n```\n"
                           f"It FAILED with:\n{fb}\n\nReturn a corrected COMPLETE solution in one ```python code block.")
                rc = extract_code(gen(tok, model, rep_ask, dev, a.max_new, 1, think=a.think)[0])
                ok, _ = run_with_feedback(it["check"](rc), a.timeout); base_c = rc
                if ok: final = rc; final_pass = True; break
        n_final += int(final_pass)
        if final:                                                    # flywheel: verified winner (+ a near-miss pair)
            fly.append({"type": "sft", "prompt": it["ask"], "response": f"```python\n{final}\n```", "verified": True})
            if near_miss:
                fly.append({"type": "dpo", "prompt": it["ask"], "chosen": f"```python\n{final}\n```", "rejected": f"```python\n{near_miss}\n```"})
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(items)}] greedy {n_greedy} | best-of-{a.n} {n_bon} | +repair {n_final}", flush=True)

    N = len(items)
    print(f"\n=== {a.bench} ({a.model})  N={a.n} repair={a.repair} ===")
    print(f"  pass@1 (greedy):       {n_greedy}/{N} = {n_greedy/N*100:.1f}%")
    print(f"  best-of-{a.n}:            {n_bon}/{N} = {n_bon/N*100:.1f}%   (+{(n_bon-n_greedy)/N*100:+.1f}pp)")
    print(f"  best-of-{a.n} + repair:   {n_final}/{N} = {n_final/N*100:.1f}%   (+{(n_final-n_greedy)/N*100:+.1f}pp over greedy)")
    print(f"  compute ~{a.n + a.repair}x greedy. The lift over pass@1 = the selection+repair headroom the SYSTEM captures for free.")
    if a.flywheel and fly:
        with open(a.flywheel, "w") as f:
            for r in fly: f.write(json.dumps(r) + "\n")
        print(f"  flywheel: {sum(r['type']=='sft' for r in fly)} verified winners (SFT) + "
              f"{sum(r['type']=='dpo' for r in fly)} pairs (DPO) -> {a.flywheel}  (distill these back into greedy next)")


if __name__ == "__main__":
    main()
