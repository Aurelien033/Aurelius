#!/usr/bin/env python3
"""grpo_train.py — RLVR via GRPO on verifiable code tasks (gym + MBPP), reusing the TESTED grpo_v3 loss.

The lever the arc points at: SFT is capped by the teacher (and v2 showed a weaker teacher DRAGS the base down).
RLVR optimizes against ground-truth correctness — no teacher ceiling — so it can exceed the base. This wires the
already-tested `src/alignment/grpo_v3.py` (GRPOLoss + GroupRewardNormalizer) to a HF model + the real execution
verifiers, fixing the 8 integration gaps in the scope (HF interface, fast generate, code reward, 8B-scale
log-probs via LoRA+grad-ckpt, disable_adapter ref, old=detached, batching, reward-hacking guard).

Per prompt: sample G completions -> verifier reward {1,0} -> group-relative advantage -> GRPO PPO-clip+KL step.
The number that must RISE is mean_reward.

Smoke (decisive plumbing check; mean_reward won't move much on a 1.5B):
  python docs/training/grpo_train.py --base Qwen/Qwen2.5-1.5B --data gym --n_tasks 16 --group_size 4 \
     --steps 50 --max_new 256 --smoke
Real (Colab A100, from the SFT'd policy):
  python docs/training/grpo_train.py --base Qwen/Qwen3-8B --adapter /content/aurelius-v1-8b \
     --data both --group_size 8 --steps 400 --max_new 512 --out /content/aurelius-rlvr
"""
import argparse, json, sys, importlib.util
from pathlib import Path
import torch, torch.nn.functional as F
REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "first_light"))
from eval_code_bench import extract_code, run_program, run_io_tests, build_msg
import first_light_runner_v4 as R


def _load_module(path, name):
    """Import grpo_v3.py directly by file (it only needs torch) — avoids the repo's heavy package __init__ chain."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m   # register so @dataclass can resolve the module
    spec.loader.exec_module(m); return m


GRPO = _load_module(REPO / "src/alignment/grpo_v3.py", "grpo_v3")


def load_gym_tasks(split_path, n):
    ids = sorted(set(json.loads(Path(split_path).read_text())["all"]))
    idx = {}
    for root, fam in [(R.RESEARCH / "gym-v0.1-FL", "F2_json"), (R.RESEARCH / "gym-v0.3", "F3_type")]:
        for sp in ["dev", "test", "smoke", "train"]:
            for f in (root / fam / sp).glob("*.json"):
                d = json.loads(f.read_text()); idx[d["instance_id"]] = d
    ids = [i for i in ids if i in idx]
    if n: ids = ids[:n]
    tasks = []
    for i in ids:
        inst = idx[i]; vf = R.VERIFIERS[inst["metadata"]["family"]]
        tasks.append((inst["prompt_context"], (lambda comp, inst=inst, vf=vf: float(bool(vf(inst, comp))))))
    return tasks


def load_mbpp_tasks(n):
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", split="train+validation")
    tasks = []
    for d in ds:
        ask = (f"Write a Python function for this task. Return ONLY the function in one ```python code block.\n\n"
               f"Task: {d['text']}\nIt must satisfy:\n{d['test_list'][0]}")
        setup, tests = (d.get("test_setup_code") or ""), "\n".join(d["test_list"])
        def rf(comp, setup=setup, tests=tests):
            code = extract_code(comp)
            return float(bool(code.strip()) and run_program(setup + "\n" + code + "\n" + tests + "\n", 12))
        tasks.append((ask, rf))
    if n: tasks = tasks[:n]
    return tasks


def load_apps_tasks(n, difficulties="interview,competition"):
    """Harder verifiable tasks: APPS competitive-programming (stdin/stdout), graded difficulty. Disjoint from HumanEval."""
    from datasets import load_dataset
    import json
    keep = set(difficulties.split(","))
    ds = load_dataset("codeparrot/apps", split="test", trust_remote_code=True)
    tasks = []
    for d in ds:
        if d.get("difficulty") not in keep: continue
        io = d.get("input_output") or ""
        if not io.strip(): continue
        try: io = json.loads(io)
        except Exception: continue
        if "fn_name" in io: continue                                  # v1: stdin/stdout problems only
        inputs, outputs = io.get("inputs", []), io.get("outputs", [])
        if not inputs or not outputs: continue
        ask = ("Solve this competitive-programming problem. Read input from stdin, write the answer to stdout. "
               "Return ONLY a complete runnable Python program in one ```python code block.\n\n" + str(d["question"])[:3500])
        def rf(comp, inputs=inputs, outputs=outputs):
            code = extract_code(comp)
            return float(bool(code.strip()) and run_io_tests(code, inputs, outputs, 8))
        tasks.append((ask, rf))
        if n and len(tasks) >= n: break
    return tasks


def seq_logprob(model, seqs, prompt_len, pad_id):
    """Sum log-prob of the COMPLETION tokens (positions >= prompt_len, non-pad) for each of the G sequences."""
    logits = model(input_ids=seqs).logits[:, :-1, :].float()       # predict tokens 1..T-1
    tgt = seqs[:, 1:]
    lp = F.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)   # (G, T-1)
    mask = torch.zeros_like(lp); mask[:, prompt_len - 1:] = 1.0    # completion region
    mask = mask * (tgt != pad_id).float()                         # drop right-pad
    return (lp * mask).sum(-1)                                    # (G,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", default=None, help="SFT LoRA to start RL from (merged in; ref = the SFT model)")
    ap.add_argument("--data", choices=["gym", "mbpp", "apps", "both", "all"], default="gym",
                    help="apps = harder competitive-programming (stdin/stdout); all = gym+mbpp+apps")
    ap.add_argument("--apps_difficulty", default="interview,competition", help="APPS difficulties (skip 'introductory' = easy)")
    ap.add_argument("--split", default="eval_data/selector_split_v0.1.json")
    ap.add_argument("--n_tasks", type=int, default=0)
    ap.add_argument("--group_size", type=int, default=8, help="G completions/prompt")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--accum", type=int, default=4, help="prompts per optimizer step")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.04, help="KL coeff")
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--think", type=int, default=0, help="1=let a reasoning base (VibeThinker/R1/QwQ) think; needs bigger --max_new")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--rank", type=int, default=16); ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--min_chars", type=int, default=1, help="reward-hacking guard: 0 reward if completion shorter")
    ap.add_argument("--save_every", type=int, default=50, help="checkpoint every N steps (survives Colab disconnects)")
    ap.add_argument("--out", default="checkpoints/rlvr")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if a.smoke: a.steps, a.group_size, a.max_new, a.accum, a.n_tasks = 4, 2, 32, 1, 6
    import random; rng = random.Random(0); torch.manual_seed(0)

    tasks = []
    if a.data in ("gym", "both", "all"): tasks += load_gym_tasks(a.split, a.n_tasks)
    if a.data in ("mbpp", "both", "all"): tasks += load_mbpp_tasks(a.n_tasks)
    if a.data in ("apps", "all"): tasks += load_apps_tasks(a.n_tasks, a.apps_difficulty)
    if not tasks: print("  no tasks"); return

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, PeftModel
    tok = AutoTokenizer.from_pretrained(a.adapter or a.base)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16).to(dev)
    if a.adapter:                                                  # start RL from the SFT model: merge it in
        model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    policy = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
    if dev == "cuda":
        policy.gradient_checkpointing_enable(); policy.enable_input_require_grads(); policy.config.use_cache = False
    policy.print_trainable_parameters()

    loss_fn = GRPO.GRPOLoss(GRPO.GRPOConfig(beta=a.beta, group_size=a.group_size, epsilon_low=0.2, epsilon_high=0.2))
    normalizer = GRPO.GroupRewardNormalizer(a.group_size)
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=a.lr)
    print(f"GRPO/RLVR {a.base}{' +'+a.adapter if a.adapter else ''} on {dev}: {len(tasks)} tasks, G={a.group_size}, "
          f"{a.steps} steps, lr {a.lr}, beta {a.beta} | reward=verifier pass-rate. WATCH mean_reward RISE.", flush=True)

    order = list(range(len(tasks))); reward_hist = []
    Path(a.out).mkdir(parents=True, exist_ok=True)
    opt.zero_grad()
    for step in range(a.steps):
        prompt, reward_fn = tasks[order[step % len(tasks)]]
        if step % len(tasks) == 0: rng.shuffle(order)
        enc = tok(build_msg(tok, prompt, think=a.think), return_tensors="pt", add_special_tokens=False).to(dev)
        plen = enc["input_ids"].shape[1]
        policy.eval()
        with torch.no_grad():
            gen = policy.generate(**enc, num_return_sequences=a.group_size, do_sample=True,
                                  temperature=a.temperature, top_p=0.95, max_new_tokens=a.max_new,
                                  pad_token_id=pad_id, use_cache=True)                       # (G, plen+comp)
        comps = [tok.decode(gen[i][plen:], skip_special_tokens=True) for i in range(a.group_size)]
        rewards = torch.tensor([(reward_fn(c) if len(c) >= a.min_chars else 0.0) for c in comps],
                               dtype=torch.float32, device=dev)
        policy.train()
        logp = seq_logprob(policy, gen, plen, pad_id)                                        # grad
        with torch.no_grad():
            with policy.disable_adapter():                                                   # ref = base/SFT (no RL-LoRA)
                ref_logp = seq_logprob(policy, gen, plen, pad_id)
        old_logp = logp.detach()
        adv = normalizer.normalize(rewards)
        loss, metrics = loss_fn(logp, old_logp, ref_logp, adv)
        (loss / a.accum).backward()
        if (step + 1) % a.accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
        reward_hist.append(rewards.mean().item())
        if step % max(1, a.steps // 20) == 0 or step == a.steps - 1:
            recent = sum(reward_hist[-20:]) / len(reward_hist[-20:])
            print(f"  step {step:4d}/{a.steps}  mean_reward {rewards.mean().item():.3f}  (run-avg {recent:.3f})  "
                  f"kl {metrics['kl_loss']:.4f}  policy_loss {metrics['policy_loss']:+.4f}", flush=True)
        if a.save_every and (step + 1) % a.save_every == 0:                  # survive Colab disconnects
            policy.save_pretrained(a.out); tok.save_pretrained(a.out)
            print(f"    [ckpt @ step {step+1} -> {a.out}]", flush=True)

    policy.save_pretrained(a.out); tok.save_pretrained(a.out)
    first = sum(reward_hist[:max(1, len(reward_hist)//5)]) / max(1, len(reward_hist)//5)
    last = sum(reward_hist[-max(1, len(reward_hist)//5):]) / max(1, len(reward_hist)//5)
    print(f"\n  mean_reward: first-fifth {first:.3f} -> last-fifth {last:.3f}  (Δ {last-first:+.3f})", flush=True)
    print(f"  saved -> {a.out}/  | RISE = RLVR works on the verifier; then eval pass@1 w/ eval_code_bench.", flush=True)


if __name__ == "__main__":
    main()
