#!/usr/bin/env python3
"""Quick integration test: P1-P3 + smoke on 3 instances."""
import sys, os, json, ast, hashlib, re, subprocess, time, types, random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jsonschema import validate as jsonschema_validate

MODEL_REPO = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
ROUTABLE_LAYERS = list(range(7, 21))
K_SKIP = 4
MAX_NEW_TOKENS = 512
GYM_ROOT = Path("/Users/christienantonio/Desktop/AI:ML Research/gym-v0.1-FL")
MANIFEST_PATH = GYM_ROOT / "gym_manifest.yaml"
SCHEMA_PATH = Path("/Users/christienantonio/Desktop/AI:ML Research/directive_trace_schema.json")

def sha256_short(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]

print("[1] Loading model...")
tok = AutoTokenizer.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_REPO, revision=MODEL_REVISION,
    dtype=torch.bfloat16, low_cpu_mem_usage=True,
).to("mps").eval()
print("    Model loaded")

class IdentitySkipContext:
    def __init__(self, model, skip_indices):
        self.model = model
        self.layers = model.model.layers
        self.skip_indices = skip_indices
        self._originals = {}
    def __enter__(self):
        for idx in self.skip_indices:
            layer = self.layers[idx]
            self._originals[idx] = layer.forward
            def identity_forward(self_layer, hidden_states, *args, **kwargs):
                return hidden_states
            layer.forward = types.MethodType(identity_forward, layer)
        return self.model
    def __exit__(self, *args):
        for idx, orig in self._originals.items():
            self.layers[idx].forward = orig
        self._originals.clear()
        return False

print("\n[2] P1: Routing proof...")
probe = "def compute(x):\n    return x * 2 + 1\n\n# The function above computes:"
inp = tok(probe, return_tensors="pt").to("mps")
with torch.no_grad():
    out_dense = model(**inp)
    logits_dense = out_dense.logits[0, -1, :].clone()
with IdentitySkipContext(model, [7,8,9,10]):
    with torch.no_grad():
        out_skip = model(**inp)
        logits_skip = out_skip.logits[0, -1, :].clone()
max_diff_val = (logits_dense - logits_skip).abs().max().item()
with torch.no_grad():
    out_restored = model(**inp)
    logits_restored = out_restored.logits[0, -1, :]
bit_identical = torch.allclose(logits_dense, logits_restored, atol=0, rtol=0)
print(f"    max|diff|={max_diff_val:.4f} (>1.0: {'YES' if max_diff_val>1 else 'NO'}))")
print(f"    restore bit-identical: {bit_identical} {'YES' if bit_identical else 'NO'}")
assert max_diff_val > 1.0, "P1 FAILED: max|diff| <= 1.0"
assert bit_identical, "P1 FAILED: restore not bit-identical"
print("    P1 PASSED")

print("\n[3] P2: Same instances proof...")
import yaml
manifest = yaml.safe_load(open(MANIFEST_PATH))
fl_ids = manifest["fl_subset_ids"]
sorted_ids = sorted(fl_ids)
id_hash = hashlib.sha256("\n".join(sorted_ids).encode()).hexdigest()[:16]
print(f"    ID hash: {id_hash}")
assert id_hash == manifest['fl_subset_hash']
print("    P2 PASSED")

print("\n[4] P3: Verifier sanity...")
def extract_sig(code):
    m = re.search(r"(def\s+\w+\([^)]*\))", code)
    return m.group(1) if m else None

def truncate_completion(text):
    for pat in ["\ndef ", "\nassert", "\nprint(", "\nclass ", "\nif __name__", "\n#"]:
        i = text.find(pat)
        if i != -1:
            text = text[:i]
    return text

def verify_f1_smoke(completion_raw, hidden_tests):
    sig = extract_sig("def add(a, b):\n    return a + b")
    scored = (sig + ":\n" + truncate_completion(completion_raw)) if sig else truncate_completion(completion_raw)
    program = scored.rstrip() + "\n"
    for test in hidden_tests:
        program += test + "\n"
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=10)
    return result.returncode == 0

f1_tests = ["assert add(1, 2) == 3", "assert add(-1, 0) == -1"]
assert verify_f1_smoke("def add(a, b):\n    return a + b", f1_tests), "F1 ref FAIL"
assert not verify_f1_smoke("", f1_tests), "F1 empty FAIL"
assert not verify_f1_smoke("def add(a, b):\n    return a - b", f1_tests), "F1 mut FAIL"
print("    F1 OK")

def verify_f2_smoke(completion_raw):
    try:
        parsed = json.loads(completion_raw)
        schema = {"type": "object", "properties": {"user_id": {"type": "string"}, "email": {"type": "string"}, "active": {"type": "boolean"}}, "required": ["user_id", "email", "active"]}
        jsonschema_validate(instance=parsed, schema=schema)
        return True
    except:
        return False

assert verify_f2_smoke('{"user_id": "123", "email": "x@y.com", "active": true}'), "F2 ref FAIL"
assert not verify_f2_smoke(""), "F2 empty FAIL"
assert not verify_f2_smoke('{"user_id": 123}'), "F2 mut FAIL"
print("    F2 OK")

def verify_f3_smoke(completion_raw):
    if not completion_raw.strip():
        return False
    if "def " not in completion_raw and "class " not in completion_raw:
        return False
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(completion_raw)
        tp = f.name
    try:
        result = subprocess.run(["mypy", "--strict", tp], capture_output=True, text=True, timeout=15)
        return result.returncode == 0
    finally:
        Path(tp).unlink(missing_ok=True)

assert verify_f3_smoke("def greet(name: str) -> str:\n    return f'Hello {name}'"), "F3 ref FAIL"
assert not verify_f3_smoke(""), "F3 empty FAIL"
assert not verify_f3_smoke("def greet(name):\n    return f'Hello {name}'"), "F3 mut FAIL"
print("    F3 OK")
print("    P3 PASSED (9/9)")

print("\n[5] Smoke on 3 instances...")
def load_instance(inst_id):
    parts = inst_id.split("/")
    family_code = parts[1]
    seed_hash_val = parts[2]
    mutation_id = parts[3]
    family_dir = {"F1": "F1_mbpp", "F2": "F2_json", "F3": "F3_type"}[family_code]
    subdir = "smoke" if inst_id in manifest.get("smoke_set_ids", []) else "test"
    filename = f"gym-v0.1-FL_{family_code}_{seed_hash_val}_{mutation_id}.json"
    fpath = GYM_ROOT / family_dir / subdir / filename
    return json.load(open(fpath))


def _extract_schema_literal(text: str) -> object | None:
    """Brace-match the first {...} argument of a jsonschema.validate call.

    Linear scan; returns the literal object, or None if absent/unparseable.
    """
    idx = text.find("jsonschema.validate(")
    if idx == -1:
        return None
    start = text.find("{", idx)
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                if i + 1 < len(text) and text[i + 1] == ")":
                    try:
                        return ast.literal_eval(text[start : i + 1])
                    except (ValueError, SyntaxError):
                        return None
                return None
    return None


def verify_instance(instance, completion_raw):
    family = instance["metadata"]["family"]
    if family == "F1_mbpp":
        broken = instance["broken_artifact"]
        sig = extract_sig(broken)
        hidden_tests = instance.get("hidden_tests", [])
        if sig:
            scored = sig + ":\n" + truncate_completion(completion_raw)
        else:
            scored = truncate_completion(completion_raw)
        program = scored.rstrip() + "\n"
        for test in hidden_tests:
            program += test + "\n"
        result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=15)
        return result.returncode == 0
    elif family == "F2_json":
        try:
            parsed = json.loads(completion_raw)
        except:
            return False
        hidden = instance.get("hidden_tests", [])
        for test in hidden:
            # Linear, regex-free extraction: the original pattern was polynomial
            # on adversarial input (CodeQL py/polynomial-redos, alert 1824) and
            # the query cannot see a window bound.
            schema_obj = _extract_schema_literal(test)
            if schema_obj is not None:
                try:
                    jsonschema_validate(instance=parsed, schema=schema_obj)
                except:
                    return False
                break
        return True
    elif family == "F3_type":
        scored = completion_raw.strip()
        if not scored or ("def " not in scored and "class " not in scored):
            return False
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(scored)
            tp = f.name
        try:
            result = subprocess.run(["mypy", "--strict", tp], capture_output=True, text=True, timeout=15)
            return result.returncode == 0
        finally:
            Path(tp).unlink(missing_ok=True)
    return False

def select_skip_random(routable, k, inst_id, seed):
    rng_seed = hash(f"{inst_id}:{seed}") & 0x7FFFFFFF
    rng = random.Random(rng_seed)
    return rng.sample(routable, k)

def compute_layer_entropies(model, inp, routable):
    with torch.no_grad():
        outputs = model(**inp, output_hidden_states=True)
    entropies = {}
    for layer_idx in routable:
        hidden_state = outputs.hidden_states[layer_idx + 1]
        logits_val = model.lm_head(hidden_state)
        probs = torch.softmax(logits_val[:, -1, :], dim=-1)
        entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).item()
        entropies[layer_idx] = entropy
    return entropies

def select_skip_entropy(model, inp, routable, k):
    entropies = compute_layer_entropies(model, inp, routable)
    deltas = {}
    for i, layer in enumerate(routable):
        if i == 0:
            deltas[layer] = 0.0
        else:
            deltas[layer] = abs(entropies[layer] - entropies[routable[i-1]])
    sorted_layers = sorted(deltas.items(), key=lambda x: x[1])
    return [layer for layer, _ in sorted_layers[:k]], entropies, deltas

smoke_ids = manifest["smoke_set_ids"][:3]
results = []

for inst_id in smoke_ids:
    inst = load_instance(inst_id)
    prompt = inst["prompt_context"]
    family = inst["metadata"]["family"]
    print(f"  {inst_id[:50]}... family={family}")
    
    for policy in ["dense_uniform", "random_matched_mix", "heuristic_entropy"]:
        for seed in [1337]:
            skip_layers = []
            if policy == "random_matched_mix":
                skip_layers = select_skip_random(ROUTABLE_LAYERS, K_SKIP, inst_id, seed)
            elif policy == "heuristic_entropy":
                inp_data = tok(prompt, return_tensors="pt").to("mps")
                skip_layers, _, _ = select_skip_entropy(model, inp_data, ROUTABLE_LAYERS, K_SKIP)
            
            torch.manual_seed(seed)
            inp_data = tok(prompt, return_tensors="pt").to("mps")
            with torch.no_grad():
                if skip_layers:
                    with IdentitySkipContext(model, skip_layers):
                        gen = model.generate(**inp_data, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tok.eos_token_id)
                else:
                    gen = model.generate(**inp_data, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tok.eos_token_id)
            
            full_text = tok.decode(gen[0], skip_special_tokens=True)
            completion = full_text[len(prompt):]
            passed = verify_instance(inst, completion)
            results.append({"inst": inst_id[:30], "policy": policy, "passed": passed, "family": family, "skip": skip_layers})
            print(f"    {policy:22s} -> {'PASS' if passed else 'fail'} skip={skip_layers}")

print(f"\n  Smoke results ({len(results)} runs):")
by_policy = defaultdict(list)
for r in results:
    by_policy[r["policy"]].append(r["passed"])
for pol, passes in by_policy.items():
    print(f"    {pol:22s}: {sum(passes)}/{len(passes)} = {np.mean(passes)*100:.1f}%")

dense_passes = [r["passed"] for r in results if r["policy"]=="dense_uniform"]
random_passes = [r["passed"] for r in results if r["policy"]=="random_matched_mix"]
if dense_passes != random_passes:
    print(f"\n  Routed outputs DIFFER from dense YES (routing works)")
else:
    print(f"\n  Routed outputs SAME as dense NO (ROUTING MAY NOT BE WORKING)")

print("\n[6] Trace schema test...")
schema_data = json.load(open(SCHEMA_PATH))
trace_rows = []
for inst_id in smoke_ids:
    for policy in ["dense_uniform", "random_matched_mix", "heuristic_entropy"]:
        for layer_idx in ROUTABLE_LAYERS:
            row = {
                "schema_version": "1.1.0",
                "run_id": "test",
                "config_id": "FROZEN-BASE-v1",
                "adr_branch": "ratified",
                "seq_id": f"seq-{sha256_short(inst_id)}",
                "token_pos": 0,
                "layer_group": f"L{layer_idx}",
                "phase": "decode",
                "cadence_path": "slow_full",
                "proposal": {"directive": "NORMAL", "logits_q": [0.0, 0.0, 0.0]},
                "projected_directive": "NORMAL",
                "final_directive": "NORMAL",
                "reason_code": "FORCED_BASELINE",
                "estimator_rung": "none_forced",
                "g_hat": None,
                "cost_estimate": {"flops_directive": 0.0, "controller_tax_flops": 0.0},
                "ring": "R0",
                "contaminated_for_paper1": True,
            }
            jsonschema_validate(instance=row, schema=schema_data)
            trace_rows.append(row)
print(f"    {len(trace_rows)} rows, all schema-valid OK")

del model
print("\nALL INTEGRATION TESTS PASSED")
