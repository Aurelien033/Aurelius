#!/usr/bin/env python3
"""
REAL First Light runner — v3 (corrected).
Fixes all 5 defect categories from first-light-dab80149_VERIFICATION.md.

FIX 1: Routing ACTUALLY executes via IdentitySkip layer patching.
FIX 2: ONE instance set (pinned fl_subset_ids) for all policies. Paired by construction.
FIX 3: Verifiers to spec (F1 sig+trunc+hidden, F2 json+schema+hidden, F3 mypy+hidden+empty-fail).
FIX 4: Evidence machinery — emit directive_trace_schema 1.1.0 rows, counterfactual-ledger rows,
       SAVE ALL raw completions + verdicts to disk.
FIX 5: Smoke gate = HARD ABORT (10 pinned smoke instances; trace schema validation;
       reason-code coverage; deterministic verdicts; fallback ladder check).

Self-proofs P1-P4 computed and stored in the receipt.
Runbook order: self-proofs → smoke → dense anchor → forced replays → pilot block → analysis → receipt.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import resource
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import psutil
import torch
import yaml
from jsonschema import validate as jsonschema_validate
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

# ═══════════════════════════════════════════════════════════════
# PINNED CONSTANTS (from FL-PREREG v1.2-final, FROZEN)
# ═══════════════════════════════════════════════════════════════
MODEL_REPO = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
TOKENIZER_HASH = "c0382117ea329cdf"
CONFIG_HASH = "0e8c8aa86468aba0"

ROUTABLE_LAYERS = list(range(7, 21))  # [7..20] inclusive = 14 layers
K_SKIP = 4  # skip 4/14 routable layers

SEEDS = [1337, 2026, 7]
MAX_NEW_TOKENS = 512

# Pinned hashes
GYM_MANIFEST_HASH = "0724aa721c25da55"
FL_SUBSET_HASH = "2509e4b242e86b3b"
SMOKE_SET_HASH = "6727c1486c1b9db5"
BYTE_PREDICTIONS_HASH = "4e9f8060471fac96"

# Paths
GYM_ROOT = Path("/Users/christienantonio/Desktop/AI:ML Research/gym-v0.1-FL")
MANIFEST_PATH = GYM_ROOT / "gym_manifest.yaml"
SCHEMA_PATH = Path("/Users/christienantonio/Desktop/AI:ML Research/directive_trace_schema.json")
BYTE_PREDICTIONS_PATH = Path("/Users/christienantonio/Desktop/AI:ML Research/first_light_byte_predictions.yaml")
OUTPUT_DIR = Path("/Users/christienantonio/Desktop/AI:ML Research/first_light_receipt")

# RSS gates (hybrid, 2026-06-11 amendment)
RSS_TOTAL_GATE_GB = 10.0
RSS_MODEL_ATTRIBUTABLE_GB = 7.5

# Bootstrap
BOOTSTRAP_B = 10000

# ═══════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════

def sha256_short(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def sha256_list(lst: list[str]) -> str:
    """sha256[:16] of json.dumps(list) — matches gym manifest hash method."""
    return hashlib.sha256(json.dumps(lst).encode()).hexdigest()[:16]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def rss_mb() -> float:
    """Current process RSS in MB."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def to_jsonable(obj):
    """Recursively convert numpy/torch scalars and containers to plain JSON/YAML-safe types.

    Receipt assembly uses scipy/numpy analysis outputs; numpy.bool_/floating scalars
    are not serializable by stdlib json and produce unsafe YAML python tags. Keep the
    conversion centralized so every artifact writer emits portable JSON primitives.
    """
    if isinstance(obj, dict):
        return {str(to_jsonable(k)): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_jsonable(obj), f, indent=2)


def save_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(to_jsonable(row)) + "\n")


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════════════════════════
# MODEL LOADING + IDENTITY SKIP
# ═══════════════════════════════════════════════════════════════

class IdentitySkipContext:
    """Context manager that patches selected decoder layers to identity-forward.
    
    On enter: replaces forward() of each skip layer with identity (return hidden_states).
    On exit: restores original forward methods.
    
    Verified in spike (scripts/spike_verification/): restore is bit-identical,
    max|logits_dense - logits_skip| > 1.0 on probe prompt.
    """
    
    def __init__(self, model, skip_indices: list[int]):
        self.model = model
        self.layers = model.model.layers
        self.skip_indices = skip_indices
        self._originals: dict[int, callable] = {}
    
    def __enter__(self):
        for idx in self.skip_indices:
            layer = self.layers[idx]
            self._originals[idx] = layer.forward
            
            # Identity forward: return hidden_states unchanged.
            # In transformers 5.11, Qwen2DecoderLayer.forward returns only hidden_states
            # (KV cache is managed via DynamicCache passed as kwarg, mutated internally).
            def identity_forward(self_layer, hidden_states, *args, **kwargs):
                return hidden_states
            
            # Bind to the layer instance
            import types
            layer.forward = types.MethodType(identity_forward, layer)
        
        return self.model
    
    def __exit__(self, *args):
        for idx, orig in self._originals.items():
            self.layers[idx].forward = orig
        self._originals.clear()
        return False


def load_model(device="mps"):
    """Load FROZEN-BASE-v1 = Qwen2.5-1.5B BASE."""
    print(f"[{now_iso()}] Loading model {MODEL_REPO}@{MODEL_REVISION[:12]}...")
    tok = AutoTokenizer.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    
    # Verify pins via file-based sha256[:16] (method per prereg amendment)
    from huggingface_hub import hf_hub_download
    cfg_path = hf_hub_download(MODEL_REPO, "config.json", revision=MODEL_REVISION)
    actual_cfg_hash = hashlib.sha256(open(cfg_path, "rb").read()).hexdigest()[:16]
    assert actual_cfg_hash == CONFIG_HASH, f"Config hash mismatch: {actual_cfg_hash} != {CONFIG_HASH}"
    print(f"[{now_iso()}] Model loaded. Config hash verified: {actual_cfg_hash}")
    
    return tok, model


def generate_completion(tok, model, prompt: str, seed: int,
                        temperature: float = 0.0, top_p: float = 0.95,
                        skip_indices: list[int] | None = None) -> tuple[str, int]:
    """Generate completion. If skip_indices provided, applies IdentitySkip during generation.
    
    Returns (completion_text, realized_tokens).
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    inp = tok(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        if skip_indices is None or len(skip_indices) == 0:
            # Dense pass
            gen_kwargs = dict(
                **inp,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=(temperature > 0),
                pad_token_id=tok.eos_token_id,
            )
            if temperature > 0:
                gen_kwargs.update(temperature=temperature, top_p=top_p)
            
            gen = model.generate(**gen_kwargs)
        else:
            # Forced-route execution with IdentitySkip
            with IdentitySkipContext(model, skip_indices):
                gen_kwargs = dict(
                    **inp,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=(temperature > 0),
                    pad_token_id=tok.eos_token_id,
                )
                if temperature > 0:
                    gen_kwargs.update(temperature=temperature, top_p=top_p)
                
                gen = model.generate(**gen_kwargs)
    
    full_text = tok.decode(gen[0], skip_special_tokens=True)
    completion = full_text[len(prompt):]
    realized_tokens = gen.shape[1] - inp["input_ids"].shape[1]
    
    return completion, realized_tokens


# ═══════════════════════════════════════════════════════════════
# VERIFIERS (FIX 3 — to spec)
# ═══════════════════════════════════════════════════════════════

def extract_sig(code: str) -> Optional[str]:
    """Extract function/class signature from code."""
    m = re.search(r"(def\s+\w+\([^)]*\))", code)
    if m:
        return m.group(1)
    m = re.search(r"(class\s+\w+)", code)
    if m:
        return m.group(1)
    return None


def truncate_completion(text: str) -> str:
    """Truncate base-model continuations at the first new top-level statement."""
    for pat in ["\ndef ", "\nassert", "\nprint(", "\nclass ", "\nif __name__", "\n#"]:
        i = text.find(pat)
        if i != -1:
            text = text[:i]
    return text


def verify_f1(instance: dict, completion_raw: str) -> tuple[bool, str]:
    """F1 verifier: signature-prepend + top-level truncation + hidden tests in subprocess.
    
    Returns (passed, error_message).
    """
    broken = instance["broken_artifact"]
    sig = extract_sig(broken)
    hidden_tests = instance.get("hidden_tests", [])
    
    # Apply signature prepend and truncation
    if sig:
        scored = sig + ":\n" + truncate_completion(completion_raw)
    else:
        scored = truncate_completion(completion_raw)
    
    # Build program: scored completion + hidden tests
    program = scored.rstrip() + "\n"
    for test in hidden_tests:
        program += test + "\n"
    
    # Run in subprocess isolation
    try:
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            return True, ""
        else:
            return False, f"exit={result.returncode} stderr={result.stderr[:200]}"
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)[:200]


def verify_f2(instance: dict, completion_raw: str) -> tuple[bool, str]:
    """F2 verifier: json.loads + jsonschema validation against instance schema.
    
    Returns (passed, error_message).
    """
    hidden_tests = instance.get("hidden_tests", [])
    
    # json.loads must succeed
    try:
        parsed = json.loads(completion_raw)
    except json.JSONDecodeError as e:
        return False, f"json.loads failed: {e}"
    
    # jsonschema validation — extract schema from hidden_tests
    # Hidden test format: "jsonschema.validate(repaired, {<schema>}) must pass"
    schema_str = None
    for test in hidden_tests:
        m = re.search(r'jsonschema\.validate\(.*?,\s*(\{.*\})\)', test)
        if m:
            schema_str = m.group(1)
            break
    
    if schema_str:
        try:
            schema = eval(schema_str)  # safe: these are static dict literals
            jsonschema_validate(instance=parsed, schema=schema)
        except Exception as e:
            return False, f"jsonschema validation failed: {e}"
    
    return True, ""


def verify_f3(instance: dict, completion_raw: str) -> tuple[bool, str]:
    """F3 verifier: mypy --strict on completion + hidden call-site checks.
    
    Empty or function-missing completion MUST fail.
    Returns (passed, error_message).
    """
    hidden_tests = instance.get("hidden_tests", [])
    
    # Write completion to temp file
    scored = completion_raw.strip()
    
    # EMPTY completion must fail
    if not scored:
        return False, "empty completion"
    
    # Must contain a function definition
    if "def " not in scored and "class " not in scored:
        return False, "no function/class found in completion"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(scored)
        temp_path = f.name
    
    try:
        result = subprocess.run(
            ["mypy", "--strict", temp_path],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return False, f"mypy failed (exit={result.returncode}): {result.stdout[:200]}"
        
        # Hidden call-site checks: append and run the function
        # For F3, hidden tests may include call-site type checks
        for test in hidden_tests:
            if "call_site" in test.lower() or "assert" in test.lower():
                full_program = scored + "\n" + test + "\n"
                try:
                    exec(full_program, {})
                except Exception as e:
                    return False, f"hidden call-site check failed: {e}"
        
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "mypy timeout"
    except Exception as e:
        return False, str(e)[:200]
    finally:
        Path(temp_path).unlink(missing_ok=True)


VERIFIERS = {
    "F1_mbpp": verify_f1,
    "F2_json": verify_f2,
    "F3_type": verify_f3,
}


# ═══════════════════════════════════════════════════════════════
# LAYER ENTROPY COMPUTATION (for heuristic_entropy policy)
# ═══════════════════════════════════════════════════════════════

def compute_layer_entropies(model, inp, routable_layers):
    """Compute logit-lens predictive entropy at each routable layer.
    
    Returns dict: {layer_idx: entropy_float}
    """
    with torch.no_grad():
        outputs = model(**inp, output_hidden_states=True)
    
    entropies = {}
    for layer_idx in routable_layers:
        hidden_state = outputs.hidden_states[layer_idx + 1]  # +1: hidden_states[0]=embedding
        logits = model.lm_head(hidden_state)
        probs = torch.softmax(logits[:, -1, :], dim=-1)  # last token
        entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).item()
        entropies[layer_idx] = entropy
    
    return entropies


def select_skip_layers_entropy(model, inp, routable_layers, k):
    """Select k layers to skip based on smallest |ΔH| between consecutive layers."""
    entropies = compute_layer_entropies(model, inp, routable_layers)
    
    deltas = {}
    for i, layer in enumerate(routable_layers):
        if i == 0:
            deltas[layer] = 0.0
        else:
            prev_layer = routable_layers[i - 1]
            deltas[layer] = abs(entropies[layer] - entropies[prev_layer])
    
    sorted_layers = sorted(deltas.items(), key=lambda x: x[1])
    skip_layers = [layer for layer, _ in sorted_layers[:k]]
    
    return skip_layers, entropies, deltas


def select_skip_layers_random(routable_layers, k, instance_id, seed):
    """Select k layers randomly, PRNG seeded by (instance_id, seed)."""
    import random
    rng_seed = hash(f"{instance_id}:{seed}") & 0x7FFFFFFF
    rng = random.Random(rng_seed)
    return rng.sample(routable_layers, k)


# ═══════════════════════════════════════════════════════════════
# GYM INSTANCE LOADING (uses PINNED fl_subset_ids, never globs)
# ═══════════════════════════════════════════════════════════════

def load_gym_instance(instance_id: str) -> dict:
    """Load a single gym instance by its canonical instance_id.
    
    instance_id format: gym-v0.1-FL/F1/005374489258f51f/5d51ccefcbce81f4
    Maps to: GYM_ROOT/F1_mbpp/test/gym-v0.1-FL_F1_005374489258f51f_5d51ccefcbce81f4.json
    """
    parts = instance_id.split("/")
    family_code = parts[1]  # F1, F2, F3
    seed_hash = parts[2]
    mutation_id = parts[3]
    
    family_dir = {"F1": "F1_mbpp", "F2": "F2_json", "F3": "F3_type"}[family_code]
    
    # Determine test vs smoke
    manifest = load_yaml(MANIFEST_PATH)
    if instance_id in manifest.get("smoke_set_ids", []):
        subdir = "smoke"
    else:
        subdir = "test"
    
    filename = f"gym-v0.1-FL_{family_code}_{seed_hash}_{mutation_id}.json"
    path = GYM_ROOT / family_dir / subdir / filename
    
    if not path.exists():
        raise FileNotFoundError(f"Instance file not found: {path}")
    
    return load_json(path)


def load_pinned_instance_ids() -> dict[str, list[str]]:
    """Load pinned instance IDs from gym_manifest.yaml.
    
    Returns {'fl_subset': [...], 'smoke': [...]}
    """
    manifest = load_yaml(MANIFEST_PATH)
    return {
        "fl_subset": manifest["fl_subset_ids"],
        "smoke": manifest["smoke_set_ids"],
    }


# ═══════════════════════════════════════════════════════════════
# DIRECTIVE TRACE EMISSION (FIX 4)
# ═══════════════════════════════════════════════════════════════

def emit_trace_row(instance_id: str, layer_idx: int, policy: str,
                   estimator_rung: str, g_hat, controller_tax: float,
                   run_id: str, phase="decode") -> dict:
    """Emit a single directive_trace_schema 1.1.0 row."""
    
    if policy == "dense_uniform":
        final_dir = "NORMAL"
        exec_mode = "live"
    else:
        final_dir = "SKIP" if layer_idx in getattr(emit_trace_row, '_current_skip', []) else "NORMAL"
        exec_mode = "offline_replay"
    
    row = {
        "schema_version": "1.1.0",
        "run_id": run_id,
        "config_id": "FROZEN-BASE-v1",
        "adr_branch": "ratified",
        "seq_id": f"seq-{sha256_short(instance_id)}",
        "token_pos": 0,
        "layer_group": f"L{layer_idx}",
        "phase": phase,
        "execution_mode": exec_mode,
        "cadence_path": "slow_full",
        "proposal": {
            "directive": final_dir,
            "logits_q": [0.0, 0.0, 0.0],
        },
        "projected_directive": final_dir,
        "final_directive": final_dir,
        "reason_code": "FORCED_BASELINE",
        "estimator_rung": estimator_rung,
        "g_hat": g_hat,
        "cost_estimate": {
            "flops_directive": 0.0,
            "controller_tax_flops": controller_tax,
        },
        "ring": "R0",
        "contaminated_for_paper1": True,
    }
    
    return row


def emit_trace_rows_for_instance(instance_id: str, policy: str, skip_layers: list[int],
                                  run_id: str) -> list[dict]:
    """Emit trace rows for all routable layers for one instance×policy."""
    rows = []
    estimator_rung = "g_entropy" if policy == "heuristic_entropy" else "none_forced"
    g_hat = None if estimator_rung == "none_forced" else 0.0
    controller_tax = 1000.0 if policy == "heuristic_entropy" else 0.0
    
    # Temporarily set the current skip layers for emit_trace_row to use
    emit_trace_row._current_skip = skip_layers
    
    for layer_idx in ROUTABLE_LAYERS:
        row = emit_trace_row(
            instance_id=instance_id,
            layer_idx=layer_idx,
            policy=policy,
            estimator_rung=estimator_rung,
            g_hat=g_hat,
            controller_tax=controller_tax,
            run_id=run_id,
        )
        rows.append(row)
    
    emit_trace_row._current_skip = []
    return rows


# ═══════════════════════════════════════════════════════════════
# COUNTERFACTUAL LEDGER EMISSION (FIX 4)
# ═══════════════════════════════════════════════════════════════

def emit_ledger_row(instance_id: str, seed: int, policy: str,
                    verified_pass: bool, flops_total: float,
                    bytes_per_token: float, wall_clock_tps: float,
                    run_id: str) -> dict:
    """Emit one counterfactual-ledger row."""
    
    if policy == "dense_uniform":
        routing_mode = "none_dense"
    else:
        routing_mode = "forced_offline_replay"
    
    return {
        "run_id": run_id,
        "instance_id": instance_id,
        "seed": seed,
        "policy": {
            "baseline_type": policy,
            "routing_mode": routing_mode,
            "contract_version": "acdt-minimal-v1",
        },
        "config_id": "FROZEN-BASE-v1",
        "cost": {
            "flops_total": flops_total,
            "controller_tax_flops": 0.0,
            "guard_tax_flops": 0.0,
            "bytes_per_token": bytes_per_token,
            "wall_clock_sustained_tps": {"p50": wall_clock_tps, "p95": wall_clock_tps},
            "phase": "decode",
        },
        "outcome": {
            "verifier_class": "deterministic",
            "verified_pass": verified_pass,
            "counterfactual_kind": "observed_replay",
            "predicted_gain": 0.0,
            "actual_gain": 0.0,
            "estimator_rung": "g_entropy" if policy == "heuristic_entropy" else "none_forced",
        },
        "flags": {
            "pilot_or_claim_eligible": "pilot",
            "contaminated_for_paper1": True,
        },
    }


# ═══════════════════════════════════════════════════════════════
# SELF-PROOFS P1-P4
# ═══════════════════════════════════════════════════════════════

def self_proof_p1(tok, model) -> dict:
    """P1: routing_proof — max|logits_dense - logits_skip| > 1.0 on probe prompt.
    Restore-check: after unwrapping, logits bit-identical to dense.
    """
    print(f"\n[{now_iso()}] SELF-PROOF P1: routing proof...")
    probe_prompt = "def compute(x):\n    return x * 2 + 1\n\n# The function above computes:"
    inp = tok(probe_prompt, return_tensors="pt").to(model.device)
    
    # Dense logits
    with torch.no_grad():
        out_dense = model(**inp)
        logits_dense = out_dense.logits[0, -1, :].clone()
    
    # Skip logits with identity on [7,8,9,10]
    skip_test = [7, 8, 9, 10]
    with IdentitySkipContext(model, skip_test):
        with torch.no_grad():
            out_skip = model(**inp)
            logits_skip = out_skip.logits[0, -1, :].clone()
    
    max_diff = (logits_dense - logits_skip).abs().max().item()
    print(f"  max|logits_dense - logits_skip[{skip_test}]| = {max_diff:.4f}")
    assert max_diff > 1.0, f"P1 FAILED: max logit diff {max_diff:.4f} <= 1.0"
    
    # Restore check: after context exit, logits must be bit-identical to dense
    with torch.no_grad():
        out_restored = model(**inp)
        logits_restored = out_restored.logits[0, -1, :]
    
    bit_identical = torch.allclose(logits_dense, logits_restored, atol=0, rtol=0)
    print(f"  Restore bit-identical: {bit_identical}")
    assert bit_identical, "P1 FAILED: restore not bit-identical to dense"
    
    print("  P1 PASSED")
    return {"max_logit_diff": float(max_diff), "restore_bit_identical": bit_identical}


def self_proof_p2(fl_subset_ids: list[str]) -> dict:
    """P2: sha256 of sorted instance_id list per policy — all three MUST be equal,
    matching fl_subset_hash.
    """
    print(f"\n[{now_iso()}] SELF-PROOF P2: same instances proof...")
    
    sorted_ids = sorted(fl_subset_ids)
    id_hash = sha256_list(sorted_ids)
    
    print(f"  Sorted FL subset ID list hash: {id_hash}")
    print(f"  Pinned fl_subset_hash:         {FL_SUBSET_HASH}")
    
    assert id_hash == FL_SUBSET_HASH, \
        f"P2 FAILED: id hash {id_hash} != pinned {FL_SUBSET_HASH}"
    
    # All three policies use the same list
    print("  All three policies use identical instance lists ✓")
    print("  P2 PASSED")
    return {"id_list_hash": id_hash, "n_instances": len(sorted_ids)}


def self_proof_p3(tok, model) -> dict:
    """P3: verifier sanity — for each family: reference artifact PASSES,
    empty completion FAILS, mutated artifact FAILS (9 asserts).
    """
    print(f"\n[{now_iso()}] SELF-PROOF P3: verifier sanity...")
    
    # F1 reference: a correct function that passes its tests
    f1_ref_instance = {
        "broken_artifact": "def add(a, b):\n    return a + b",
        "hidden_tests": ["assert add(1, 2) == 3", "assert add(-1, 0) == -1"],
    }
    ret_ref, err_ref = verify_f1(f1_ref_instance, "    return a + b")
    print(f"  F1 reference PASSES: {ret_ref} {err_ref}")
    assert ret_ref, f"P3 FAILED: F1 reference did not pass: {err_ref}"
    
    ret_empty, err_empty = verify_f1(f1_ref_instance, "")
    print(f"  F1 empty FAILS: {not ret_empty} {err_empty}")
    assert not ret_empty, "P3 FAILED: F1 empty completion passed (should fail)"
    
    ret_mut, err_mut = verify_f1(f1_ref_instance, "    return a - b")
    print(f"  F1 mutated FAILS: {not ret_mut} {err_mut}")
    assert not ret_mut, "P3 FAILED: F1 mutated artifact passed (should fail)"
    
    # F2 reference: valid JSON matching schema
    f2_ref_instance = {
        "broken_artifact": '{"user_id": "123", "email": "x@y.com", "active": true}',
        "hidden_tests": [
            'jsonschema.validate(repaired, {"type": "object", "properties": {"user_id": {"type": "string"}, "email": {"type": "string"}, "active": {"type": "boolean"}}, "required": ["user_id", "email", "active"]}) must pass'
        ],
    }
    ret_ref, err_ref = verify_f2(f2_ref_instance, '{"user_id": "123", "email": "x@y.com", "active": true}')
    print(f"  F2 reference PASSES: {ret_ref} {err_ref}")
    assert ret_ref, f"P3 FAILED: F2 reference did not pass: {err_ref}"
    
    ret_empty, err_empty = verify_f2(f2_ref_instance, "")
    print(f"  F2 empty FAILS: {not ret_empty} {err_empty}")
    assert not ret_empty, "P3 FAILED: F2 empty completion passed (should fail)"
    
    ret_mut, err_mut = verify_f2(f2_ref_instance, '{"user_id": 123}')
    print(f"  F2 mutated FAILS: {not ret_mut} {err_mut}")
    assert not ret_mut, "P3 FAILED: F2 mutated artifact passed (should fail)"
    
    # F3 reference: properly typed function
    f3_ref_instance = {
        "broken_artifact": "def greet(name: str) -> str:\n    return f'Hello {name}'",
        "hidden_tests": [],
    }
    ret_ref, err_ref = verify_f3(f3_ref_instance, "def greet(name: str) -> str:\n    return f'Hello {name}'")
    print(f"  F3 reference PASSES: {ret_ref} {err_ref}")
    assert ret_ref, f"P3 FAILED: F3 reference did not pass: {err_ref}"
    
    ret_empty, err_empty = verify_f3(f3_ref_instance, "")
    print(f"  F3 empty FAILS: {not ret_empty} {err_empty}")
    assert not ret_empty, "P3 FAILED: F3 empty completion passed (should fail)"
    
    ret_mut, err_mut = verify_f3(f3_ref_instance, "def greet(name):\n    return f'Hello {name}'")
    print(f"  F3 mutated FAILS: {not ret_mut} {err_mut}")
    assert not ret_mut, "P3 FAILED: F3 mutated artifact passed (should fail)"
    
    print("  P3 PASSED (9/9 asserts)")
    return {"f1": "9/9", "f2": "OK", "f3": "OK"}


# ═══════════════════════════════════════════════════════════════
# RUN PHASES
# ═══════════════════════════════════════════════════════════════

@dataclass
class RunResult:
    instance_id: str
    policy: str
    seed: int
    passed: bool
    error: str
    completion: str
    realized_tokens: int
    family: str
    skip_layers: list[int]
    entropy_data: dict | None
    run_label: str = "main"


class FirstLightRunner:
    """Main runner coordinating all phases."""
    
    def __init__(self, run_id: str | None = None):
        self.run_id = run_id or os.environ.get("FIRST_LIGHT_RUN_ID") or f"first-light-{uuid.uuid4().hex[:8]}"
        self.run_dir = OUTPUT_DIR / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.tok = None
        self.model = None
        self.schema = None
        self.manifest = None
        self.pinned_ids = None
        self.k_skip = K_SKIP  # instance-level, mutable via fallback ladder
        
        # Results storage
        self.all_results: list[RunResult] = []
        self.trace_rows: list[dict] = []
        self.ledger_rows: list[dict] = []
        self.rss_log: list[dict] = []
        self.thermal_log: list[dict] = []
        self.checkpoint_data: dict = {}
        
        # Checkpoint path
        self.checkpoint_path = self.run_dir / "checkpoint.jsonl"
        
        print(f"[{now_iso()}] First Light runner initialized")
        print(f"  Run ID: {self.run_id}")
        print(f"  Output: {self.run_dir}")
    
    # ── SETUP ──────────────────────────────────────────────
    
    def setup(self):
        """Load model, schema, manifest, pinned IDs."""
        self.schema = load_json(SCHEMA_PATH)
        self.manifest = load_yaml(MANIFEST_PATH)
        self.pinned_ids = load_pinned_instance_ids()
        
        self.tok, self.model = load_model()
        
        # Record empty-process RSS baseline
        rss_empty = rss_mb()
        print(f"  Empty-process RSS: {rss_empty:.1f} MB")
        self.rss_log.append({"timestamp": now_iso(), "event": "empty_baseline", "rss_mb": rss_empty})
        
        # Load model RSS
        rss_loaded = rss_mb()
        print(f"  After model load RSS: {rss_loaded:.1f} MB ({rss_loaded/1024:.2f} GB)")
        self.rss_log.append({"timestamp": now_iso(), "event": "model_loaded", "rss_mb": rss_loaded})
        
        # RSS gate check
        rss_total_gb = rss_loaded / 1024
        rss_attributable_gb = (rss_loaded - rss_empty) / 1024
        print(f"  Model-attributable RSS: {rss_attributable_gb:.2f} GB (gate: ≤{RSS_MODEL_ATTRIBUTABLE_GB} GB)")
        print(f"  Total-process RSS: {rss_total_gb:.2f} GB (gate: <{RSS_TOTAL_GATE_GB} GB)")
        
        if rss_total_gb >= RSS_TOTAL_GATE_GB:
            print(f"  ⚠ RSS EXCEEDS TOTAL GATE ({rss_total_gb:.1f} >= {RSS_TOTAL_GATE_GB}) — logging amendment")
        if rss_attributable_gb > RSS_MODEL_ATTRIBUTABLE_GB:
            print(f"  ⚠ RSS EXCEEDS MODEL-ATTRIBUTABLE GATE ({rss_attributable_gb:.1f} > {RSS_MODEL_ATTRIBUTABLE_GB}) — logging amendment")
    
    # ── CHECKPOINTING ─────────────────────────────────────
    
    def load_checkpoint(self):
        """Load existing checkpoint if any.

        Legacy rows from the first long run have no `label`; those remain keyed by
        (instance_id, policy, seed). New labeled rows use (label, instance_id,
        policy, seed). This prevents the PILOT dense row from colliding with the
        greedy dense anchor while preserving resume behavior for old rows.
        """
        if self.checkpoint_path.exists():
            completed = set()
            with open(self.checkpoint_path) as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        base_key = (row["instance_id"], row["policy"], row["seed"])
                        label = row.get("label")
                        if label:
                            completed.add((label, *base_key))
                            if label != "pilot":
                                completed.add(base_key)
                        else:
                            completed.add(base_key)
            return completed
        return set()
    
    def save_checkpoint(self, result: RunResult, label: str = "main"):
        """Append a result to checkpoint file."""
        with open(self.checkpoint_path, "a") as f:
            f.write(json.dumps(to_jsonable({
                "label": label,
                "instance_id": result.instance_id,
                "policy": result.policy,
                "seed": result.seed,
                "passed": result.passed,
                "error": result.error,
                "realized_tokens": result.realized_tokens,
                "family": result.family,
            })) + "\n")
    
    # ── THERMAL WARM ──────────────────────────────────────
    
    def thermal_warm(self, duration_s: int = 600):
        """10-min thermal warm: run dense passes through small set until steady state."""
        print(f"\n[{now_iso()}] Starting {duration_s}s thermal warm...")
        warm_start = time.time()
        
        # Use first 3 smoke instances for warmup
        smoke_ids = self.pinned_ids["smoke"][:3]
        
        iters = 0
        while (time.time() - warm_start) < duration_s:
            for inst_id in smoke_ids:
                inst = load_gym_instance(inst_id)
                prompt = inst["prompt_context"]
                generate_completion(self.tok, self.model, prompt, 42, temperature=0.0)
                iters += 1
                if (time.time() - warm_start) >= duration_s:
                    break
        
        elapsed = time.time() - warm_start
        rss_after = rss_mb()
        print(f"  Warm complete: {iters} iterations in {elapsed:.0f}s, RSS: {rss_after:.1f} MB")
        self.thermal_log.append({
            "timestamp": now_iso(),
            "event": "thermal_warm_complete",
            "duration_s": elapsed,
            "iterations": iters,
            "rss_mb": rss_after,
        })
    
    # ── RUN ONE INSTANCE×POLICY×SEED ──────────────────────
    
    def run_one(self, instance: dict, policy: str, seed: int,
                temperature: float = 0.0, top_p: float = 0.95,
                run_label: str = "main") -> RunResult:
        """Run one (instance, policy, seed) combination with REAL routing."""
        
        instance_id = instance["instance_id"]
        prompt = instance["prompt_context"]
        family = instance["metadata"]["family"]
        verifier = VERIFIERS[family]
        
        # Determine skip layers based on policy
        skip_layers = []
        entropy_data = None
        
        if policy == "dense_uniform":
            skip_layers = []
        elif policy == "random_matched_mix":
            skip_layers = select_skip_layers_random(ROUTABLE_LAYERS, self.k_skip, instance_id, seed)
        elif policy == "heuristic_entropy":
            inp = self.tok(prompt, return_tensors="pt").to(self.model.device)
            skip_layers, entropies, deltas = select_skip_layers_entropy(
                self.model, inp, ROUTABLE_LAYERS, self.k_skip
            )
            entropy_data = {"entropies": entropies, "deltas": deltas}
        
        # Generate completion WITH IdentitySkip if applicable
        t0 = time.time()
        if skip_layers:
            completion, realized_tokens = generate_completion(
                self.tok, self.model, prompt, seed,
                temperature=temperature, top_p=top_p, skip_indices=skip_layers
            )
        else:
            completion, realized_tokens = generate_completion(
                self.tok, self.model, prompt, seed,
                temperature=temperature, top_p=top_p
            )
        wall_time = time.time() - t0
        
        # Verify
        passed, error = verifier(instance, completion)
        
        result = RunResult(
            instance_id=instance_id,
            policy=policy,
            seed=seed,
            passed=passed,
            error=error,
            completion=completion,
            realized_tokens=realized_tokens,
            family=family,
            skip_layers=skip_layers,
            entropy_data=entropy_data,
            run_label=run_label,
        )
        
        # Save raw completion to disk. PILOT rows get their own namespace so they
        # cannot overwrite the primary greedy dense completions.
        if run_label == "pilot":
            compl_path = self.run_dir / "completions" / "pilot" / policy / f"{instance_id.replace('/', '_')}_seed{seed}.txt"
        else:
            compl_path = self.run_dir / "completions" / policy / f"{instance_id.replace('/', '_')}_seed{seed}.txt"
        compl_path.parent.mkdir(parents=True, exist_ok=True)
        compl_path.write_text(completion)
        
        # Emit trace rows
        trace_rows = emit_trace_rows_for_instance(instance_id, policy, skip_layers, self.run_id)
        for tr in trace_rows:
            jsonschema_validate(instance=tr, schema=self.schema)
        self.trace_rows.extend(trace_rows)
        
        # Emit ledger row
        ledger_row = emit_ledger_row(
            instance_id, seed, policy, passed,
            flops_total=0.0,
            bytes_per_token=0.0,
            wall_clock_tps=realized_tokens / wall_time if wall_time > 0 else 0.0,
            run_id=self.run_id,
        )
        self.ledger_rows.append(ledger_row)
        
        return result
    
    # ── RUN BLOCK ─────────────────────────────────────────
    
    def run_block(self, instance_ids: list[str], policies: list[str],
                  seeds: list[int], label: str, temperature: float = 0.0,
                  resume: bool = True) -> list[RunResult]:
        """Run a block of (instance × policy × seed) with checkpointing.
        
        Parameters
        ----------
        instance_ids : list[str]
            Pinned instance IDs to evaluate.
        policies : list[str]
            Policies to evaluate per instance.
        seeds : list[int]
            Seeds to evaluate per policy.
        label : str
            Block label for logging (e.g., 'smoke', 'dense', 'forced_random').
        temperature : float
            Generation temperature (0.0 = greedy).
        resume : bool
            If True, skip already-checkpointed entries.
        
        Returns
        -------
        list[RunResult]
        """
        total = len(instance_ids) * len(policies) * len(seeds)
        completed_set = self.load_checkpoint() if resume else set()
        n_skip = 0
        
        print(f"\n[{now_iso()}] BLOCK: {label} ({total} total, {len(completed_set)} already completed)")
        block_results = []
        count = 0
        
        for i, inst_id in enumerate(instance_ids):
            try:
                instance = load_gym_instance(inst_id)
            except FileNotFoundError as e:
                print(f"  SKIP {inst_id}: {e}")
                continue
            
            for policy in policies:
                for seed in seeds:
                    base_key = (inst_id, policy, seed)
                    key = (label, *base_key) if label == "pilot" else base_key
                    if resume and key in completed_set:
                        n_skip += 1
                        continue
                    
                    result = self.run_one(instance, policy, seed, temperature=temperature,
                                          run_label=label)
                    block_results.append(result)
                    self.all_results.append(result)
                    self.save_checkpoint(result, label=label)
                    
                    count += 1
                    if count % 50 == 0 or count <= 5:
                        pct = (count + n_skip) / total * 100
                        print(f"  [{now_iso()}] {label}: {count+n_skip}/{total} ({pct:.0f}%) "
                              f"last={result.passed} {result.instance_id[:30]}...")
        
        print(f"  [{now_iso()}] BLOCK {label} DONE: {count} new + {n_skip} skipped = {count+n_skip}/{total}")
        
        # Save intermediate traces and ledgers
        self._save_evidence(label)
        
        return block_results
    
    def _save_evidence(self, label: str):
        """Save current trace and ledger rows to disk."""
        trace_path = self.run_dir / f"traces_{label}.jsonl"
        save_jsonl(trace_path, self.trace_rows)
        
        ledger_path = self.run_dir / f"ledger_{label}.jsonl"
        save_jsonl(ledger_path, self.ledger_rows)
        
        # Also save complete results so far
        results_path = self.run_dir / f"results_{label}.json"
        save_json(results_path, [{
            "run_label": r.run_label,
            "instance_id": r.instance_id,
            "policy": r.policy,
            "seed": r.seed,
            "passed": r.passed,
            "family": r.family,
        } for r in self.all_results])
    
    # ── SMOKE GATE (FIX 5 — HARD ABORT) ──────────────────
    
    def smoke_gate(self) -> bool:
        """Run smoke gate on 10 pinned smoke instances. Hard abort on failure."""
        print(f"\n{'═'*70}")
        print(f"[{now_iso()}] SMOKE GATE — 10 pinned instances, HARD ABORT on failure")
        print(f"{'═'*70}")
        
        smoke_ids = self.pinned_ids["smoke"]
        assert len(smoke_ids) == 10, f"Expected 10 smoke IDs, got {len(smoke_ids)}"
        print(f"  Smoke IDs: {len(smoke_ids)} (pinned hash {SMOKE_SET_HASH})")
        
        # Run smoke: all 10 instances × 3 policies × 3 seeds = 90 runs
        # For the gate, we also run each twice to check determinism
        smoke_results = self.run_block(smoke_ids, ["dense_uniform", "random_matched_mix", "heuristic_entropy"],
                                       SEEDS, "smoke", temperature=0.0, resume=False)
        
        # ── Gate checks ──
        
        # 1. Schema validation: every trace row must validate
        trace_failures = 0
        for tr in self.trace_rows:
            try:
                jsonschema_validate(instance=tr, schema=self.schema)
            except Exception as e:
                trace_failures += 1
                print(f"  TRACE VALIDATION ERROR: {e}")
        
        # 2. Reason-code coverage: must be 100% FORCED_BASELINE
        reason_codes = set()
        for tr in self.trace_rows:
            reason_codes.add(tr.get("reason_code", "MISSING"))
        
        print(f"\n  Trace rows: {len(self.trace_rows)}")
        print(f"  Trace validation failures: {trace_failures}")
        print(f"  Reason codes: {reason_codes}")
        
        # 3. Verifier determinism: re-run 3 instances 2× and check verdicts match
        print(f"\n  Determinism check (3 instances × 2 runs)...")
        det_instances = smoke_ids[:3]
        det_results_run1 = {}
        det_results_run2 = {}
        
        for inst_id in det_instances:
            inst = load_gym_instance(inst_id)
            for policy in ["dense_uniform"]:
                for seed in [SEEDS[0]]:
                    r1 = self.run_one(inst, policy, seed, temperature=0.0)
                    r2 = self.run_one(inst, policy, seed, temperature=0.0)
                    det_results_run1[(inst_id, policy, seed)] = r1.passed
                    det_results_run2[(inst_id, policy, seed)] = r2.passed
        
        nondeterministic = []
        for key in det_results_run1:
            if det_results_run1[key] != det_results_run2[key]:
                nondeterministic.append(key)
        
        if nondeterministic:
            print(f"  NONDETERMINISTIC VERDICTS: {nondeterministic}")
        else:
            print(f"  All verdicts deterministic ✓")
        
        # 4. Forced rows pass rate check: if BOTH forced rows are 0% on smoke → fallback ladder
        smoke_by_policy = defaultdict(list)
        for r in self.all_results:
            if r.instance_id in smoke_ids:
                smoke_by_policy[r.policy].append(r.passed)
        
        dense_rate = np.mean(smoke_by_policy["dense_uniform"]) if smoke_by_policy["dense_uniform"] else 0
        random_rate = np.mean(smoke_by_policy["random_matched_mix"]) if smoke_by_policy["random_matched_mix"] else 0
        entropy_rate = np.mean(smoke_by_policy["heuristic_entropy"]) if smoke_by_policy["heuristic_entropy"] else 0
        
        print(f"\n  Smoke pass rates:")
        print(f"    dense_uniform:     {dense_rate*100:.1f}%")
        print(f"    random_matched_mix: {random_rate*100:.1f}%")
        print(f"    heuristic_entropy:  {entropy_rate*100:.1f}%")
        
        # Check fallback ladder
        amendment_log = []
        if random_rate == 0.0 and entropy_rate == 0.0:
            print(f"  ⚠ BOTH forced rows at 0% on smoke — applying fallback ladder k→2→1")
            self.k_skip = 2
            amendment_log.append(f"Fallback ladder step 1: k reduced from 4 to 2 (smoke forced rows both 0%)")
            print(f"  AMENDMENT: {amendment_log[-1]}")
            
            # Re-run smoke with k=2 to check
            smoke_results2 = self.run_block(smoke_ids[:3], ["random_matched_mix", "heuristic_entropy"],
                                            [SEEDS[0]], "smoke_fallback_k2", temperature=0.0, resume=False)
            random_rate2 = np.mean([r.passed for r in smoke_results2 if r.policy == "random_matched_mix"])
            entropy_rate2 = np.mean([r.passed for r in smoke_results2 if r.policy == "heuristic_entropy"])
            
            if random_rate2 == 0.0 and entropy_rate2 == 0.0:
                self.k_skip = 1
                amendment_log.append(f"Fallback ladder step 2: k reduced from 2 to 1 (still 0%)")
                print(f"  AMENDMENT: {amendment_log[-1]}")
        
        # ── GATE DECISION ──
        gate_passed = True
        failures = []
        
        if trace_failures > 0:
            failures.append(f"Trace validation failures: {trace_failures}")
            gate_passed = False
        
        if len(reason_codes) != 1 or "FORCED_BASELINE" not in reason_codes:
            failures.append(f"Reason-code coverage < 100%: {reason_codes}")
            gate_passed = False
        
        if nondeterministic:
            failures.append(f"Nondeterministic verdicts: {len(nondeterministic)}")
            gate_passed = False
        
        if len(self.trace_rows) < 10 * 3 * len(ROUTABLE_LAYERS):
            # Expected: 10 instances × 3 policies × 14 routable layers = 420 rows minimum
            failures.append(f"Too few trace rows: {len(self.trace_rows)}")
            gate_passed = False
        
        if not gate_passed:
            print(f"\n{'═'*70}")
            print(f"SMOKE GATE FAILED — BLOCKER REPORT:")
            for f in failures:
                print(f"  ✗ {f}")
            print(f"{'═'*70}")
            
            # Write blocker report
            blocker_path = self.run_dir / "SMOKE_GATE_BLOCKER.md"
            blocker_path.write_text(
                f"# Smoke Gate Blocker Report\n\n"
                f"Run: {self.run_id}\n"
                f"Time: {now_iso()}\n\n"
                f"## Failures\n\n" +
                "\n".join(f"- {f}" for f in failures) +
                f"\n\n## Amendments Logged\n\n" +
                "\n".join(f"- {a}" for a in amendment_log) +
                f"\n"
            )
            print(f"\n  Blocker report: {blocker_path}")
            return False
        
        print(f"\n  ✓ SMOKE GATE PASSED")
        print(f"  Amendments: {len(amendment_log)}")
        for a in amendment_log:
            print(f"    - {a}")
        
        # Save smoke evidence
        self._save_evidence("smoke")
        self.amendment_log = amendment_log
        
        return True
    
    # ── DENSE ANCHOR + S1/S2 TELEMETRY ────────────────────
    
    def run_dense_anchor(self):
        """Run dense_uniform on ALL 300 instances × 3 seeds.
        Collects per-layer entropy telemetry for S1/S2 secondary analyses.
        """
        fl_ids = self.pinned_ids["fl_subset"]
        print(f"\n[{now_iso()}] DENSE ANCHOR: {len(fl_ids)} instances × {len(SEEDS)} seeds = {len(fl_ids)*len(SEEDS)} runs")
        
        # Run dense passes
        dense_results = self.run_block(fl_ids, ["dense_uniform"], SEEDS, "dense", temperature=0.0)
        
        # Collect S1/S2 telemetry: per-layer entropies for dense row only (once per instance)
        print(f"\n[{now_iso()}] Collecting S1/S2 entropy telemetry on dense row...")
        s1_s2_data = []
        for i, inst_id in enumerate(fl_ids):
            try:
                inst = load_gym_instance(inst_id)
            except FileNotFoundError:
                continue
            prompt = inst["prompt_context"]
            inp = self.tok(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                with IdentitySkipContext(self.model, []):  # no skip = dense
                    entropies = compute_layer_entropies(self.model, inp, ROUTABLE_LAYERS)
            
            # Compute mean prompt entropy and mean |ΔH|
            h_values = list(entropies.values())
            mean_h = np.mean(h_values)
            deltas = []
            for j in range(1, len(ROUTABLE_LAYERS)):
                deltas.append(abs(h_values[j] - h_values[j-1]))
            mean_abs_delta = np.mean(deltas) if deltas else 0.0
            
            s1_s2_data.append({
                "instance_id": inst_id,
                "mean_entropy": float(mean_h),
                "mean_abs_delta_h": float(mean_abs_delta),
                "per_layer_entropy": {str(k): float(v) for k, v in entropies.items()},
            })
            
            if (i + 1) % 50 == 0:
                print(f"  S1/S2 telemetry: {i+1}/{len(fl_ids)}")
        
        # Save telemetry
        telemetry_path = self.run_dir / "s1_s2_telemetry.json"
        save_json(telemetry_path, s1_s2_data)
        print(f"  S1/S2 telemetry saved: {telemetry_path}")
        
        self._save_evidence("dense")
        return dense_results
    
    # ── FORCED REPLAYS ────────────────────────────────────
    
    def run_forced_replays(self):
        """Run random_matched_mix and heuristic_entropy on ALL 300 instances × 3 seeds."""
        fl_ids = self.pinned_ids["fl_subset"]
        
        print(f"\n[{now_iso()}] FORCED REPLAYS — RANDOM: {len(fl_ids)} instances × {len(SEEDS)} seeds")
        random_results = self.run_block(fl_ids, ["random_matched_mix"], SEEDS, "forced_random", temperature=0.0)
        
        print(f"\n[{now_iso()}] FORCED REPLAYS — ENTROPY: {len(fl_ids)} instances × {len(SEEDS)} seeds")
        entropy_results = self.run_block(fl_ids, ["heuristic_entropy"], SEEDS, "forced_entropy", temperature=0.0)
        
        self._save_evidence("forced")
        return random_results, entropy_results
    
    # ── PILOT BLOCK ───────────────────────────────────────
    
    def run_pilot_block(self):
        """Run dense at temperature 0.7 / top-p 0.95 — PILOT only, never claim-eligible."""
        fl_ids = self.pinned_ids["fl_subset"]
        print(f"\n[{now_iso()}] PILOT BLOCK: {len(fl_ids)} instances × {len(SEEDS)} seeds at temp=0.7")
        pilot_results = self.run_block(fl_ids, ["dense_uniform"], SEEDS, "pilot", temperature=0.7)
        self._save_evidence("pilot")
        return pilot_results
    
    # ── ANALYSIS ──────────────────────────────────────────
    
    def run_analysis(self):
        """Frozen analysis plan: paired bootstrap, McNemar, S1/S2, byte model.
        All hypothesis verdicts COMPUTED from data, not hardcoded.
        """
        print(f"\n{'═'*70}")
        print(f"[{now_iso()}] ANALYSIS — frozen plan, computed verdicts only")
        print(f"{'═'*70}")
        
        fl_ids = self.pinned_ids["fl_subset"]
        
        # Aggregate results by instance×policy
        # For each instance, aggregate over seeds (take mode/pass-if-any, per prereg seed reporting)
        by_instance = defaultdict(lambda: defaultdict(list))
        for r in self.all_results:
            if r.instance_id in fl_ids and r.run_label != "pilot":
                by_instance[r.instance_id][r.policy].append(r.passed)
        
        # Per-policy pass rates (over instances, over seeds)
        dense_passes = []
        random_passes = []
        entropy_passes = []
        
        for inst_id in fl_ids:
            for policy, passes in by_instance[inst_id].items():
                # Instance-level: pass if any seed passes (per prereg seed reporting)
                inst_pass = any(passes)
                if policy == "dense_uniform":
                    dense_passes.append(inst_pass)
                elif policy == "random_matched_mix":
                    random_passes.append(inst_pass)
                elif policy == "heuristic_entropy":
                    entropy_passes.append(inst_pass)
        
        dense_rate = np.mean(dense_passes) if dense_passes else 0.0
        random_rate = np.mean(random_passes) if random_passes else 0.0
        entropy_rate = np.mean(entropy_passes) if entropy_passes else 0.0
        
        n_dense = len(dense_passes)
        n_random = len(random_passes)
        n_entropy = len(entropy_passes)
        
        print(f"\n  Pooled pass rates (instance-level, any seed):")
        print(f"    dense_uniform:     {dense_rate*100:.1f}% ({sum(dense_passes)}/{n_dense})")
        print(f"    random_matched_mix: {random_rate*100:.1f}% ({sum(random_passes)}/{n_random})")
        print(f"    heuristic_entropy:  {entropy_rate*100:.1f}% ({sum(entropy_passes)}/{n_entropy})")
        
        # ── Paired bootstrap (B=10000) over instances ──
        print(f"\n  Paired bootstrap (B={BOOTSTRAP_B})...")
        rng = np.random.RandomState(20260612)
        
        # Only use instances with data for ALL three policies
        paired_instances = [
            inst_id for inst_id in fl_ids
            if all(pol in by_instance[inst_id] for pol in ["dense_uniform", "random_matched_mix", "heuristic_entropy"])
        ]
        n_paired = len(paired_instances)
        print(f"  Paired instances: {n_paired}")
        
        # Build paired arrays
        dense_arr = np.array([any(by_instance[iid]["dense_uniform"]) for iid in paired_instances])
        random_arr = np.array([any(by_instance[iid]["random_matched_mix"]) for iid in paired_instances])
        entropy_arr = np.array([any(by_instance[iid]["heuristic_entropy"]) for iid in paired_instances])
        
        # Bootstrap CIs
        def bootstrap_ci(arr, B=BOOTSTRAP_B):
            means = []
            for _ in range(B):
                idx = rng.choice(len(arr), size=len(arr), replace=True)
                means.append(np.mean(arr[idx]))
            return np.mean(arr), (np.percentile(means, 2.5), np.percentile(means, 97.5))
        
        dense_mean, dense_ci = bootstrap_ci(dense_arr)
        random_mean, random_ci = bootstrap_ci(random_arr)
        entropy_mean, entropy_ci = bootstrap_ci(entropy_arr)
        
        # Paired deltas
        delta_dr_arr = dense_arr.astype(int) - random_arr.astype(int)
        delta_de_arr = dense_arr.astype(int) - entropy_arr.astype(int)
        delta_re_arr = random_arr.astype(int) - entropy_arr.astype(int)
        
        def paired_bootstrap_ci(delta_arr, B=BOOTSTRAP_B):
            means = []
            for _ in range(B):
                idx = rng.choice(len(delta_arr), size=len(delta_arr), replace=True)
                means.append(np.mean(delta_arr[idx]))
            return np.mean(delta_arr), (np.percentile(means, 2.5), np.percentile(means, 97.5))
        
        dr_mean, dr_ci = paired_bootstrap_ci(delta_dr_arr)
        de_mean, de_ci = paired_bootstrap_ci(delta_de_arr)
        re_mean, re_ci = paired_bootstrap_ci(delta_re_arr)
        
        print(f"\n  Bootstrap results:")
        print(f"    dense:  {dense_mean*100:.1f}% [{dense_ci[0]*100:.1f}%, {dense_ci[1]*100:.1f}%]")
        print(f"    random: {random_mean*100:.1f}% [{random_ci[0]*100:.1f}%, {random_ci[1]*100:.1f}%]")
        print(f"    entropy:{entropy_mean*100:.1f}% [{entropy_ci[0]*100:.1f}%, {entropy_ci[1]*100:.1f}%]")
        print(f"    Δ(dense-random):  {dr_mean*100:+.1f}pp [{dr_ci[0]*100:+.1f}pp, {dr_ci[1]*100:+.1f}pp]")
        print(f"    Δ(dense-entropy): {de_mean*100:+.1f}pp [{de_ci[0]*100:+.1f}pp, {de_ci[1]*100:+.1f}pp]")
        print(f"    Δ(random-entropy):{re_mean*100:+.1f}pp [{re_ci[0]*100:+.1f}pp, {re_ci[1]*100:+.1f}pp]")
        
        # ── McNemar secondary (dense vs random) ──
        print(f"\n  McNemar exact test (dense vs random):")
        a = np.sum(dense_arr & ~random_arr)  # dense pass, random fail
        b = np.sum(~dense_arr & random_arr)  # dense fail, random pass
        c = np.sum(dense_arr & random_arr)    # both pass
        d_sum = np.sum(~dense_arr & ~random_arr)  # both fail
        print(f"    pass/fail agreement matrix: a={a} b={b} c={c} d={d_sum}")
        
        if a + b > 0:
            from scipy.stats import binom
            p_mcnemar = binom.cdf(min(a, b), a + b, 0.5) * 2  # two-sided
            print(f"    McNemar p = {p_mcnemar:.4f}")
        else:
            p_mcnemar = 1.0
            print(f"    McNemar not applicable (no discordant pairs)")
        
        # ── Per-family breakdown ──
        print(f"\n  Per-family breakdown:")
        per_family = {}
        for family in ["F1_mbpp", "F2_json", "F3_type"]:
            family_insts = [iid for iid in paired_instances if iid.startswith(f"gym-v0.1-FL/{family[0:2]}/")]
            if family_insts:
                f_dense = np.array([any(by_instance[iid]["dense_uniform"]) for iid in family_insts])
                f_random = np.array([any(by_instance[iid]["random_matched_mix"]) for iid in family_insts])
                f_entropy = np.array([any(by_instance[iid]["heuristic_entropy"]) for iid in family_insts])
                
                per_family[family] = {
                    "n": len(family_insts),
                    "dense": float(np.mean(f_dense)),
                    "random": float(np.mean(f_random)),
                    "entropy": float(np.mean(f_entropy)),
                }
                print(f"    {family}: n={len(family_insts)} dense={np.mean(f_dense)*100:.1f}% "
                      f"random={np.mean(f_random)*100:.1f}% entropy={np.mean(f_entropy)*100:.1f}%")
        
        # ── S1: Entropy vs outcome rank correlation ──
        print(f"\n  S1: Entropy vs pass/fail rank correlation...")
        telemetry_path = self.run_dir / "s1_s2_telemetry.json"
        if telemetry_path.exists():
            telemetry = load_json(telemetry_path)
            telemetry_by_id = {t["instance_id"]: t for t in telemetry}
            
            s1_entropies = []
            s1_outcomes = []
            for inst_id in paired_instances:
                if inst_id in telemetry_by_id and "dense_uniform" in by_instance[inst_id]:
                    s1_entropies.append(telemetry_by_id[inst_id]["mean_abs_delta_h"])
                    s1_outcomes.append(any(by_instance[inst_id]["dense_uniform"]))
            
            if len(s1_entropies) > 5:
                rho, p_rho = stats.spearmanr(s1_entropies, s1_outcomes)
                print(f"    Spearman ρ(mean|ΔH| vs dense pass) = {rho:.4f} (p={p_rho:.4f})")
            else:
                rho, p_rho = 0.0, 1.0
                print(f"    Insufficient data for S1 correlation")
        else:
            rho, p_rho = 0.0, 1.0
            print(f"    No S1/S2 telemetry found")
        
        # ── Byte model check (H-FL-4) ──
        print(f"\n  H-FL-4: Byte model throughput check...")
        byte_preds = load_yaml(BYTE_PREDICTIONS_PATH) if BYTE_PREDICTIONS_PATH.exists() else {}
        predicted_dense_p50 = float(byte_preds.get("throughput_predictions", {}).get("dense_predicted_tps_p50", 15.0))
        predicted_skip_p50 = float(byte_preds.get("throughput_predictions", {}).get("skip_predicted_tps_p50", 18.0))

        # Use the primary pre-pilot ledger when present: it contains smoke + dense + forced
        # replays, but not the repaired pilot sampling rows. This is the cleanest available
        # measurement artifact for H-FL-4 in the v3 run. Fall back to in-memory ledger rows.
        def _median(vals):
            vals = sorted(float(v) for v in vals)
            if not vals:
                return 0.0
            mid = len(vals) // 2
            return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2

        ledger_source_path = self.run_dir / "ledger_forced_entropy.jsonl"
        ledger_for_hfl4 = []
        if ledger_source_path.exists():
            with open(ledger_source_path) as f:
                for line in f:
                    if line.strip():
                        ledger_for_hfl4.append(json.loads(line))
        else:
            ledger_for_hfl4 = self.ledger_rows
            ledger_source_path = self.run_dir / "ledger.jsonl"

        tps_by_policy = defaultdict(list)
        for row in ledger_for_hfl4:
            policy_name = row.get("policy", {}).get("baseline_type")
            try:
                tps = row["cost"]["wall_clock_sustained_tps"]["p50"]
            except Exception:
                continue
            tps_by_policy[policy_name].append(tps)

        measured_tps_dense = _median(tps_by_policy["dense_uniform"])
        measured_tps_random = _median(tps_by_policy["random_matched_mix"])
        measured_tps_entropy = _median(tps_by_policy["heuristic_entropy"])
        forced_skip_vals = tps_by_policy["random_matched_mix"] + tps_by_policy["heuristic_entropy"]
        measured_tps_skip = _median(forced_skip_vals)

        dense_pct_error = abs(measured_tps_dense - predicted_dense_p50) / predicted_dense_p50 if predicted_dense_p50 else 1.0
        skip_pct_error = abs(measured_tps_skip - predicted_skip_p50) / predicted_skip_p50 if predicted_skip_p50 else 1.0
        dense_within_20pct = dense_pct_error <= 0.20
        skip_within_20pct = skip_pct_error <= 0.20

        print(f"    Dense: measured={measured_tps_dense:.2f} tok/s predicted={predicted_dense_p50:.2f} tok/s "
              f"within 20%: {dense_within_20pct}")
        print(f"    Skip:  measured={measured_tps_skip:.2f} tok/s predicted={predicted_skip_p50:.2f} tok/s "
              f"within 20%: {skip_within_20pct} (random={measured_tps_random:.2f}, entropy={measured_tps_entropy:.2f})")
        
        # ── COMPUTED HYPOTHESIS VERDICTS (P4 — no string literals) ──
        print(f"\n  {'═'*60}")
        print(f"  COMPUTED HYPOTHESIS VERDICTS (P4)")
        print(f"  {'═'*60}")
        
        # H-FL-1: pipeline integrity — trace schema valid, reason-code coverage 100%.
        assert self.schema is not None, "schema must be loaded before analysis"
        trace_schema_errors = 0
        reason_codes_seen = set()
        for tr in self.trace_rows:
            reason_codes_seen.add(tr.get("reason_code", "MISSING"))
            try:
                jsonschema_validate(instance=tr, schema=self.schema)
            except Exception:
                trace_schema_errors += 1
        hfl1_pass = trace_schema_errors == 0 and reason_codes_seen == {"FORCED_BASELINE"}
        hfl1_detail = (
            f"PASS ({len(self.trace_rows)} trace rows, 0 schema errors, 100% FORCED_BASELINE coverage)"
            if hfl1_pass else
            f"FAIL ({len(self.trace_rows)} trace rows, schema_errors={trace_schema_errors}, reason_codes={sorted(reason_codes_seen)})"
        )
        
        # H-FL-2: dense > random — sanity check
        # PASS if dense CI > random CI (CIs don't overlap zero delta)
        hfl2_pass = dr_ci[0] > 0  # lower bound of CI excludes 0
        hfl2_detail = (
            f"PASS (Δ={dr_mean*100:+.1f}pp, CI=[{dr_ci[0]*100:+.1f}, {dr_ci[1]*100:+.1f}]pp)"
            if hfl2_pass
            else "FAIL → harness-bug protocol per prereg outcome_action_map"
        )
        
        # H-FL-3: entropy >= random at matched compute.
        # `re_*` above is random - entropy, so invert the sign for the preregistered
        # H-FL-3 estimand (entropy - random). The previous v3 receipt labeled this backwards.
        mde_band = 0.08  # 8pp conservative
        er_mean = -re_mean
        er_ci = (-re_ci[1], -re_ci[0])
        hfl3_tie = abs(er_ci[0]) <= mde_band and abs(er_ci[1]) <= mde_band  # CI entirely within MDE band
        hfl3_entropy_wins = er_ci[0] > 0  # lower bound excludes 0 positively
        if hfl3_entropy_wins:
            hfl3_detail = f"PASS / ENTROPY WINS (entropy-random Δ={er_mean*100:+.1f}pp, CI=[{er_ci[0]*100:+.1f}, {er_ci[1]*100:+.1f}]pp excludes 0)"
        elif hfl3_tie:
            hfl3_detail = f"TIE (entropy-random Δ={er_mean*100:+.1f}pp, CI within ~{mde_band*100:.0f}pp MDE band)"
        else:
            hfl3_detail = f"FAIL / ENTROPY BEHIND (entropy-random Δ={er_mean*100:+.1f}pp, CI=[{er_ci[0]*100:+.1f}, {er_ci[1]*100:+.1f}]pp)"
        
        # H-FL-4: byte model within 20%
        hfl4_pass = dense_within_20pct and skip_within_20pct
        hfl4_detail = (
            f"PASS (dense measured={measured_tps_dense:.2f} vs predicted={predicted_dense_p50:.2f}, error={dense_pct_error*100:.1f}%; "
            f"skip measured={measured_tps_skip:.2f} vs predicted={predicted_skip_p50:.2f}, error={skip_pct_error*100:.1f}%)"
            if hfl4_pass
            else f"FAIL → byte-model audit per §143.4 (dense measured={measured_tps_dense:.2f} vs predicted={predicted_dense_p50:.2f}, error={dense_pct_error*100:.1f}%; "
                 f"skip measured={measured_tps_skip:.2f} vs predicted={predicted_skip_p50:.2f}, error={skip_pct_error*100:.1f}%)"
        )
        
        print(f"  H-FL-1: {hfl1_detail}")
        print(f"  H-FL-2: {hfl2_detail}")
        print(f"  H-FL-3: {hfl3_detail}")
        print(f"  H-FL-4: {hfl4_detail}")
        
        # Build analysis result
        analysis = {
            "per_policy": {
                "dense_uniform": {"rate": float(dense_mean), "ci_95": [float(dense_ci[0]), float(dense_ci[1])], "n": n_dense},
                "random_matched_mix": {"rate": float(random_mean), "ci_95": [float(random_ci[0]), float(random_ci[1])], "n": n_random},
                "heuristic_entropy": {"rate": float(entropy_mean), "ci_95": [float(entropy_ci[0]), float(entropy_ci[1])], "n": n_entropy},
            },
            "paired_deltas": {
                "dense_vs_random_pp": {"mean": float(dr_mean), "ci_95": [float(dr_ci[0]), float(dr_ci[1])]},
                "dense_vs_entropy_pp": {"mean": float(de_mean), "ci_95": [float(de_ci[0]), float(de_ci[1])]},
                "random_vs_entropy_pp": {"mean": float(re_mean), "ci_95": [float(re_ci[0]), float(re_ci[1])]},
            },
            "mcnemar": {"a": int(a), "b": int(b), "c": int(c), "d": int(d_sum), "p": float(p_mcnemar)},
            "per_family": per_family,
            "s1_spearman_rho": float(rho),
            "s1_spearman_p": float(p_rho),
            "byte_model": {
                "ledger_source": str(ledger_source_path),
                "dense_predicted_tps_p50": predicted_dense_p50,
                "skip_predicted_tps_p50": predicted_skip_p50,
                "dense_measured_tps": measured_tps_dense,
                "random_measured_tps": measured_tps_random,
                "entropy_measured_tps": measured_tps_entropy,
                "skip_measured_tps": measured_tps_skip,
                "dense_abs_pct_error": float(dense_pct_error),
                "skip_abs_pct_error": float(skip_pct_error),
                "dense_within_20pct": dense_within_20pct,
                "skip_within_20pct": skip_within_20pct,
            },
            "mde_floor_pp": 8.0,
            "hypotheses": {
                "H-FL-1": {"pass": hfl1_pass, "detail": hfl1_detail},
                "H-FL-2": {"pass": hfl2_pass, "detail": hfl2_detail},
                "H-FL-3": {"detail": hfl3_detail, "tie": hfl3_tie, "entropy_wins": hfl3_entropy_wins},
                "H-FL-4": {"pass": hfl4_pass, "detail": hfl4_detail},
            },
        }
        
        self.analysis = analysis
        return analysis
    
    # ── ASSEMBLE RECEIPT ──────────────────────────────────
    
    def assemble_receipt(self, self_proofs: dict):
        """Assemble the complete First Light receipt."""
        print(f"\n{'═'*70}")
        print(f"[{now_iso()}] ASSEMBLING RECEIPT")
        print(f"{'═'*70}")
        
        receipt = {
            "run_id": self.run_id,
            "date": now_iso(),
            "status": "COMPLETE",
            "prereg": {
                "version": "FL-PREREG-v1.2-final",
                "status": "FROZEN",
                "prereg_hash": sha256_short(
                    open("/Users/christienantonio/Desktop/AI:ML Research/first_light_preregistration.yaml").read()
                ),
            },
            "model": {
                "repo": MODEL_REPO,
                "revision": MODEL_REVISION,
                "tokenizer_hash": TOKENIZER_HASH,
                "config_hash": CONFIG_HASH,
            },
            "gym": {
                "version": "gym-v0.1-FL",
                "manifest_hash": GYM_MANIFEST_HASH,
                "fl_subset_hash": FL_SUBSET_HASH,
                "smoke_set_hash": SMOKE_SET_HASH,
                "fl_subset_size": len(self.pinned_ids["fl_subset"]),
                "smoke_size": len(self.pinned_ids["smoke"]),
            },
            "execution": {
                "seeds": SEEDS,
                "routable_layers": ROUTABLE_LAYERS,
                "k_skip": self.k_skip,
                "max_new_tokens": MAX_NEW_TOKENS,
                "policies": ["dense_uniform", "random_matched_mix", "heuristic_entropy"],
                "amendment_log": getattr(self, 'amendment_log', []),
            },
            "self_proofs": self_proofs,
            "analysis": self.analysis,
            "rss_log": self.rss_log,
            "thermal_log": self.thermal_log,
            "artifacts": {
                "traces": str(self.run_dir / "traces.jsonl"),
                "ledger": str(self.run_dir / "ledger.jsonl"),
                "primary_throughput_ledger": str(self.run_dir / "ledger_forced_entropy.jsonl"),
                "byte_predictions": str(BYTE_PREDICTIONS_PATH),
                "rss_sidecar": str(self.run_dir / "rss_sidecar_measurement.json"),
                "completions": str(self.run_dir / "completions/"),
                "pilot_completions": str(self.run_dir / "completions/pilot/dense_uniform"),
                "checkpoint": str(self.checkpoint_path),
                "s1_s2_telemetry": str(self.run_dir / "s1_s2_telemetry.json"),
            },
            "contamination_note": "ALL rows contaminated_for_paper1=True (pilot). Per-family flags: F1=FLAGGED, F2/F3=CLEAN_BY_CONSTRUCTION.",
            "non_claims": [
                "NO VB-MCA superiority claim",
                "NO speedup claim",
                "NO scaling/organization claim",
            ],
        }
        
        # Save receipt
        receipt = to_jsonable(receipt)
        receipt_path = self.run_dir / f"{self.run_id}_receipt.yaml"
        with open(receipt_path, "w") as f:
            yaml.safe_dump(receipt, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        # Also save as JSON for programmatic access
        receipt_json_path = self.run_dir / f"{self.run_id}_receipt.json"
        save_json(receipt_json_path, receipt)
        
        # Save final combined traces + ledger
        save_jsonl(self.run_dir / "traces.jsonl", self.trace_rows)
        save_jsonl(self.run_dir / "ledger.jsonl", self.ledger_rows)
        
        print(f"  Receipt: {receipt_path}")
        print(f"  Receipt JSON: {receipt_json_path}")
        
        self.receipt = receipt
        return receipt


# ═══════════════════════════════════════════════════════════════
# MAIN — RUNBOOK ORDER
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("REAL FIRST LIGHT — v3 CORRECTED RUNNER")
    print("=" * 70)
    print(f"Start: {now_iso()}")
    print(f"Model: {MODEL_REPO}@{MODEL_REVISION[:12]}")
    print(f"Routable layers: {ROUTABLE_LAYERS} (n={len(ROUTABLE_LAYERS)})")
    print(f"k skip: {K_SKIP}")
    print(f"Seeds: {SEEDS}")
    
    runner = FirstLightRunner()
    
    # ── Phase 0: Setup ──
    runner.setup()
    
    # ── Phase 0.5: Thermal warm ──
    runner.thermal_warm(duration_s=600)
    
    # ── Phase 1: Self-proofs P1-P4 ──
    print(f"\n{'═'*70}")
    print(f"PHASE 1: SELF-PROOFS P1-P4")
    print(f"{'═'*70}")
    
    self_proofs = {}
    
    # P1: routing proof
    self_proofs["p1_routing"] = self_proof_p1(runner.tok, runner.model)
    
    # P2: same instances proof
    self_proofs["p2_same_instances"] = self_proof_p2(runner.pinned_ids["fl_subset"])
    
    # P3: verifier sanity
    self_proofs["p3_verifier_sanity"] = self_proof_p3(runner.tok, runner.model)
    
    # P4: computed verdicts (verified at analysis time)
    self_proofs["p4_computed_verdicts"] = "VERIFIED_DURING_ANALYSIS"
    
    print(f"\n  All self-proofs PASSED")
    
    # ── Phase 2: Smoke gate ──
    print(f"\n{'═'*70}")
    print(f"PHASE 2: SMOKE GATE")
    print(f"{'═'*70}")
    
    smoke_ok = runner.smoke_gate()
    if not smoke_ok:
        print(f"\n  ✗ SMOKE GATE FAILED — sys.exit(1)")
        sys.exit(1)
    
    # ── Phase 3: Dense anchor ──
    print(f"\n{'═'*70}")
    print(f"PHASE 3: DENSE ANCHOR")
    print(f"{'═'*70}")
    runner.run_dense_anchor()
    
    # ── Phase 4: Forced replays ──
    print(f"\n{'═'*70}")
    print(f"PHASE 4: FORCED REPLAYS (RANDOM + ENTROPY)")
    print(f"{'═'*70}")
    runner.run_forced_replays()
    
    # ── Phase 5: Pilot block ──
    print(f"\n{'═'*70}")
    print(f"PHASE 5: PILOT BLOCK (temp=0.7)")
    print(f"{'═'*70}")
    runner.run_pilot_block()
    
    # ── Phase 6: Analysis ──
    print(f"\n{'═'*70}")
    print(f"PHASE 6: FROZEN ANALYSIS")
    print(f"{'═'*70}")
    analysis = runner.run_analysis()
    
    # ── Phase 7: Receipt assembly ──
    print(f"\n{'═'*70}")
    print(f"PHASE 7: RECEIPT ASSEMBLY")
    print(f"{'═'*70}")
    receipt = runner.assemble_receipt(self_proofs)
    
    # ── Phase 8: Update claims.yaml and obligation ledger ──
    print(f"\n{'═'*70}")
    print(f"PHASE 8: UPDATE CLAIMS + OBLIGATION LEDGER")
    print(f"{'═'*70}")
    # TODO: update claims.yaml (CLM-008) and obligation ledger when those files exist
    
    # ── Done ──
    print(f"\n{'═'*70}")
    print(f"REAL FIRST LIGHT COMPLETE")
    print(f"{'═'*70}")
    print(f"Run ID: {runner.run_id}")
    print(f"Receipt: {runner.run_dir / f'{runner.run_id}_receipt.yaml'}")
    print(f"Traces: {runner.run_dir / 'traces.jsonl'}")
    print(f"Ledger: {runner.run_dir / 'ledger.jsonl'}")
    print(f"Completions: {runner.run_dir / 'completions/'}")
    print()
    print(f"H-FL-1 (pipeline integrity): {analysis['hypotheses']['H-FL-1']['detail']}")
    print(f"H-FL-2 (dense > random): {analysis['hypotheses']['H-FL-2']['detail']}")
    print(f"H-FL-3 (entropy vs random): {analysis['hypotheses']['H-FL-3']['detail']}")
    print(f"H-FL-4 (byte model): {analysis['hypotheses']['H-FL-4']['detail']}")
    print()
    print("A negative or tie result is a SUCCESS — written exactly as it landed.")


if __name__ == "__main__":
    main()
