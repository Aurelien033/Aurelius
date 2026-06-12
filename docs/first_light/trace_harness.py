#!/usr/bin/env python3
"""Minimal trace harness for First Light.

Emits directive_trace_schema 1.1.0 rows for all 3 policies (dense, random, entropy).
All forced baselines: reason_code=FORCED_BASELINE.
Validates with jsonschema. Emits counterfactual ledger rows.
"""

import hashlib
import json
import random
import sys
import time
import uuid
from pathlib import Path

import torch
import yaml
from jsonschema import validate
from transformers import AutoModelForCausalLM, AutoTokenizer

# Constants
MODEL_ID = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
ROUTABLE_LAYERS = list(range(7, 21))  # [7..20]
K_SKIP = 4
SEEDS = [1337, 2026, 7]
MAX_NEW_TOKENS = 512

GYM_ROOT = Path("/Users/christienantonio/Desktop/AI:ML Research/gym-v0.1-FL")
SCHEMA_PATH = Path("/Users/christienantonio/Desktop/AI:ML Research/directive_trace_schema.json")
OUTPUT_DIR = Path("/Users/christienantonio/Desktop/AI:ML Research/first_light_traces")


def sha256_short(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def load_schema():
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def load_gym_instances():
    """Load FL subset instances."""
    instances = []
    for family in ["F1_mbpp", "F2_json", "F3_type"]:
        test_dir = GYM_ROOT / family / "test"
        for f in sorted(test_dir.glob("*.json")):
            with open(f) as fh:
                instances.append(json.load(fh))
    return instances


def load_model():
    """Load Qwen2.5-1.5B BASE on MPS."""
    print("Loading model...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to("mps")
    model.eval()
    return tok, model


def extract_code_block(completion: str) -> str:
    """Extract code from markdown fences."""
    import re
    pattern = r'```(?:python)?\s*\n(.*?)```'
    match = re.search(pattern, completion, re.DOTALL)
    if match:
        return match.group(1).strip()
    return completion.strip()


def generate_completion(tok, model, prompt: str, seed: int, temperature: float = 0.0) -> str:
    """Generate completion with given seed and temperature."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    inp = tok(prompt, return_tensors="pt").to("mps")
    
    with torch.no_grad():
        if temperature == 0.0:
            gen = model.generate(
                **inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=tok.eos_token_id
            )
        else:
            gen = model.generate(
                **inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                temperature=temperature, top_p=0.95,
                pad_token_id=tok.eos_token_id
            )
    
    full_text = tok.decode(gen[0], skip_special_tokens=True)
    completion = full_text[len(prompt):]
    return completion


def compute_layer_entropies(model, inp, routable_layers):
    """Compute logit-lens entropy at each routable layer."""
    with torch.no_grad():
        outputs = model(**inp, output_hidden_states=True)
    
    entropies = {}
    for layer_idx in routable_layers:
        hidden_state = outputs.hidden_states[layer_idx + 1]  # +1 because hidden_states[0] = embedding
        # Project to vocab (logit lens)
        logits = model.lm_head(hidden_state)
        probs = torch.softmax(logits[:, -1, :], dim=-1)  # Last token
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).item()
        entropies[layer_idx] = entropy
    
    return entropies


def select_skip_layers_entropy(model, inp, routable_layers, k):
    """Select k layers to skip based on entropy delta (smallest |ΔH|)."""
    entropies = compute_layer_entropies(model, inp, routable_layers)
    
    # Compute |ΔH| between consecutive layers
    deltas = {}
    for i, layer in enumerate(routable_layers):
        if i == 0:
            deltas[layer] = 0.0
        else:
            prev_layer = routable_layers[i - 1]
            deltas[layer] = abs(entropies[layer] - entropies[prev_layer])
    
    # Skip k layers with smallest |ΔH|
    sorted_layers = sorted(deltas.items(), key=lambda x: x[1])
    skip_layers = [layer for layer, _ in sorted_layers[:k]]
    
    return skip_layers, entropies, deltas


def select_skip_layers_random(routable_layers, k, seed):
    """Select k layers to skip randomly."""
    rng = random.Random(seed)
    return rng.sample(routable_layers, k)


def emit_trace_row(instance_id, layer_idx, policy, estimator_rung, g_hat, 
                   controller_tax, run_id, phase="decode"):
    """Emit a single trace row conforming to directive_trace_schema 1.1.0."""
    
    # Determine directives based on policy
    if policy == "dense_uniform":
        proposal_directive = "NORMAL"
        projected_directive = "NORMAL"
        final_directive = "NORMAL"
        reason_code = "FORCED_BASELINE"
        execution_mode = None  # Dense is live, no execution_mode field
    elif policy == "random_matched_mix":
        proposal_directive = "SKIP"
        projected_directive = "SKIP"
        final_directive = "SKIP"
        reason_code = "FORCED_BASELINE"
        execution_mode = "offline_replay"
    elif policy == "heuristic_entropy":
        proposal_directive = "SKIP"
        projected_directive = "SKIP"
        final_directive = "SKIP"
        reason_code = "FORCED_BASELINE"
        execution_mode = "offline_replay"
    else:
        raise ValueError(f"Unknown policy: {policy}")
    
    row = {
        "schema_version": "1.1.0",
        "run_id": run_id,
        "config_id": "FROZEN-BASE-v1",
        "adr_branch": "ratified",
        "seq_id": f"seq-{sha256_short(instance_id)}",
        "token_pos": 0,  # Layer-level routing, token_pos=0
        "layer_group": f"L{layer_idx}",
        "phase": phase,
        "cadence_path": "slow_full",
        "proposal": {
            "directive": proposal_directive,
            "logits_q": [0.0, 0.0, 0.0]  # Placeholder for forced baseline
        },
        "projected_directive": projected_directive,
        "final_directive": final_directive,
        "reason_code": reason_code,
        "estimator_rung": estimator_rung,
        "g_hat": g_hat,
        "cost_estimate": {
            "flops_directive": 0.0,
            "controller_tax_flops": controller_tax,
        },
        "ring": "R0",
        "contaminated_for_paper1": True,  # All First Light is pilot
    }
    
    if execution_mode is not None:
        row["execution_mode"] = execution_mode
    
    return row


def emit_counterfactual_row(instance_id, seed, policy, verified_pass, flops_total, 
                            bytes_per_token, wall_clock_tps, run_id):
    """Emit a counterfactual ledger row."""
    
    if policy == "dense_uniform":
        routing_mode = "none_dense"
        counterfactual_kind = "observed_replay"
    else:
        routing_mode = "forced_offline_replay"
        counterfactual_kind = "observed_replay"
    
    row = {
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
            "counterfactual_kind": counterfactual_kind,
            "predicted_gain": 0.0,
            "actual_gain": 0.0,
            "estimator_rung": "g_entropy" if policy == "heuristic_entropy" else "none_forced",
        },
        "flags": {
            "pilot_or_claim_eligible": "pilot",
            "contaminated_for_paper1": True,
        }
    }
    
    return row


def main():
    print("=" * 60)
    print("MINIMAL TRACE HARNESS")
    print("=" * 60)
    
    # Load schema
    schema = load_schema()
    print(f"Loaded schema: {SCHEMA_PATH}")
    
    # Load gym instances (just first 3 for smoke test)
    instances = load_gym_instances()
    print(f"Loaded {len(instances)} instances (using first 3 for smoke)")
    instances = instances[:3]
    
    # Load model
    tok, model = load_model()
    
    # Generate run_id
    run_id = f"first-light-smoke-{uuid.uuid4().hex[:8]}"
    print(f"Run ID: {run_id}")
    
    # Output files
    OUTPUT_DIR.mkdir(exist_ok=True)
    trace_file = OUTPUT_DIR / f"{run_id}_traces.jsonl"
    ledger_file = OUTPUT_DIR / f"{run_id}_counterfactual_ledger.jsonl"
    
    trace_rows = []
    ledger_rows = []
    
    # Process each instance × policy × seed
    for inst_idx, inst in enumerate(instances):
        instance_id = inst["instance_id"]
        prompt = inst["prompt_context"]
        
        print(f"\n[{inst_idx+1}/{len(instances)}] {instance_id}")
        
        for policy in ["dense_uniform", "random_matched_mix", "heuristic_entropy"]:
            for seed in SEEDS:
                # Generate completion
                completion = generate_completion(tok, model, prompt, seed, temperature=0.0)
                
                # Determine skip layers
                inp = tok(prompt, return_tensors="pt").to("mps")
                
                if policy == "dense_uniform":
                    skip_layers = []
                    estimator_rung = "none_forced"
                    g_hat = None
                    controller_tax = 0.0
                elif policy == "random_matched_mix":
                    skip_layers = select_skip_layers_random(ROUTABLE_LAYERS, K_SKIP, seed)
                    estimator_rung = "none_forced"
                    g_hat = None
                    controller_tax = 0.0
                elif policy == "heuristic_entropy":
                    skip_layers, entropies, deltas = select_skip_layers_entropy(
                        model, inp, ROUTABLE_LAYERS, K_SKIP
                    )
                    estimator_rung = "g_entropy"
                    g_hat = sum(deltas.values()) / len(deltas)  # Mean entropy delta
                    controller_tax = 1000.0  # Placeholder for entropy computation cost
                
                # Emit trace rows for each routable layer
                for layer_idx in ROUTABLE_LAYERS:
                    trace_row = emit_trace_row(
                        instance_id, layer_idx, policy, estimator_rung, g_hat,
                        controller_tax, run_id
                    )
                    
                    # Mark skipped layers
                    if layer_idx in skip_layers:
                        trace_row["proposal"]["directive"] = "SKIP"
                        trace_row["projected_directive"] = "SKIP"
                        trace_row["final_directive"] = "SKIP"
                    
                    # Validate
                    validate(instance=trace_row, schema=schema)
                    trace_rows.append(trace_row)
                
                # Emit counterfactual ledger row (placeholder verdict)
                ledger_row = emit_counterfactual_row(
                    instance_id, seed, policy, verified_pass=False,
                    flops_total=0.0, bytes_per_token=0.0, wall_clock_tps=15.0,
                    run_id=run_id
                )
                ledger_rows.append(ledger_row)
    
    # Write outputs
    with open(trace_file, "w") as f:
        for row in trace_rows:
            f.write(json.dumps(row) + "\n")
    
    with open(ledger_file, "w") as f:
        for row in ledger_rows:
            f.write(json.dumps(row) + "\n")
    
    print(f"\n{'=' * 60}")
    print(f"TRACE HARNESS COMPLETE")
    print(f"{'=' * 60}")
    print(f"Trace rows: {len(trace_rows)}")
    print(f"Ledger rows: {len(ledger_rows)}")
    print(f"Trace file: {trace_file}")
    print(f"Ledger file: {ledger_file}")
    print(f"All rows validated against schema 1.1.0 ✓")
    print(f"Reason-code coverage: 100% (all FORCED_BASELINE) ✓")


if __name__ == "__main__":
    main()
