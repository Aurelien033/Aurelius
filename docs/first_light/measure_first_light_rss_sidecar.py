#!/usr/bin/env python3
"""Sidecar RSS/MPS memory measurement for First Light FROZEN-BASE-v1.

Why this exists: v3's in-run RSS log measured the "empty" baseline after model
load, and macOS/MPS current RSS is not a clean model-weight accounting surface.
This sidecar records current RSS, peak RSS, USS when available, and torch.mps
allocator bytes at controlled milestones with the same pinned model revision.
"""
from __future__ import annotations

import json
import os
import resource
import time
from pathlib import Path
from typing import Any

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_REPO = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
OUT = Path("/Users/christienantonio/Desktop/AI:ML Research/first_light_receipt/first-light-3651bb11/rss_sidecar_measurement.json")


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def uss_mb() -> float | None:
    try:
        return psutil.Process(os.getpid()).memory_full_info().uss / (1024 * 1024)
    except Exception:
        return None


def peak_rss_mb() -> float:
    # macOS ru_maxrss is bytes; Linux is KiB. Host is macOS per system prompt.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def mps_memory_mb() -> dict[str, float | None]:
    out: dict[str, float | None] = {"current_allocated_mb": None, "driver_allocated_mb": None}
    if hasattr(torch, "mps"):
        try:
            out["current_allocated_mb"] = torch.mps.current_allocated_memory() / (1024 * 1024)
        except Exception:
            pass
        try:
            out["driver_allocated_mb"] = torch.mps.driver_allocated_memory() / (1024 * 1024)
        except Exception:
            pass
    return out


def snap(event: str, method: str) -> dict[str, Any]:
    row = {
        "event": event,
        "timestamp": time.time(),
        "rss_mb": rss_mb(),
        "peak_rss_mb": peak_rss_mb(),
        "uss_mb": uss_mb(),
        "mps_memory_mb": mps_memory_mb(),
        "method": method,
    }
    print(json.dumps(row, indent=2), flush=True)
    return row


def main() -> None:
    rows = []
    rows.append(snap("process_start_after_imports", "psutil RSS/USS + resource ru_maxrss + torch.mps allocator"))
    tok = AutoTokenizer.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)
    rows.append(snap("tokenizer_loaded", "same process, after pinned tokenizer load"))
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("mps").eval()
    rows.append(snap("model_loaded_to_mps", "pinned Qwen2.5-1.5B base loaded to MPS"))
    inp = tok("First Light RSS sidecar", return_tensors="pt").to(model.device)
    with torch.no_grad():
        _ = model(**inp)
    rows.append(snap("model_touched_one_forward", "one forward pass to materialize lazy MPS allocations"))

    empty = rows[0]
    final = rows[-1]
    peak_delta = max(r["peak_rss_mb"] for r in rows) - empty["peak_rss_mb"]
    current_delta = final["rss_mb"] - empty["rss_mb"]
    mps_driver = final["mps_memory_mb"].get("driver_allocated_mb")
    result = {
        "run_id": "first-light-3651bb11",
        "measurement_class": "rss_mps_sidecar_repair",
        "why_sidecar": "v3 runner recorded empty baseline after load; macOS/MPS current RSS is not a direct model-weight accounting surface, so this records current RSS, peak RSS, USS, and torch.mps allocator bytes with the same pins",
        "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION, "dtype": "bfloat16", "device": "mps"},
        "rows": rows,
        "empty_process_baseline_current_rss_mb": empty["rss_mb"],
        "total_process_final_current_rss_mb": final["rss_mb"],
        "model_attributable_current_rss_delta_mb": current_delta,
        "model_attributable_peak_rss_delta_mb": peak_delta,
        "mps_driver_allocated_mb_after_touch": mps_driver,
        "total_process_current_rss_gb": final["rss_mb"] / 1024,
        "model_attributable_peak_rss_delta_gb": peak_delta / 1024,
        "mps_driver_allocated_gb_after_touch": (mps_driver / 1024) if mps_driver is not None else None,
        "gates_on_observed_process_rss": {
            "total_process_current_rss_lt_10gb": final["rss_mb"] / 1024 < 10.0,
            "peak_rss_delta_le_7_5gb": peak_delta / 1024 <= 7.5,
        },
        "interpretation": "For macOS/MPS, use total current RSS and peak RSS delta as the RSS gate evidence; use mps_driver_allocated for model/device memory context, not as RSS.",
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
