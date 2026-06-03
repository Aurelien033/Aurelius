"""Tests for Ring 1 DreamBank runner (Tranche 3)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from src.eval.ring1_dreambank_runner import (
    ZeroGradProof,
    extract_seeds_from_traces,
    hash_model_params,
    run_sleep_cycles,
    write_dreambank_artifacts,
)
from src.memory.hlm_bank import HLMPreferenceBank, HLMPreferenceBankConfig


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight


def test_extract_seeds_from_traces() -> None:
    traces = [
        {
            "trace_id": "abc",
            "steps": [
                {"step_id": 1, "observation": "Question?", "reflection": {"summary": "ok"}},
            ],
        }
    ]
    seeds = extract_seeds_from_traces(traces, max_seeds=10)
    assert len(seeds) == 2
    assert seeds[0].prompt == "Question?"


def test_zero_grad_proof_unchanged_with_model() -> None:
    model = _TinyModel()
    bank = HLMPreferenceBank(HLMPreferenceBankConfig(bank_size=4, bank_dim=8))
    from src.alignment.dreambank import DreamSeed

    result = run_sleep_cycles(
        [DreamSeed(prompt="test prompt")],
        bank=bank,
        num_cycles=1,
        model=model,
    )
    assert result.zero_grad_proof is not None
    assert result.zero_grad_proof.param_hash_unchanged
    assert result.total_writes >= 0


def test_hash_model_params_stable() -> None:
    model = _TinyModel()
    assert hash_model_params(model) == hash_model_params(model)


def test_write_dreambank_artifacts() -> None:
    proof = ZeroGradProof(
        pre_cycle_param_hash="aaa",
        post_cycle_param_hash="aaa",
        param_hash_unchanged=True,
        inference_mode=True,
    )
    from src.alignment.dreambank import DreamCycleResult

    result_type = DreamCycleResult
    with tempfile.TemporaryDirectory() as tmpdir:
        from src.eval.ring1_dreambank_runner import DreamBankRunResult

        result = DreamBankRunResult(
            cycles=[
                result_type(
                    seeds=1,
                    candidates=4,
                    pairs=1,
                    writes=1,
                    mean_margin=0.1,
                    bank_fill=1,
                )
            ],
            total_writes=1,
            bank_fill=1,
            sleep_cycle_log=[{"cycle": 0, "writes": 1}],
            zero_grad_proof=proof,
        )
        out = Path(tmpdir)
        write_dreambank_artifacts(result, out)
        assert (out / "pre_cycle_model_hash.txt").exists()
        assert (out / "post_cycle_model_hash.txt").exists()
        payload = json.loads((out / "dreambank_result.json").read_text(encoding="utf-8"))
        assert payload["zero_grad_proof"]["param_hash_unchanged"] is True
