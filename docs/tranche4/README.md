# Ring 1 Tranche 4 — Gate Verification & Scale Runs

Tranche 4 scales to **200+ traces per condition** and runs automated **R1-GA / R1-GB / R1-GC** verification. Official gate pass still requires maintainer sign-off per the Gate Verification Procedures v0.1.

## One-command workflow

```bash
python scripts/ring1_tranche4_workflow.py --config configs/ring1_tranche4.yaml
```

Full verification:

```bash
bash verification/verify_ring1_gates.sh
```

## What it does

1. Collect **200 AMC traces** (`Ring1Agent`) → `data/ring1_traces/tranche4/amc/`
2. Collect **200 no-memory traces** (`Ring1DummyAgent`) → `data/ring1_traces/tranche4/no_memory/`
3. Run **DreamBank sleep** on AMC traces + shuffled control
4. Run **dual-condition eval harness** with bootstrap CI
5. Build **reproducibility pack**
6. Emit **`gate_verification_report.json`** with step-by-step pass/fail evidence

## Gate thresholds (configurable)

| Gate | Key check |
|------|-----------|
| **R1-GA** | ≥200 traces/condition, memory promote+reuse audit, ≥5pp holdout lift |
| **R1-GB** | Zero-grad hash proof, DreamBank lift ≥1pp vs shuffled control |
| **R1-GC** | Repro pack validates, traces + hashes included |

## Output

```
docs/reproducibility/ring1_tranche4/
  gate_verification_report.json
  workflow_summary.json
  dreambank_run/
  metrics/
  ring1-repro-*.tar.gz
```

## Tranche 4 scope

- New: `ring1_gate_verifier.py`, `ring1_tranche4_workflow.py`, `verify_ring1_gates.sh`
- Does **not** modify core AMC, DreamBank, or model training code
