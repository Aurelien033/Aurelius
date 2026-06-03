# Evidence Manifest — NEW-04: Shard Diversity Audit

## Status: PARTIAL (tooling ready; no local shards)

## Tool
`scripts/audit_tokenized_shards.py` — samples `shard_*.npy` files and asserts unique-row ratio ≥ 0.9 per shard.

## Run
```bash
python scripts/audit_tokenized_shards.py /path/to/tokenized/shards \
  --output docs/remediation/evidence/2026-05-27/NEW-04-shard-audit.json
```

## Local result (2026-05-27)
```json
{
  "shard_dir": "training_data/tokenized",
  "status": "SKIP",
  "reason": "no shard_*.npy files found"
}
```

## Residual
Re-run audit on production/training shard directories after any historical `n_workers>1` tokenization. Failures require re-tokenization per runbook.
