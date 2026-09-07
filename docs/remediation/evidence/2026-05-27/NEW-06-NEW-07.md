# Evidence Manifest — NEW-06 / NEW-07: DPO/GRPO Trainer Crash Triage

## Row Metadata
- **Row IDs**: NEW-06, NEW-07
- **Previous status**: NOT_RECHECKED
- **New status**: NOT_REPRODUCED (tests pass)

## Validation
```text
.venv/bin/python -m pytest tests/training/test_dpo_trainer.py tests/training/test_grpo_trainer.py -q
# all passed
```

## Notes
CODE_REVIEW_RESCAN crash claims appear stale for synthetic-batch paths; no code change required at this time.
