# Evidence Manifest — R1-SoftMoE: SoftMoELayer missing _softmax

## Row Metadata
- **Row ID**: R1-SoftMoE
- **Title**: SoftMoELayer missing _softmax
- **Severity / Theme**: High / E-Model
- **Primary file**: `src/model/moe.py`
- **Previous status**: OPEN
- **New status**: PARTIAL

## Problem Statement
`SoftMoELayer.forward()` at line 465 called `TopKRouter._softmax(logits)` — a static method that does not exist on `TopKRouter`. Runtime crash.

## Fix Applied
Replaced `TopKRouter._softmax()` calls with an inline `_mock_softmax()` function using `math.exp` — numerically stable softmax over lists.

## What Was NOT Fixed
`SoftMoELayer` is an incomplete mock (list-of-lists interface calling torch-based `ExpertFFN.forward()`). The full class needs a proper torch-tensor implementation before it can be used in training. This is a deeper refactor beyond the scope of this tranche.

## Validation
```text
.venv/bin/python -m pytest tests/model/test_moe.py tests/model/test_moe_router.py
# 37 passed (all existing MoE tests unaffected)
# SoftMoELayer.forward() gets past softmax without error
# Remaining crash: ExpertFFN expects torch.Tensor, not list — deferred
```

## Files Modified
- `src/model/moe.py`: Replaced `TopKRouter._softmax()` with `_mock_softmax()` in `SoftMoELayer.forward()`
