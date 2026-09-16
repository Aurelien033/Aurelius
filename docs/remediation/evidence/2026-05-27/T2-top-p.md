# Evidence Manifest — T2: Top-p Single Helper

## Rows Covered
- **NEW-05**: Top-p helper consolidation
- **C-17**: generate() top-p correctness
- **C-18**: generate_stream() top-p correctness

## Previous Status
OPEN → **REGRESSION_LOCKED** (shared `_apply_top_p_filter` already in production; evidence tests pass)

## Production Behavior
`src/model/transformer.py`:
- Single `_apply_top_p_filter()` helper used by both `generate()` and `generate_stream()`
- Boundary uses `cumulative_probs - sorted_probs >= top_p` (HuggingFace-compatible nucleus)

## Validation
```text
.venv/bin/python -m pytest tests/model/test_t2_top_p_evidence.py tests/model/test_top_p_correctness.py -q
............  [12 passed]
```

Static analysis tests confirm no inline duplicated top-p logic in generate paths.

## Files Modified
None this session.

## Remaining Risk
- Other modules (inference/, speculative/) may have separate top-p implementations (out of T2 scope)
