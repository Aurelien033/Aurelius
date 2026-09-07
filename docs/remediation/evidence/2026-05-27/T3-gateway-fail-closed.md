# Evidence Manifest — T3: Gateway Fail-Closed Perimeter

## Rows Covered
- **C-03**: Unauthenticated Prometheus metrics
- **C-04**: Rate limiter fail-open when uninitialized
- **M-06**: Public /metrics information disclosure

## Previous Status
OPEN → **REGRESSION_LOCKED / FIXED** (production already fail-closed; tests completed)

## Production Behavior (`gateway/aurelius_api.py`)
1. **C-04**: `rate_limit` middleware returns 503 when `_rate_limiter is None` (fail-closed)
2. **C-03/M-06**: `/metrics` requires `AURELIUS_METRICS_API_KEY` env and matching `X-API-Key` header; returns 503 if unconfigured, 401 if wrong/missing

## Test Changes
- Fixed skipped positive control in `tests/gateway/test_t3_gateway_fail_closed.py::test_metrics_with_valid_auth`

## Validation
```text
.venv/bin/python -m pytest tests/gateway/test_t3_gateway_fail_closed.py -q
....  [4 passed]
```

## Files Modified
- `tests/gateway/test_t3_gateway_fail_closed.py` (enabled positive metrics auth test)

## Remaining Risk
- Workspace/session endpoints on gateway still lack API key middleware (separate rows: C-24 marked FIXED in register)
