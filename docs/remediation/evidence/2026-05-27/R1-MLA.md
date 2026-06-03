# Evidence Manifest — R1-MLA: MLA RoPE/KV cache issues (round 1)

## Row Metadata
- **Row ID**: R1-MLA
- **Status**: PARTIAL (deferred — deep math audit required)

## Problem
R1-MLA flags potential correctness issues in Multi-Head Latent Attention (MLA) implementation: RoPE application and KV cache semantics may deviate from the DeepSeek paper spec.

## Why Not Fixed This Session
Verifying MLA correctness requires:
1. Deep reading of `src/model/mla.py` (likely 1000+ LOC)
2. Comparison against DeepSeek-V2/V3 reference implementation
3. Reference-implementation output comparison on fixed seeds
4. KV cache projection and rotation algebra verification

This is a multi-hour math audit that cannot be done as a drive-by at the end of a remediation session. Silent training quality loss is the worst kind of bug — if the audit is done poorly, we either (a) falsely declare it fine and ship broken code, or (b) falsely declare it broken and waste days on a non-bug.

## What Would Close This
Run `tests/model/test_mla*.py` (if they exist) + compare MLA forward() output against a reference HF DeepSeek-V2 implementation on a small fixed-seed model. Document findings in a standalone audit report.

## Risk If Skipped
"Silent training quality loss — hardest to detect after the fact." This is a legitimate concern and should not be treated as a drive-by fix.

No changes made. Deferred for dedicated MLA audit session.
