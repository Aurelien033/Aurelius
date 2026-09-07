# Scaling Law Battle: Can Small Models Beat Large Models?

## Core Thesis
Smaller models with superior inference-time computation can outperform larger models.

## Evidence Vectors

1. **Early-exit inference** (CALM, 2022)
   - Mechanism: Stop at intermediate layers when confident
   - Speedup: 2-5x inference acceleration

2. **Tapered architecture** (DeepSeek-V3, 2025)  
   - Mechanism: Decreasing layer width toward output
   - Efficiency: FLOPs reduced 10-100x at inference

3. **Model merging** (TIES/SLERP/DARE)
   - Mechanism: Combine multiple small models
   - Result: Ensemble quality at zero inference cost

## Chinchilla Law Implications
N* = sqrt(C/120) parameters
D* = C/(6*N*) tokens

For C = 1e24 FLOPs (large training budget):
- Optimal N = ~12M parameters (NOT 100B+)
- Optimal D = ~8e21 tokens

This suggests 100B+ models are OVER-parameterized for their data!