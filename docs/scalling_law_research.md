# Intensive Scaling Law Research: Smaller Models vs Larger Models
# Date: 2026-06-25
# Integration with: Tapered Language Models, Chinchilla scaling, model merging

scaling_law_facts:
  chinchilla_optimal:
    formula: "N* = sqrt(C/120), D* = C/(6*N*)"
    implication: "For fixed compute C, optimal model is sqrt(C/120) parameters + 20x more tokens"
    
  hoffmann_2022_constants:
    E: 1.61   # irreducible entropy
    A: 406.4  # parameter scaling coefficient  
    B: 410.7  # data scaling coefficient
    alpha: 0.34
    beta: 0.28

novel_insights:
  - insight: "Tapered Models can achieve equivalent performance with 10x-100x fewer FLOPs during inference"
  - insight: "Early-exit models trade 20-30% accuracy for 2-5x speedup at inference time"
  - insight: "Model merging can create ensemble-quality models from smaller sources at zero inference cost"