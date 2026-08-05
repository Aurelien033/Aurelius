# Scaling Law Battle: Small vs Large Models
# Key Question: Can <10B models beat 100B+ models in practice?

analysis:
  size_is_not_fate:
    thesis: "Smaller models with superior inference-time computation can outperform larger models"
    
  evidence_vectors:
    - vector: "Early-exit inference"
      source: "Schuster et al. CALM (NeurIPS 2022)"
      mechanism: "Stop computation at intermediate layers when confident"
      speedup: "2-5x inference acceleration"
      
    - vector: "Tapered architecture"
      source: "DeepSeek-V3 technical report, 2025"
      mechanism: "Decreasing layer width toward output"
      efficiency: "FLOPs reduced 10-100x at inference time"
      
    - vector: "Model merging ensembles"
      source: "TAU & Google merging papers (2023-2024)"
    - vector: "TIES, SLERP, DARE methods in model_merging.py"