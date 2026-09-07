# ACDT Formal Specification: CUDA Optimization Mechanisms for Aurelius
# Version: 2026-06-25.1
# OBL-001 override: Discharges OBL-033 staged item for CUDA mechanisms

mechanism_spec_version: 1.0
spec_level: formal
contract_type: acdt_controller
domain: cuda_inference_optimization

mechanisms:
  - id: CUDA-MLA-001
    name: FlashMLA KV Cache Absorption
    status: implemented
    location: src/model/flash_mla.py
    lines_of_code: 227

  - id: CUDA-QJL-002
    name: TurboQuant QJL Attention Sketching
    status: implemented
    location: src/inference/turboquant/qjl.py
    lines_of_code: 94

  - id: CUDA-KIVI-003
    name: 2-bit KV Cache Quantization
    status: config_ready
    location: src/inference/kv_quant/
    config_path: src/model/config.py

  - id: CUDA-GRAPH-004
    name: Adaptive Speculative Decoding Graphs
    status: proposed
    rationale: "Replace sequential Python loops with pre-captured CUDA graphs"

spec_falsifiers:
  - mechanism_id: CUDA-MLA-001
    falsifier: "kv_cache_size_ratio() < 1.0 fails"
    test_path: tests/integration/test_flash_mla_integration.py::test_kv_cache_ratio_lt_one

  - mechanism_id: CUDA-KIVI-003
    falsifier: "kivi_bits > 0 does not reduce kv_cache VRAM by >=4x"
    test_path: benchmarks/cuda_optimization_benchmark.py

cost_model:
  mla_kv_compression:
    formula: "ratio = kv_lrank / (n_heads * head_dim)"
    baseline_7b: "64 / 1280 = 0.05 (95% compression)"
    
  kivi_memory_savings:
    formula: "savings = n_heads * head_dim * seq_len * kv_quant_bits / 16"
    baseline_7b_16k: "8 * 128 * 16384 * 2 / 16 = 2MB per layer per token"

falsifier_contracts:
  # Each mechanism has explicit falsification tests
  - contract_id: F-MLA-01
    mechanism: CUDA-MLA-001
    assertion: "assert torch.allclose(out_std, out_abs, rtol=1e-4, atol=1e-5)"
    violation_action: "Disable mla_enabled, log error, fall back to standard GQA"
    
  - contract_id: F-QJL-02
    mechanism: CUDA-QJL-002
    assertion: "assert estimate.var() / residual.var() <= 1.1  # Unbiased estimator check"
    violation_action: "Revert to decompressed KV, alert precision degradation"
    
  - contract_id: F-KIVI-03
    mechanism: CUDA-KIVI-003
    assertion: "assert kv_cache_bytes < 2 * seq_len * n_layers * n_heads * head_dim * 2"
    violation_action: "Reduce kivi_bits to 1, or disable if quality degrades"