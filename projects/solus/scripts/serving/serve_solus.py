#!/usr/bin/env python
"""Solus-7B — vLLM-based inference server (OpenAI-compatible API).

Run:
    python scripts/serving/serve_solus.py \
            --checkpoint checkpoints/final/ \
            --port 8000

Then talk to it:
    curl http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"solus","messages":[{"role":"user","content":"Hello!"}],"max_tokens":256}'
"""
from __future__ import annotations
import argparse
import asyncio
from fastapi import FastAPI
from uvicorn import Config, Server
import torch
from pathlib import Path

try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    from src.modeling.model import SolusForCausalLM
    from src.serving.fallback_server import create_fastapi_app
    print("  ! vLLM not installed — using PyTorch fallback server (slower)")

from src.modeling.config import SolusConfig


def serve(checkpoint: str, port: int = 8000):
    if not HAS_VLLM:
        raise ImportError("Install vLLM: pip install vllm>=0.4.0")

    print(f"\n  Loading Solus-7B from {checkpoint} ...")
    llm = LLM(
        model=checkpoint,
        tokenizer=checkpoint,
        dtype="auto",
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.90,
        max_num_seqs=128,
    )
    print(f"  ✓ Loaded onto {torch.cuda.device_count()} GPU(s)")

    app = FastAPI(title="Solus-7B Inference API")

    @app.post("/v1/chat/completions")
    async def chat(request: dict):
        messages = request.get("messages", [])
        max_tokens  = request.get("max_tokens", 256)
        temperature = max(request.get("temperature", 0.7), 0.0)
        top_p       = request.get("top_p", 0.9)
        top_k       = request.get("top_k", 50)

        prompt = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in messages
        ) + "\nAssistant:"

        sp = SamplingParams(
            temperature=temperature if not temperature == 0 else 0,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
        )
        [output] = llm.generate([prompt], sp)
        return {
            "id":       output.request_id,
            "choices": [{"message": {"role": "assistant", "content": output.outputs[0].text}}],
            "usage":    {"completion_tokens": len(output.outputs[0].token_ids)},
        }

    config = Config(app=app, host="0.0.0.0", port=port, log_level="info")
    server = Server(config)
    asyncio.run(server.serve())


if __name__ == "__main__":
    pa = argparse.ArgumentParser("Solus-7B vLLM inference server")
    pa.add_argument("--checkpoint", type=str, required=True)
    pa.add_argument("--port", type=int, default=8000)
    args = pa.parse_args()
    print("\n{'─'*50}")
    print("  SOLUS-7B — vLLM Inference Server")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Port       : {args.port}")
    print(f"{'─'*50}\n")
    serve(args.checkpoint, args.port)
