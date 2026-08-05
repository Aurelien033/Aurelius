"""Model adapter for the alignment release gate.

Turns a model (local HF weights) or an OpenAI-compatible API into the
`generate(prompt) -> str` callable that `run_release_gate` expects. Unlike the
data-generation teacher, the eval adapter is UNCONSTRAINED by the firewall — you
are *measuring* a model's alignment, not ingesting its outputs into a release,
so evaluating any model (including closed ones) is fine.

Two backends:
  make_hf_generate(model_dir)  — local transformers weights (lazy import; the
                                 same loading pattern as eval_code_bench)
  make_api_generate(model)     — OpenAI-compatible chat endpoint

Both accept greedy decoding by default with a modest token budget (these are
short forced-choice answers). Import is torch-free so the module (and tests)
load without a GPU.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Callable

Generate = Callable[[str], str]


def make_api_generate(
    model: str,
    api_base: str = "https://openrouter.ai/api/v1",
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> Generate:
    """Return generate(prompt)->str backed by an OpenAI-compatible endpoint."""
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    url = api_base.rstrip("/") + "/chat/completions"

    def generate(prompt: str) -> str:
        if not key:
            raise RuntimeError("set OPENROUTER_API_KEY (or OPENAI_API_KEY)")
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "max_tokens": max_tokens,
        }).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]

    return generate


def make_hf_generate(
    model_dir: str,
    max_new_tokens: int = 512,
    think: int = 0,
    device: str = "cuda",
    load_4bit: bool = False,
) -> Generate:
    """Return generate(prompt)->str backed by local transformers weights.

    Mirrors eval_code_bench: greedy decode, optional Qwen3 thinking flag off by
    default (these prompts want a short 'FINAL: X', not a CoT dump). Heavy imports
    are deferred to call time so this module stays importable without torch.
    """
    import torch  # noqa: F401  (deferred)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw: dict = {"trust_remote_code": True, "dtype": __import__("torch").bfloat16}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=__import__("torch").float16)
        kw["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(model_dir, **kw)
    if not load_4bit:
        model = model.to(device)
    model.eval()

    def generate(prompt: str) -> str:
        import torch
        msgs = [{"role": "user", "content": prompt}]
        kwargs = {}
        try:  # Qwen3 thinking toggle; ignored by templates that don't support it
            kwargs["enable_thinking"] = bool(think)
        except Exception:
            pass
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", **kwargs).to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=tok.pad_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    return generate


def make_generate(
    model: str | None = None,
    api: bool = False,
    **kw,
) -> Generate:
    """Convenience factory: api=True -> API backend, else local HF weights."""
    if model is None:
        raise ValueError("provide a model id (API) or model dir (local)")
    return make_api_generate(model, **kw) if api else make_hf_generate(model, **kw)
