#!/usr/bin/env python3
"""Robust text-generation model loader — handles plain CausalLM AND multimodal (VLM) teachers.

Some strong teachers (e.g. Qwen/Qwen3.6-27B) are registered as image-text-to-text, so a bare
AutoModelForCausalLM.from_pretrained raises "Unrecognized configuration class". This tries the text-gen
Auto classes in order and returns the first that loads. Text-only generation works the same on all of them.
"""
import torch


def load_causal_lm(name, load_4bit=False, dev="cuda"):
    """Returns (model.eval(), loader_class_name). 4-bit uses device_map='auto' (don't .to(dev))."""
    import transformers
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kw = dict(quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                  bnb_4bit_compute_dtype=torch.bfloat16), device_map="auto")
    else:
        kw = dict(dtype=torch.bfloat16)
    last = None
    for cls in ["AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModel"]:
        Loader = getattr(transformers, cls, None)
        if Loader is None:
            continue
        try:
            m = Loader.from_pretrained(name, trust_remote_code=True, **kw)
            return (m if load_4bit else m.to(dev)).eval(), cls
        except Exception as e:        # wrong Auto class -> try the next; keep the last error if all fail
            last = e
    raise RuntimeError(f"could not load {name} with any text-gen Auto class: {last}")
