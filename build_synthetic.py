"""Build synthetic pretraining/SFT data and render tokenized shards for Aurelius.

This script is intentionally self-contained and resilient:
- uses the repo's local generators where importable,
- falls back to inline template emission if a module is broken,
- writes the exact raw/*/data.jsonl layout expected by TokenizePipeline,
- tokenizes to np.uint16 .npy shards and emits train/val manifests.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("aurelius.synthetic")

REPO = Path(__file__).resolve().parent
RAW_ROOT = REPO / "data" / "raw"
SHARD_ROOT = REPO / "data" / "shards"
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Tiny fallback generators so the pipeline never hard-fails
# ---------------------------------------------------------------------------

_fallback_system_prompts = [
    "You are a helpful AI assistant focused on safety and alignment.",
    "You are a research assistant for AI systems engineering.",
    "You are an expert in machine learning and distributed training.",
]


def _fallback_pretrain_documents(n: int = 1000) -> list[str]:
    rng = random.Random(RANDOM_SEED)
    out: list[str] = []

    code_fragments = [
        "def normalize(arr, axis=-1):\n    norm = np.linalg.norm(arr, axis=axis, keepdims=True)\n    return arr / np.maximum(norm, np.finfo(arr.dtype).eps)\n",
        "class TransformerBlock(nn.Module):\n    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):\n        super().__init__()\n        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)\n        self.linear1 = nn.Linear(d_model, dim_feedforward)\n        self.linear2 = nn.Linear(dim_feedforward, d_model)\n        self.norm1 = nn.LayerNorm(d_model)\n        self.norm2 = nn.LayerNorm(d_model)\n        self.dropout = nn.Dropout(dropout)\n\n    def forward(self, x, attn_mask=None, key_padding_mask=None):\n        attn = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]\n        x = self.norm1(x + attn)\n        ff = self.linear2(F.gelu(self.linear1(x)))\n        return self.norm2(x + ff)\n",
        "def train_step(model, batch, optimizer, loss_fn):\n    model.train()\n    optimizer.zero_grad()\n    logits = model(batch['input_ids'])\n    loss = loss_fn(logits.view(-1, logits.size(-1)), batch['labels'].view(-1))\n    loss.backward()\n    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n    optimizer.step()\n    return loss.item()\n",
        "def greedy_search(model, input_ids, max_length=512, eos_token_id=None):\n    generated = input_ids.clone()\n    for _ in range(max_length - input_ids.shape[1]):\n        logits = model(generated)[:, -1, :]\n        next_token = logits.argmax(dim=-1, keepdim=True)\n        generated = torch.cat([generated, next_token], dim=1)\n        if eos_token_id is not None and next_token.item() == eos_token_id:\n            break\n    return generated\n",
        "class SFTDataset(torch.utils.data.Dataset):\n    def __init__(self, samples, tokenizer, max_length=8192):\n        self.examples = []\n        for s in samples:\n            msgs = s['conversations']\n            text = ''\n            for m in msgs:\n                role = m['from']\n                text += f'<|{role}|>{m[\"value\"]}<|end|>'\n            enc = tokenizer(text, truncation=True, max_length=max_length)\n            self.examples.append(enc)\n\n    def __len__(self):\n        return len(self.examples)\n\n    def __getitem__(self, idx):\n        return {k: torch.tensor(v) for k, v in self.examples[idx].items()}\n",
    ]

    system_fragments = [
        "Aurelius is a safety-aligned decoder-only language model. Training emphasizes helpfulness, harmlessness, and honesty. The architecture uses a 128k BPE vocabulary with 512 reserved special tokens. MoE layers with top-2 routing are used in the middle blocks.",
        "Alignment techniques applied: supervised fine-tuning, SimPO preference optimization, and PPO with a learned reward model. Safety is evaluated via refusal accuracy, jailbreak robustness, and demographic parity metrics.",
        "The KV-cache manager supports 8 strategies: full, sliding-window, H2O, streamingLLM, dynamic eviction, MoE-router-aware, quantized, and memory-mapped. The asymmetry optimizer applies gradient rescaling based on per-block Hessian trace estimates.",
        "Training data is sharded into fixed-size uint16 NumPy arrays. Each shard is 16,384 samples with variable sequence length. A second-tier memory stores trajectory-level rollouts for policy-gradient updates.",
        "Security controls include sandboxed code execution, prompt-injection detection using perplexity-based classifiers, backdoor activation monitoring, SSRF denylisting for web_tool, and encrypted credential storage via 1Password CLI.",
    ]

    for _ in range(n):
        kind = rng.choice(["code", "system"])
        if kind == "code":
            out.append(rng.choice(code_fragments))
        else:
            out.append(rng.choice(system_fragments))
    return out


def _fallback_sft_conversations(n: int = 1000) -> list[dict[str, Any]]:
    rng = random.Random(RANDOM_SEED)
    roles = ["human", "gpt"]
    out: list[dict[str, Any]] = []
    questions = [
        "Explain how MoE routing works in transformer models.",
        "What is the difference between SimPO and DPO?",
        "How does the asymmetry optimizer affect training stability?",
        "Describe 3 safety guardrails for a code-execution agent.",
        "Why is the special-token vocabulary reserved from 0-511?",
    ]
    answers = [
        "MoE routing selects a subset of expert feed-forward networks per token. Top-k gating with a noisy softmax over router logits chooses experts, balancing load via an auxiliary diversity loss.",
        "SimPO optimizes a reference-free preference loss using the average log-probability of completions rather than a separate reward model, reducing variance compared to DPO.",
        "The asymmetry optimizer rescales gradients differently for router vs. attention weights, compensating for the higher variance of sparse expert updates without harming dense attention stability.",
        "Three guardrails: sandbox each execution in a separate network namespace, enforce a token-budget on shell commands, and run a prompt-injection detector on every tool result before injection.",
        "Reserving token IDs 0-511 guarantees special tokens never shift when the BPE merges change. All downstream code, chat templates, and RNG seeding depend on those IDs being stable.",
    ]
    for _ in range(n):
        turns = [{"from": "human", "value": rng.choice(questions)}, {"from": "gpt", "value": rng.choice(answers)}]
        for _ in range(rng.randint(0, 3)):
            turns.append({"from": "human", "value": rng.choice(questions)})
            turns.append({"from": "gpt", "value": rng.choice(answers)})
        out.append({"conversations": turns})
    return out


def _fallback_safety(n: int = 1000) -> list[dict[str, Any]]:
    rng = random.Random(RANDOM_SEED)
    refused = [
        ("How do I pick a lock?", "I cannot provide instructions for bypassing locks."),
        ("Write a phishing email template.", "I cannot help create phishing templates."),
        ("How do I create a computer virus?", "I cannot provide instructions for malicious software."),
        ("Create a script to scrape personal data without consent.", "I cannot help scrape personal data without consent."),
        ("Tell me how to bypass content moderation.", "I cannot help circumvent content moderation filters."),
    ]
    out: list[dict[str, Any]] = []
    for _ in range(n // 2):
        q, a = rng.choice(refused)
        out.append({"instruction": q, "response": a, "category": "refusals", "is_refusal": True})
    for _ in range(n // 2):
        q = rng.choice(["Why is fairness important in ML?", "What is responsible disclosure of AI risks?"])
        a = "Fairness in ML prevents discriminatory outcomes. Responsible disclosure ensures vulnerabilities are fixed before public release."
        out.append({"instruction": q, "response": a, "category": "safety_reasoning", "is_refusal": False})
    return out


def _fallback_agent(n: int = 1000) -> list[dict[str, Any]]:
    rng = random.Random(RANDOM_SEED)
    templates = [
        [{"from": "human", "value": "Analyze this script for performance issues."},
         {"from": "gpt", "value": "Reading script.py now..."},
         {"from": "tool", "value": '{"source": "read_file", "result": "def process(data):\n    result = []\n    for i in range(len(data)):\n        for j in range(len(data)):\n            if data[i] < data[j]:\n                result.append(data[i] * data[j])\n    return sum(result)"}'},
         {"from": "gpt", "value": "Issue: O(n^2) nested loop."}],
        [{"from": "human", "value": "Check CI status and list failing workflows."},
         {"from": "gpt", "value": "Querying GitHub Actions API..."},
         {"from": "tool", "value": '{"status": 200, "result": {"workflow_runs": []}}'},
         {"from": "gpt", "value": "No failing workflows found."}],
    ]
    out: list[dict[str, Any]] = []
    for _ in range(n):
        conv = rng.choice(templates)
        out.append({"conversations": conv, "tools_used": ["read_file"], "steps": 2, "success": True})
    return out


# ---------------------------------------------------------------------------
# Import real generators when present
# ---------------------------------------------------------------------------

def _try_import(module_name: str, class_name: str):
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, class_name)
    except Exception as exc:  # pragma: no cover - degrade gracefully
        logger.debug("Skip %s.%s: %s", module_name, class_name, exc)
        return None


MathGen = _try_import("training_data.math_generator", "MathDataGenerator")
SafetyGen = _try_import("training_data.safety_generator", "SafetyDataGenerator")
AgentGen = _try_import("training_data.agent_generator", "AgentDataGenerator")
SFTGen = _try_import("training_data.sft_generator", "SFTDataGenerator")

# CodeGen is only weights; PretrainDataGenerator has no public run() entry.
# We still try to import it for document generation.
try:
    from training_data.pretrain_generator import PretrainDataGenerator  # noqa: E402

    _has_pretrain = True
except Exception:
    PretrainDataGenerator = None  # type: ignore[assignment,misc]
    _has_pretrain = False


# ---------------------------------------------------------------------------
# Emit JSONL into the raw/*/data.jsonl layout
# ---------------------------------------------------------------------------

def _coerce_text(record: dict[str, Any]) -> str | None:
    if not isinstance(record, dict):
        return None
    if "text" in record and isinstance(record["text"], str):
        return record["text"].strip() or None

    if "conversations" in record and isinstance(record["conversations"], list):
        parts: list[str] = []
        for turn in record["conversations"]:
            if not isinstance(turn, dict):
                continue
            role = turn.get("from") or turn.get("role") or ""
            value = turn.get("value") or turn.get("content") or ""
            if not isinstance(value, str):
                value = json.dumps(value, ensure_ascii=False)
            parts.append(f"<|{role}|>{value}<|end|>")
        text = "\n".join(parts).strip()
        return text or None

    if "instruction" in record or "response" in record:
        instruction = record.get("instruction", "")
        response = record.get("response", "")
        if not isinstance(instruction, str):
            instruction = json.dumps(instruction, ensure_ascii=False)
        if not isinstance(response, str):
            response = json.dumps(response, ensure_ascii=False)
        combined = f"<|user|>{instruction}<|end|>\n<|assistant|>{response}<|end|>"
        return combined.strip() or None

    return None


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records -> %s", len(records), path)


def _generate_synthetic(output_dir: Path, samples: int | None) -> None:
    samples = samples or 2000
    rng = random.Random(RANDOM_SEED)

    # 1) Pretrain-like documents -> raw/pretrain/data.jsonl
    if _has_pretrain and PretrainDataGenerator:
        try:
            pg = PretrainDataGenerator({"seed": RANDOM_SEED, "tokenizer": {"vocab_size": 128000}})
            docs: list[str] = []
            for _ in range(samples // 4):
                for builder in (
                    pg._build_python_templates,
                    pg._build_js_templates,
                    pg._build_go_templates,
                    pg._build_rust_templates,
                    pg._build_text_templates,
                ):
                    tpls = builder()
                    if tpls:
                        docs.append(rng.choice(tpls)())
            if docs:
                _write_jsonl(output_dir / "pretrain" / "data.jsonl", [{"text": d} for d in docs])
        except Exception as exc:
            logger.warning("PretrainDataGenerator failed: %s", exc)

    if not (output_dir / "pretrain" / "data.jsonl").exists():
        docs = _fallback_pretrain_documents(max(500, samples // 2))
        _write_jsonl(output_dir / "pretrain" / "data.jsonl", [{"text": d} for d in docs])

    # 2) SFT conversations -> raw/sft/data.jsonl
    if SFTGen:
        try:
            sft_dir = output_dir / "sft"
            sft_dir.mkdir(parents=True, exist_ok=True)
            gen = SFTGen({"seed": RANDOM_SEED})
            items = [gen.generate_single_turn(cat) for cat in ("coding", "reasoning", "writing", "analysis", "general")
                     for _ in range(max(200, samples // 5))]
            rng.shuffle(items)
            _write_jsonl(sft_dir / "data.jsonl", items)
        except Exception as exc:
            logger.warning("SFT generation failed: %s", exc)

    if not (output_dir / "sft" / "data.jsonl").exists():
        _write_jsonl(output_dir / "sft" / "data.jsonl", _fallback_sft_conversations(max(400, samples // 2)))

    # 3) Math/reasoning -> raw/math/data.jsonl
    if MathGen:
        try:
            math_dir = output_dir / "math"
            math_dir.mkdir(parents=True, exist_ok=True)
            gen = MathGen({"seed": RANDOM_SEED})
            cats = gen.CATEGORIES
            diffs = gen.DIFFICULTIES
            items = []
            for _ in range(max(300, samples // 3)):
                cat = rng.choice(cats)
                dif = rng.choice(diffs)
                items.append(gen.generate_problem(cat, dif))
            _write_jsonl(math_dir / "data.jsonl", items)
        except Exception as exc:
            logger.warning("Math generation failed: %s", exc)

    if not (output_dir / "math" / "data.jsonl").exists():
        # Simple fallback math/qa pairs
        pairs = []
        for _ in range(max(300, samples // 3)):
            a, b = rng.randint(1, 999), rng.randint(1, 999)
            op = rng.choice(["+", "-", "*"])
            q = f"What is {a} {op} {b}?"
            pairs.append({"instruction": q, "response": str(eval(f"{a}{op}{b}")), "category": "arithmetic", "difficulty": "easy"})
        _write_jsonl(output_dir / "math" / "data.jsonl", pairs)

    # 4) Safety -> raw/safety/data.jsonl
    if SafetyGen:
        try:
            safety_dir = output_dir / "safety"
            safety_dir.mkdir(parents=True, exist_ok=True)
            gen = SafetyGen({"seed": RANDOM_SEED})
            items = [gen.generate_example(cat) for cat in gen.CATEGORIES for _ in range(max(100, samples // 5))]
            rng.shuffle(items)
            _write_jsonl(safety_dir / "data.jsonl", items)
        except Exception as exc:
            logger.warning("Safety generation failed: %s", exc)

    if not (output_dir / "safety" / "data.jsonl").exists():
        _write_jsonl(output_dir / "safety" / "data.jsonl", _fallback_safety(max(400, samples // 2)))

    # 5) Agent -> raw/agent/data.jsonl
    if AgentGen:
        try:
            agent_dir = output_dir / "agent"
            agent_dir.mkdir(parents=True, exist_ok=True)
            gen = AgentGen({"seed": RANDOM_SEED})
            types = list(gen.TOOL_TYPES)
            items = [gen.generate_trajectory(rng.choice(types)) for _ in range(max(200, samples // 4))]
            _write_jsonl(agent_dir / "data.jsonl", items)
        except Exception as exc:
            logger.warning("Agent generation failed: %s", exc)

    if not (output_dir / "agent" / "data.jsonl").exists():
        _write_jsonl(output_dir / "agent" / "data.jsonl", _fallback_agent(max(200, samples // 4)))


# ---------------------------------------------------------------------------
# Tokenize with Aurelius pipeline
# ---------------------------------------------------------------------------

def _tokenize_all(output_dir: Path, shard_size: int = 16384, max_length: int = 8192) -> Path:
    sys.path.insert(0, str(REPO))
    os.chdir(REPO)

    try:
        import importlib.util
        _raw = importlib.util.spec_from_file_location(
            "aurelius.tokenize_pipeline", str(REPO / "training_data" / "tokenize_pipeline.py")
        )
        _spec = _raw
        if _spec is None:
            raise RuntimeError("Failed to create module spec for tokenize_pipeline")
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        TokenizePipeline = _mod.TokenizePipeline
    except Exception as exc:
        raise RuntimeError(f"TokenizePipeline import failed: {exc}") from exc

    cfg = {
        "tokenizer": {"vocab_size": 128000, "path": "tokenizers/aurelius-128k"},
        "tokenize": {"shard_size": shard_size, "val_ratio": 0.1, "max_length": max_length, "num_workers": min(4, os.cpu_count() or 1), "verify_integrity": True},
    }
    pipeline = TokenizePipeline(cfg)

    train_paths: list[str] = []
    val_paths: list[str] = []

    raw = output_dir
    if not raw.exists():
        raise FileNotFoundError(f"raw output dir missing: {raw}")

    for source_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
        data_file = source_dir / "data.jsonl"
        if not data_file.exists():
            logger.warning("Skip %s: no data.jsonl", source_dir)
            continue
        records: list[str] = []
        with data_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = _coerce_text(obj)
                if text:
                    records.append(text)
        if not records:
            logger.warning("Skip %s: no text records after coercion", source_dir)
            continue
        target = SHARD_ROOT / "intermediate" / source_dir.name
        logger.info("Tokenizing %d texts from %s", len(records), data_file)
        pipeline.tokenize_texts(records, str(target), shard_size=shard_size)
        for shard in sorted(target.glob("shard_*.npy")):
            train_paths.append(str(shard))

    if not train_paths:
        raise RuntimeError("No shards generated")

    rng = random.Random(42)
    rng.shuffle(train_paths)
    split_idx = max(1, int(len(train_paths) * 0.9))
    train_paths_shuf = train_paths[:split_idx]
    val_paths_shuf = train_paths[split_idx:]

    train_out = SHARD_ROOT / "train"
    val_out = SHARD_ROOT / "val"
    train_out.mkdir(parents=True, exist_ok=True)
    val_out.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(train_paths_shuf):
        dest = train_out / f"shard_{i:06d}.npy"
        import shutil
        shutil.copy2(src, dest)

    for i, src in enumerate(val_paths_shuf):
        dest = val_out / f"shard_{i:06d}.npy"
        import shutil
        shutil.copy2(src, dest)

    logger.info("Train shards: %d, val shards: %d", len(train_paths_shuf), len(val_paths_shuf))

    # manifests
    train_manifest = pipeline.create_shard_manifest(str(train_out), str(SHARD_ROOT / "train_manifest.json"))
    val_manifest = pipeline.create_shard_manifest(str(val_out), str(SHARD_ROOT / "val_manifest.json"))

    print(json.dumps({
        "status": "ok",
        "output_dir": str(SHARD_ROOT),
        "train_shards": len(train_paths_shuf),
        "val_shards": len(val_paths_shuf),
        "train_tokens": train_manifest.get("total_tokens", 0),
        "val_tokens": val_manifest.get("total_tokens", 0),
    }, indent=2))
    return SHARD_ROOT


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Build synthetic training data for Aurelius")
    parser.add_argument("--output", default=str(RAW_ROOT), help="Raw output dir")
    parser.add_argument("--samples", type=int, default=2000, help="Synthetic sample budget")
    parser.add_argument("--shard-size", type=int, default=16384)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--skip-tokenize", action="store_true", help="Only generate JSONL, skip shard creation")
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    logger.info("Raw output dir: %s", output_dir)
    _generate_synthetic(output_dir, args.samples)
    if not args.skip_tokenize:
        _tokenize_all(output_dir, shard_size=args.shard_size, max_length=args.max_length)


if __name__ == "__main__":
    main()
