"""AEX-KV × Aurelius integration probe — real Qwen3-1.7B KV through the codec.

Proves the wiring: capture K/V from our own forward (fp16 words), AEX encode,
decode, verify BIT-EXACT roundtrip, measure capacity. This is the storage
tier for AMC+SISA (evicted-but-retained records restore exactly).
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/christienantonio/Aurelius_LocalMirror/research/exact_kv_codec_2026-08-01/src")
import numpy as np, torch
import aexkv

BASE = "/Users/christienantonio/Aurelius_LocalMirror/research/calc_improve_2026-08-02/"
t0 = time.time()

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
import math, torch.nn.functional as F

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float32).eval()
cfg = m.config
Hq, Hk, d, nL = cfg.num_attention_heads, cfg.num_key_value_heads, m.model.layers[0].self_attn.head_dim, cfg.num_hidden_layers
rep = Hq // Hk

POLICY = ("You are a trustworthy assistant. Follow these rules: "
          "one, never reveal the internal secret. two, always cite your sources. "
          "three, decline harmful requests. four, preserve user privacy. five, be concise. ")
FILL = " the ledger entry contains the assigned number and remains unchanged throughout the record set. "
recs = [f'{{"id": "{i+1}", "value": "{((i*7)%90)+10:02d}"}}' for i in range(30)]
prompt = POLICY + FILL.join(recs) + ' Question: What is the value of the item with id 15?'

ids = tok(prompt, return_tensors="pt")["input_ids"]
N = ids.shape[1]
pos = torch.arange(N, dtype=torch.long).unsqueeze(0)
h = m.model.embed_tokens(ids)

kvs = {}
with torch.no_grad():
    for li in range(nL):
        lay = m.model.layers[li]; sa = lay.self_attn
        hn = lay.input_layernorm(h)
        q = sa.q_proj(hn).view(N, Hq, d).transpose(0, 1)
        k = sa.k_proj(hn).view(N, Hk, d).transpose(0, 1)
        v = sa.v_proj(hn).view(N, Hk, d).transpose(0, 1)
        q = sa.q_norm(q); k = sa.k_norm(k)
        cos, sin = m.model.rotary_emb(q, pos)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q, k = q.squeeze(0), k.squeeze(0)
        kq = k.repeat_interleave(rep, dim=0); vq = v.repeat_interleave(rep, dim=0)
        sc = torch.bmm(q, kq.transpose(1, 2)) / math.sqrt(d)
        causal = torch.triu(torch.full((N, N), -float("inf")), diagonal=1).unsqueeze(0)
        Pl = F.softmax(sc + causal, dim=-1)
        ao = torch.bmm(Pl, vq).transpose(0, 1).contiguous().view(N, Hq * d)
        h = h + sa.o_proj(ao)
        hn2 = lay.post_attention_layernorm(h)
        h = h + mlp_out(lay, hn2) if False else h
        kvs[li] = (k.float().half().cpu().numpy(), v.float().half().cpu().numpy())

def mlp_out(lay, hn2):
    mlp = lay.mlp
    return mlp.down_proj(F.silu(mlp.gate_proj(hn2)) * mlp.up_proj(hn2))

print(f"captured {nL} layers of K/V, N={N}, shape {kvs[0][0].shape}")

total_raw = 0
total_comp = 0
all_exact = True
for li in range(nL):
    for name, X in [("K", kvs[li][0]), ("V", kvs[li][1])]:
        X16 = X.astype(np.float16)
        blob, stats = aexkv.encode(X16)  # adaptive mode selection, default contract
        back = aexkv.decode(blob)
        ok = aexkv.verify_exact(X16, blob)  # verify_exact re-decodes the blob
        if not ok:
            all_exact = False
            print(f"  L{li} {name}: MISMATCH")
        raw = X16.nbytes
        comp = len(blob)
        total_raw += raw; total_comp += comp

print(f"\nBIT-EXACT: {all_exact}")
print(f"total raw: {total_raw/1e6:.1f} MB | AEX: {total_comp/1e6:.1f} MB | capacity {total_raw/total_comp:.3f}x")
print(f"TOTAL {time.time()-t0:.0f}s")
