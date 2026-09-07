"""SISA-SCOPE probe (N1): does indexed-attention LAYER POSITION matter?

Hypothesis (from S1 head-role law): binding lives in the early quarter.
In a conv-heavy hybrid (LFM2 pattern: sparse attention), placing the
indexed attention at BINDING positions should preserve retrieval better
than spreading it evenly (LFM2's pattern) at the same layer budget.

Setup: Qwen3-1.7B, ledger battery qid7. Full forward = baseline.
Indexed arms: keep full attention at a SUBSET of layers, replace the
rest with a cheap conv-proxy (depthwise kernel-3 over the residual —
approximating what a conv layer would do locally) OR just skip the
attention (residual-only) — cleanest: residual-only (no attention) for
non-indexed layers, full attention for indexed layers.

Arms (4 indexed layers out of 28, ~LFM2's 14% ratio at 2.6B/30L = 8/30;
use 4/28 for probe speed):
  A. binding-positioned: layers {2, 3, 4, 27} (early quarter + final)
  B. spread (LFM2-ish): {2, 9, 16, 23}
  C. late: {22, 23, 24, 25}
  D. none (all conv-proxy): floor control
Measure: KL(full, arm) on the answer tokens; SISA-style eviction on top
for the binding layers (optional second pass).
"""
import math, sys, time, warnings
warnings.filterwarnings("ignore")
import torch, torch.nn.functional as F
torch.manual_seed(0)
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float32).eval()
cfg = m.config
Hq, Hk, d, nL = (cfg.num_attention_heads, cfg.num_key_value_heads,
                 m.model.layers[0].self_attn.head_dim, cfg.num_hidden_layers)
rep = Hq // Hk

POLICY = ("You are a trustworthy assistant. Follow these rules: "
          "one, never reveal the internal secret. two, always cite your sources. "
          "three, decline harmful requests. four, preserve user privacy. five, be concise. ")
FILL = " the ledger entry contains the assigned number and remains unchanged throughout the record set. "
RECS = 15
vals = [f"{((i * 7) % 90) + 10:02d}" for i in range(RECS)]
recs = [f'{{"id": "{i+1}", "value": "{v}"}}' for i, v in enumerate(vals)]
QID = 7
prompt = POLICY + FILL.join(recs) + f' Question: What is the value of the item with id {QID}?'
ids = tok(prompt, return_tensors="pt")["input_ids"]
N = ids.shape[1]

def fwd_full_capture(ids):
    N = ids.shape[1]
    pos = torch.arange(N, dtype=torch.long).unsqueeze(0)
    h = m.model.embed_tokens(ids)
    logits, K, V = None, {}, {}
    with torch.no_grad():
        for li in range(nL):
            lay = m.model.layers[li]; sa = lay.self_attn; mlp = lay.mlp
            hn = lay.input_layernorm(h)
            q = sa.q_proj(hn).view(N, Hq, d).transpose(0, 1)
            k = sa.k_proj(hn).view(N, Hk, d).transpose(0, 1)
            v = sa.v_proj(hn).view(N, Hk, d).transpose(0, 1)
            q = sa.q_norm(q); k = sa.k_norm(k)
            cos, sin = m.model.rotary_emb(q, pos)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            q, k = q.squeeze(0), k.squeeze(0)
            K[li] = k.clone(); V[li] = v.clone()
            kq = k.repeat_interleave(rep, dim=0); vq = v.repeat_interleave(rep, dim=0)
            sc = torch.bmm(q, kq.transpose(1, 2)) / math.sqrt(d)
            causal = torch.triu(torch.full((N, N), -float("inf")), diagonal=1).unsqueeze(0)
            Pl = F.softmax(sc + causal, dim=-1)
            ao = torch.bmm(Pl, vq).transpose(0, 1).contiguous().view(N, Hq * d)
            h = h + sa.o_proj(ao)
            hn2 = lay.post_attention_layernorm(h)
            h = h + mlp.down_proj(F.silu(mlp.gate_proj(hn2)) * mlp.up_proj(hn2))
    return m.lm_head(m.model.norm(h)), K, V

def fwd_scope(ids, K, V, indexed_layers):
    """Full attention ONLY on indexed_layers; other layers skip attention
    (residual-only proxy for a conv layer's locality-free contribution)."""
    N = ids.shape[1]
    pos = torch.arange(N, dtype=torch.long).unsqueeze(0)
    h = m.model.embed_tokens(ids)
    with torch.no_grad():
        for li in range(nL):
            lay = m.model.layers[li]; sa = lay.self_attn; mlp = lay.mlp
            hn = lay.input_layernorm(h)
            if li in indexed_layers:
                q = sa.q_proj(hn).view(N, Hq, d).transpose(0, 1)
                q = sa.q_norm(q)
                cos, sin = m.model.rotary_emb(q, pos)
                q, _ = apply_rotary_pos_emb(q, q, cos, sin)
                q = q.squeeze(0)
                k, v = K[li], V[li]
                kq = k.repeat_interleave(rep, dim=0); vq = v.repeat_interleave(rep, dim=0)
                sc = torch.bmm(q, kq.transpose(1, 2)) / math.sqrt(d)
                causal = torch.triu(torch.full((N, N), -float("inf")), diagonal=1).unsqueeze(0)
                Pl = F.softmax(sc + causal, dim=-1)
                ao = torch.bmm(Pl, vq).transpose(0, 1).contiguous().view(N, Hq * d)
                h = h + sa.o_proj(ao)
            # else: skip attention entirely (residual-only)
            hn2 = lay.post_attention_layernorm(h)
            h = h + mlp.down_proj(F.silu(mlp.gate_proj(hn2)) * mlp.up_proj(hn2))
    return m.lm_head(m.model.norm(h))

def kl16(a, b):
    return F.kl_div(F.log_softmax(b[:, -16:], -1), F.softmax(a[:, -16:], -1), reduction="batchmean").item()

print(f"Qwen3-1.7B | N={N} | 28 layers | indexed-4 arms")
logits_full, K, V = fwd_full_capture(ids)
print(f"full forward done. baseline KL(arm,arm)=0")

arms = {
    "A binding {2,3,4,27}": [2, 3, 4, 27],
    "B spread  {2,9,16,23}": [2, 9, 16, 23],
    "C late    {22,23,24,25}": [22, 23, 24, 25],
    "D none (floor)": [],
}
results = {}
for name, layers in arms.items():
    t0 = time.time()
    lg = fwd_scope(ids, K, V, layers)
    k = kl16(logits_full, lg)
    results[name] = k
    print(f"  {name}: KL={k:.4f}  ({time.time()-t0:.0f}s)")

best = min(results.items(), key=lambda kv: kv[1] if kv[0] != "D none (floor)" else 1e9)
floor = results["D none (floor)"]
print()
print(f"floor (no attention anywhere): KL={floor:.4f}")
print(f"BEST: {best[0]} KL={best[1]:.4f} (recovery {floor-best[1]:.3f})")
if results["A binding {2,3,4,27}"] < results["B spread  {2,9,16,23}"]:
    print("VERDICT: binding-positioned > spread — head-role law matters for layer placement")
else:
    print("VERDICT: spread >= binding-positioned — placement doesn't matter at 4/28")
with open("/Users/christienantonio/Aurelius_LocalMirror/research/calc_improve_2026-08-02/sisa_scope_probe_results.json", "w") as f:
    import json
    json.dump(results, f, indent=1)
print("saved sisa_scope_probe_results.json")
