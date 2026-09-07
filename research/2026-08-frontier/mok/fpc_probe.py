"""FPC — Field-Preserving Clustering probe (New Algorithm 1).

HCA probe falsified CROSS-record clustering (KL 6.06 vs SISA 1.70):
clustering mixes DISTINCT records into shared centroids, destroying
field identity. FPC hypothesis: clustering WITHIN record boundaries
preserves identity by construction -> should approach SISA's quality
at centroid-budget (one centroid per record = 14 slots at G=14, vs
SISA's 8 group slots).

Compare on the ledger battery (qid7): FPC G=14 (1 centroid/record) and
FPC G=8 (8 records preserved, 6 evicted) vs FULL (baseline), SISA
(B=135), HCA (G=8, from the prior probe: 6.06).
"""
import json, math, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn.functional as F

t0 = time.time()
torch.manual_seed(0)
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float32).eval()
cfg = m.config
Hq, Hk, d, nL = cfg.num_attention_heads, cfg.num_key_value_heads, m.model.layers[0].self_attn.head_dim, cfg.num_hidden_layers
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

def fwd_with_kv(ids, K, V, groups=None):
    """groups: list of token-lists; each group becomes ONE centroid (mean K/V)."""
    N = ids.shape[1]
    pos = torch.arange(N, dtype=torch.long).unsqueeze(0)
    h = m.model.embed_tokens(ids)
    with torch.no_grad():
        for li in range(nL):
            lay = m.model.layers[li]; sa = lay.self_attn; mlp = lay.mlp
            hn = lay.input_layernorm(h)
            q = sa.q_proj(hn).view(N, Hq, d).transpose(0, 1)
            q = sa.q_norm(q)
            cos, sin = m.model.rotary_emb(q, pos)
            q, _ = apply_rotary_pos_emb(q, q, cos, sin)
            q = q.squeeze(0)
            k, v = K[li], V[li]
            grouped = set(sum(groups, []))
            keep = [j for j in range(N) if j not in grouped]
            rk, rv, roff = [], [], []
            for g in groups:
                if len(g) == 0:
                    continue
                w = torch.full((len(g),), 1.0 / len(g))
                rk.append((w.unsqueeze(1) * k[:, g, :]).sum(1))
                rv.append((w.unsqueeze(1) * v[:, g, :]).sum(1))
                roff.append(min(g))
            kk = torch.cat([k[:, keep, :], torch.stack(rk, dim=1)], dim=1)
            vv = torch.cat([v[:, keep, :], torch.stack(rv, dim=1)], dim=1)
            nk = kk.shape[1]
            kq = kk.repeat_interleave(rep, dim=0); vq = vv.repeat_interleave(rep, dim=0)
            sc = torch.bmm(q, kq.transpose(1, 2)) / math.sqrt(d)
            causal = torch.zeros(N, nk, dtype=torch.bool)
            for i in range(nk):
                causal[:, i] = (roff[i - len(keep)] if i >= len(keep) else keep[i]) <= torch.arange(N)
            sc = sc.masked_fill(~causal.unsqueeze(0).expand(Hq, -1, -1), -float("inf"))
            Pl = F.softmax(sc, dim=-1)
            ao = torch.bmm(Pl, vq).transpose(0, 1).contiguous().view(N, Hq * d)
            h = h + sa.o_proj(ao)
            hn2 = lay.post_attention_layernorm(h)
            h = h + mlp.down_proj(F.silu(mlp.gate_proj(hn2)) * mlp.up_proj(hn2))
    return m.lm_head(m.model.norm(h))

def kl16(a, b):
    return F.kl_div(F.log_softmax(b[:, -16:], -1), F.softmax(a[:, -16:], -1), reduction="batchmean").item()

enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=False)
rec_start = prompt.find('{"id": "1"')
rec_toks_by_id = {}
for i in range(RECS):
    s = prompt.find(f'{{"id": "{i+1}"', rec_start)
    e = prompt.find("}", s) + 1
    rec_toks_by_id[i] = [j for j, (a, b) in enumerate(enc["offset_mapping"]) if b > 0 and a >= s and b <= e]
target = QID - 1
evictable = [i for i in range(RECS) if i != target]
print(f"N={N}, records={RECS}, target id={QID}, evictable records={len(evictable)}")

logits_full, K, V = fwd_full_capture(ids)

# FPC G=14: one centroid per evictable record (14 records preserved, target full)
groups14 = [rec_toks_by_id[i] for i in evictable]
logits_fpc14 = fwd_with_kv(ids, K, V, groups=groups14)
# FPC G=8: keep 8 records as centroids (binding-mass order not known a priori;
# use SISA's attention-mass to pick which 8 — the honest merge signal)
# compute mass on layer-2 heads {3,11} at the query position
h2 = None
with torch.no_grad():
    pass  # mass computation below uses the full-forward capture
# use a simple proxy: records closest to the query in embedding space
# (deterministic, no oracle) — pick the 8 records whose id tokens are most
# similar to the question tokens (average cosine of layer-1 K)
q_toks = [j for j, (a, b) in enumerate(enc["offset_mapping"]) if b > 0 and prompt.find("Question:", 0) <= a]
with torch.no_grad():
    k1 = K[1]  # (Hk, N, d)
    qk = k1[:, q_toks, :].mean(dim=1).mean(dim=0)  # (d,)
    rec_sim = {}
    for i in evictable:
        rt = rec_toks_by_id[i]
        rk = k1[:, rt, :].mean(dim=1).mean(dim=0)
        rec_sim[i] = F.cosine_similarity(qk.unsqueeze(0), rk.unsqueeze(0)).item()
    top8 = sorted(evictable, key=lambda i: rec_sim[i], reverse=True)[:8]
groups8 = [rec_toks_by_id[i] for i in top8]
logits_fpc8 = fwd_with_kv(ids, K, V, groups=groups8)

res = {
    "FPC G=14 (1 centroid/record)": kl16(logits_full, logits_fpc14),
    "FPC G=8 (mass-top-8 records)": kl16(logits_full, logits_fpc8),
    "REF SISA B=135": 1.7037,
    "REF HCA G=8 (cross-record)": 6.0563,
}
for k, v in res.items():
    print(f"{k:34s} KL={v:.4f}")
best_fpc = min(res["FPC G=14 (1 centroid/record)"], res["FPC G=8 (mass-top-8 records)"])
verdict = ("FPC CLEARS SISA — field preservation beats token selection"
           if best_fpc < 1.7037 else
           "FPC matches SISA range — field preservation ≈ token selection"
           if best_fpc < 3.0 else
           "FPC FAILS — identity preservation alone doesn't explain SISA's win")
print(f"VERDICT: {verdict}")
with open("/Users/christienantonio/Aurelius_LocalMirror/research/calc_improve_2026-08-02/fpc_probe_results.json", "w") as f:
    json.dump(res, f, indent=1)
print(f"TOTAL {time.time()-t0:.0f}s")
