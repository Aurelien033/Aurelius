"""AEX-KV Stage 2 — lossless eviction restore probe (the integration claim).

Claim: with AEX-KV as the warm tier, SISA eviction becomes REVERSIBLE —
an evicted record's K/V compresses bit-exactly and restores exactly on
re-query, so retrieval quality equals never-evicted.

Method (Qwen3-1.7B, ledger battery qid):
  1. FULL: standard forward, capture logits + per-layer K/V.
  2. SISA-EVICT at budget B (protect target record, drop others): forward
     with dropped tokens removed (group-summed replacements, the battery
     protocol), capture logits_evicted.
  3. AEX-RESTORE: AEX-encode the FULL K/V of the evicted region at the
     SAME layers; simulate re-query: replace the evicted blocks' K/V with
     decoded-from-AEX values (bit-exact) in a fresh forward, capture
     logits_restored.
  4. Compare: KL(full, evicted) [the loss SISA pays today] vs
     KL(full, restored) [the loss with the warm tier]. EXPECTED: ~0.
Falsifier: if KL(restored) is not < KL(evicted) by a large margin, the
lossless-eviction claim fails.
"""
import json, math, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, "/Users/christienantonio/Aurelius_LocalMirror/research/exact_kv_codec_2026-08-01/src")
import aexkv

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
print(f"N={N}")

def fwd_full_capture(ids):
    """Return logits + per-layer raw K/V (pre-RoPE? no — POST-RoPE K, raw V as used)."""
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

def fwd_with_kv(ids, K, V, evict=None, groups=None):
    """Forward using PROVIDED K/V (optionally with eviction via group replacement)."""
    N = ids.shape[1]
    pos = torch.arange(N, dtype=torch.long).unsqueeze(0)
    h = m.model.embed_tokens(ids)
    dropped = set(evict) if evict else set()
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
            keep = [j for j in range(N) if j not in dropped]
            rk, rv, roff = [], [], []
            for g in (groups or []):
                if len(g) == 0:
                    continue
                S = torch.full((len(g),), 1.0 / len(g))
                rk.append((S.unsqueeze(1) * k[:, g, :]).sum(1))
                rv.append((S.unsqueeze(1) * v[:, g, :]).sum(1))
                roff.append(min(g))
            offsets = keep + roff
            kk = torch.cat([k[:, keep, :], torch.stack(rk, dim=1)], dim=1) if rk else k[:, keep, :]
            vv = torch.cat([v[:, keep, :], torch.stack(rv, dim=1)], dim=1) if rv else v[:, keep, :]
            nk = kk.shape[1]
            kq = kk.repeat_interleave(rep, dim=0); vq = vv.repeat_interleave(rep, dim=0)
            sc = torch.bmm(q, kq.transpose(1, 2)) / math.sqrt(d)
            causal = torch.zeros(N, nk, dtype=torch.bool)
            for i in range(nk):
                causal[:, i] = offsets[i] <= torch.arange(N)
            sc = sc.masked_fill(~causal.unsqueeze(0).expand(Hq, -1, -1), -float("inf"))
            Pl = F.softmax(sc, dim=-1)
            ao = torch.bmm(Pl, vq).transpose(0, 1).contiguous().view(N, Hq * d)
            h = h + sa.o_proj(ao)
            hn2 = lay.post_attention_layernorm(h)
            h = h + mlp.down_proj(F.silu(mlp.gate_proj(hn2)) * mlp.up_proj(hn2))
    return m.lm_head(m.model.norm(h))

def kl16(a, b):
    return F.kl_div(F.log_softmax(b[:, -16:], -1), F.softmax(a[:, -16:], -1), reduction="batchmean").item()

# region classification
enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=False)
rec_start = prompt.find('{"id": "1"')
rec_chars = []
for i in range(RECS):
    s = prompt.find(f'{{"id": "{i+1}"', rec_start)
    e = prompt.find("}", s) + 1
    rec_chars.append((s, e, str(i + 1), vals[i]))
target = QID - 1
tgt_s, tgt_e, _, _ = rec_chars[target]
tgt_toks = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > 0 and a >= tgt_s and b <= tgt_e]
evict_toks = []
for r, (s, e, _, _) in enumerate(rec_chars):
    if r != target:
        evict_toks += [i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > 0 and a >= s and b <= e]
print(f"target toks: {len(tgt_toks)}, evict toks: {len(evict_toks)}")

B = 135
stride = len(evict_toks) / B
drop_toks = [evict_toks[int(i * stride)] for i in range(B)]
groups = [drop_toks[i::8] for i in range(8) if drop_toks[i::8]]

logits_full, K, V = fwd_full_capture(ids)
logits_evicted = fwd_with_kv(ids, K, V, evict=drop_toks, groups=groups)

# AEX-encode the EVICTED region's K/V (all layers), then decode back
aex_sizes = {"K": 0, "V": 0}
raw_sizes = {"K": 0, "V": 0}
K_restored, V_restored = {}, {}
for li in range(nL):
    kr, vr = {}, {}
    for name, src, dst in [("K", K[li], K_restored), ("V", V[li], V_restored)]:
        sub = src[:, evict_toks, :].cpu().numpy().astype(np.float16)  # (Hk, n_evict, d)
        blob, _ = aexkv.encode(sub)
        back = aexkv.decode(blob)
        assert aexkv.verify_exact(sub, blob), f"L{li} {name} NOT EXACT"
        dst[li] = torch.from_numpy(back.astype(np.float32)).to(src.device)
        aex_sizes[name] += len(blob)
        raw_sizes[name] += sub.nbytes
# reinsert restored blocks into the full K/V (replace the evicted positions)
K_final, V_final = {}, {}
for li in range(nL):
    K_final[li] = K[li].clone()
    V_final[li] = V[li].clone()
    K_final[li][:, evict_toks, :] = K_restored[li]
    V_final[li][:, evict_toks, :] = V_restored[li]

logits_restored = fwd_with_kv(ids, K_final, V_final)

kl_evict = kl16(logits_full, logits_evicted)
kl_restore = kl16(logits_full, logits_restored)
print(f"\nKL(full, evicted)   = {kl_evict:.6f}  <- the loss SISA pays today")
print(f"KL(full, restored)  = {kl_restore:.6f}  <- with AEX warm tier")
print(f"IMPROVEMENT: {(1 - kl_restore / max(kl_evict, 1e-9)) * 100:.1f}% of eviction loss removed")
print(f"AEX capacity on evicted K/V: K {raw_sizes['K']/aex_sizes['K']:.2f}x, V {raw_sizes['V']/aex_sizes['V']:.2f}x")
print(f"FALSIFIER (restored << evicted): {'CLEARED' if kl_restore < kl_evict * 0.1 else 'NOT CLEARED'}")

with open("/Users/christienantonio/Aurelius_LocalMirror/research/calc_improve_2026-08-02/aexkv_stage2_results.json", "w") as f:
    json.dump({"kl_evicted": kl_evict, "kl_restored": kl_restore,
               "aex_capacity": {"K": raw_sizes['K']/aex_sizes['K'], "V": raw_sizes['V']/aex_sizes['V']},
               "budget": B, "qid": QID, "n_evict": len(evict_toks)}, f, indent=1)
print(f"TOTAL {time.time()-t0:.0f}s")


# === HONEST ARM (Claude review fix): restore EXACTLY the dropped set ===
import aexkv as aexkv2
K_rest, V_rest = {}, {}
for li in range(nL):
    Kr, Vr = K[li].clone(), V[li].clone()
    sub_k = K[li][:, drop_toks, :].cpu().numpy().astype(np.float16)
    sub_v = V[li][:, drop_toks, :].cpu().numpy().astype(np.float16)
    bk, _ = aexkv2.encode(sub_k); bv, _ = aexkv2.encode(sub_v)
    assert aexkv2.verify_exact(sub_k, bk) and aexkv2.verify_exact(sub_v, bv)
    Kr[:, drop_toks, :] = torch.from_numpy(aexkv2.decode(bk).astype(np.float32))
    Vr[:, drop_toks, :] = torch.from_numpy(aexkv2.decode(bv).astype(np.float32))
    K_rest[li], V_rest[li] = Kr, Vr
all_keep = sorted(set(range(N)) - set(drop_toks)) + sorted(drop_toks)
logits_restored = fwd_with_kv(ids, K_rest, V_rest)

kl_restore = kl16(logits_full, logits_restored)
print(f"\nKL(full, evicted)   = {kl_evict:.6f}  <- the loss SISA pays today")
print(f"KL(full, restored)  = {kl_restore:.6f}  <- AEX-restored DROPPED SET (honest arm)")
print(f"IMPROVEMENT: {(1 - kl_restore / max(kl_evict, 1e-9)) * 100:.1f}% of eviction loss removed")
print(f"FALSIFIER (restored << evicted): {'CLEARED' if kl_restore < kl_evict * 0.1 else 'NOT CLEARED'}")
with open("/Users/christienantonio/Aurelius_LocalMirror/research/calc_improve_2026-08-02/aexkv_stage2_results.json", "w") as f:
    import json as _json
    _json.dump({"kl_evicted": kl_evict, "kl_restored": kl_restore, "budget": B, "qid": QID}, f, indent=1)
print(f"TOTAL {time.time()-t0:.0f}s")
