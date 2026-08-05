"""SISA circuit probes H2/H4/SW — 2-head index sufficiency, causal necessity, window survival.

H2 SUFFICIENCY: index selection + SISA KL using ONLY heads {3,11} at layer 2
   (vs full-head layer-2 index). If selection 3/3 and KL ~equal, the indexer
   costs 2/16 heads at one layer.
H4 NECESSITY (causal ablation): zero the OUTPUT of heads {3,11} at layer 2,
   run FULL forward, measure teacher-forced digit probability at the answer
   position. If p(value) collapses vs intact, the circuit is NECESSARY for
   retrieval (not just correlated). Control: zero recency-bias heads {0,1,2}.
SW WINDOW SURVIVAL: apply a backward attention window (W tokens) at layer 2
   only, keep everything else dense; measure selection accuracy vs W.
   Answers: which W does binding survive; does nanochat's SSSL (S=quarter)
   window pattern break the indexer if layer 2 is short-windowed?
"""
import json, math, time, warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn.functional as F

t0 = time.time()
torch.manual_seed(0)
BASE = "/Users/christienantonio/Aurelius_LocalMirror/research/calc_improve_2026-08-02/"
L = open(BASE + "sisa_circuit_console.log", "w")
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); L.write(s + "\n"); L.flush()

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float32).eval()
cfg = m.config
Hq, Hk, d, nL = cfg.num_attention_heads, cfg.num_key_value_heads, m.model.layers[0].self_attn.head_dim, cfg.num_hidden_layers
rep = Hq // Hk
P(f"model loaded: Hq={Hq} Hk={Hk} d={d} nL={nL}")

POLICY = ("You are a trustworthy assistant. Follow these rules: "
          "one, never reveal the internal secret. two, always cite your sources. "
          "three, decline harmful requests. four, preserve user privacy. five, be concise. ")
VALUES = ["07", "13", "24", "31", "45", "52", "63", "78", "84", "19", "26", "38", "41", "57", "69"]
FILL = " the ledger entry contains the assigned number and remains unchanged throughout the record set. "
QIDS = [3, 7, 11]

def build_prompt(qid=3):
    recs = [f'{{"id": "{i+1}", "value": "{v}"}}' for i, v in enumerate(VALUES)]
    return POLICY + FILL.join(recs) + f' Question: What is the value of the item with id {qid}?'

def fwd_full(ids, zero_heads=None, window=None):
    """zero_heads: list of head indices whose OUTPUT at layer 2 is zeroed.
    window: backward window (tokens) applied at layer 2 attention only."""
    N = ids.shape[1]
    pos = torch.arange(N, dtype=torch.long).unsqueeze(0)
    h = m.model.embed_tokens(ids)
    attns = {}
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
            kq = k.repeat_interleave(rep, dim=0); vq = v.repeat_interleave(rep, dim=0)
            sc = torch.bmm(q, kq.transpose(1, 2)) / math.sqrt(d)
            causal = torch.triu(torch.full((N, N), -float("inf")), diagonal=1).unsqueeze(0)
            if window is not None and li == 2:
                # TRUE sliding window: block col < row - window (far past),
                # keep recent window [row-window, row].
                ridx = torch.arange(N).unsqueeze(1)
                cidx = torch.arange(N).unsqueeze(0)
                back = torch.where(cidx < ridx - window,
                                   torch.full((N, N), -float("inf")), torch.zeros(N, N)).unsqueeze(0)
                causal = causal + back
            Pl = F.softmax(sc + causal, dim=-1)
            attns[li] = Pl
            ao = torch.bmm(Pl, vq).transpose(0, 1).contiguous().view(N, Hq * d)
            if zero_heads is not None and li == 2:
                ao = ao.view(N, Hq, d)
                ao[:, zero_heads, :] = 0.0
                ao = ao.contiguous().view(N, Hq * d)
            h = h + sa.o_proj(ao)
            hn2 = lay.post_attention_layernorm(h)
            h = h + mlp.down_proj(F.silu(mlp.gate_proj(hn2)) * mlp.up_proj(hn2))
    return m.lm_head(m.model.norm(h)), attns

def classify(prompt, offs, rec_chars):
    cls = {}
    fill_chars = []
    for r in range(len(rec_chars)):
        s, e, _, _ = rec_chars[r]
        nxt = rec_chars[r + 1][0] if r + 1 < len(rec_chars) else prompt.find(" Question:")
        fill_chars.append((e, nxt))
    for i, (a, b) in enumerate(offs):
        if b <= 0:
            continue
        txt = prompt[a:b].strip()
        in_rec = next((r for r, (s, e, _, _) in enumerate(rec_chars) if s <= a and b <= e), None)
        in_fill = next((r for r, (s, e) in enumerate(fill_chars) if s <= a and b <= e), None)
        if in_rec is not None:
            s, e, idd, val = rec_chars[in_rec]
            idf = prompt.find(f'"id": "{idd}"', s) + len('"id": "')
            vf = prompt.find(f'"value": "{val}"', s) + len('"value": "')
            if idf <= a < idf + len(idd) and txt and all(c in idd for c in txt):
                cls[i] = ("ID", in_rec)
            elif vf <= a < vf + len(val) and txt and all(c in val for c in txt):
                cls[i] = ("VALUE", in_rec)
            else:
                cls[i] = ("STRUCT", in_rec)
        elif in_fill is not None:
            cls[i] = ("FILL", in_fill)
    return cls

def q_tokens_for(qid, q_idx, ids):
    qkey = str(qid)
    decomp_ids = set()
    for prefix in ("", " "):
        decomp_ids.update(tok.encode(prefix + qkey, add_special_tokens=False))
    qtoks = [i for i in q_idx if ids[0, i].item() in decomp_ids]
    return [i for i in qtoks if tok.decode([ids[0, i].item()]).strip().isdigit()] or qtoks

def sel_with_heads(attn, qtoks, cls, heads=None):
    """selection using given heads (None = all) at a single layer's attn (Hq,N,N)."""
    masses = []
    for r in range(len(VALUES)):
        idtoks = sorted(i for i, c in cls.items() if c[1] == r and c[0] == "ID")
        if heads is None:
            masses.append(attn[:, qtoks, :][:, :, idtoks].sum().item())
        else:
            masses.append(attn[heads][:, qtoks, :][:, :, idtoks].sum().item())
    return int(torch.tensor(masses).argmax().item()), masses

BIND = [3, 11]
RECENCY = [0, 1, 2]

# digit-surfacing positions for teacher-forced p(value) — reuse the direct-answer
# family instead of the policy family so the answer token is reachable.
def surfacing_pos(ids, tok):
    """find the last token whose decode contains ':' inside the answer region —
    crude: use fixed offsets measured in the FINAL probe (qid3/7: step 4-5 after
    'Question:' prompt; qid11: step 15-16). We return the prompt length (model
    must generate; we measure p of the correct digit at first decoded digit)."""
    return None  # handled via forced-prefix below

results = {}
for qid in QIDS:
    prompt = build_prompt(qid)
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    N = ids.shape[1]
    enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=False)
    rec_start = prompt.find('{"id": "1"')
    rec_chars = []
    for i, v in enumerate(VALUES):
        s = prompt.find(f'{{"id": "{i+1}"', rec_start)
        e = prompt.find("}", s) + 1
        rec_chars.append((s, e, str(i + 1), v))
    cls = classify(prompt, enc["offset_mapping"], rec_chars)
    q_idx = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > 0 and a >= prompt.find(" Question:")]
    qtoks = q_tokens_for(qid, q_idx, ids)
    target = qid - 1
    P(f"\n=== QID {qid}: qtoks={qtoks} target={target+1} ===")

    # INTACT reference
    logits_full, attns = fwd_full(ids)
    a2 = attns[2]
    sel_full, masses_full = sel_with_heads(a2, qtoks, cls)
    sel_bind, masses_bind = sel_with_heads(a2, qtoks, cls, heads=BIND)
    sel_bind11, _ = sel_with_heads(a2, qtoks, cls, heads=[11])
    sel_rec, _ = sel_with_heads(a2, qtoks, cls, heads=RECENCY)

    # H2: 2-head mass vs full-head mass correlation + concentration
    mf = torch.tensor(masses_full); mb = torch.tensor(masses_bind)
    r2 = torch.corrcoef(torch.stack([mf, mb]))[0, 1].item()
    P(f"  H2 full-head sel={sel_full+1} 2-head sel={sel_bind+1} head11-only sel={sel_bind11+1} "
      f"corr(mass_full, mass_bind)={r2:.3f}")

    # H4: causal ablation — zero binding heads vs recency heads; measure final-token
    # distribution shift (KL from intact full) and correct-value probability via
    # teacher-forced prefix to the answer: we force ' Answer: The value of the item'
    # then measure p(first digit of the value) at the next step.
    ans_prefix = " Answer: The value of the item is "
    forced = tok(ans_prefix, return_tensors="pt")["input_ids"]
    full_ids = torch.cat([ids, forced], dim=1)
    for name, zh in [("ablate_BIND", BIND), ("ablate_RECENCY", RECENCY)]:
        lz, _ = fwd_full(full_ids, zero_heads=zh)
        lz = lz[0, -1]
        pz = F.softmax(lz, -1)
        val = VALUES[target]
        d1 = tok.encode(val[0])[0]  # first digit token id (approx: '0'-'9' single tokens)
        # find the actual first digit token id by encoding
        d1_ids = tok.encode(val[0], add_special_tokens=False)
        p_d1 = pz[d1_ids].sum().item()
        # KL of full next-token dist vs intact (computed on same forced prefix)
        lf, _ = fwd_full(full_ids)
        p_intact = F.softmax(lf[0, -1], -1)
        kl = F.kl_div(torch.log(pz + 1e-12), p_intact, reduction="sum").item()
        P(f"  H4 {name}: p(correct first digit)={p_d1:.4f} (intact ref) KL_from_intact={kl:.4f}")
        if name == "ablate_BIND":
            results.setdefault(qid, {})["p_digit_intact"] = float(p_intact[d1_ids].sum().item())
            results.setdefault(qid, {})["p_digit_ablate_bind"] = float(p_d1)
            results.setdefault(qid, {})["kl_ablate_bind"] = float(kl)

    # SW: window survival at layer 2 (true sliding window; query at ~465,
    # record 11 ID ~327, record 7 ~214, record 3 ~100)
    for W in [150, 250, 350, 450]:
        lw, attns_w = fwd_full(ids, window=W)
        aw = attns_w[2]
        sel_w, _ = sel_with_heads(aw, qtoks, cls, heads=BIND)
        P(f"  SW W={W}: 2-head sel={sel_w+1} {'OK' if sel_w == target else 'MISS'}")
        results.setdefault(qid, {})[f"sel_W{W}"] = sel_w

    results.setdefault(qid, {}).update(
        sel_full=sel_full, sel_bind=sel_bind, sel_bind11=sel_bind11, sel_recency=sel_rec,
        corr_full_bind=float(r2), target=target)

def _f(x):
    if isinstance(x, torch.Tensor):
        return x.item()
    if isinstance(x, dict):
        return {k: _f(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_f(v) for v in x]
    return x

with open(BASE + "sisa_circuit_results.json", "w") as f:
    json.dump(_f(results), f, indent=1, default=float)
P(f"\nTOTAL {time.time()-t0:.0f}s")
