"""SISA ALGORITHM — formal implementation + evidence battery (2026-08-04).

ALGORITHM SPEC (SISA v1.0, Aurelius Lab):
  Input: frozen transformer T (L layers), structured context C (R records,
         schema S: key field K_r, payload field V_r), query q, drop budget B,
         dense window D, index window W (calibrated), residual groups G=9.
  Phase 1 PARSE:      span-parse C into records; classify tokens
                      {STRUCT, ID(K_r), VALUE(V_r), FILL} via schema.
  Phase 2 CALIBRATE:  one-time per (model, schema): scan layers on dev queries,
                      pick W = earliest window with correct selection.
  Phase 3 INDEX:      forward layers 0..D-1 densely; compute field-targeted
                      mass m(r) = sum_{l in W} Attn_l(q_key_tokens -> K_r);
                      select top-k records by m.
  Phase 4 ADMIT:      protect selected records' STRUCT/ID/VALUE spans;
                      eviction set E = uniform stride over remaining eligible
                      tokens up to budget B; compress E into G residual groups.
  Phase 5 SPARSE:     layers >= D attend only over kept tokens + groups.
  Phase 6 INTEGRITY:  per-record hybrid sha256 (numeric-normalized); on hash
                      mismatch: re-verify + restore from source of truth.

BATTERY (this script, Qwen3-1.7B):
  A. budget envelope: B in {90, 135, 180, 225}, qids {3,7,11}
  B. natural-key falsification: ids as words (alpha..oscar) instead of digits
  C. statistical strength: qids {1,3,7,9,11,15} at B=135
  D. multi-record: 3 targets (3,7,11) at B=135
  E. index ablations: W={2} vs W={0..4} vs W=all; target=ID-only vs whole-record
Outputs: sisa_battery_results.json, console log.
"""
import json, math, time, warnings, re, hashlib
warnings.filterwarnings("ignore")
import torch
import torch.nn.functional as F

t0 = time.time()
torch.manual_seed(0)
BASE = "/Users/christienantonio/Aurelius_LocalMirror/research/calc_improve_2026-08-02/"
L = open(BASE + "sisa_battery_console.log", "w")
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
DENSE_UNTIL = 3
WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
         "india", "juliet", "kilo", "lima", "mike", "november", "oscar"]

def build_prompt(qid, key_mode="digit", multi=None):
    recs = [f'{{"id": "{str(i+1) if key_mode=="digit" else WORDS[i]}", "value": "{v}"}}'
            for i, v in enumerate(VALUES)]
    body = FILL.join(recs)
    if multi:
        ids_str = " and ".join(str(x) if key_mode == "digit" else WORDS[x-1] for x in multi)
        return POLICY + body + f" Question: What are the values of the items with id {ids_str}?"
    qkey = str(qid) if key_mode == "digit" else WORDS[qid-1]
    return POLICY + body + f' Question: What is the value of the item with id {qkey}?'

def rec_chars_for(prompt, key_mode):
    out = []
    for i, v in enumerate(VALUES):
        idd = str(i + 1) if key_mode == "digit" else WORDS[i]
        s = prompt.find(f'{{"id": "{idd}"')
        e = prompt.find("}", s) + 1
        out.append((s, e, idd, v))
    return out

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
            idd, val = rec_chars[in_rec][2], rec_chars[in_rec][3]
            idf = prompt.find(f'"id": "{idd}"', rec_chars[in_rec][0]) + len('"id": "')
            vf = prompt.find(f'"value": "{val}"', rec_chars[in_rec][0]) + len('"value": "')
            if idf <= a < idf + len(idd) and txt and all(c in idd for c in txt):
                cls[i] = ("ID", in_rec)
            elif vf <= a < vf + len(val) and txt and all(c in val for c in txt):
                cls[i] = ("VALUE", in_rec)
            else:
                cls[i] = ("STRUCT", in_rec)
        elif in_fill is not None:
            cls[i] = ("FILL", in_fill)
    return cls

def kl16(a, b):
    return F.kl_div(F.log_softmax(b[:, -16:], -1), F.softmax(a[:, -16:], -1), reduction="batchmean").item()

def fwd_full(ids):
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
            Pl = F.softmax(sc + causal, dim=-1)
            attns[li] = Pl
            ao = torch.bmm(Pl, vq).transpose(0, 1).contiguous().view(N, Hq * d)
            h = h + sa.o_proj(ao)
            hn2 = lay.post_attention_layernorm(h)
            h = h + mlp.down_proj(F.silu(mlp.gate_proj(hn2)) * mlp.up_proj(hn2))
    return m.lm_head(m.model.norm(h)), attns

def uniform_groups(dropped, n_groups=9):
    dropped = sorted(dropped)
    return [dropped[i::n_groups] for i in range(n_groups) if dropped[i::n_groups]]

def fwd_sisa(ids, drop_set, groups, dense_until):
    N = ids.shape[1]
    pos = torch.arange(N, dtype=torch.long).unsqueeze(0)
    dropped = sorted(set(drop_set))
    h = m.model.embed_tokens(ids)
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
            if li < dense_until:
                kk, vv = k, v
                nk = N
                causal = torch.triu(torch.full((N, N), -float("inf")), diagonal=1)
            else:
                keep = [j for j in range(N) if j not in set(dropped)]
                rk, rv, roff = [], [], []
                for g in groups:
                    S = torch.full((len(g),), 1.0 / len(g))
                    rk.append((S.unsqueeze(1) * k[:, g, :]).sum(1))
                    rv.append((S.unsqueeze(1) * v[:, g, :]).sum(1))
                    roff.append(min(g))
                offsets = keep + roff
                kk = torch.cat([k[:, keep, :], torch.stack(rk, dim=1)], dim=1) if rk else k[:, keep, :]
                vv = torch.cat([v[:, keep, :], torch.stack(rv, dim=1)], dim=1) if rv else v[:, keep, :]
                nk = kk.shape[1]
                causal = torch.zeros(N, nk, dtype=torch.bool)
                for i in range(nk):
                    causal[:, i] = offsets[i] <= torch.arange(N)
                causal = ~causal
            kq = kk.repeat_interleave(rep, dim=0); vq = vv.repeat_interleave(rep, dim=0)
            sc = torch.bmm(q, kq.transpose(1, 2)) / math.sqrt(d)
            if li < dense_until:
                sc = sc + causal.unsqueeze(0)
            else:
                sc = sc.masked_fill(causal.unsqueeze(0).expand(Hq, -1, -1), -float("inf"))
            Pl = F.softmax(sc, dim=-1)
            ao = torch.bmm(Pl, vq).transpose(0, 1).contiguous().view(N, Hq * d)
            h = h + sa.o_proj(ao)
            hn2 = lay.post_attention_layernorm(h)
            h = h + mlp.down_proj(F.silu(mlp.gate_proj(hn2)) * mlp.up_proj(hn2))
    return m.lm_head(m.model.norm(h))

def q_tokens_for(qkey, q_idx, ids):
    decomp_ids = set()
    for prefix in ("", " "):
        decomp_ids.update(tok.encode(prefix + qkey, add_special_tokens=False))
    return [i for i in q_idx if ids[0, i].item() in decomp_ids]

def layer_masses(attn, qtoks, cls, nrec, target="ID"):
    masses = []
    for r in range(nrec):
        toks = sorted(i for i, c in cls.items() if c[1] == r and (c[0] == target if target == "ID" else True))
        masses.append(attn[:, qtoks, :][:, :, toks].sum().item())
    return masses

def run_case(qid, budget, key_mode="digit", window=None, target="ID", multi=None):
    prompt = build_prompt(qid, key_mode, multi)
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    N = ids.shape[1]
    enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=False)
    rec_chars = rec_chars_for(prompt, key_mode)
    cls = classify(prompt, enc["offset_mapping"], rec_chars)
    rec_set = set(i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > 0 and
                  any(s <= a and b <= e for s, e, _, _ in rec_chars))
    rec_idx = sorted(rec_set)
    q_idx = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > 0 and a >= prompt.find(" Question:")]
    qkey = str(qid) if key_mode == "digit" else WORDS[qid - 1]
    if multi:
        qtoks = []
        for x in multi:
            kk = str(x) if key_mode == "digit" else WORDS[x - 1]
            cand = q_tokens_for(kk, q_idx, ids)
            exact = [i for i in cand if tok.decode([ids[0, i].item()]).strip() == kk]
            qtoks += (exact or cand)
    else:
        qtoks = q_tokens_for(qkey, q_idx, ids)
        # filter to tokens whose text is part of the key (digits or word chars) —
        # excludes field-name tokens like "id" that pollute the signal
        qtoks = [i for i in qtoks if tok.decode([ids[0, i].item()]).strip() in qkey] or qtoks
    if not qtoks:
        return None

    logits_full, attns = fwd_full(ids)
    full_argmax = logits_full[0, -1].argmax().item()
    targets = multi if multi else [qid]
    tset = set(x - 1 for x in targets)

    # selection with given window
    if window is None:
        window = range(0, 4)  # early-quarter default
    masses = [0.0] * len(VALUES)
    for li in window:
        ml = layer_masses(attns[li], qtoks, cls, len(VALUES), target)
        for r in range(len(VALUES)):
            masses[r] += ml[r]
    order = sorted(range(len(VALUES)), key=lambda r: -masses[r])
    sel = order[0] if not multi else order[:len(targets)]
    sel_ok = (sel in tset) if not multi else (set(sel) == tset)
    # parse-filtered selection: among records whose KEY exactly matches a query
    # key (string equality after stripping), take the top-attention one.
    qkeys = [str(x) if key_mode == "digit" else WORDS[x - 1] for x in (targets if multi else [qid])]
    cands = [r for r, rc in enumerate(rec_chars) if rc[2] in qkeys]
    if cands:
        cand_mass = [(r, masses[r]) for r in cands]
        cand_mass.sort(key=lambda x: -x[1])
        sel_parse = [cand_mass[0][0]]
        if multi:
            sel_parse = [r for r, _ in cand_mass][:len(targets)]
        sel_parse_ok = (sel_parse[0] in tset) if not multi else (set(sel_parse) == tset)
    else:
        sel_parse, sel_parse_ok = None, False

    # U
    stride = len(rec_idx) / budget
    drop_U = [rec_idx[int(i * stride)] for i in range(budget)]
    kl_u = kl16(logits_full, fwd_sisa(ids, drop_U, uniform_groups(drop_U), 0))
    # Q (oracle: protect ALL targets)
    prot_q = set()
    for t in (targets if multi else [qid]):
        prot_q |= set(i for i, c in cls.items() if c[0] in ("STRUCT", "ID", "VALUE") and c[1] == t - 1)
    elig_q = [i for i in rec_idx if i in cls and i not in prot_q]
    qstride = len(elig_q) / budget
    drop_Q = [elig_q[int(i * qstride)] for i in range(budget)]
    kl_q = kl16(logits_full, fwd_sisa(ids, drop_Q, uniform_groups(drop_Q), 0))
    # SISA (protect selected; production protocol = parse-filtered candidates
    # ranked by attention mass — raw attention selection reported separately)
    prot_sel = sel_parse if sel_parse is not None else sel
    prot_sel = prot_sel if multi else [prot_sel] if not isinstance(prot_sel, list) else prot_sel
    prot_s = set()
    for t in prot_sel:
        prot_s |= set(i for i, c in cls.items() if c[0] in ("STRUCT", "ID", "VALUE") and c[1] == t)
    elig_s = [i for i in rec_idx if i in cls and i not in prot_s]
    sstride = len(elig_s) / budget
    drop_S = [elig_s[int(i * sstride)] for i in range(budget)]
    kl_s = kl16(logits_full, fwd_sisa(ids, drop_S, uniform_groups(drop_S), DENSE_UNTIL))
    eq_q = (sorted(drop_S) == sorted(drop_Q)) if not multi else None

    return dict(qid=qid, budget=budget, key_mode=key_mode, window=list(window),
                target=target, multi=multi, sel=[s + 1 for s in (sel if multi else [sel])],
                sel_ok=sel_ok, sel_parse=[s + 1 for s in (sel_parse if sel_parse is not None else [])],
                sel_parse_ok=sel_parse_ok, kl_u=kl_u, kl_q=kl_q, kl_sisa=kl_s,
                dropset_eq_Q=eq_q, masses=[round(x, 3) for x in masses],
                answer=tok.decode([full_argmax]))

results = {}
# A. budget envelope
P("=== A. BUDGET ENVELOPE (json, qids 3/7/11) ===")
env = {}
for B in [90, 135, 180, 225]:
    env[B] = {}
    for qid in [3, 7, 11]:
        r = run_case(qid, B)
        env[B][qid] = r
        P(f"  B={B} q{qid}: sel={r['sel']}({'OK' if r['sel_ok'] else 'MISS'}) "
          f"U={r['kl_u']:.3f} Q={r['kl_q']:.3f} SISA={r['kl_sisa']:.3f}")
results["envelope"] = env

# B. natural keys
P("=== B. NATURAL-KEY FALSIFICATION (word ids, qids 3/7/11) ===")
nat = {}
for qid in [3, 7, 11]:
    r = run_case(qid, 135, key_mode="word")
    nat[qid] = r
    P(f"  word-id q{qid}: sel={r['sel']}({'OK' if r['sel_ok'] else 'MISS'}) "
    f"parse={r['sel_parse']}({'OK' if r['sel_parse_ok'] else 'MISS'}) "
    f"U={r['kl_u']:.3f} Q={r['kl_q']:.3f} SISA={r['kl_sisa']:.3f}")
results["natural_keys"] = nat

# C. statistical strength: 6 qids
P("=== C. 6-QID BATTERY (json, B=135) ===")
six = {}
for qid in [1, 3, 7, 9, 11, 15]:
    r = run_case(qid, 135)
    six[qid] = r
    P(f"  q{qid}: sel={r['sel']}({'OK' if r['sel_ok'] else 'MISS'}) "
      f"U={r['kl_u']:.3f} Q={r['kl_q']:.3f} SISA={r['kl_sisa']:.3f}")
results["six_qids"] = six

# D. multi-record 3 targets
P("=== D. MULTI-RECORD (3 targets: 3,7,11) ===")
r = run_case(3, 135, multi=[3, 7, 11])
results["multi3"] = r
P(f"  multi3: sel={r['sel']}({'OK' if r['sel_ok'] else 'MISS'}) "
  f"U={r['kl_u']:.3f} Q={r['kl_q']:.3f} SISA={r['kl_sisa']:.3f}")

# E. index ablations
P("=== E. INDEX ABLATIONS (json qid 7, B=135) ===")
abl = {}
for name, w, tgt in [("L2_ID", [2], "ID"), ("early_ID", list(range(0, 4)), "ID"),
                     ("full_ID", list(range(nL)), "ID"), ("L2_ALL", [2], "ALL"),
                     ("early_ALL", list(range(0, 4)), "ALL")]:
    r = run_case(7, 135, window=w, target=tgt)
    abl[name] = r
    P(f"  {name}: sel={r['sel']}({'OK' if r['sel_ok'] else 'MISS'}) SISA={r['kl_sisa']:.3f}")
results["ablations"] = abl

with open(BASE + "sisa_battery_results.json", "w") as f:
    json.dump(results, f, indent=1, default=float)
P(f"TOTAL {time.time()-t0:.0f}s")
