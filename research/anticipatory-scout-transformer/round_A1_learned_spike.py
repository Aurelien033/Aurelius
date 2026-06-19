"""
Round A1 — learned confirmation of the anisotropic equilibrium operator.
Spec: round-A-anisotropic-operator-spike.md v2  (Phase 1).

Question: when the operator is LEARNED (not hand-planted), can it stay simultaneously
  STABLE (sigma_max(J) < 1, bounded transient growth),
  TRAINABLE (cond(I-J) tame, trains without divergence),
  BENEFIT-CAPABLE (per-token iteration-count variance; useful along-manifold work)?

T1 (default): a LINEAR input-injected equilibrium map  z_{k+1} = M z_k + W x , trained
  on a planted-manifold denoising task. Recurrent Jacobian = M (constant) -> exact
  diagnostics. Strict-margin stability enforced by clipping sigma_max(M) <= 1-eps.
T3 (--task t3): a small nonlinear self-attention + LoRA block over a short sequence;
  Jacobian measured at sampled states (the joint N*d Jacobian, per R2). Tests whether
  attention+LoRA can EMERGE into the stable anisotropic regime, with NO planted manifold.

Authored without execution — smoke-test with --d 8 --m 2 --steps 50 first.
"""
import argparse, math
import torch, torch.nn as nn

torch.set_default_dtype(torch.float64)  # diagnostics want precision; small models

# ----------------------------- diagnostics (CORRECTED) -----------------------------
def sigma_max(M):            return torch.linalg.svdvals(M)[0].item()
def sigma_min(M):            return torch.linalg.svdvals(M)[-1].item()
def spectral_radius(M):      return torch.linalg.eigvals(M).abs().max().item()
def num_abscissa(M):         return torch.linalg.eigvalsh(0.5*(M+M.T)).max().item()

def transient_kappa(M, K):   # sup_{k<=K} ||M^k||_2  (non-normal transient growth)
    P, best = torch.eye(M.shape[0], dtype=M.dtype), 1.0   # ||M^0||=1 is the floor
    for _ in range(K):
        P = M @ P
        best = max(best, sigma_max(P))
    return best

def cond_I_minus(M):
    I = torch.eye(M.shape[0], dtype=M.dtype)
    s = torch.linalg.svdvals(I - M)
    return (s[0] / s[-1].clamp_min(1e-18)).item(), s[-1].item()

def principal_angle_deg(U, V):   # smallest principal angle between subspaces (cols)
    Uq, _ = torch.linalg.qr(U); Vq, _ = torch.linalg.qr(V)
    s = torch.linalg.svdvals(Uq.T @ Vq).clamp(-1, 1)
    return math.degrees(math.acos(s[0].item()))

# ----------------------------- T1: linear planted-manifold denoiser ----------------
def run_T1(args):
    d, m, K, dev = args.d, args.m, args.K, args.device
    torch.manual_seed(args.seed)                      # seed the global RNG -> M,W init reproducible
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    U_true, _ = torch.linalg.qr(torch.randn(d, d, generator=g)); U_true = U_true[:, :m]  # planted manifold
    P_along = U_true @ U_true.T

    def batch(n):
        c = torch.randn(n, m, generator=g) @ U_true.T            # on-manifold target z*
        off = torch.randn(n, d, generator=g) @ (torch.eye(d) - P_along)
        scale = (torch.rand(n, 1, generator=g) * args.noise_max)  # per-token off-manifold noise (difficulty variance)
        x = c + scale * off
        return x.to(dev), c.to(dev)

    M = nn.Parameter(0.5 * torch.randn(d, d, device=dev))
    W = nn.Parameter(0.5 * torch.randn(d, d, device=dev))
    opt = torch.optim.Adam([M, W], lr=args.lr)
    target_smax = 1 - args.eps

    for step in range(args.steps):
        x, zstar = batch(args.bs)
        z = torch.zeros_like(x)
        for _ in range(K):
            z = z @ M.T + x @ W.T
        loss = ((z - zstar) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():                                    # strict-margin projection: sigma_max(M) <= 1-eps
            Um, S, Vm = torch.linalg.svd(M)
            M.copy_(Um @ torch.diag(S.clamp(max=target_smax)) @ Vm)
        if step % max(1, args.steps // 5) == 0:
            print(f"  step {step:4d}  loss {loss.item():.4e}  sigma_max(M) {sigma_max(M.detach()):.3f}")

    Md = M.detach()
    # benefit: iterate to tolerance, measure per-token iters + along-manifold work
    x, zstar = batch(args.benefit_tokens)
    z = torch.zeros_like(x); z0n = zstar.norm(dim=-1) + 1e-9
    along0 = (z - zstar) @ P_along.to(dev); along0 = along0.norm(dim=-1)
    iters = torch.zeros(x.shape[0], device=dev)
    done = torch.zeros(x.shape[0], dtype=torch.bool, device=dev)
    for k in range(args.K_eval):
        z = z @ Md.T + x @ W.detach().T
        res = (z - zstar).norm(dim=-1) / z0n
        newly = (res < args.tol) & (~done); iters[newly] = k + 1; done |= newly
    iters[~done] = args.K_eval
    along_work = 1 - ((z - zstar) @ P_along.to(dev)).norm(dim=-1) / (along0 + 1e-9)
    it = iters.float()

    # diagnostics on the constant recurrent Jacobian M
    smax = sigma_max(Md); kap = transient_kappa(Md, args.K); cond, smin_IJ = cond_I_minus(Md)
    # slow singular subspace of M vs planted manifold
    _, _, Vt = torch.linalg.svd(Md); slow = Vt[-m:].T
    angle = principal_angle_deg(slow.cpu(), U_true[:, :m])
    report_decision("T1", smax, spectral_radius(Md), num_abscissa(Md), kap, cond, smin_IJ,
                    it.std().item()/(it.mean().item()+1e-9), it.mean().item(),
                    along_work.mean().item(), args, extra=f"slow-subspace∠planted={angle:.1f}°")

# ----------------------------- T3: nonlinear attention + LoRA, emergent ------------
class AttnLoRABlock(nn.Module):
    def __init__(self, dim, rank=2, n=4):
        super().__init__()
        self.n, self.dim = n, dim
        self.q = nn.Linear(dim, dim, bias=False); self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.lora_a = nn.Linear(dim, rank, bias=False); self.lora_b = nn.Linear(rank, dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.3))
    def forward(self, z):                         # z: (..., n*dim) flattened
        s = z.reshape(-1, self.n, self.dim)
        att = torch.softmax(self.q(s) @ self.k(s).transpose(-1, -2) / self.dim**0.5, -1)
        upd = att @ self.v(s) + self.lora_b(torch.tanh(self.lora_a(s)))
        out = s - self.alpha * (s - upd)          # subtractive/residual update
        return out.reshape(z.shape)

def run_T3(args):
    dev, n, dim = args.device, args.n_tok, args.d
    D = n * dim
    torch.manual_seed(args.seed)                      # seed before block init for reproducibility
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    block = AttnLoRABlock(dim, rank=args.rank, n=n).to(dev)
    # synthetic task: map a noisy sequence to its per-position running-mean (needs iteration)
    def batch(b):
        x = torch.randn(b, n, dim, generator=g).to(dev)
        tgt = x.cumsum(1) / torch.arange(1, n+1, device=dev).view(1, n, 1)
        return x.reshape(b, D), tgt.reshape(b, D)
    opt = torch.optim.Adam(block.parameters(), lr=args.lr)
    for step in range(args.steps):
        x, tgt = batch(args.bs); z = x.clone()
        for _ in range(args.K): z = block(z)
        loss = ((z - tgt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, args.steps // 5) == 0:
            print(f"  step {step:4d}  loss {loss.item():.4e}")
    # measured JOINT Jacobian at a sampled converged-ish state (per R2)
    x, _ = batch(1); z = x.clone()
    for _ in range(args.K): z = block(z)
    z = z.reshape(D).detach().requires_grad_(True)
    J = torch.autograd.functional.jacobian(lambda zz: block(zz.reshape(1, D)).reshape(D), z)
    smax = sigma_max(J); kap = transient_kappa(J, args.K); cond, smin_IJ = cond_I_minus(J)
    print(f"\n[T3] measured at N={n} (joint dim {D})")
    report_decision("T3", smax, spectral_radius(J), num_abscissa(J), kap, cond, smin_IJ,
                    float("nan"), float("nan"), float("nan"), args,
                    extra="benefit metrics: add a difficulty-varying task + homogeneous control")

# ----------------------------- decision rule (separable verdicts) ------------------
def report_decision(tag, smax, rho, omega, kappa, cond, smin_IJ, it_cv, it_mean, along, args, extra=""):
    print(f"\n[{tag} diagnostics]  {extra}")
    print(f"  sigma_max(J)={smax:.3f}  rho(J)={rho:.3f}  num_abscissa={omega:.3f}")
    print(f"  transient kappa(k<={args.K})={kappa:.3f}  cond(I-J)={cond:.1f}  sigma_min(I-J)={smin_IJ:.4f}")
    if it_cv == it_cv:  # not nan
        print(f"  iters mean={it_mean:.1f} CV={it_cv:.2f}  along-work-frac={along:.2f}")
    stable   = (smax < 1.0) and (kappa <= args.kappa_max)
    trainable = cond <= args.cond_max
    benefit  = (it_cv != it_cv) or (it_cv >= args.cv_min)   # T3 leaves benefit to a later task
    verdict = ("GREEN" if (stable and trainable and benefit) else
               "RED-STABILITY" if not stable else
               "RED-GRADIENT" if not trainable else "RED-BENEFIT")
    print(f"  => STABLE={stable} TRAINABLE={trainable} BENEFIT={benefit}  VERDICT={verdict}")
    print("     (guards: refine grid before declaring empty; require BOTH D-arms; "
          "phantom-grad rescue before RED-GRADIENT)")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["t1", "t3"], default="t1")
    p.add_argument("--d", type=int, default=16); p.add_argument("--m", type=int, default=4)
    p.add_argument("--n_tok", type=int, default=4); p.add_argument("--rank", type=int, default=2)
    p.add_argument("--K", type=int, default=8); p.add_argument("--K_eval", type=int, default=300)
    p.add_argument("--eps", type=float, default=0.05)        # strict singular-value margin (sweep this)
    p.add_argument("--noise_max", type=float, default=2.0)
    p.add_argument("--steps", type=int, default=1500); p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3); p.add_argument("--tol", type=float, default=1e-2)
    p.add_argument("--benefit_tokens", type=int, default=512)
    p.add_argument("--kappa_max", type=float, default=2.0)   # thresholds (pre-register!)
    p.add_argument("--cond_max", type=float, default=1e3); p.add_argument("--cv_min", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--device", default="cpu")
    p.add_argument("--sweep_eps", default="", help="comma list, e.g. 0.02,0.05,0.1,0.2 (phase LINE)")
    args = p.parse_args()
    epss = [float(e) for e in args.sweep_eps.split(",")] if args.sweep_eps else [args.eps]
    for e in epss:
        args.eps = e
        print(f"\n================ {args.task.upper()}  eps={e} ================")
        (run_T1 if args.task == "t1" else run_T3)(args)

if __name__ == "__main__":
    main()
