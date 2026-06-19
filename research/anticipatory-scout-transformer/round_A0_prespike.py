"""
Round A — Phase A0 analytic pre-spike (CPU, no learning).
Implements the CORRECTED diagnostics from round-A-anisotropic-operator-spike.md v2:
  - contraction certificate = sigma_max(J)  (NOT |lambda_max|)
  - on the FULL coupled Jacobian
  - transient growth kappa = max_k ||J^k||_2 over the few-loop budget
  - numerical abscissa = lambda_max((J+J^T)/2); Kreiss lower bound
  - cond_2(I-J) = sigma_max/sigma_min(I-J)
  - benefit-capability: iteration-count CV, per-iter loss drop, along-manifold work fraction
Run:  pip install numpy   then   python round_A0_prespike.py
Small matrices -> exact dense SVD/eig (power iteration only needed at scale).
"""
import numpy as np

def build_J(d, m, delta_along, eps_off, gamma, enforce_margin=None, seed=0):
    """J = Q (Lambda + gamma*N_strict) Q^T.
    Lambda: diag contraction factors -- along-manifold (first m) = 1-delta_along,
            off-manifold = 1-eps_off.  N_strict: strictly-upper-triangular non-normal
            coupling (the injected, attention-like non-normality).
    enforce_margin: if set, rescale gamma so sigma_max(J) == 1-enforce_margin
            (simulates spectral normalization = the §10.4 headline arm)."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))          # fixed orthonormal basis
    lam = np.where(np.arange(d) < m, 1 - delta_along, 1 - eps_off)
    Nst = np.triu(rng.standard_normal((d, d)), k=1)           # strictly upper-tri
    def J_of(g):
        return Q @ (np.diag(lam) + g * Nst) @ Q.T
    g = gamma
    if enforce_margin is not None:
        target = 1 - enforce_margin
        # bisect gamma so sigma_max == target (sigma_max is monotone increasing in g here)
        lo, hi = 0.0, gamma
        if smax(J_of(hi)) <= target:
            g = hi
        else:
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if smax(J_of(mid)) > target: hi = mid
                else: lo = mid
            g = lo
    Jf = J_of(g)
    if enforce_margin is not None:                # verify the bisection actually hit the margin
        assert smax(Jf) <= (1 - enforce_margin) + 1e-3, "gamma bisection failed (sigma_max not monotone in g?)"
    return Jf, Q, lam, g

def smax(M):  return np.linalg.svd(M, compute_uv=False)[0]
def smin(M):  return np.linalg.svd(M, compute_uv=False)[-1]
def rho(M):   return np.max(np.abs(np.linalg.eigvals(M)))
def num_abscissa(M): return np.max(np.linalg.eigvalsh(0.5 * (M + M.T)))

def transient_kappa(J, K):
    P, best = np.eye(J.shape[0]), 0.0
    for _ in range(K):
        P = J @ P
        best = max(best, smax(P))
    return best

def benefit(J, P_along, n_tokens=400, tol=1e-2, Kmax=400, seed=1):
    rng = np.random.default_rng(seed)
    d = J.shape[0]
    iters, along_frac = [], []
    for _ in range(n_tokens):
        z = rng.standard_normal(d); z /= np.linalg.norm(z)
        z0n = np.linalg.norm(z)
        k = 0; prev = z.copy()
        along_prog, total_prog = 0.0, 0.0          # fraction of the PATH that is along-manifold
        while np.linalg.norm(z) > tol * z0n and k < Kmax:
            z = J @ z; k += 1
            step = z - prev; prev = z.copy()
            total_prog += np.linalg.norm(step)
            along_prog += np.linalg.norm(P_along @ step)
        iters.append(k)
        along_frac.append(along_prog / (total_prog + 1e-12))
    it = np.array(iters, float)
    return dict(it_mean=it.mean(), it_cv=it.std() / (it.mean() + 1e-12),
                along_work=float(np.mean(along_frac)))

def report(tag, J, P_along, K=5):
    sM, sm_IJ = smax(J), smin(np.eye(J.shape[0]) - J)
    cond_IJ = smax(np.eye(J.shape[0]) - J) / (sm_IJ + 1e-18)
    b = benefit(J, P_along)
    print(f"\n[{tag}]")
    print(f"  rho(J)={rho(J):.3f}  sigma_max(J)={sM:.3f}  num_abscissa={num_abscissa(J):.3f}")
    print(f"  transient kappa (k<= {K}) = {transient_kappa(J,K):.3f}")
    print(f"  sigma_min(I-J)={sm_IJ:.4f}  cond_2(I-J)={cond_IJ:.1f}")
    print(f"  iters: mean={b['it_mean']:.1f} CV={b['it_cv']:.2f}  along-work-frac={b['along_work']:.2f}")
    print(f"  STABLE(sigma_max<1 & kappa bounded): {sM < 1 and transient_kappa(J,K) < 2}")

if __name__ == "__main__":
    d, m = 16, 4
    # build_J rotates the operator into a RANDOM basis Q, so the true along-manifold is
    # span(Q[:, :m]) -- transport the projector accordingly (a fixed identity-basis Pal
    # would measure the WRONG subspace).
    def P_of(Q): return Q[:, :m] @ Q[:, :m].T

    print("=== CASE 1: eig<1 but non-normal (the v1 false-GREEN) ===")
    J1,Q1,_,g1 = build_J(d, m, delta_along=0.1, eps_off=0.1, gamma=3.0, enforce_margin=None)
    report("eig~0.9, large gamma", J1, P_of(Q1))

    print("\n=== CASE 2: §10.4 headline arm (enforce sigma_max <= 1-eps) ===")
    for eps in (0.05, 0.1):
        J2,Q2,_,g2 = build_J(d, m, delta_along=eps, eps_off=0.2, gamma=3.0, enforce_margin=eps)
        report(f"strict margin eps={eps}, surviving gamma={g2:.3f}", J2, P_of(Q2))

    print("\n=== CASE 3: contraction-vs-benefit sweep (margin arm) ===")
    for eps in (0.02, 0.05, 0.1, 0.2):
        J3,Q3,_,_ = build_J(d, m, delta_along=eps, eps_off=0.2 if eps < 0.2 else 0.4,
                            gamma=0.05, enforce_margin=eps)
        report(f"eps={eps}", J3, P_of(Q3))
