"""Symmetry-Compatible Optimizers for MoE Router Matrices.

Implements the centered row-norm and left-spectral update rules from
arXiv:2605.18106 (Lau & Su, May 2026).  These optimizers preserve the
expert-permutation symmetry and shared-logit-shift invariance of MoE
router weight matrices, unlike standard AdamW which breaks them.

Classes:
    CenteredRowNorm:  Local per-expert row-normalized update.
    LeftSpectral:     Global left-spectral (polar-like) update.
    build_symmetry_optimizer:  Factory that finds MoE router params
                               in a model and returns the appropriate
                               symmetry-aware optimizer.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer

__all__ = [
    "CenteredRowNorm",
    "LeftSpectral",
    "build_symmetry_optimizer",
]


# ---------------------------------------------------------------------------
# Helper: center rows (remove shared-logit component)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _center_rows(W: torch.Tensor) -> torch.Tensor:
    """Remove the shared-logit component from rows.

    Π⊥ = I - (1/e)·1·1^T  where e = number of rows (experts).

    Args:
        W: (e, d) matrix.

    Returns:
        Row-centered copy of W (each row has the column-wise mean subtracted).
    """
    return W - W.mean(dim=0, keepdim=True)


# ---------------------------------------------------------------------------
# Centered Row-Norm Optimizer
# ---------------------------------------------------------------------------


class CenteredRowNorm(Optimizer):
    """Symmetry-compatible optimizer using centered row-norm updates.

    Implements the update from arXiv:2605.18106 §3.6:

        D_c = Π⊥ · D          (center rows — remove shared component)
        η_i = 1 / (‖D_c[i]‖₂ + ε)
        ΔW  = -lr · Π⊥ · diag(η) · D_c   (row-norm scaling + re-center)

    where D is the EMA momentum of the gradient.

    This update is equivariant under expert-permutation and invariant
    under shared-logit-shift — matching the MoE router's symmetry group.

    Args:
        params: Iterable of parameters (typically MoE gate weights).
        lr: Learning rate.
        beta: Momentum coefficient (default 0.95).
        weight_decay: Decoupled weight decay (default 0.0).
        eps: Small constant for numerical stability in row-norm scaling.
        row_mode: Normalization strategy — ``"inverse_eps"``
            (default, 1/(norm + eps)) or ``"unit"`` (1/clamp(norm, min=eps)).
    """

    def __init__(
        self,
        params,
        lr: float = 5e-4,
        beta: float = 0.95,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        row_mode: str = "inverse_eps",
    ) -> None:
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        if row_mode not in ("inverse_eps", "unit"):
            raise ValueError(
                f"row_mode must be 'inverse_eps' or 'unit', got {row_mode!r}"
            )

        defaults = dict(
            lr=lr,
            beta=beta,
            weight_decay=weight_decay,
            eps=eps,
            row_mode=row_mode,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            wd = group["weight_decay"]
            eps = group["eps"]
            row_mode = group["row_mode"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad.float()
                state = self.state[p]

                # EMA momentum
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(g)
                buf = state["momentum"]
                buf.mul_(beta).add_(g, alpha=1.0 - beta)

                # Decoupled weight decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # Step 1: Center rows (remove shared-logit component)
                m_c = _center_rows(buf)

                # Step 2: Compute row norms and scaling factor
                row_norms = m_c.norm(dim=-1, keepdim=True)  # (e, 1)
                if row_mode == "inverse_eps":
                    scale = 1.0 / (row_norms + eps)
                else:  # "unit"
                    scale = 1.0 / row_norms.clamp_min(eps)

                # Step 3: Apply row-norm scaling
                update = scale * m_c  # (e, d)

                # Step 4: Re-center so update stays in the quotient subspace
                update = _center_rows(update)

                # Apply update
                p.add_(update.to(p.dtype), alpha=-lr)

        return loss


# ---------------------------------------------------------------------------
# Left-Spectral Optimizer
# ---------------------------------------------------------------------------


class LeftSpectral(Optimizer):
    """Left-spectral (polar-like) optimizer for MoE router matrices.

    Implements the update from arXiv:2605.18106 §3.6:

        D_c = Π⊥ · D                     (center rows)
        C   = D_c @ D_c^T                (centered Gram matrix, e×e)
        L   = C^(-1/2)                   (inverse matrix square root)
        ΔW  = -lr · tr(C·L)^α · L @ D_c  (left-spectral scaling)

    This globally mixes information across all experts through the
    centered Gram matrix before applying the update.

    NOTE: Uses eigendecomposition (``torch.linalg.eigh``) for the matrix
    inverse square root, which is O(e³) in the number of experts.
    For e ≤ 256 this is negligible; for larger e consider Newton-Schulz
    iterations.

    Args:
        params: Iterable of parameters.
        lr: Learning rate.
        beta: Momentum coefficient (default 0.95).
        alpha: Spectral scaling exponent (default 0.5).
            0 = no spectral scaling, 0.5 = sqrt scaling, 1.0 = linear.
        weight_decay: Decoupled weight decay (default 0.0).
        eps: Small constant for eigenvalue clamping.
    """

    def __init__(
        self,
        params,
        lr: float = 5e-4,
        beta: float = 0.95,
        alpha: float = 0.5,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0, 1), got {beta}")

        defaults = dict(
            lr=lr,
            beta=beta,
            alpha=alpha,
            weight_decay=weight_decay,
            eps=eps,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _matrix_inv_sqrt(M: torch.Tensor, eps: float) -> torch.Tensor:
        """Compute M^(-1/2) via eigendecomposition.

        Args:
            M: (e, e) symmetric positive-semidefinite matrix.
            eps: Clamp floor for eigenvalues.

        Returns:
            (e, e) matrix square root inverse.
        """
        L, Q = torch.linalg.eigh(M)
        L = L.clamp_min(eps)
        return (Q * L.rsqrt().unsqueeze(-2)) @ Q.transpose(-2, -1)

    @torch.no_grad()
    def step(self, closure=None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            alpha = group["alpha"]
            wd = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad.float()
                state = self.state[p]

                # EMA momentum
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(g)
                buf = state["momentum"]
                buf.mul_(beta).add_(g, alpha=1.0 - beta)

                # Decoupled weight decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # Step 1: Center rows (remove shared-logit component)
                m_c = _center_rows(buf)

                # Step 2: Centered Gram matrix D_c @ D_c^T, shape (e, e)
                C = m_c @ m_c.transpose(-1, -2)

                # Step 3: Inverse square root of Gram
                L_inv_sqrt = self._matrix_inv_sqrt(C, eps=eps)

                # Step 4: Spectral scaling factor ν = tr(C · C^(-1/2))
                if alpha != 0.0:
                    nu = torch.trace(C @ L_inv_sqrt)
                    scale = nu.clamp_min(eps).pow(alpha)
                else:
                    scale = 1.0

                # Step 5: Apply left-spectral update
                update = scale * (L_inv_sqrt @ m_c)  # (e, d)

                # Step 6: Re-center for numerical safety
                update = _center_rows(update)

                p.add_(update.to(p.dtype), alpha=-lr)

        return loss


# ---------------------------------------------------------------------------
# Factory: build symmetry-aware optimizer for MoE router params
# ---------------------------------------------------------------------------


def build_symmetry_optimizer(
    model: torch.nn.Module,
    mode: str = "centered_row_norm",
    lr: float = 5e-4,
    beta: float = 0.95,
    weight_decay: float = 0.0,
    **kwargs,
) -> CenteredRowNorm | LeftSpectral | None:
    """Identify MoE router parameters in *model* and return a
    symmetry-compatible optimizer for them.

    Searches for parameters whose name contains ``gate.weight``,
    ``router.weight``, or ``proj.weight`` — the typical MoE router
    linear projections.

    Args:
        model: The PyTorch model (trainable parameters are inspected).
        mode: ``"centered_row_norm"`` (default) or ``"left_spectral"``.
        lr: Learning rate for the router optimizer.
        beta: Momentum coefficient.
        weight_decay: Decoupled weight decay.
        **kwargs: Additional keyword arguments forwarded to the optimizer
            constructor (e.g. ``row_mode`` for CenteredRowNorm, ``alpha``
            for LeftSpectral).

    Returns:
        A symmetry-compatible optimizer instance, or ``None`` if no
        router parameters were found.
    """
    router_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "gate.weight" in name or "router.weight" in name or "proj.weight" in name:
            router_params.append(param)

    if not router_params:
        return None

    if mode == "centered_row_norm":
        return CenteredRowNorm(
            router_params, lr=lr, beta=beta, weight_decay=weight_decay, **kwargs
        )
    elif mode == "left_spectral":
        return LeftSpectral(
            router_params, lr=lr, beta=beta, weight_decay=weight_decay, **kwargs
        )
    else:
        raise ValueError(
            f"Unknown symmetry optimizer mode: {mode!r}. "
            f"Expected 'centered_row_norm' or 'left_spectral'."
        )
