"""Mamba-2 Selective State Space Model block.

Reference
---------
Dao & Gu (2024) "Transformers are SSMs: Generalized Models and Efficient
Algorithms Through Structured State Space Duality". arXiv:2405.21060.

This is a minimal, self-contained implementation for use as the
per-layer working-memory substrate (Tier-1) of the Aurelian Memory
Core. It exposes ``get_state()`` / ``set_state()`` / ``reset_state()``
so the AMC layer can inspect and checkpoint the recurrent state.

Implementation notes
--------------------
This implementation uses a sequential scan for clarity and easy
debugging. A production Mamba-2 uses associative-parallel scan
(``mamba_ssm``) for O(L) parallel-trainable speed. For research
iteration on a 1B model this sequential reference is sufficient —
swap in ``mamba_ssm`` later without changing the public API.

State shape
-----------
``state: (B, nheads, headdim, d_state)``

This is the standard Mamba-2/SSD shape: each head carries a
``d_state``-slot memory, with a ``headdim``-wide input/output
interface. The state is updated via zero-order-hold (ZOH)
discretization of the continuous SSM:

    A_disc = exp(Δ * A)                       (diagonal, per-head)
    B_disc = (exp(Δ * A) - I) * A^-1 * Δ * B  ≈  Δ * B  (ZOH)

    h_t = A_disc * h_{t-1}  +  B_disc * x_t
    y_t = C_t @ h_t         +  D * x_t
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


@dataclass
class Mamba2Config:
    """Configuration for a single Mamba-2 block.

    ``d_inner = d_model * expand`` is the working dimension after the
    input projection. Multi-head attention-style parallelism splits
    ``d_inner`` into ``nheads = d_inner // headdim`` heads, each with
    a ``d_state``-wide recurrent state.
    """

    d_model: int
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    A_init_range: tuple[float, float] = (1.0, 16.0)

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.d_state <= 0:
            raise ValueError(f"d_state must be positive, got {self.d_state}")
        if self.headdim <= 0:
            raise ValueError(f"headdim must be positive, got {self.headdim}")
        if self.d_inner % self.headdim != 0:
            raise ValueError(
                f"d_inner ({self.d_inner}) must be divisible by headdim ({self.headdim})"
            )
        if self.expand <= 0:
            raise ValueError(f"expand must be positive, got {self.expand}")
        if self.d_conv < 1:
            raise ValueError(f"d_conv must be >= 1, got {self.d_conv}")
        if not (0.0 < self.dt_min <= self.dt_max):
            raise ValueError(f"dt_min/dt_max invalid: {self.dt_min}/{self.dt_max}")
        if not (0.0 < self.A_init_range[0] <= self.A_init_range[1]):
            raise ValueError(f"A_init_range invalid: {self.A_init_range}")

    @property
    def d_inner(self) -> int:
        return self.d_model * self.expand

    @property
    def nheads(self) -> int:
        return self.d_inner // self.headdim


class Mamba2Block(nn.Module):
    """Mamba-2 Selective State Space block.

    Input:  ``(B, L, d_model)``
    Output: ``(B, L, d_model)``

    Internal state (``state``) has shape
    ``(B, nheads, headdim, d_state)`` and is introspectable via
    :meth:`get_state` for AMC checkpointing.

    The forward pass has four stages:

    1. **Joint projection** — a single linear maps ``x`` to
       ``(x_proj, z, B, C, dt)``.
    2. **Short causal convolution + SiLU** on ``x_proj``.
    3. **Selective recurrent scan** per head, per token.
    4. **Gated output projection** — ``y = SiLU(z) * (C @ h + D * x)``
       projected back to ``d_model``.
    """

    def __init__(self, config: Mamba2Config) -> None:
        super().__init__()
        self.config = config
        d_in = config.d_model
        d_inner = config.d_inner
        nheads = config.nheads
        d_state = config.d_state
        ngroups = config.ngroups

        # --- 1. Joint input projection ----------------------------------
        # Sizes in the concatenated output:
        #   x_proj : d_inner                      (main stream)
        #   z      : d_inner                      (gate)
        #   B      : ngroups * d_state             (input SSM matrix)
        #   C      : ngroups * d_state             (output SSM matrix)
        #   dt     : nheads                        (per-head discretization)
        in_proj_dim = d_inner + d_inner + ngroups * d_state + ngroups * d_state + nheads
        self.in_proj = nn.Linear(d_in, in_proj_dim, bias=False)

        # --- 2. Short causal depth-wise convolution ---------------------
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=config.d_conv,
            groups=d_inner,
            padding=config.d_conv - 1,
            bias=True,
        )

        # --- 3. SSM parameters ------------------------------------------
        # A is stored in log space for stability: the continuous-time
        # A is always negative (diagonal), so we parameterize A_log
        # with exp(A_log) > 0 and negate at use time.
        A = torch.empty(nheads).uniform_(*config.A_init_range).log()
        self.A_log = nn.Parameter(A)

        # D is a per-head skip-residual.
        self.D = nn.Parameter(torch.zeros(nheads))

        # dt_bias is initialized so softplus(dt_bias) lands in
        # [dt_min, dt_max]. Inverse softplus: dt = log(exp(x) - 1).
        dt_init = torch.exp(
            torch.rand(nheads) * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        )
        inv_softplus = torch.log(torch.exp(dt_init) - 1.0)
        self.dt_bias = nn.Parameter(inv_softplus)
        self.dt_init_floor = config.dt_init_floor

        # --- 4. Output projection ---------------------------------------
        self.out_proj = nn.Linear(d_inner, d_in, bias=False)

        self.reset_parameters()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def reset_parameters(self) -> None:
        # Conv init: small weights to behave like identity at start.
        nn.init.zeros_(self.conv1d.bias)
        nn.init.zeros_(self.conv1d.weight)
        # Set the "current token" tap (last position in the kernel)
        # to 1 so x passes through the convolution unchanged at init.
        with torch.no_grad():
            self.conv1d.weight[:, 0, -1] = 1.0
        # Identity-init the input projection's x_proj slice by scaling
        # its weight block. Full identity on a fat projection is not
        # possible, so we scale down to keep activations modest.
        nn.init.kaiming_uniform_(self.in_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.out_proj.weight, a=math.sqrt(5))

    # ------------------------------------------------------------------
    # State introspection (AMC checkpoint surface)
    # ------------------------------------------------------------------
    def reset_state(self) -> None:
        """Clear cached recurrent state between sessions."""
        self._state: torch.Tensor | None = None

    def get_state(self) -> torch.Tensor | None:
        """Return the cached recurrent state, or ``None`` if not set."""
        return getattr(self, "_state", None)

    def set_state(self, state: torch.Tensor) -> None:
        """Set the cached recurrent state (for checkpoint resumption)."""
        expected = (self.config.nheads, self.config.headdim, self.config.d_state)
        if state.dim() == 3:
            # (nheads, headdim, d_state) — broadcast over batch later.
            pass
        elif state.dim() == 4:
            expected = (state.shape[0],) + expected
        else:
            raise ValueError(
                f"state must be 3-(H,HD,N) or 4-(B,H,HD,N), got shape {tuple(state.shape)}"
            )
        if tuple(state.shape) != expected:
            raise ValueError(f"state shape {tuple(state.shape)} != expected {expected}")
        self._state = state

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        *,
        step: int = 0,
        prev_state: torch.Tensor | None = None,
        return_state: bool = False,
    ):
        """Run one block forward.

        Parameters
        ----------
        x : (B, L, d_model)
        step : monotonic step counter. Recorded on the returned state
            so checkpoints can be correlated.
        prev_state : optional ``(B, H, HD, N)`` recurrent state. If
            ``None``, state starts at zero for this call.
        return_state : if ``True``, return ``(output, state_dict)``.

        Returns
        -------
        output : ``(B, L, d_model)`` if ``return_state`` is False.
        (output, state_dict) : tuple if ``return_state`` is True.
            ``state_dict["ssm_state"]`` has shape ``(B, H, HD, N)``.
        """
        cfg = self.config
        B, L, D = x.shape
        if D != cfg.d_model:
            raise ValueError(f"expected last dim {cfg.d_model}, got {D}")

        # 1. Joint projection -------------------------------------------------
        zxbcdt = self.in_proj(x)
        splits = (
            cfg.d_inner,
            cfg.d_inner,
            cfg.ngroups * cfg.d_state,
            cfg.ngroups * cfg.d_state,
            cfg.nheads,
        )
        x_proj, z, B_raw, C_raw, dt_raw = zxbcdt.split(splits, dim=-1)

        # 2. Short causal depth-wise conv + SiLU on x_proj -------------------
        x_conv = x_proj.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[..., :L]  # truncate right padding
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # Reshape x_conv to multi-head: (B, L, H, HD)
        x_h = rearrange(x_conv, "b l (h hd) -> b l h hd", h=cfg.nheads, hd=cfg.headdim)

        # 3. Broadcast B and C to heads via ``ngroups``. ----------------------
        # When ngroups == nheads each head has its own B/C. When ngroups == 1
        # all heads share one. General case: group index = h // (nheads // ngroups).
        heads_per_group = cfg.nheads // cfg.ngroups
        B_g = rearrange(B_raw, "b l (g n) -> b l g n", g=cfg.ngroups, n=cfg.d_state)
        C_g = rearrange(C_raw, "b l (g n) -> b l g n", g=cfg.ngroups, n=cfg.d_state)
        # Broadcast group → head: (B, L, H, N)
        B_h = repeat(B_g, "b l g n -> b l (g rep) n", rep=heads_per_group)
        C_h = repeat(C_g, "b l g n -> b l (g rep) n", rep=heads_per_group)

        # 4. Discretization Δ via softplus on learned bias --------------------
        dt = F.softplus(dt_raw + self.dt_bias)  # (B, L, H)
        dt = torch.clamp(dt, min=self.dt_init_floor)

        # 5. Continuous-time A (diagonal, per-head) ---------------------------
        A = -torch.exp(self.A_log.float())  # (H,)

        # 6. Sequential selective scan ---------------------------------------
        # state: (B, H, HD, N). Initialize from prev_state or zeros.
        if prev_state is not None:
            if prev_state.shape[0] != B:
                raise ValueError(f"prev_state batch {prev_state.shape[0]} != input batch {B}")
            state = prev_state.clone()
        else:
            state = torch.zeros(
                B,
                cfg.nheads,
                cfg.headdim,
                cfg.d_state,
                device=x.device,
                dtype=x.dtype,
            )

        outputs: list[torch.Tensor] = []
        # Pre-compute per-head A_disc, B_disc, C per token inside the loop.
        for t in range(L):
            x_t = x_h[:, t]  # (B, H, HD)
            B_t = B_h[:, t]  # (B, H, N)
            C_t = C_h[:, t]  # (B, H, N)
            dt_t = dt[:, t]  # (B, H)

            # ZOH discretization: A_disc = exp(Δ*A). Δ is (B, H), A is (H,).
            # Result shape: (B, H, 1, 1) for broadcasting across (HD, N).
            A_disc = torch.exp(dt_t.unsqueeze(-1).unsqueeze(-1) * A.view(1, cfg.nheads, 1, 1)).to(
                x.dtype
            )

            # B_disc ≈ Δ * B, lifted to (B, H, 1, N) for multiplication with
            # x_t which is (B, H, HD). We form the outer product per head:
            #   (B, H, HD, N) = x_t[...,None] * B_t[:, :, None, :]
            B_t_scaled = (dt_t.unsqueeze(-1) * B_t).to(x.dtype)  # (B, H, N)
            Bx_t = x_t.unsqueeze(-1) * B_t_scaled.unsqueeze(-2)  # (B, H, HD, N)

            # Recurrent update.
            state = A_disc * state + Bx_t

            # Output: y_t = C_t @ h_t + D * x_t
            # C_t: (B, H, N), state: (B, H, HD, N) → y: (B, H, HD)
            y_t = torch.einsum("bhn,bhkn->bhk", C_t, state)
            y_t = y_t + self.D.view(1, cfg.nheads, 1) * x_t

            outputs.append(y_t)

        # Stack → (B, L, H, HD) → (B, L, d_inner)
        y = torch.stack(outputs, dim=1)
        y = rearrange(y, "b l h hd -> b l (h hd)")

        # 7. Gate with SiLU(z) -----------------------------------------------
        y = y * F.silu(z)

        # 8. Output projection back to d_model --------------------------------
        out = self.out_proj(y)

        state_dict = {"ssm_state": state, "step": step + L}
        self._state = state.detach()

        if return_state:
            return out, state_dict
        return out

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        cfg = self.config
        return (
            f"d_model={cfg.d_model}, d_inner={cfg.d_inner}, "
            f"nheads={cfg.nheads}, headdim={cfg.headdim}, "
            f"d_state={cfg.d_state}, d_conv={cfg.d_conv}, "
            f"ngroups={cfg.ngroups}"
        )


__all__ = ["Mamba2Block", "Mamba2Config"]
