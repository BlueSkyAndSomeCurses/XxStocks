"""Pure-PyTorch Mamba (S6) implementation tailored for tabular time-series.

Design notes / references
-------------------------
* Shi 2024 ("MambaStock", arXiv:2402.18959): drives a Mamba block at state
  size ``N = 16`` over a window of historical bars, projects to a 1-D
  prediction. We keep the same S6 selection mechanism (data-dependent
  ``B, C, Δ``) and the same N=16 default.
* Hu et al. KDD 2026 ("How to Train Your Mamba for Time Series Forecasting"):
  for TSF, the *only* component of Mamba that matters is the time-varying
  selective SSM kernel. The auxiliary short-conv / gating layers in the
  Mamba block "introduce task-agnostic inductive bias and lead to overfitting
  in time-series forecasting tasks". We therefore expose ``use_short_conv``
  and ``use_gate`` so the playground can disable them, and we default to
  ``use_short_conv=False`` for the small TSF window we operate on.

We deliberately avoid the ``mamba-ssm`` package: it requires custom CUDA
kernels and Triton, which collides with the project's pinned ``torch==2.10``
on Windows. Sequence length here is ``window_size = 32``, so the naive O(L)
recurrent scan is not a bottleneck.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class MambaConfig:
    """Hyper-parameters for ``MambaModel``.

    Attributes mirror Mamba paper notation:
      * ``input_dim``: number of input channels per time step (D in S4 paper).
      * ``d_model``: post-embedding hidden dimension that the SSM operates on.
      * ``d_state``: state-space size N (MambaStock uses 16).
      * ``d_conv``: kernel size of the optional short causal conv.
      * ``expand``: inner-block expansion factor; inner dim = expand * d_model.
      * ``n_layers``: stacked Mamba blocks (Hu et al. recommend 1-2 for TSF).
      * ``dt_rank``: low-rank dim used to project x -> Δ. ``"auto"`` => ceil(d_inner/16).
      * ``use_short_conv`` / ``use_gate``: ablations from Hu et al. (RQ1).
    """

    input_dim: int
    d_model: int = 64
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    n_layers: int = 2
    dropout: float = 0.1
    dt_rank: int | str = "auto"
    use_short_conv: bool = False
    use_gate: bool = True


class SelectiveSSM(nn.Module):
    """Time-varying (selective) S6 kernel.

    Implements
        h_t = exp(Δ_t · A) · h_{t-1} + Δ_t · B_t · x_t
        y_t = C_t · h_t + D ⊙ x_t
    with
        A     ∈ R^{D_inner × N}   (real diagonal, parameterised in log-space
                                   so that A = -exp(A_log) is always negative,
                                   initialised to -1, -2, ..., -N as in
                                   Mamba / Hu et al. eq. (6))
        B_t   ∈ R^{B × L × N}      (linear projection of x_t)
        C_t   ∈ R^{B × L × N}      (linear projection of x_t)
        Δ_t   ∈ R^{B × L × D_inner} (low-rank projection + softplus)
        D     ∈ R^{D_inner}        (per-channel skip)

    The recurrent scan is unrolled with a Python loop over the (short) time
    axis. For ``L = 32`` and the dataset we have, this is well within budget.
    """

    def __init__(
        self,
        d_inner: int,
        d_state: int = 16,
        dt_rank: int | str = "auto",
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
    ) -> None:
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = (
            math.ceil(d_inner / 16) if dt_rank == "auto" else int(dt_rank)
        )

        # x -> [Δ_pre | B | C], all data-dependent (the "selection mechanism").
        self.x_proj = nn.Linear(
            d_inner, self.dt_rank + 2 * d_state, bias=False
        )
        # Δ_pre -> Δ before softplus.
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)

        # Mamba's Δ initialisation: bias chosen so that softplus(b) is
        # uniformly distributed in [dt_min, dt_max]. Stops the system from
        # decaying too fast / saturating immediately at init.
        dt_init = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt_init + torch.log1p(-torch.exp(-dt_init).clamp(max=1.0 - 1e-6))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Match Mamba: don't decay this bias via Adam's weight decay heuristics.
        self.dt_proj.bias._no_weight_decay = True  # type: ignore[attr-defined]

        # A in log-space so it stays negative.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]

        # Per-channel skip (the "D" term in S4).
        self.D = nn.Parameter(torch.ones(d_inner))
        self.D._no_weight_decay = True  # type: ignore[attr-defined]

    def forward(self, x: Tensor) -> Tensor:
        """``x``: (B, L, D_inner) → ``y``: (B, L, D_inner)."""
        bsz, seq_len, d_inner = x.shape
        assert d_inner == self.d_inner

        # Generate Δ_pre, B, C from x (the selection mechanism).
        x_dbl = self.x_proj(x)
        delta_pre, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta_pre))  # (B, L, D_inner)

        # A negative diagonal of shape (D_inner, N).
        A = -torch.exp(self.A_log)

        # Discretisation:
        #   Â = exp(Δ ⊗ A)    (zero-order hold on A)
        #   B̂_t · x_t = (Δ_t · B_t) · x_t   (forward Euler on B, channel-wise)
        # Shapes:
        #   delta:      (B, L, D_inner)        -> (B, L, D_inner, 1)
        #   A:          (D_inner, N)           -> (1, 1, D_inner, N)
        #   B:          (B, L, N)              -> (B, L, 1, N)
        #   x:          (B, L, D_inner)        -> (B, L, D_inner, 1)
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        deltaB_x = (
            delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)
        )  # (B, L, D_inner, N)

        # Sequential scan along time (windows are short here; Δ-A is data
        # dependent so the cheap parallel-prefix trick from S4 doesn't apply).
        h = x.new_zeros(bsz, d_inner, self.d_state)
        ys: list[Tensor] = []
        for t in range(seq_len):
            h = deltaA[:, t] * h + deltaB_x[:, t]  # (B, D_inner, N)
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # (B, D_inner)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, L, D_inner)

        return y + self.D * x


class MambaBlock(nn.Module):
    """One Mamba block: in-projection (+ optional gate / short conv) → S6 → out-proj.

    Set ``use_short_conv=False`` and/or ``use_gate=False`` to recover the
    "pure SSM kernel" configuration that the KDD 2026 paper recommends for
    time-series forecasting (RQ1: those auxiliary modules tend to overfit).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_short_conv: bool = False,
        use_gate: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.use_short_conv = use_short_conv
        self.use_gate = use_gate
        self.d_conv = d_conv

        proj_out = 2 * self.d_inner if use_gate else self.d_inner
        self.in_proj = nn.Linear(d_model, proj_out, bias=False)

        if use_short_conv:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                bias=True,
            )

        self.ssm = SelectiveSSM(d_inner=self.d_inner, d_state=d_state)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        proj = self.in_proj(x)
        if self.use_gate:
            x_in, gate = proj.chunk(2, dim=-1)
        else:
            x_in, gate = proj, None

        if self.use_short_conv:
            # (B, L, D) → (B, D, L) → conv → trim right-padding for causality.
            x_in = x_in.transpose(1, 2)
            x_in = self.conv1d(x_in)[..., :seq_len]
            x_in = x_in.transpose(1, 2)

        x_in = F.silu(x_in)
        y = self.ssm(x_in)

        if gate is not None:
            y = y * F.silu(gate)

        return self.dropout(self.out_proj(y))


class MambaModel(nn.Module):
    """Stack of Mamba blocks → last-token linear head, returning (out, None, None).

    The ``(out, None, None)`` return signature mirrors ``LSTMModel`` so the
    training / rolling-forecast utilities in ``models_playground.py`` can use
    this model interchangeably with the LSTM and FinCast adapters.

    For binary tasks the head outputs raw logits (BCEWithLogitsLoss expected).
    For continuous tasks the head outputs a raw scalar (MSELoss expected). The
    target standardisation in the playground (see ``TargetScaler``) makes the
    continuous head behave well even at log-return scale.
    """

    def __init__(self, config: MambaConfig) -> None:
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(config.input_dim, config.d_model)
        self.input_norm = nn.LayerNorm(config.d_model)

        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    d_model=config.d_model,
                    d_state=config.d_state,
                    d_conv=config.d_conv,
                    expand=config.expand,
                    dropout=config.dropout,
                    use_short_conv=config.use_short_conv,
                    use_gate=config.use_gate,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(config.d_model) for _ in range(config.n_layers)]
        )

        self.final_norm = nn.LayerNorm(config.d_model)
        self.head_dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.d_model, 1)

    def forward(
        self, x: Tensor, hidden: object = None
    ) -> tuple[Tensor, None, None]:
        # ``hidden`` is accepted only to mirror ``LSTMModel.forward``.
        del hidden
        h = self.input_norm(self.input_proj(x))
        for block, norm in zip(self.blocks, self.norms):
            # Pre-norm residual, in line with modern SSM stacks.
            h = h + block(norm(h))
        h = self.final_norm(h)
        h_last = h[:, -1, :]
        out = self.head(self.head_dropout(h_last))  # (B, 1)
        return out, None, None


class BinaryMamba(MambaModel):
    """Alias kept for symmetry with ``BinaryFinCast``.

    The architecture is identical to ``MambaModel``; the binary head simply
    outputs raw logits to be paired with ``BCEWithLogitsLoss``.
    """


class ContinuousMamba(MambaModel):
    """Alias kept for symmetry with ``ContinuousFinCast``.

    Outputs a raw scalar to be paired with ``MSELoss`` over a standardised
    target (see ``TargetScaler`` in ``models/evaluation.py``).
    """
