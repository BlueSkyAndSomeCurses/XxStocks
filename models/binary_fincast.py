from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class BinaryFinCastConfig:
    input_dim: int
    patch_len: int = 16
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 12
    ff_mult: int = 4
    dropout: float = 0.1
    n_freqs: int = 8


class ResidualMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.dropout(self.net(x))


class BinaryClassificationHead(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, 1)

    def forward(self, h_last: Tensor) -> Tensor:
        h_last = self.norm(h_last)
        h_last = self.dropout(h_last)
        return self.proj(h_last).squeeze(-1)


class SparseMoE(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, n_experts: int = 4, top_k: int = 2) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, d_model),
                )
                for _ in range(n_experts)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, N, D]
        gate_scores = torch.softmax(self.gate(x), dim=-1)
        top_vals, top_idx = torch.topk(gate_scores, k=self.top_k, dim=-1)

        out = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx_k = top_idx[..., k]  # [B, N]
            gate_k = top_vals[..., k].unsqueeze(-1)  # [B, N, 1]

            expert_out = torch.zeros_like(x)
            for expert_id, expert in enumerate(self.experts):
                mask = expert_idx_k == expert_id
                if mask.any():
                    expert_out[mask] = expert(x[mask])

            out = out + gate_k * expert_out

        return out


class DecoderMoEBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_mult: int,
        dropout: float = 0.1,
        n_experts: int = 4,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.moe = SparseMoE(
            d_model=d_model,
            hidden_dim=d_model * ff_mult,
            n_experts=n_experts,
            top_k=top_k,
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout1(attn_out)

        h = self.norm2(x)
        x = x + self.dropout2(self.moe(h))
        return x


class BinaryFinCast(nn.Module):
    def __init__(self, config: BinaryFinCastConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_len = config.patch_len

        patch_dim = config.patch_len * config.input_dim
        self.input_residual = ResidualMLP(
            dim=patch_dim,
            hidden_dim=max(patch_dim * 2, config.d_model),
            dropout=config.dropout,
        )
        self.patch_to_model = nn.Linear(patch_dim, config.d_model)
        self.freq_embed = nn.Embedding(config.n_freqs, config.d_model)

        self.backbone = nn.ModuleList(
            [
                DecoderMoEBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    ff_mult=config.ff_mult,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

        self.classification_head = BinaryClassificationHead(
            d_model=config.d_model,
            dropout=config.dropout,
        )

    @staticmethod
    def _make_causal_mask(seq_len: int, device: torch.device) -> Tensor:
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    def _patchify(self, x: Tensor) -> Tensor:
        bsz, seq_len, channels = x.shape
        n_patches = seq_len // self.patch_len
        if n_patches < 1:
            raise ValueError("Sequence length must be >= patch_len.")

        x = x[:, : n_patches * self.patch_len, :]
        x = x.reshape(bsz, n_patches, self.patch_len * channels)
        return x

    def _instance_norm(self, patches: Tensor, mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor, Tensor]:
        """
        Instance normalization at patch level.
        """
        eps = 1e-6
        if mask is None:
            mu = patches.mean(dim=-1, keepdim=True)
            var = patches.var(dim=-1, unbiased=False, keepdim=True)
        else:
            valid = 1.0 - mask.float()
            denom = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
            mu = (patches * valid).sum(dim=-1, keepdim=True) / denom
            var = ((patches - mu) ** 2 * valid).sum(dim=-1, keepdim=True) / denom

        sigma = torch.sqrt(var + eps)
        x_norm = (patches - mu) / sigma
        return x_norm, mu, sigma

    def load_pretrained_backbone(self, state_dict: dict, strict: bool = False) -> None:
        self.load_state_dict(state_dict, strict=strict)

    def set_lightweight_finetune(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

        for p in self.classification_head.parameters():
            p.requires_grad = True

        n_layers = len(self.backbone)
        n_unfreeze = max(1, int(0.1 * n_layers))
        for block in self.backbone[-n_unfreeze:]:
            for p in block.parameters():
                p.requires_grad = True

    def forward(
        self,
        x: Tensor,
        freq_id: Tensor,
        target: Optional[Tensor] = None,
        patch_mask: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        patches = self._patchify(x)  # [B, N, Dp]
        patches, _, _ = self._instance_norm(patches, mask=patch_mask)

        h = self.input_residual(patches)
        h = self.patch_to_model(h)  # [B, N, D]
        h = h + self.freq_embed(freq_id).unsqueeze(1)

        seq_len = h.size(1)
        causal_mask = self._make_causal_mask(seq_len=seq_len, device=h.device)
        for block in self.backbone:
            h = block(h, attn_mask=causal_mask)
        h = self.final_norm(h)

        h_last = h[:, -1, :]
        logits = self.classification_head(h_last)  # [B]

        out = {
            "logits": logits,
            "probs": torch.sigmoid(logits),
        }

        if target is not None:
            y = target.float().view(-1)
            out["loss"] = F.binary_cross_entropy_with_logits(logits, y)

        return out
