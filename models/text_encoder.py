"""
Neural text encoder for tweet feature extraction.

    B - batch of tweets / time-bins
    L - tweet token sequence length
    K - number of tweets per time bin (variable, padded with mask)
    D - ``d_model`` embedding dimension
    V - vocabulary size
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class TweetEncoderConfig:
    vocab_size: int
    max_seq_len: int = 64
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    pad_token_id: int = 0
    cls_token_id: int = 1
    mask_token_id: int = 2


class _PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1)]


class _TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, ff_mult: int, dropout: float
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
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor]) -> Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(
            h, h, h, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = x + self.dropout1(attn_out)

        h = self.norm2(x)
        x = x + self.dropout2(self.ff(h))
        return x


class TweetEncoder(nn.Module):
    """Bidirectional Transformer encoder for a single (tokenised) tweet.

    The first position is reserved for a learnable ``[CLS]`` token. Its
    final hidden state is the tweet embedding consumed downstream.
    """

    def __init__(self, config: TweetEncoderConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )
        self.pos_embed = _PositionalEmbedding(
            max_seq_len=config.max_seq_len + 1, d_model=config.d_model
        )
        self.embed_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                _TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    ff_mult=config.ff_mult,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

    @property
    def d_model(self) -> int:
        return self.config.d_model

    def _prepend_cls(self, token_ids: Tensor) -> tuple[Tensor, Tensor]:
        bsz = token_ids.size(0)
        cls = torch.full(
            (bsz, 1),
            self.config.cls_token_id,
            dtype=token_ids.dtype,
            device=token_ids.device,
        )
        return torch.cat([cls, token_ids], dim=1), cls

    def forward(
        self,
        token_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_token_states: bool = False,
    ) -> dict[str, Tensor]:
        """Encode a batch of tweets.

        Args:
            token_ids: ``[B, L]`` token ids without the CLS prefix.
            attention_mask: ``[B, L]`` with 1 for real tokens and 0 for
                padding. If ``None`` it is inferred from ``pad_token_id``.
            return_token_states: also return per-token hidden states (used
                by the MLM head during pretraining).
        """
        if attention_mask is None:
            attention_mask = (token_ids != self.config.pad_token_id).long()

        token_ids_with_cls, cls = self._prepend_cls(token_ids)
        cls_mask = torch.ones_like(cls, dtype=attention_mask.dtype)
        attn_mask = torch.cat([cls_mask, attention_mask], dim=1)
        # ``MultiheadAttention`` expects True for positions to ignore.
        key_padding_mask = attn_mask == 0

        h = self.token_embed(token_ids_with_cls)
        h = self.pos_embed(h)
        h = self.embed_dropout(h)

        for block in self.blocks:
            h = block(h, key_padding_mask=key_padding_mask)
        h = self.final_norm(h)

        out: dict[str, Tensor] = {"cls_embedding": h[:, 0, :]}
        if return_token_states:
            # Strip the CLS prefix so the caller can align with original
            # ``token_ids`` directly.
            out["token_states"] = h[:, 1:, :]
        return out


class MLMHead(nn.Module):
    """Tied-weights masked-language-modelling head used for pretraining."""

    def __init__(self, encoder: TweetEncoder) -> None:
        super().__init__()
        d_model = encoder.d_model
        self.transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        # The output bias is independent; the matrix is shared with the
        # token embedding to halve the parameter count.
        self.bias = nn.Parameter(torch.zeros(encoder.config.vocab_size))
        self._embedding = encoder.token_embed

    def forward(self, token_states: Tensor) -> Tensor:
        h = self.transform(token_states)
        return F.linear(h, self._embedding.weight, self.bias)


@dataclass
class TimeBinAggregatorConfig:
    d_model: int
    n_heads: int = 4
    dropout: float = 0.1
    out_dim: Optional[int] = None


class TimeBinAggregator(nn.Module):
    """Aggregate a variable-length set of tweet embeddings into one vector.

    Implements multi-head attention pooling against a single learnable query
    token. Padding positions (where ``tweet_mask == 0``) are ignored.
    """

    def __init__(self, config: TimeBinAggregatorConfig) -> None:
        super().__init__()
        self.config = config
        self.query = nn.Parameter(torch.zeros(1, 1, config.d_model))
        nn.init.trunc_normal_(self.query, std=0.02)

        self.attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(config.d_model)

        out_dim = config.out_dim or config.d_model
        self.proj = (
            nn.Identity() if out_dim == config.d_model else nn.Linear(config.d_model, out_dim)
        )

    @property
    def out_dim(self) -> int:
        return self.config.out_dim or self.config.d_model

    def forward(self, tweet_embeddings: Tensor, tweet_mask: Tensor) -> Tensor:
        """
        Args:
            tweet_embeddings: ``[B, K, D]`` - K tweets per bin, padded.
            tweet_mask: ``[B, K]`` with 1 for real tweets and 0 for padding.
        Returns:
            ``[B, out_dim]`` - one embedding per time bin.
        """
        bsz = tweet_embeddings.size(0)
        query = self.query.expand(bsz, -1, -1)

        # Bins with zero real tweets would yield NaNs from softmax-over-mask.
        # We patch them by attending over a fake token (zeros) and zeroing
        # the result afterwards.
        real_counts = tweet_mask.sum(dim=1)
        empty_bins = real_counts == 0
        safe_mask = tweet_mask.clone()
        if empty_bins.any():
            safe_mask[empty_bins, 0] = 1

        key_padding_mask = safe_mask == 0
        pooled, _ = self.attn(
            query,
            tweet_embeddings,
            tweet_embeddings,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        pooled = self.norm(pooled.squeeze(1))
        if empty_bins.any():
            pooled = pooled.masked_fill(empty_bins.unsqueeze(-1), 0.0)
        return self.proj(pooled)


class TextEncoderPipeline(nn.Module):
    """Convenience wrapper used at extraction time.

    Given a batch of tweets grouped by time bin (already padded), it returns
    one embedding per bin.
    """

    def __init__(
        self,
        encoder: TweetEncoder,
        aggregator: TimeBinAggregator,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.aggregator = aggregator

    @property
    def out_dim(self) -> int:
        return self.aggregator.out_dim

    @torch.no_grad()
    def encode_tweets(
        self, token_ids: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        return self.encoder(token_ids, attention_mask=attention_mask)["cls_embedding"]

    def forward(
        self,
        token_ids: Tensor,
        attention_mask: Tensor,
        tweet_mask: Tensor,
    ) -> Tensor:
        """
        Args:
            token_ids: ``[B, K, L]`` - K tweets per bin, each L tokens.
            attention_mask: ``[B, K, L]`` token padding mask.
            tweet_mask: ``[B, K]`` per-tweet validity mask within a bin.
        Returns:
            ``[B, out_dim]`` time-bin embedding.
        """
        bsz, k, seq = token_ids.shape
        flat_ids = token_ids.view(bsz * k, seq)
        flat_attn = attention_mask.view(bsz * k, seq)
        cls = self.encoder(flat_ids, attention_mask=flat_attn)["cls_embedding"]
        cls = cls.view(bsz, k, -1)
        return self.aggregator(cls, tweet_mask=tweet_mask)
