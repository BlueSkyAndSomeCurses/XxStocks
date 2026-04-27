"""
Time-bin text pooling with a frozen or fine-tunable Hugging Face BERT encoder.

Uses ``transformers.AutoModel`` (BERT-style pooler or CLS hidden state) plus
:class:`TimeBinAggregator` for variable tweet counts per bin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from transformers import AutoModel


@dataclass
class TimeBinAggregatorConfig:
    d_model: int
    n_heads: int = 12
    dropout: float = 0.1
    out_dim: Optional[int] = None


class TimeBinAggregator(nn.Module):
    """Aggregate a variable-length set of tweet embeddings into one vector.

    Multi-head attention pooling against a learnable query; padding
    (``tweet_mask == 0``) is ignored.
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
        bsz = tweet_embeddings.size(0)
        query = self.query.expand(bsz, -1, -1)

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


def _pooled_sequence_output(
    last_hidden_state: Tensor,
    pooler_output: Optional[Tensor],
) -> Tensor:
    if pooler_output is not None:
        return pooler_output
    # Models without a pooler (e.g. DistilBERT): use first token representation.
    return last_hidden_state[:, 0]


class BertTimeBinPipeline(nn.Module):
    """Encode tweets with HF BERT (per tweet CLS / pooler) and pool per time bin."""

    def __init__(
        self,
        model_name: str,
        aggregator: TimeBinAggregator,
        *,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        d_model = int(self.bert.config.hidden_size)
        if aggregator.config.d_model != d_model:
            raise ValueError(
                f"Aggregator d_model={aggregator.config.d_model} must match "
                f"BERT hidden_size={d_model} for `{model_name}`."
            )
        self.aggregator = aggregator

    @property
    def out_dim(self) -> int:
        return self.aggregator.out_dim

    @property
    def hidden_size(self) -> int:
        return int(self.bert.config.hidden_size)

    def tweet_embeddings(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return _pooled_sequence_output(out.last_hidden_state, out.pooler_output)

    def forward(
        self,
        token_ids: Tensor,
        attention_mask: Tensor,
        tweet_mask: Tensor,
    ) -> Tensor:
        """
        Args:
            token_ids: ``[B, K, L]`` — K tweets per bin.
            attention_mask: ``[B, K, L]``.
            tweet_mask: ``[B, K]`` — 1 for real tweets, 0 for padding slots.
        """
        bsz, k, seq = token_ids.shape
        flat_ids = token_ids.view(bsz * k, seq)
        flat_attn = attention_mask.view(bsz * k, seq)
        per_tweet = self.tweet_embeddings(flat_ids, flat_attn)
        per_tweet = per_tweet.view(bsz, k, -1)
        return self.aggregator(per_tweet, tweet_mask)
