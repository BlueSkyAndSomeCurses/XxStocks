"""
PyTorch dataset for grouping tweets into time bins and tokenising with
Hugging Face models (BERT, etc.) for :class:`models.text_encoder.BertTimeBinPipeline`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


@dataclass
class TimeBinBatch:
    """Batch container for :class:`TimeBinTweetDataset`."""

    time_bin: list  # length B
    token_ids: torch.Tensor  # [B, K, L]
    attention_mask: torch.Tensor  # [B, K, L]
    tweet_mask: torch.Tensor  # [B, K]


class TimeBinTweetDataset(Dataset):
    """Group tweets by time bin; pad to ``max_tweets_per_bin`` × ``max_seq_len``."""

    def __init__(
        self,
        tweets: pl.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        bins: pl.DataFrame,
        *,
        max_seq_len: int = 128,
        text_col: str = "Text",
        date_col: str = "Date",
        bin_col: str = "TimeBin",
        max_tweets_per_bin: int = 32,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_tweets_per_bin = max_tweets_per_bin

        joined = tweets.sort(date_col).join_asof(
            bins.select(pl.col(bin_col)).sort(bin_col),
            strategy="forward",
            left_on=date_col,
            right_on=bin_col,
        )

        grouped = (
            joined.drop_nulls(bin_col)
            .group_by(bin_col, maintain_order=True)
            .agg(pl.col(text_col).alias("texts"))
        )

        self._bins: list = grouped[bin_col].to_list()
        self._texts: list[list[str]] = grouped["texts"].to_list()

    def __len__(self) -> int:
        return len(self._bins)

    def _encode_one(self, text: str | None) -> tuple[list[int], list[int]]:
        enc: dict[str, Any] = self.tokenizer(
            text or "",
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors=None,
        )
        ids = enc["input_ids"]
        if isinstance(ids[0], str):
            raise TypeError("Tokenizer must return integer input_ids (set return_tensors=None).")
        attn = enc["attention_mask"]
        return list(ids), list(attn)

    def __getitem__(self, index: int) -> dict:
        texts = self._texts[index][: self.max_tweets_per_bin]
        k = self.max_tweets_per_bin
        max_len = self.max_seq_len

        pad_id = int(self.tokenizer.pad_token_id)
        ids = np.full((k, max_len), pad_id, dtype=np.int64)
        attn = np.zeros((k, max_len), dtype=np.int64)
        tweet_mask = np.zeros((k,), dtype=np.int64)

        for j, raw in enumerate(texts):
            row_ids, row_attn = self._encode_one(raw)
            if not any(row_attn):
                continue
            ids[j] = np.asarray(row_ids[:max_len], dtype=np.int64)
            attn[j] = np.asarray(row_attn[:max_len], dtype=np.int64)
            tweet_mask[j] = 1

        return {
            "time_bin": self._bins[index],
            "token_ids": torch.from_numpy(ids),
            "attention_mask": torch.from_numpy(attn),
            "tweet_mask": torch.from_numpy(tweet_mask),
        }


def collate_time_bin_batch(items: list[dict]) -> TimeBinBatch:
    return TimeBinBatch(
        time_bin=[item["time_bin"] for item in items],
        token_ids=torch.stack([item["token_ids"] for item in items], dim=0),
        attention_mask=torch.stack([item["attention_mask"] for item in items], dim=0),
        tweet_mask=torch.stack([item["tweet_mask"] for item in items], dim=0),
    )
