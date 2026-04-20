"""
Tokeniser and PyTorch datasets for the neural text-encoder pipeline.

Two datasets are provided:

* :class:`MLMTweetDataset` – self-supervised masked-language-modelling
  pretraining over individual tweets.
* :class:`TimeBinTweetDataset` – inference dataset that groups tweets into
  fixed-size buckets by time bin (matching the 30-min aggregation used in
  ``dataset/preprocessing.py``).

The tokeniser is intentionally lightweight: the input text has already been
normalised by ``data_preparation.py`` (lowercased, no URLs, no punctuation,
stop-words removed), so a whitespace tokeniser with a frequency-pruned
vocabulary is enough and keeps the encoder portable.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


PAD_TOKEN = "<pad>"
CLS_TOKEN = "<cls>"
MASK_TOKEN = "<mask>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = (PAD_TOKEN, CLS_TOKEN, MASK_TOKEN, UNK_TOKEN)


@dataclass
class TokenizerConfig:
    max_seq_len: int = 64
    min_token_freq: int = 5
    max_vocab_size: int = 30_000


class WhitespaceTokenizer:
    """
    Tiny whitespace tokeniser with frequency-pruned vocabulary.
    """

    def __init__(
        self,
        token_to_id: dict[str, int],
        config: TokenizerConfig,
    ) -> None:
        self.token_to_id = token_to_id
        self.id_to_token = {idx: tok for tok, idx in token_to_id.items()}
        self.config = config

        for tok in SPECIAL_TOKENS:
            if tok not in self.token_to_id:
                raise ValueError(f"Special token {tok!r} missing from vocabulary.")

    @classmethod
    def fit(
        cls, texts: Iterable[str], config: TokenizerConfig | None = None
    ) -> "WhitespaceTokenizer":
        config = config or TokenizerConfig()

        counter: Counter[str] = Counter()
        for text in texts:
            if text is None:
                continue
            counter.update(tok for tok in text.split() if tok)

        token_to_id: dict[str, int] = {}
        for tok in SPECIAL_TOKENS:
            token_to_id[tok] = len(token_to_id)

        # Sort by frequency (desc) then by token for determinism.
        kept = [
            tok
            for tok, freq in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
            if freq >= config.min_token_freq
        ]
        budget = max(0, config.max_vocab_size - len(token_to_id))
        for tok in kept[:budget]:
            token_to_id[tok] = len(token_to_id)

        return cls(token_to_id=token_to_id, config=config)

    def __len__(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def cls_token_id(self) -> int:
        return self.token_to_id[CLS_TOKEN]

    @property
    def mask_token_id(self) -> int:
        return self.token_to_id[MASK_TOKEN]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    @property
    def special_token_ids(self) -> set[int]:
        return {self.token_to_id[tok] for tok in SPECIAL_TOKENS}

    def encode(self, text: str | None) -> list[int]:
        if text is None:
            return []
        unk = self.unk_token_id
        ids = [self.token_to_id.get(tok, unk) for tok in text.split() if tok]
        return ids[: self.config.max_seq_len]

    def encode_batch(self, texts: Sequence[str | None]) -> tuple[np.ndarray, np.ndarray]:
        max_len = self.config.max_seq_len
        ids = np.full((len(texts), max_len), self.pad_token_id, dtype=np.int64)
        attn = np.zeros((len(texts), max_len), dtype=np.int64)
        for i, text in enumerate(texts):
            enc = self.encode(text)
            if not enc:
                continue
            ids[i, : len(enc)] = enc
            attn[i, : len(enc)] = 1
        return ids, attn

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "max_seq_len": self.config.max_seq_len,
                "min_token_freq": self.config.min_token_freq,
                "max_vocab_size": self.config.max_vocab_size,
            },
            "token_to_id": self.token_to_id,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "WhitespaceTokenizer":
        payload = json.loads(Path(path).read_text())
        config = TokenizerConfig(**payload["config"])
        return cls(token_to_id=payload["token_to_id"], config=config)


class MLMTweetDataset(Dataset):
    """
    Self-supervised MLM dataset over individual tweets.
    """

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer: WhitespaceTokenizer,
        mlm_probability: float = 0.15,
    ) -> None:
        ids, attn = tokenizer.encode_batch(texts)
        self.token_ids = torch.from_numpy(ids)
        self.attention_mask = torch.from_numpy(attn)
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def __len__(self) -> int:
        return self.token_ids.size(0)

    def _mask_tokens(self, ids: torch.Tensor, attn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        labels = ids.clone()
        special_ids = torch.tensor(
            sorted(self.tokenizer.special_token_ids), dtype=ids.dtype
        )

        # Probability matrix for masking (only over real, non-special tokens).
        prob = torch.full(ids.shape, self.mlm_probability)
        prob = prob * attn.float()
        special_mask = torch.isin(ids, special_ids)
        prob = prob.masked_fill(special_mask, 0.0)

        masked = torch.bernoulli(prob).bool()
        labels[~masked] = -100  # CrossEntropy ignore_index

        # 80% [MASK], 10% random, 10% unchanged - the canonical BERT recipe.
        replace_with_mask = torch.bernoulli(torch.full(ids.shape, 0.8)).bool() & masked
        replace_with_random = (
            torch.bernoulli(torch.full(ids.shape, 0.5)).bool()
            & masked
            & ~replace_with_mask
        )

        ids = ids.clone()
        ids[replace_with_mask] = self.tokenizer.mask_token_id
        if replace_with_random.any():
            random_tokens = torch.randint(
                low=len(SPECIAL_TOKENS),
                high=len(self.tokenizer),
                size=ids.shape,
                dtype=ids.dtype,
            )
            ids[replace_with_random] = random_tokens[replace_with_random]
        return ids, labels

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ids = self.token_ids[index]
        attn = self.attention_mask[index]
        masked_ids, labels = self._mask_tokens(ids, attn)
        return {
            "token_ids": masked_ids,
            "attention_mask": attn,
            "labels": labels,
        }


@dataclass
class TimeBinBatch:
    """
    Container yielded by :class:`TimeBinTweetDataset`.
    """

    time_bin: list  # list of bin keys (datetime / str) of length B
    token_ids: torch.Tensor  # [B, K, L]
    attention_mask: torch.Tensor  # [B, K, L]
    tweet_mask: torch.Tensor  # [B, K]


class TimeBinTweetDataset(Dataset):
    """
    Group tweets by time bin and return fixed-size padded batches.
    """

    def __init__(
        self,
        tweets: pl.DataFrame,
        tokenizer: WhitespaceTokenizer,
        bins: pl.DataFrame,
        text_col: str = "Text",
        date_col: str = "Date",
        bin_col: str = "TimeBin",
        max_tweets_per_bin: int = 32,
    ) -> None:
        self.tokenizer = tokenizer
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

    def __getitem__(self, index: int) -> dict:
        texts = self._texts[index][: self.max_tweets_per_bin]
        max_len = self.tokenizer.config.max_seq_len
        k = self.max_tweets_per_bin

        ids = np.full((k, max_len), self.tokenizer.pad_token_id, dtype=np.int64)
        attn = np.zeros((k, max_len), dtype=np.int64)
        tweet_mask = np.zeros((k,), dtype=np.int64)

        for j, text in enumerate(texts):
            enc = self.tokenizer.encode(text)
            if not enc:
                continue
            ids[j, : len(enc)] = enc
            attn[j, : len(enc)] = 1
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
