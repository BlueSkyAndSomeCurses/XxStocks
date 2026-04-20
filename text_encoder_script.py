"""End-to-end script for the neural text-encoder feature extractor.

Pipeline:
    1. Load tweets from ``data/final_data/train/twitter_final.csv``.
    2. Fit (or load) the whitespace tokeniser.
    3. Pretrain a small bidirectional Transformer (:class:`TweetEncoder`)
       with masked language modelling.
    4. Build the 30-min ``TimeBin`` schedule from the SPY OHLCV file.
    5. Encode every tweet with the trained encoder and aggregate per bin
       via :class:`TimeBinAggregator` (attention pooling against a learnable
       query token).
    6. Write the per-bin dense embeddings to
       ``data/final_data/train/text_encoder_embeddings_30m.parquet`` so that
       they can be joined alongside the OHLCV features in the same way the
       lexicon-based bag-of-words features are joined today.

Run with::

    uv run python text_encoder_script.py --epochs 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset.preprocessing import augment_dataset, downsample_to_interval
from dataset.text_dataset import (
    MLMTweetDataset,
    TimeBinTweetDataset,
    TokenizerConfig,
    WhitespaceTokenizer,
    collate_time_bin_batch,
)
from models.text_encoder import (
    MLMHead,
    TextEncoderPipeline,
    TimeBinAggregator,
    TimeBinAggregatorConfig,
    TweetEncoder,
    TweetEncoderConfig,
)


DEFAULT_TWEETS_PATH = "data/final_data/train/twitter_final.csv"
DEFAULT_OHLCV_PATH = "data/1_min_SPY_2008-2021.csv"
DEFAULT_OUT_PATH = "data/final_data/train/text_encoder_embeddings_30m.parquet"
DEFAULT_TOKENIZER_PATH = "data/models_checkpoints/text_encoder_tokenizer.json"
DEFAULT_ENCODER_CKPT = "data/models_checkpoints/text_encoder.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tweets-path", default=DEFAULT_TWEETS_PATH)
    p.add_argument("--ohlcv-path", default=DEFAULT_OHLCV_PATH)
    p.add_argument("--out-path", default=DEFAULT_OUT_PATH)
    p.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    p.add_argument("--encoder-ckpt", default=DEFAULT_ENCODER_CKPT)
    p.add_argument("--interval", default="30m")
    p.add_argument("--max-seq-len", type=int, default=64)
    p.add_argument("--max-tweets-per-bin", type=int, default=32)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--inference-batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--mlm-probability", type=float, default=0.15)
    p.add_argument("--skip-pretraining", action="store_true")
    p.add_argument("--device", default=None)
    return p.parse_args()


def select_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tweets(path: str) -> pl.DataFrame:
    df = pl.read_csv(path)
    date_dtype = df.schema.get("Date")
    if "Date" in df.columns and not (date_dtype is not None and date_dtype.is_temporal()):
        df = df.with_columns(
            pl.col("Date").str.to_datetime(
                format="%Y-%m-%dT%H:%M:%S.%6f%z",
                strict=False,
            )
        )
        new_dtype = df.schema.get("Date")
        if getattr(new_dtype, "time_zone", None) is not None:
            df = df.with_columns(pl.col("Date").dt.convert_time_zone("US/Mountain"))
    return df.drop_nulls("Text").filter(pl.col("Text").str.len_chars() > 0)


def fit_or_load_tokenizer(
    texts: list[str], path: str, max_seq_len: int
) -> WhitespaceTokenizer:
    p = Path(path)
    if p.exists():
        tokenizer = WhitespaceTokenizer.load(p)
        if tokenizer.config.max_seq_len != max_seq_len:
            tokenizer.config.max_seq_len = max_seq_len
        return tokenizer
    tokenizer = WhitespaceTokenizer.fit(
        texts, TokenizerConfig(max_seq_len=max_seq_len)
    )
    tokenizer.save(p)
    return tokenizer


def pretrain_encoder(
    encoder: TweetEncoder,
    head: MLMHead,
    dataset: MLMTweetDataset,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> None:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    encoder.train()
    head.train()
    for epoch in range(epochs):
        running = 0.0
        seen = 0
        for batch in loader:
            ids = batch["token_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            out = encoder(ids, attention_mask=attn, return_token_states=True)
            logits = head(out["token_states"])
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            running += loss.item() * ids.size(0)
            seen += ids.size(0)

        avg = running / max(seen, 1)
        print(f"[mlm] epoch {epoch + 1}/{epochs}  loss={avg:.4f}")


def build_time_bins(ohlcv_path: str, interval: str) -> pl.DataFrame:
    raw = pl.read_csv(ohlcv_path)
    downsampled = downsample_to_interval(raw, interval=interval)
    augmented = augment_dataset(downsampled)
    return augmented.select(pl.col("date").alias("TimeBin")).sort("TimeBin")


def encode_time_bins(
    pipeline: TextEncoderPipeline,
    dataset: TimeBinTweetDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> pl.DataFrame:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_time_bin_batch,
    )

    pipeline.eval()
    all_bins: list = []
    all_emb: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            emb = pipeline(
                batch.token_ids.to(device),
                batch.attention_mask.to(device),
                batch.tweet_mask.to(device),
            )
            all_bins.extend(batch.time_bin)
            all_emb.append(emb.cpu().numpy())

    embeddings = np.concatenate(all_emb, axis=0) if all_emb else np.zeros((0, pipeline.out_dim), dtype=np.float32)

    embed_cols = {
        f"text_embed_{i}": embeddings[:, i].astype(np.float32)
        for i in range(embeddings.shape[1])
    }
    return pl.DataFrame({"TimeBin": all_bins, **embed_cols}).sort("TimeBin")


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    print(f"[setup] device={device}")

    print(f"[load] tweets from {args.tweets_path}")
    tweets = load_tweets(args.tweets_path)
    print(f"[load] {tweets.height} tweets")

    texts: list[str] = tweets["Text"].to_list()
    tokenizer = fit_or_load_tokenizer(
        texts, args.tokenizer_path, max_seq_len=args.max_seq_len
    )
    print(f"[tokenizer] vocab_size={len(tokenizer)}")

    encoder_config = TweetEncoderConfig(
        vocab_size=len(tokenizer),
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        pad_token_id=tokenizer.pad_token_id,
        cls_token_id=tokenizer.cls_token_id,
        mask_token_id=tokenizer.mask_token_id,
    )
    encoder = TweetEncoder(encoder_config).to(device)

    ckpt_path = Path(args.encoder_ckpt)
    if ckpt_path.exists() and args.skip_pretraining:
        print(f"[mlm] loading existing checkpoint {ckpt_path}")
        encoder.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print("[mlm] pretraining encoder via masked language modelling")
        head = MLMHead(encoder).to(device)
        mlm_dataset = MLMTweetDataset(
            texts, tokenizer, mlm_probability=args.mlm_probability
        )
        pretrain_encoder(
            encoder,
            head,
            mlm_dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(encoder.state_dict(), ckpt_path)
        print(f"[mlm] saved encoder checkpoint to {ckpt_path}")

    aggregator = TimeBinAggregator(
        TimeBinAggregatorConfig(d_model=args.d_model, n_heads=args.n_heads)
    ).to(device)
    pipeline = TextEncoderPipeline(encoder=encoder, aggregator=aggregator).to(device)

    print(f"[bins] reading 30m TimeBin schedule from {args.ohlcv_path}")
    bins = build_time_bins(args.ohlcv_path, interval=args.interval)

    print("[encode] encoding tweets and aggregating per bin")
    bin_dataset = TimeBinTweetDataset(
        tweets=tweets,
        tokenizer=tokenizer,
        bins=bins,
        max_tweets_per_bin=args.max_tweets_per_bin,
    )
    embeddings = encode_time_bins(
        pipeline,
        bin_dataset,
        batch_size=args.inference_batch_size,
        device=device,
    )

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings.write_parquet(out_path)
    print(f"[done] wrote {embeddings.height} bin embeddings to {out_path}")


if __name__ == "__main__":
    main()
