"""Encode tweets per 30-minute time bin using Hugging Face BERT.

Pipeline:
    1. Load tweets from ``data/final_data/train/twitter_final.csv``.
    2. Load ``AutoTokenizer`` + ``AutoModel`` (default ``bert-base-uncased``).
    3. Build the ``TimeBin`` schedule from the SPY OHLCV file.
    4. For each bin, embed tweets with BERT and aggregate via
       :class:`TimeBinAggregator` (attention pooling).
    5. Write per-bin vectors to
       ``data/final_data/train/text_encoder_embeddings_30m.parquet``.

Run::

    uv run python text_encoder_script.py --model-name bert-base-uncased
"""

from __future__ import annotations

import argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from dataset.preprocessing import augment_dataset, downsample_to_interval
from dataset.text_dataset import TimeBinTweetDataset, collate_time_bin_batch
from models.text_encoder import BertTimeBinPipeline, TimeBinAggregator, TimeBinAggregatorConfig


DEFAULT_TWEETS_PATH = "data/final_data/train/twitter_for_encoder_final.csv"
DEFAULT_OHLCV_PATH = "data/1_min_SPY_2008-2021.csv"
DEFAULT_OUT_PATH = "data/final_data/train/text_encoder_embeddings_30m.parquet"
DEFAULT_MODEL_NAME = "bert-base-uncased"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tweets-path", default=DEFAULT_TWEETS_PATH)
    p.add_argument("--ohlcv-path", default=DEFAULT_OHLCV_PATH)
    p.add_argument("--out-path", default=DEFAULT_OUT_PATH)
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    p.add_argument("--interval", default="30m")
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--max-tweets-per-bin", type=int, default=32)
    p.add_argument("--n-heads", type=int, default=12)
    p.add_argument("--inference-batch-size", type=int, default=8)
    p.add_argument("--trust-remote-code", action="store_true")
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


def build_time_bins(ohlcv_path: str, interval: str) -> pl.DataFrame:
    raw = pl.read_csv(ohlcv_path)
    downsampled = downsample_to_interval(raw, interval=interval)
    augmented = augment_dataset(downsampled)
    return augmented.select(pl.col("date").alias("TimeBin")).sort("TimeBin")


def encode_time_bins(
    pipeline: BertTimeBinPipeline,
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
        for batch in tqdm(loader):
            emb = pipeline(
                batch.token_ids.to(device),
                batch.attention_mask.to(device),
                batch.tweet_mask.to(device),
            )
            all_bins.extend(batch.time_bin)
            all_emb.append(emb.cpu().numpy())

    embeddings = (
        np.concatenate(all_emb, axis=0)
        if all_emb
        else np.zeros((0, pipeline.out_dim), dtype=np.float32)
    )

    embed_cols = {
        f"text_embed_{i}": embeddings[:, i].astype(np.float32)
        for i in range(embeddings.shape[1])
    }
    return pl.DataFrame({"TimeBin": all_bins, **embed_cols}).sort("TimeBin")


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    print(f"[setup] device={device} model={args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tweets = load_tweets(args.tweets_path)
    print(f"[load] {tweets.height} tweets from {args.tweets_path}")

    hf_cfg = AutoConfig.from_pretrained(
        args.model_name, trust_remote_code=args.trust_remote_code
    )
    d_model = int(hf_cfg.hidden_size)

    aggregator = TimeBinAggregator(
        TimeBinAggregatorConfig(d_model=d_model, n_heads=args.n_heads)
    ).to(device)
    pipeline = BertTimeBinPipeline(
        args.model_name,
        aggregator,
        trust_remote_code=args.trust_remote_code,
    ).to(device)

    bins = build_time_bins(args.ohlcv_path, interval=args.interval)
    print(f"[bins] {bins.height} rows from {args.ohlcv_path}")

    bin_dataset = TimeBinTweetDataset(
        tweets=tweets,
        tokenizer=tokenizer,
        bins=bins,
        max_seq_len=args.max_seq_len,
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
