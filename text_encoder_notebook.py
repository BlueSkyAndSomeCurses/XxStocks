# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.23.3",
# ]
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # BERT per-bin tweet embeddings

    Interactive version of `text_encoder_script.py`. Uses **Hugging Face**
    `AutoModel` / `AutoTokenizer` (default `bert-base-uncased`) — no custom
    encoder training. Each tweet is embedded with BERT; tweets in the same
    30-minute `TimeBin` are pooled with `TimeBinAggregator`.
    """)
    return


@app.cell
def _():
    import numpy as np
    import polars as pl
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoTokenizer

    from dataset.preprocessing import augment_dataset, downsample_to_interval
    from dataset.text_dataset import TimeBinTweetDataset, collate_time_bin_batch
    from models.text_encoder import (
        BertTimeBinPipeline,
        TimeBinAggregator,
        TimeBinAggregatorConfig,
    )

    return (
        AutoConfig,
        AutoTokenizer,
        BertTimeBinPipeline,
        DataLoader,
        TimeBinAggregator,
        TimeBinAggregatorConfig,
        TimeBinTweetDataset,
        augment_dataset,
        collate_time_bin_batch,
        downsample_to_interval,
        np,
        pl,
        torch,
    )


@app.cell
def _(torch):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    device
    return (device,)


@app.cell
def _():
    MODEL_NAME = "bert-base-uncased"
    MAX_SEQ_LEN = 128
    MAX_TWEETS_PER_BIN = 32
    INFER_BATCH = 8
    return INFER_BATCH, MAX_SEQ_LEN, MAX_TWEETS_PER_BIN, MODEL_NAME


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load tweets
    """)
    return


@app.cell
def _(pl):
    tweets = (
        pl.read_csv("data/final_data/train/twitter_final.csv")
        .with_columns(
            pl.col("Date")
            .str.to_datetime(format="%Y-%m-%dT%H:%M:%S.%6f%z", strict=False)
            .dt.convert_time_zone("US/Mountain")
        )
        .drop_nulls("Text")
        .filter(pl.col("Text").str.len_chars() > 0)
    )
    tweets.head()
    return (tweets,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Tokenizer and BERT + aggregator
    """)
    return


@app.cell
def _(
    AutoConfig,
    AutoTokenizer,
    BertTimeBinPipeline,
    MODEL_NAME,
    TimeBinAggregator,
    TimeBinAggregatorConfig,
    device,
    mo,
):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    hf_cfg = AutoConfig.from_pretrained(MODEL_NAME)
    d_model = int(hf_cfg.hidden_size)

    aggregator = TimeBinAggregator(
        TimeBinAggregatorConfig(d_model=d_model, n_heads=12)
    ).to(device)
    pipeline = BertTimeBinPipeline(MODEL_NAME, aggregator).to(device)
    pipeline.eval()

    mo.md(f"Hidden size **{d_model}**, output dim **{pipeline.out_dim}**.")
    return pipeline, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Thirty-minute `TimeBin` schedule
    """)
    return


@app.cell
def _(augment_dataset, downsample_to_interval, pl):
    raw = pl.read_csv("data/1_min_SPY_2008-2021.csv")
    bins = (
        augment_dataset(downsample_to_interval(raw, interval="30m"))
        .select(pl.col("date").alias("TimeBin"))
        .sort("TimeBin")
    )
    bins.head()
    return (bins,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Encode and aggregate per bin
    """)
    return


@app.cell
def _(
    MAX_SEQ_LEN,
    MAX_TWEETS_PER_BIN,
    TimeBinTweetDataset,
    bins,
    tokenizer,
    tweets,
):
    bin_dataset = TimeBinTweetDataset(
        tweets=tweets,
        tokenizer=tokenizer,
        bins=bins,
        max_seq_len=MAX_SEQ_LEN,
        max_tweets_per_bin=MAX_TWEETS_PER_BIN,
    )
    len(bin_dataset)
    return (bin_dataset,)


@app.cell
def _(
    DataLoader,
    INFER_BATCH,
    bin_dataset,
    collate_time_bin_batch,
    device,
    np,
    pipeline,
    pl,
    torch,
):
    loader_inf = DataLoader(
        bin_dataset,
        batch_size=INFER_BATCH,
        shuffle=False,
        collate_fn=collate_time_bin_batch,
    )

    all_bins = []
    all_emb = []
    with torch.no_grad():
        for batch in loader_inf:
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
    text_embed_30m = pl.DataFrame({"TimeBin": all_bins, **embed_cols}).sort("TimeBin")
    text_embed_30m.head()
    return (text_embed_30m,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Save parquet

    Same schema as before (`TimeBin` + `text_embed_*`) for joins with OHLCV.
    """)
    return


@app.cell
def _(text_embed_30m):
    text_embed_30m.write_parquet(
        "data/final_data/train/text_encoder_embeddings_30m.parquet"
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
