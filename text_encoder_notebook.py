import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Neural text encoder for tweet features

    This notebook is the interactive twin of `text_encoder_script.py`. It is
    an alternative to the lexicon / bag-of-words pipeline implemented in
    `text_feature_extraction.py`.

    Instead of mapping each token to a Harvard IV-4 category and counting,
    we:

    1. Pretrain a small bidirectional Transformer (`TweetEncoder`) on the
       tweet corpus with **masked language modelling**.
    2. Embed each individual tweet with the trained encoder.
    3. Aggregate the per-tweet embeddings inside every 30-minute `TimeBin`
       with `TimeBinAggregator` – attention pooling against a learnable
       query token.
    4. Save one dense vector per bin to a parquet file that can be joined
       alongside the OHLCV features just like the existing bag-of-words
       parquet.
    """
    )
    return


@app.cell
def _():
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

    return (
        DataLoader,
        MLMHead,
        MLMTweetDataset,
        TextEncoderPipeline,
        TimeBinAggregator,
        TimeBinAggregatorConfig,
        TimeBinTweetDataset,
        TokenizerConfig,
        TweetEncoder,
        TweetEncoderConfig,
        WhitespaceTokenizer,
        augment_dataset,
        collate_time_bin_batch,
        downsample_to_interval,
        nn,
        np,
        optim,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 1. Load tweets""")
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
    mo.md(
        r"""
    ## 2. Fit the whitespace tokenizer

    Text was already lowercased / cleaned in `data_preparation.py`, so a
    plain whitespace split with a frequency-pruned vocabulary is enough.
    """
    )
    return


@app.cell
def _(TokenizerConfig, WhitespaceTokenizer, tweets):
    tokenizer_config = TokenizerConfig(
        max_seq_len=64,
        min_token_freq=5,
        max_vocab_size=30_000,
    )
    tokenizer = WhitespaceTokenizer.fit(tweets["Text"].to_list(), tokenizer_config)
    print(f"vocab_size = {len(tokenizer)}")
    return (tokenizer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 3. Build the encoder""")
    return


@app.cell
def _(TweetEncoder, TweetEncoderConfig, device, tokenizer):
    encoder_config = TweetEncoderConfig(
        vocab_size=len(tokenizer),
        max_seq_len=tokenizer.config.max_seq_len,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
        pad_token_id=tokenizer.pad_token_id,
        cls_token_id=tokenizer.cls_token_id,
        mask_token_id=tokenizer.mask_token_id,
    )
    encoder = TweetEncoder(encoder_config).to(device)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"encoder parameters: {n_params:,}")
    return (encoder,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4. Self-supervised pretraining (MLM)

    Standard 80 / 10 / 10 BERT recipe over the tweet corpus.
    """
    )
    return


@app.cell
def _(MLMHead, MLMTweetDataset, encoder, device, tokenizer, tweets):
    head = MLMHead(encoder).to(device)
    mlm_dataset = MLMTweetDataset(
        tweets["Text"].to_list(), tokenizer, mlm_probability=0.15
    )
    print(f"MLM dataset size = {len(mlm_dataset)}")
    return head, mlm_dataset


@app.cell
def _(DataLoader, encoder, head, mlm_dataset, nn, optim, torch, device):
    EPOCHS = 1
    BATCH_SIZE = 256
    LR = 3e-4

    loader = DataLoader(mlm_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = optim.AdamW(params, lr=LR, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    encoder.train()
    head.train()
    for epoch in range(EPOCHS):
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

        print(f"epoch {epoch + 1}/{EPOCHS}  mlm_loss={running / max(seen, 1):.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 5. Build 30-min `TimeBin` schedule""")
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
    mo.md(
        r"""
    ## 6. Aggregate tweet embeddings per bin

    `TimeBinAggregator` performs multi-head attention pooling against a
    learnable query so that bins with many tweets do not blur into a mean
    vector – the model can up-weight the more informative ones.
    """
    )
    return


@app.cell
def _(
    TextEncoderPipeline,
    TimeBinAggregator,
    TimeBinAggregatorConfig,
    TimeBinTweetDataset,
    bins,
    device,
    encoder,
    tokenizer,
    tweets,
):
    aggregator = TimeBinAggregator(
        TimeBinAggregatorConfig(
            d_model=encoder.d_model,
            n_heads=encoder.config.n_heads,
        )
    ).to(device)
    pipeline = TextEncoderPipeline(encoder=encoder, aggregator=aggregator).to(device)

    bin_dataset = TimeBinTweetDataset(
        tweets=tweets,
        tokenizer=tokenizer,
        bins=bins,
        max_tweets_per_bin=32,
    )
    print(f"bins with at least one tweet: {len(bin_dataset)}")
    return bin_dataset, pipeline


@app.cell
def _(DataLoader, bin_dataset, collate_time_bin_batch, device, np, pipeline, pl, torch):
    INFER_BATCH = 32
    loader_inf = DataLoader(
        bin_dataset,
        batch_size=INFER_BATCH,
        shuffle=False,
        collate_fn=collate_time_bin_batch,
    )

    pipeline.eval()
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
    mo.md(
        r"""
    ## 7. Persist the per-bin embeddings

    The output schema mirrors `bow_2stages_30m.parquet` (one row per
    `TimeBin`) so it can be joined with `combine_numerical_and_text_data`
    by simply pointing `category_text_data` at this new parquet.
    """
    )
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
