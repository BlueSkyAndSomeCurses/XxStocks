import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # BERT embeddings + time series dataloader

    Builds augmented SPY bars, joins ``text_encoder_embeddings_30m.parquet`` via
    :func:`dataset.preprocessing.combine_numerical_and_bert_embeddings`, then
    :class:`dataset.bert_ts_loader.BertTimeSeriesDataset` (numeric window +
    BERT vector at the last bar + target).
    """
    )
    return


@app.cell
def _():
    from pathlib import Path

    import polars as pl
    from torch.utils.data import DataLoader

    from dataset.bert_ts_loader import BertTimeSeriesDataset
    from dataset.preprocessing import (
        augment_dataset,
        combine_numerical_and_bert_embeddings,
        downsample_to_interval,
    )

    return (
        BertTimeSeriesDataset,
        DataLoader,
        Path,
        augment_dataset,
        combine_numerical_and_bert_embeddings,
        downsample_to_interval,
        pl,
    )


@app.cell
def _(Path, pl):
    bert_path = Path("data/final_data/train/text_encoder_embeddings_30m.parquet")
    if not bert_path.exists():
        raise FileNotFoundError(
            f"Missing {bert_path}; run text_encoder_script.py or text_encoder_notebook.py first."
        )
    bert_emb = pl.read_parquet(bert_path)
    return bert_emb,


@app.cell
def _(augment_dataset, downsample_to_interval, pl):
    raw = pl.read_csv("data/1_min_SPY_2008-2021.csv")
    stock = augment_dataset(downsample_to_interval(raw, interval="30m"))
    stock.head(3)
    return (stock,)


@app.cell
def _(bert_emb, combine_numerical_and_bert_embeddings, stock):
    combined = combine_numerical_and_bert_embeddings(stock, bert_emb)
    combined.head(3)
    return (combined,)


@app.cell
def _(BertTimeSeriesDataset, DataLoader, combined):
    ds = BertTimeSeriesDataset(combined, window_size=16, task="binary")
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    x_seq, x_bert, y = next(iter(loader))
    x_seq.shape, x_bert.shape, y.shape
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
