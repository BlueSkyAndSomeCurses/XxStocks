import polars as pl

# ,date,open,high,low,close,volume,barCount,average


def downsample_to_interval(df: pl.DataFrame, interval: str = "30m") -> pl.DataFrame:
    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column for time-based downsampling.")

    date_dtype = df.schema["date"]
    if date_dtype.is_temporal():
        parsed = df.sort("date")
    else:
        parsed = df.with_columns(
            pl.col("date").str.to_datetime(strict=False).alias("date")
        ).sort("date")

    aggregations = []
    if "open" in parsed.columns:
        aggregations.append(pl.col("open").first().alias("open"))
    if "high" in parsed.columns:
        aggregations.append(pl.col("high").max().alias("high"))
    if "low" in parsed.columns:
        aggregations.append(pl.col("low").min().alias("low"))
    if "close" in parsed.columns:
        aggregations.append(pl.col("close").last().alias("close"))
    if "volume" in parsed.columns:
        aggregations.append(pl.col("volume").sum().alias("volume"))
    if "barCount" in parsed.columns:
        aggregations.append(pl.col("barCount").sum().alias("barCount"))
    if "average" in parsed.columns:
        aggregations.append(pl.col("average").mean().alias("average"))

    required_cols = [c for c in ["open", "high", "low", "close", "average"] if c in parsed.columns]

    downsampled = (
        parsed.group_by_dynamic("date", every=interval)
        .agg(aggregations)
        .drop_nulls(required_cols)
    )
    return downsampled


def augment_dataset(df: pl.DataFrame) -> pl.DataFrame:
    new_cols = [
        (pl.col("high") - pl.col("low")).alias("highLow"),
        (pl.col("open") - pl.col("close")).alias("openClose"),
        (pl.col("high") - pl.col("open")).alias("highOpen"),
        (pl.col("close") - pl.col("low")).alias("closeLow"),
    ]
    return df.with_columns(new_cols)


def label_data(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        target=(pl.col("average").shift(-1) > pl.col("average")).cast(pl.Int8)
    ).drop_nulls()


def split_features_target(df: pl.DataFrame):
    df = label_data(df)
    X = df.drop(["date", "average", "target"])
    y = df.select("target")
    return X, y


def time_train_test_split(df: pl.DataFrame, test_ratio: float = 0.2):
    n_rows = df.height
    split_idx = int(n_rows * (1 - test_ratio))

    train_df = df.slice(0, split_idx)
    test_df = df.slice(split_idx, n_rows - split_idx)
    return train_df, test_df
