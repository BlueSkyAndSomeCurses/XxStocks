import polars as pl

# ,date,open,high,low,close,volume,barCount,average


def downsample_to_interval(df: pl.DataFrame, interval: str = "30m") -> pl.DataFrame:
    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column for time-based downsampling.")

    date_dtype = df.schema["date"]
    if date_dtype.is_temporal():
        parsed = df.sort("date")
    else:
        date_str = (
            pl.col("date")
            .cast(pl.Utf8)
            .str.strip_chars()
            .str.replace_all(r"\s+", " ")
        )
        parsed_date = pl.coalesce(
            [
                date_str.str.strptime(
                    pl.Datetime, format="%Y%m%d %H:%M:%S", strict=False
                ),
                date_str.str.strptime(
                    pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False
                ),
                date_str.str.strptime(pl.Datetime, format="%Y-%m-%d", strict=False),
            ]
        )
        parsed = df.with_columns(
            parsed_date.alias("date")
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


def add_binary_target(
    df: pl.DataFrame,
    price_col: str = "average",
    target_col: str = "target_binary",
) -> pl.DataFrame:
    return df.with_columns(
        (
            pl.when(pl.col(price_col).shift(-1) > pl.col(price_col))
            .then(1)
            .otherwise(-1)
            .cast(pl.Int8)
        ).alias(target_col)
    ).drop_nulls([target_col])


def add_continuous_target(
    df: pl.DataFrame,
    price_col: str = "average",
    target_col: str = "target_continuous",
) -> pl.DataFrame:
    return df.with_columns(
        (
            (pl.col(price_col).shift(-1) - pl.col(price_col))
            / pl.col(price_col)
        ).alias(target_col)
    ).drop_nulls([target_col])


def add_prediction_targets(df: pl.DataFrame, price_col: str = "average") -> pl.DataFrame:
    with_binary = add_binary_target(df, price_col=price_col)
    with_both = add_continuous_target(with_binary, price_col=price_col)
    return with_both


def label_data(df: pl.DataFrame) -> pl.DataFrame:
    return add_binary_target(df).rename({"target_binary": "target"})


def split_features_target(
    df: pl.DataFrame,
    task: str = "binary",
    price_col: str = "average",
):
    prepared = add_prediction_targets(df, price_col=price_col)

    if task == "binary":
        target_col = "target_binary"
    elif task in {"continuous", "non-binary", "non_binary"}:
        target_col = "target_continuous"
    else:
        raise ValueError("task must be either 'binary' or 'continuous'.")

    drop_cols = ["date", price_col, "target_binary", "target_continuous"]
    existing_drop_cols = [col for col in drop_cols if col in prepared.columns]

    X = prepared.drop(existing_drop_cols)
    y = prepared.select(target_col)
    return X, y


def prepare_arima_data(
    df: pl.DataFrame,
    task: str = "binary",
    price_col: str = "average",
):
    prepared = add_prediction_targets(df, price_col=price_col)

    if task == "binary":
        target_col = "target_binary"
    elif task in {"continuous", "non-binary", "non_binary"}:
        target_col = "target_continuous"
    else:
        raise ValueError("task must be either 'binary' or 'continuous'.")

    drop_cols = ["date", "target_binary", "target_continuous"]
    exog_cols = [c for c in prepared.columns if c not in drop_cols and c != target_col]

    y = prepared.get_column(target_col)
    X = prepared.select(exog_cols)
    return y, X


def time_train_test_split(df: pl.DataFrame, test_ratio: float = 0.2):
    n_rows = df.height
    split_idx = int(n_rows * (1 - test_ratio))

    train_df = df.slice(0, split_idx)
    test_df = df.slice(split_idx, n_rows - split_idx)
    return train_df, test_df
