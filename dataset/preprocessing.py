import polars as pl

# ,date,open,high,low,close,volume,barCount,average


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
