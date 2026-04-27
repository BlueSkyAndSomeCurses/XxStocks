import polars as pl

# ,date,open,high,low,close,volume,barCount,average

# Absolute price / level columns that must never be fed to a model whose
# target is a return: they are essentially the denominator of the target and
# create a trivial-looking dependency that masquerades as predictive power
# (this was the source of the suspiciously strong "baseline" SARIMAX).
PRICE_LEVEL_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "average",
    "volume",
    "barCount",
)


def downsample_to_interval(df: pl.DataFrame, interval: str = "30m") -> pl.DataFrame:
    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column for time-based downsampling.")

    date_dtype = df.schema["date"]
    if date_dtype.is_temporal():
        parsed = df.sort("date")
    else:
        date_str = (
            pl.col("date").cast(pl.Utf8).str.strip_chars().str.replace_all(r"\s+", " ")
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
            parsed_date.dt.replace_time_zone("US/Mountain").alias("date")
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

    required_cols = [
        c for c in ["open", "high", "low", "close", "average"] if c in parsed.columns
    ]

    downsampled = (
        parsed.group_by_dynamic("date", every=interval)
        .agg(aggregations)
        .drop_nulls(required_cols)
    )
    return downsampled


def augment_dataset(df: pl.DataFrame) -> pl.DataFrame:
    new_cols = [
        # Bar-shape features expressed as ratios so they are scale-invariant
        # and do not leak the absolute price level into downstream models.
        ((pl.col("high") - pl.col("low")) / pl.col("average")).alias("hl_range_pct"),
        ((pl.col("open") - pl.col("close")) / pl.col("average")).alias("oc_body_pct"),
        ((pl.col("high") - pl.col("open")) / pl.col("average")).alias("upper_wick_pct"),
        ((pl.col("close") - pl.col("low")) / pl.col("average")).alias("lower_wick_pct"),
        pl.max_horizontal(
            pl.col("high").sub(pl.col("low")),
            pl.col("high").sub(pl.col("close").shift(1)).abs(),
            pl.col("low").sub(pl.col("close").shift(1)).abs(),
        )
        .truediv(pl.col("average"))
        .alias("average_true_range"),
        pl.col("close").diff().alias("change"),
        # Log-returns at multiple lags – the only safe way to expose price
        # dynamics to the model without leaking the level itself.
        (pl.col("average").log() - pl.col("average").shift(1).log()).alias("ret_1"),
        (pl.col("average").log() - pl.col("average").shift(2).log()).alias("ret_2"),
        (pl.col("average").log() - pl.col("average").shift(4).log()).alias("ret_4"),
        (pl.col("average").log() - pl.col("average").shift(8).log()).alias("ret_8"),
        (pl.col("average").log() - pl.col("average").shift(16).log()).alias("ret_16"),
        # Log-volume keeps order-flow info but normalises scale.
        (pl.col("volume").cast(pl.Float64) + 1.0).log().alias("log_volume"),
    ]

    dataset_with_features = (
        df.with_columns(new_cols)
        .with_columns(
            pl.when(pl.col("change").ge(0))
            .then(pl.col("change"))
            .otherwise(pl.lit(0))
            .alias("gain"),
            pl.when(pl.col("change").ge(0))
            .then(pl.lit(0))
            .otherwise(pl.lit(-1.0).mul(pl.col("change")))
            .alias("loss"),
        )
        .with_columns(
            pl.col("gain").ewm_mean(span=14, adjust=False).alias("average_gain"),
            pl.col("loss").ewm_mean(span=14, adjust=False).alias("average_loss"),
            pl.col("close").rolling_std(window_size=10).alias("std_dev_10"),
        )
        .with_columns(
            pl.lit(100)
            .sub(
                pl.lit(100).truediv(
                    pl.lit(1).add(
                        pl.col("average_gain").truediv(pl.col("average_loss"))
                    )
                )
            )
            .alias("RSI"),
            pl.when(pl.col("close").ge(pl.col("close").shift(1)))
            .then(pl.col("std_dev_10"))
            .otherwise(pl.lit(0))
            .alias("up_std"),
            pl.when(pl.col("close").ge(pl.col("close").shift(1)))
            .then(pl.lit(0))
            .otherwise(pl.col("std_dev_10"))
            .alias("down_std"),
        )
        .with_columns(
            pl.col("up_std").ewm_mean(span=14).alias("average_up_std"),
            pl.col("down_std").ewm_mean(span=14).alias("average_down_std"),
        )
        .with_columns(
            pl.lit(100)
            .mul(
                pl.col("average_up_std").truediv(
                    pl.col("average_up_std").add(pl.col("average_down_std"))
                )
            )
            .alias("RVI")
        )
        .drop(
            "gain",
            "loss",
            "average_gain",
            "average_loss",
            "std_dev_10",
            "up_std",
            "down_std",
            "average_up_std",
            "average_down_std",
            "change",
        )
        .drop_nulls()
    )

    return dataset_with_features


def get_bag_of_words(
    twitter_posts: pl.DataFrame, dictionary: pl.DataFrame, time_window: str = "15"
) -> pl.DataFrame:
    category_columns = [
        col_name for col_name in dictionary.columns if col_name != "EntryCleaned"
    ]

    dict_words = dictionary["EntryCleaned"].to_list()
    dict_flags = dictionary.with_columns(
        [pl.col(column).is_not_null().alias(column) for column in category_columns]
    )

    twitter_posts_tokenized = twitter_posts.with_columns(
        pl.col("Text").str.split(" ").alias("Tokens")
    )

    bag_of_words_prep = (
        twitter_posts_tokenized.explode("Tokens")
        .filter(pl.col("Tokens").is_not_null())
        .filter(pl.col("Tokens").ne(""))
        .filter(pl.col("Tokens").is_in(dict_words))
        .sort("Date")
    )

    bow_with_categories = (
        bag_of_words_prep.join(
            dict_flags, left_on="Tokens", right_on="EntryCleaned", how="left"
        )
        .with_columns(
            [pl.col(c).fill_null(False).cast(pl.UInt32) for c in category_columns]
        )
        .sort("Date")
    )

    rolling_bag_of_words = (
        bag_of_words_prep.rolling("Date", period=time_window, group_by="Tokens")
        .agg(pl.len().alias("Count"))
        .unique()
        .pivot("Tokens", values="Count", index="Date")
        .fill_null(0)
        .sort("Date")
    )

    rolling_category_counts = (
        bow_with_categories.rolling("Date", period=time_window)
        .agg([pl.col(column).sum() for column in category_columns])
        .fill_null(0)
        .sort("Date")
    )

    rolling_features = rolling_bag_of_words.join(
        rolling_category_counts, on="Date", how="left"
    ).sort("Date")

    return rolling_features


def get_category_features(
    rolling_features: pl.DataFrame, dictionary: pl.DataFrame
) -> pl.DataFrame:
    category_columns = [
        col_name for col_name in dictionary.columns if col_name != "EntryCleaned"
    ]

    category_features = (
        rolling_features.select(*category_columns, "Date")
        .select(
            "Date",
            pl.sum_horizontal("Econ@", "Exch", "ECON").alias("EconomicWords"),
            "Legal",
            "Milit",
            "Polit@",
            "PowTot",
        )
        .unique()
        .sort("Date")
    )

    return category_features


def combine_numerical_and_text_data(
    stock_augmented_data: pl.DataFrame, text_with_features: pl.DataFrame
) -> pl.DataFrame:
    combined_data = stock_augmented_data.join(
        text_with_features, left_on="date", right_on="TimeBin", how="left"
    ).rename({"date": "TradeDate"}).fill_null(0)

    # Building <col> * RSI / <col> * RVI products on absolute price columns
    # would simply re-introduce the leakage we just removed via
    # PRICE_LEVEL_COLUMNS, so we restrict the cross-products to safe inputs.
    excluded = set(PRICE_LEVEL_COLUMNS) | {"TradeDate", "RSI", "RVI"}
    category_columns = [
        col_name
        for col_name in combined_data.columns
        if "Rudementary" not in col_name and col_name not in excluded
    ]

    combined_data = combined_data.with_columns(
        *[
            pl.col(category_column)
            .mul(pl.col(index_column))
            .alias(f"{category_column} * {index_column}")
            for category_column in category_columns
            for index_column in ["RSI", "RVI"]
        ],
        pl.col("TradeDate").diff().truediv(pl.duration(minutes=30)).floor().cast(pl.UInt64).fill_null(0).alias("DateDiff")
    )

    return combined_data


def combine_numerical_and_bert_embeddings(
    stock_augmented_data: pl.DataFrame,
    bert_embeddings: pl.DataFrame,
) -> pl.DataFrame:
    """Left-join per-bar BERT vectors (``text_embed_*``) onto OHLCV rows on ``date`` / ``TimeBin``.

    Rows without tweets get zeros on all ``text_embed_*`` columns.
    """
    joined = stock_augmented_data.join(
        bert_embeddings,
        left_on="date",
        right_on="TimeBin",
        how="left",
    )
    text_cols = [c for c in joined.columns if c.startswith("text_embed_")]
    if text_cols:
        joined = joined.with_columns([pl.col(c).fill_null(0.0) for c in text_cols])
    return joined


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
        (pl.col(price_col).shift(-1).truediv(pl.col(price_col))).log().alias(target_col)
    ).drop_nulls([target_col])


def add_prediction_targets(
    df: pl.DataFrame, price_col: str = "average"
) -> pl.DataFrame:
    with_binary = add_binary_target(df, price_col=price_col)
    with_both = add_continuous_target(with_binary, price_col=price_col)
    return with_both


def label_data(df: pl.DataFrame) -> pl.DataFrame:
    return add_binary_target(df).rename({"target_binary": "target"})


def _resolve_target_col(task: str) -> str:
    if task == "binary":
        return "target_binary"
    if task in {"continuous", "non-binary", "non_binary"}:
        return "target_continuous"
    raise ValueError("task must be either 'binary' or 'continuous'.")


def _drop_leaky_columns(df: pl.DataFrame, extra: list[str]) -> pl.DataFrame:
    """Strip absolute-level price columns plus any caller-supplied date /
    target columns. Keeping ``average`` etc. would let the model (especially
    SARIMAX with linear exog) regress the target onto its own denominator."""
    candidates = list(PRICE_LEVEL_COLUMNS) + extra
    existing = [c for c in candidates if c in df.columns]
    return df.drop(existing)


def split_features_target(
    df: pl.DataFrame,
    task: str = "binary",
    price_col: str = "average",
):
    prepared = add_prediction_targets(df, price_col=price_col)
    target_col = _resolve_target_col(task)

    X = _drop_leaky_columns(
        prepared,
        extra=["date", "TimeBin", "TradeDate", "target_binary", "target_continuous"],
    )
    y = prepared.select(target_col)
    return X, y


def prepare_arima_data(
    df: pl.DataFrame,
    task: str = "binary",
    price_col: str = "average",
):
    prepared = add_prediction_targets(df, price_col=price_col)
    target_col = _resolve_target_col(task)

    X = _drop_leaky_columns(
        prepared,
        extra=["date", "TimeBin", "TradeDate", "target_binary", "target_continuous"],
    )
    y = prepared.get_column(target_col)
    return y, X


def time_train_test_split(df: pl.DataFrame, test_ratio: float = 0.2):
    n_rows = df.height
    split_idx = int(n_rows * (1 - test_ratio))

    train_df = df.slice(0, split_idx)
    test_df = df.slice(split_idx, n_rows - split_idx)
    return train_df, test_df
