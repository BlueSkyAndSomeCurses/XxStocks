import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import polars as pl

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import plotly.express as px

    from sklearn.preprocessing import StandardScaler
    import umap

    import pandas as pd
    import numpy as np

    return StandardScaler, TSNE, np, pd, pl, px, umap


@app.cell
def _(pl):
    dictionary = pl.read_csv("data/final_data/dictionary/cleaned_dict.csv")
    twitter_posts= pl.read_csv("data/final_data/train/twitter_final.csv").with_columns(
        pl.col("Date").str.to_datetime()
    )
    return dictionary, twitter_posts


@app.cell
def _(twitter_posts):
    twitter_posts
    return


@app.cell
def _(dictionary):
    dictionary
    return


@app.cell
def _(dictionary, pl):
    category_columns = [
        col_name for col_name in dictionary.columns if col_name != "EntryCleaned"
    ]

    dict_words = dictionary["EntryCleaned"].to_list()
    dict_flags = dictionary.with_columns(
        [pl.col(column).is_not_null().alias(column) for column in category_columns]
    )
    return category_columns, dict_flags, dict_words


@app.cell
def _(mo):
    mo.md(r"""
    # Core data feature extraction preparation
    """)
    return


@app.cell
def _(dict_words, pl, twitter_posts):
    twitter_posts_tokenized = twitter_posts.with_columns(
        pl.col("Text").str.split(" ").alias("Tokens")
    ).with_columns(
        pl.col("Tokens")
        .map_elements(lambda tokens: any(t in dict_words for t in tokens))
        .alias("HasDictWord")
    )
    return (twitter_posts_tokenized,)


@app.cell
def _(category_columns, dict_flags, dict_words, pl, twitter_posts_tokenized):
    bag_of_words_prep = (
        twitter_posts_tokenized.explode("Tokens")
        .filter(pl.col("Tokens").is_not_null())
        .filter(pl.col("Tokens").ne(""))
        .filter(pl.col("Tokens").is_in(dict_words))
        .sort("Date")
    )

    bow_with_categories = bag_of_words_prep.join(
        dict_flags, left_on="Tokens", right_on="EntryCleaned", how="left"
    ).with_columns([pl.col(c).fill_null(False).cast(pl.UInt32) for c in category_columns])
    return bag_of_words_prep, bow_with_categories


@app.cell
def _(bow_with_categories):
    bow_with_categories
    return


@app.cell
def _(bag_of_words_prep, bow_with_categories, category_columns, pl):
    rolling_bag_of_words = (
        bag_of_words_prep.rolling("Date", period="1mo", group_by="Tokens")
        .agg(pl.len().alias("Count"))
        .group_by("Date", "Tokens")
        .agg(pl.len().alias("Count"))
        .pivot("Tokens", values="Count", index="Date")
        .fill_null(0)
        .sort("Date")
    )

    rolling_category_counts = (
        bow_with_categories.rolling("Dat", period="1mo")    
        .agg([pl.col(column).sum().alias(f"{column}_count") for column in category_columns])
        .fill_null(0)
        .sort("Date")
    )

    rolling_features = rolling_bag_of_words.join(rolling_category_counts, on="Date", how="left")
    return (rolling_bag_of_words,)


@app.cell
def _(rolling_bag_of_words):
    rolling_bag_of_words

    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Coverage check
    """)
    return


@app.cell
def _(mo, twitter_posts_tokenized):
    coverage = twitter_posts_tokenized["HasDictWord"].mean()
    mo.md(f"Coverage {coverage:.3f}")
    return


@app.cell
def _(dict_words, pl, twitter_posts_tokenized):
    freqencies = twitter_posts_tokenized.explode("Tokens")

    freqencies = (
        freqencies.group_by("Tokens")
        .len()
        .with_columns(pl.col("len").truediv(freqencies.height).alias("freq"))
        .filter(pl.col("Tokens").is_in(dict_words))
    )
    return (freqencies,)


@app.cell
def _(freqencies, mo):
    mo.ui.dataframe(freqencies)
    return


@app.cell
def _(dict_words, pl, twitter_posts_tokenized):
    twitter_posts_tokenized.with_columns(
        pl.col("Tokens")
        .list.eval(pl.element().is_in(dict_words))
        .list.sum()
        .alias("num_dict_words")
    ).select("num_dict_words").describe()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Cluster check
    """)
    return


@app.cell
def _(StandardScaler, TSNE, rolling_bag_of_words):
    rolling_bow_no_date = rolling_bag_of_words.drop("Date").to_pandas()

    perplexity = min(30, max(5, rolling_bow_no_date.shape[0] // 3 - 1))
    X_tsne = TSNE(
        n_components=3, perplexity=perplexity, init="pca", random_state=42
    ).fit_transform(StandardScaler().fit_transform(rolling_bow_no_date))
    # X_pca = PCA(n_components=3, random_state=42).fit_transform(dynamic_bow_no_date)
    return X_tsne, rolling_bow_no_date


@app.cell
def _(dynamic_bag_of_words, np, pd, rolling_bow_no_date):
    TOP_K_WORDS_IN_HOVER = 6
    feature_names = dynamic_bag_of_words.columns[1:]
    month_index = dynamic_bag_of_words.to_pandas()["Date"].astype(str)

    X_counts_pd = pd.DataFrame(
        rolling_bow_no_date.values.astype(float),
        index=month_index,
        columns=feature_names,
    )

    def top_k_words(row, k=TOP_K_WORDS_IN_HOVER):
        vals = row.values
        if vals.sum() == 0:
            return ""
        idx = np.argsort(vals)[-k:][::-1]
        return ", ".join(
            [f"{feature_names[i]}({int(vals[i])})" for i in idx if vals[i] > 0]
        )

    hover_top = X_counts_pd.apply(
        lambda r: top_k_words(r, TOP_K_WORDS_IN_HOVER), axis=1
    )
    return hover_top, month_index


@app.cell
def _(X_tsne, hover_top, mo, month_index, pd, px):
    plot_df = pd.DataFrame(
        {
            "tsne1": X_tsne[:, 0],
            "tsne2": X_tsne[:, 1],
            "tsne3": X_tsne[:, 2],
            "month": month_index,
            "top_words": hover_top.values,
        }
    )

    fig = px.scatter_3d(
        plot_df,
        x="tsne1",
        y="tsne2",
        z="tsne3",
        hover_data={"month": True, "top_words": True},
        text="month",
        title="Monthly dictionary-word usage: t-SNE visualization",
        labels={"tsne1": "t-SNE 1", "tsne2": "t-SNE 2"},
    )

    fig.update_traces(marker=dict(size=10, opacity=0.8), textposition="top center")

    mo.ui.plotly(fig)
    return


@app.cell
def _(StandardScaler, dynamic_bag_of_words, umap):
    matrix_pd = dynamic_bag_of_words.to_pandas()

    dates = matrix_pd["Date"]
    X = matrix_pd.drop(columns=["Date"]).values

    X_scaled = StandardScaler().fit_transform(X)

    reducer = umap.UMAP(
        n_neighbors=10, min_dist=0.3, n_components=3, metric="cosine", random_state=42
    )

    embedding = reducer.fit_transform(X_scaled)
    return dates, embedding


@app.cell
def _(dates, embedding, mo, pd, px):
    umap_table = pd.DataFrame(
        {
            "umap1": embedding[:, 0],
            "umap2": embedding[:, 1],
            "umap3": embedding[:, 2],
            "Date": dates,
        }
    )

    figure_umap = px.scatter_3d(
        umap_table,
        x="umap1",
        y="umap2",
        z="umap3",
        text="Date",
        title="UMAP projection of monthly political vocabulary",
    )

    figure_umap.update_traces(textposition="top center")
    mo.ui.plotly(figure_umap)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
