import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import polars as pl
    import xdk

    import unicodedata
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    from sklearn.preprocessing import StandardScaler

    from cleantext import clean
    import json

    import os

    return Path, clean, json, nltk, os, pl, unicodedata, xdk


@app.cell
def _(mo):
    mo.md(r"""
    # Twitter
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Preparation
    """)
    return


@app.cell
def _(os, xdk):
    bearer_token = os.environ.get("BEARER_TOKEN")
    client = xdk.Client(bearer_token=bearer_token)
    return (client,)


@app.cell
def _(Path):
    random_entry = Path("data/EliteTwitterActivity")
    return (random_entry,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Parsing one post per author
    To obtain author twitter id
    """)
    return


@app.cell
def _(pl, random_entry):
    post_ids = []

    for data_file in random_entry.rglob("*.xlsx"):
        parsed_table = pl.read_excel(data_file)

        new_post_id = parsed_table["tweet_id"][0]
        new_post_id = new_post_id[1:] if new_post_id[0] == "'" else new_post_id
        new_post_id = new_post_id[:-1] if new_post_id[-1] == "'" else new_post_id

        post_ids.append(new_post_id)
    return (post_ids,)


@app.cell
def _(post_ids):
    post_ids
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Parsing all posts per author
    """)
    return


@app.cell
def _(client, post_ids):
    i = 0

    authors = []
    includes = []

    while i * 100 < len(post_ids):
        current_ids = post_ids[i * 100 : (i + 1) * 100]

        try:
            response = client.posts.get_by_ids(
                ids=current_ids,
                tweet_fields=["author_id"],
                expansions=["author_id"],
                user_fields=["username", "name"],
            )
        except Exception as err:
            print(err)

        authors.append(response.data)
        includes.append(response.includes)

        i += 1
    return (includes,)


@app.cell
def _(includes):
    user_names_with_ids = []

    for include in includes:
        user_names_with_ids.extend(include["users"])
    return (user_names_with_ids,)


@app.cell
def _(pl, user_names_with_ids):
    user_names_with_ids_table = pl.DataFrame(user_names_with_ids).drop("withheld")
    return (user_names_with_ids_table,)


@app.cell
def _(random_entry, user_names_with_ids_table):
    user_names_with_ids_table.write_csv(random_entry / "authors_ids.csv")
    return


@app.cell
def _(mo, user_names_with_ids_table):
    mo.ui.dataframe(user_names_with_ids_table)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Flattening responses
    """)
    return


@app.cell
def _(client, user_names_with_ids):
    all_posts = []
    for author in user_names_with_ids:
        posts_response = client.users.get_posts(
            author["id"], tweet_fields=["author_id", "created_at", "id", "text"]
        )

        for post in posts_response:
            post_data = getattr(post, "data", []) or []
            all_posts.extend(post_data)
    return (all_posts,)


@app.cell
def _(all_posts, pl, random_entry):
    pl.DataFrame(all_posts).drop("edit_history_tweet_ids").write_csv(
        random_entry / "one_author_posts.csv"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Dictionary preparation
    """)
    return


@app.cell
def _(pl):
    general_inquirer = pl.read_excel("data/final_data/dictionary/inquirerbasic.xls")
    return (general_inquirer,)


@app.cell
def _():
    interest_topics = ["Econ@", "Exch", "ECON", "Legal", "Milit", "Polit@", "PowTot"]

    excluded_topics = ["PowPt", "PowOth"]
    return excluded_topics, interest_topics


@app.cell
def _(excluded_topics, general_inquirer, interest_topics, pl):
    general_inquirer_relevant_words = (
        general_inquirer.filter(
            pl.any_horizontal(pl.col(interest_topics).is_null().not_())
            & pl.any_horizontal(pl.col(excluded_topics).is_null())
        )
        .select(["Entry", *interest_topics])
        .with_columns(
            pl.col("Entry")
            .str.to_lowercase()
            .str.splitn("#", 2)
            .struct.field("field_0")
            .alias("EntryCleaned")
        )
        .unique("EntryCleaned", keep="first")
        .drop("Entry")
    )
    return (general_inquirer_relevant_words,)


@app.cell
def _(general_inquirer_relevant_words, mo):
    mo.ui.dataframe(general_inquirer_relevant_words)
    return


@app.cell
def _(general_inquirer_relevant_words):
    general_inquirer_relevant_words.write_csv(
        "data/final_data/dictionary/cleaned_dict.csv"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Texts preparation
    """)
    return


@app.cell
def _(nltk):
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    return


@app.cell
def _(pl):
    political_tweets = pl.read_csv("data/final_data/train/Political_tweets 2.csv")
    return (political_tweets,)


@app.cell
def _(clean, pl, political_tweets):
    political_tweets_cleaned = political_tweets.select(
        "user_name",
        pl.col("date").str.to_datetime(),
        pl.col("text")
        .str.to_lowercase()
        .map_elements(
            lambda text: clean(
                text,
                extra_spaces=True,
                stopwords=True,
                numbers=True,
                punct=True,
            ), return_dtype = pl.String
        ),
    ).rename(
        {
            "user_name": "UserName",
            "date": "Date",
            "text": "Text",
        }
    )
    return (political_tweets_cleaned,)


@app.cell
def _(pl, political_tweets_cleaned, unicodedata):
    def normalize_text(text: str) -> str:
        if not text:
            return ""

        text = unicodedata.normalize("NFKC", text)
        return text

    political_tweets_norm = political_tweets_cleaned.with_columns(
        pl.col("Text").map_elements(normalize_text)
    )
    return normalize_text, political_tweets_norm


@app.cell
def _(mo, political_tweets_norm):
    mo.ui.dataframe(political_tweets_norm)
    return


@app.cell
def _(clean, normalize_text, pl):
    trump_tweets = pl.read_csv("data/final_data/train/trump_tweets_01-08-2021.csv")
    trump_tweets_norm = (
        trump_tweets
            .select(
            pl.lit("Trump").alias("UserName"),
            pl.col("date").str.to_datetime(),
            pl.col("text")
            .str.to_lowercase()
            .map_elements(
                lambda text: clean(
                    text,
                    extra_spaces=True,
                    stopwords=True,
                    numbers=True,
                punct=True,
            ), return_dtype = pl.String
        ),
        ).rename(
            {
                "date": "Date",
                "text": "Text",
            }
        )
        .with_columns(
            pl.col("Text").map_elements(normalize_text)
        )
    )
    return (trump_tweets_norm,)


@app.cell
def _(mo, trump_tweets_norm):
    mo.ui.dataframe(trump_tweets_norm)
    return


@app.cell
def _(json):
    congress_tweets = []

    with open("data/final_data/train/tweets.json", "r") as congress_tweets_file:
        for line in congress_tweets_file:
            congress_tweets.append(json.loads(line))
    return (congress_tweets,)


@app.cell
def _(congress_tweets, pl):
    tweets_congress_table = pl.DataFrame(congress_tweets, schema=["created_at", "screen_name", "text", "user_id"]).rename({
        "created_at": "Date",
        "screen_name": "UserName",
        "text": "Text"
    }).with_columns(pl.from_epoch("Date", time_unit="s"))
    return (tweets_congress_table,)


@app.cell
def _(clean, normalize_text, pl, tweets_congress_table):
    tweets_congress_table_norm = (
        tweets_congress_table
        .with_columns(
            pl.col("Text")
            .str.to_lowercase()
            .map_elements( 
                lambda text: clean(
                    text,
                    extra_spaces=True,
                    stopwords=True,
                    numbers=True,
                    punct=True,
                ), return_dtype = pl.String
            ),
        )
        .with_columns(
            pl.col("Text").map_elements(normalize_text)
        )
    )
    return (tweets_congress_table_norm,)


@app.cell
def _(mo, tweets_congress_table_norm):
    mo.ui.dataframe(tweets_congress_table_norm)
    return


@app.cell
def _(
    pl,
    political_tweets_norm,
    trump_tweets_norm,
    tweets_congress_table_norm,
):
    full_twitter_data = pl.concat(
        [
            political_tweets_norm,
            trump_tweets_norm[political_tweets_norm.columns],
            tweets_congress_table_norm[political_tweets_norm.columns]
        ],
        how="vertical"
    )
    return (full_twitter_data,)


@app.cell
def _(full_twitter_data, mo):

    mo.ui.dataframe(full_twitter_data)
    return


@app.cell
def _(full_twitter_data):
    full_twitter_data.write_csv("data/final_data/train/twitter_final.csv")
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
