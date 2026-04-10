import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from decouple import config

    import nltk
    from nltk.corpus import stopwords, words
    from nltk.stem import WordNetLemmatizer
    import re

    from cleantext import clean
    from spellchecker import SpellChecker

    from mlx_lm import load, generate
    import csv

    from dataset.preprocessing import (
        augment_dataset,
        downsample_to_interval,
        prepare_arima_data,
        split_features_target,
        time_train_test_split,
        get_bag_of_words,
        get_category_features,
    )

    return (
        SpellChecker,
        augment_dataset,
        csv,
        downsample_to_interval,
        generate,
        load,
        mo,
        pl,
        re,
        words,
    )


@app.cell
def _(pl):
    dictionary = pl.read_csv("data/final_data/dictionary/cleaned_dict.csv")
    twitter_posts = pl.read_csv("data/final_data/train/twitter_final.csv").with_columns(
        pl.col("Date").str.to_datetime(format="%Y-%m-%dT%H:%M:%S.%6f%z").dt.convert_time_zone("US/Mountain")
    )
    return dictionary, twitter_posts


@app.cell
def _(dictionary, mo):
    mo.ui.dataframe(dictionary.head())
    return


@app.cell
def _(dictionary, pl):
    category_columns = [col_name for col_name in dictionary.columns if col_name != "EntryCleaned"]

    dict_words = dictionary["EntryCleaned"].to_list()
    dict_flags = dictionary.with_columns([pl.col(column).is_not_null().alias(column) for column in category_columns])
    return category_columns, dict_flags, dict_words


@app.cell
def _(pl, twitter_posts):
    twitter_posts_tokenized = twitter_posts.with_columns(pl.col("Text").str.split(" ").alias("Tokens"))
    return (twitter_posts_tokenized,)


@app.cell
def _(category_columns, dict_flags, dict_words, pl, twitter_posts_tokenized):
    bag_of_words_prep = (
        twitter_posts_tokenized.explode("Tokens")
        .filter(pl.col("Tokens").is_not_null())
        .filter(pl.col("Tokens").ne(""))
        .with_columns(pl.col("Tokens").is_in(dict_words).alias("TokensStage1"))
        .sort("Date")
    )

    bow_with_categories = (
        bag_of_words_prep.join(dict_flags, left_on="Tokens", right_on="EntryCleaned", how="left")
        .with_columns([pl.col(c).fill_null(False).cast(pl.UInt32) for c in category_columns])
        .rename({col_name: f"{col_name}Stage1" for col_name in category_columns})
    )
    return (bow_with_categories,)


@app.cell
def _(bow_with_categories):
    bow_with_categories
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Filling up Stage2
    Words, that are not found in harward dictionary are marked as Stage 2 and classified by chat gpt.
    """)
    return


@app.cell
def _(category_columns, dictionary, pl):
    sampled_dictionary = (
        dictionary.with_columns(pl.col(column_name).is_not_null() for column_name in category_columns)
        .unpivot(index="EntryCleaned", on=category_columns, variable_name="category", value_name="is_in_category")
        .filter(pl.col("is_in_category"))  # 1. Keep only where the boolean is True
        .sample(fraction=1.0, shuffle=True)  # 2. Shuffle everything randomly
        .group_by("category")  # 3. Group by the new 'category' column
        .head(10)  # 4. Take the first 10 for each
        .drop("is_in_category")  # 5. Clean up (they are all True anyway)
    )
    return (sampled_dictionary,)


@app.cell
def _():
    PROMPT_SYSTEM = """
    You are a proffesional linguist. Your specialization is dictionary collection and word's category classification.
    You are given an example of classifying word using Harvard IV-4 dictionary. There are 7 categories used: 
    Economics related -- Econ@, Exch, ECON
    Law related - Legal
    Military related - Milit
    Politics related - Polit, PowTot

    Your task is to classify the word given to you in form
    Classify the word: <the word to be classified>
    You MUST assign this word to ONLY ONE category: Econ@, Exch, ECON, Legal, Milit, Polit, PowTot, Rudementary(In case word does not fit in any of it)\n

    Your answer MUST be only ONE word -- the class of the word.
    """
    return (PROMPT_SYSTEM,)


@app.cell
def _(load):
    model, tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")
    return model, tokenizer


@app.cell
def _(PROMPT_SYSTEM, generate, model, sampled_dictionary, tokenizer):
    words_to_classify = sampled_dictionary["EntryCleaned"].to_list()[25000:]
    categories = sampled_dictionary["category"].to_list()[25000:]

    examples_of_classification = []
    for word_, category_ in zip(words_to_classify, categories):
        examples_of_classification.append({"role": "user", "content": f"Classify word: {word_}"})
        examples_of_classification.append({"role": "assistant", "content": category_})

    messages = [
        {"role": "system", "content": PROMPT_SYSTEM},
        *examples_of_classification,
    ]


    def get_words_category(word: str) -> str:
        curr_messages = messages + [{"role": "user", "content": f"Classify word: {word}"}]

        prompt = tokenizer.apply_chat_template(curr_messages, tokenize=False, add_generation_prompt=True)

        response = generate(model, tokenizer, prompt=prompt, verbose=False, max_tokens=6)

        return response

    return (get_words_category,)


@app.cell
def _(bow_with_categories, pl):
    stage2_token = bow_with_categories.filter(pl.col("TokensStage1").eq(False))["Tokens"].unique().to_list()
    return (stage2_token,)


@app.cell
def _(SpellChecker, re, words):
    english_vocab = set(w.lower() for w in words.words())
    spell = SpellChecker()


    def is_normal_word(word):
        if len(word) <= 2:
            if word in {"up", "us", "uk", "go", "no", "qe"}:
                return True
            return False

        if word in english_vocab:
            return True

        vowels = "aeiouy"
        if not any(char in vowels for char in word):
            return False

        if re.search(r"(.)\1\1\1", word):
            return False

        unknown = spell.unknown([word])
        if len(unknown) == 0:
            return True

        return False

    return (is_normal_word,)


@app.cell
def _(is_normal_word, stage2_token):
    stage2_approved_tokens = [token for token in stage2_token if is_normal_word(token)]
    return (stage2_approved_tokens,)


@app.cell
def _(csv, get_words_category, mo, stage2_approved_tokens):
    with open("./data/stage2_dictionary_other_half.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(["word", "category"])

        with mo.status.progress_bar(total=len(stage2_approved_tokens), title="Starting...") as bar:
            for index, word in enumerate(stage2_approved_tokens):
                category = get_words_category(word)
                writer.writerow([word, category])

                bar.update(increment=1, title=f"Processing index {index}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Combining with numerical data
    """)
    return


@app.cell
def _(augment_dataset, downsample_to_interval, pl):
    CSV_PATH = "./data/1_min_SPY_2008-2021.csv"
    df = pl.read_csv(CSV_PATH)
    df = pl.read_csv(CSV_PATH)
    df = downsample_to_interval(df, interval="30m")
    df_augmented = augment_dataset(df)
    return (df_augmented,)


@app.cell
def _(bow_with_categories):
    bow_with_categories
    return


@app.cell
def _(bow_with_categories, df_augmented, pl):
    bow_with_time_stamps = bow_with_categories.join_asof(
        df_augmented.select(pl.col("date").alias("TimeBin")), strategy="forward", left_on="Date", right_on="TimeBin"
    )
    return (bow_with_time_stamps,)


@app.cell
def _(bow_with_time_stamps):
    bow_with_time_stamps
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
