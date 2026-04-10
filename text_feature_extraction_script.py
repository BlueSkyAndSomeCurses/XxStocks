import polars as pl
from decouple import config

import nltk
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
import re

from cleantext import clean
from spellchecker import SpellChecker

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import csv


dictionary = pl.read_csv("data/final_data/dictionary/cleaned_dict.csv")
twitter_posts = pl.read_csv("data/final_data/train/twitter_final.csv").with_columns(
    pl.col("Date").str.to_datetime()
)

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
    .with_columns(pl.col("Tokens").is_in(dict_words).alias("TokensStage1"))
    .sort("Date")
)

bow_with_categories = (
    bag_of_words_prep.join(
        dict_flags, left_on="Tokens", right_on="EntryCleaned", how="left"
    )
    .with_columns(
        [pl.col(c).fill_null(False).cast(pl.UInt32) for c in category_columns]
    )
    .rename({col_name: f"{col_name}Stage1" for col_name in category_columns})
)

sampled_dictionary = (
    dictionary.with_columns(
        pl.col(column_name).is_not_null() for column_name in category_columns
    )
    .unpivot(
        index="EntryCleaned",
        on=category_columns,
        variable_name="category",
        value_name="is_in_category",
    )
    .filter(pl.col("is_in_category"))  # 1. Keep only where the boolean is True
    .sample(fraction=1.0, shuffle=True)  # 2. Shuffle everything randomly
    .group_by("category")  # 3. Group by the new 'category' column
    .head(10)  # 4. Take the first 10 for each
    .drop("is_in_category")  # 5. Clean up (they are all True anyway)
)


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

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# Configure 4-bit quantization for NVIDIA GPUs
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

# device_map="auto" automatically loads the model onto your NVIDIA GPU(s)
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map="auto"
)

words_to_classify = (
    sampled_dictionary["EntryCleaned"].to_list()[25000:]
    + sampled_dictionary["EntryCleaned"].to_list()[:1334]
)
categories = (
    sampled_dictionary["category"].to_list()[25000:]
    + sampled_dictionary["category"].to_list()[:1334]
)

examples_of_classification = []
for word_, category_ in zip(words_to_classify, categories):
    examples_of_classification.append(
        {"role": "user", "content": f"Classify word: {word_}"}
    )
    examples_of_classification.append({"role": "assistant", "content": category_})

messages = [
    {"role": "system", "content": PROMPT_SYSTEM},
    *examples_of_classification,
]


def get_words_category(word: str) -> str:
    curr_messages = messages + [{"role": "user", "content": f"Classify word: {word}"}]

    input_ids = tokenizer.apply_chat_template(
        curr_messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    # Llama 3 specific stopping criteria
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    # Generate response
    outputs = model.generate(
        input_ids,
        max_new_tokens=6,
        eos_token_id=terminators,
        do_sample=False,  # Keeps responses deterministic
        pad_token_id=tokenizer.eos_token_id,
    )

    response_ids = outputs[0][input_ids.shape[-1] :]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)

    return response.strip()


stage2_token = (
    bow_with_categories.filter(pl.col("TokensStage1").eq(False))["Tokens"]
    .unique()
    .to_list()
)
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


stage2_approved_tokens = [token for token in stage2_token if is_normal_word(token)]


with open(
    "./data/stage2_dictionary_other_half.csv",
    mode="w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.writer(file)

    writer.writerow(["word", "category"])

    for index, word in enumerate(stage2_approved_tokens):
        category = get_words_category(word)
        writer.writerow([word, category])

        print(index, word, category)
