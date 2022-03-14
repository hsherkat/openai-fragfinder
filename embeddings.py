import pickle
from typing import ChainMap

import pandas as pd

import config
from openai.embeddings_utils import get_embedding

df_reviews = pd.read_csv(config.CLEAN_REVIEW_PATH)

if config.TESTING_BOOL:
    print(f"Before: {df_reviews.shape}")
    df_reviews = df_reviews.sample(frac=config.SAMPLE_RATE, random_state=config.SEED)
    print(f"After: {df_reviews.shape}")

try:
    embedding_train_cache = pd.read_pickle(config.EMBEDDING_TRAIN_CACHE_PATH)
except FileNotFoundError:
    embedding_train_cache = {}

try:
    embedding_query_cache = pd.read_pickle(config.EMBEDDING_QUERY_CACHE_PATH)
except FileNotFoundError:
    embedding_query_cache = {}

embedding_cache = ChainMap(embedding_query_cache, embedding_train_cache)


def embedding(
    text: str,
    engine: str = "text-similarity-babbage-001",
    cache=embedding_cache,
    training=False,
):
    """Returns embedding of text. Memoized to avoid recomputing.
    Text should be cleaned to not have newlines.
    Queries and training are cached separately.
    """
    query_cache, train_cache = cache.maps
    if (text, engine) not in cache:
        if training:
            train_cache[(text, engine)] = get_embedding(text, engine)
        else:
            query_cache[(text, engine)] = get_embedding(text, engine)
            with open(config.EMBEDDING_QUERY_CACHE_PATH, "wb") as cache_file:
                pickle.dump(query_cache, cache_file)
        print("NOT FOUND")
    return cache[(text, engine)]


def main():
    """Run to get and save embeddings.
    """
    df_reviews["babbage_similarity"] = df_reviews.cleaned_review.apply(
        lambda x: embedding(x, cache=embedding_cache, training=True)
    )
    with open(config.EMBEDDING_TRAIN_CACHE_PATH, "wb") as cache_file:
        pickle.dump(embedding_train_cache, cache_file)
    df_reviews.to_csv(config.REVIEWS_AND_EMBEDDINGS_PATH, index=False)


if __name__ == "__main__":
    main()
