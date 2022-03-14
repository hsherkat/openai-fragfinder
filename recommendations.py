import numpy as np
import pandas as pd

import config
from embeddings import embedding
from openai.embeddings_utils import (
    distances_from_embeddings,
    indices_of_nearest_neighbors_from_distances,
)

df_reviews_and_embeddings = pd.read_csv(config.REVIEWS_AND_EMBEDDINGS_PATH)
df_reviews_and_embeddings[
    "babbage_similarity"
] = df_reviews_and_embeddings.babbage_similarity.apply(eval).apply(np.array)

fragrance_embeddings = df_reviews_and_embeddings.groupby(
    ["brand", "name"]
).babbage_similarity.apply(np.mean)


def get_recommendations_from_strings(
    strings: list[str],
    query: str,
    k_nearest_neighbors: int = 1,
    engine="text-similarity-babbage-001",
) -> list[int]:
    """Get the k nearest neighbors of a given string."""
    embeddings = [embedding(string, engine) for string in strings]
    query_embedding = embedding(query)
    distances = distances_from_embeddings(
        query_embedding, embeddings, distance_metric="cosine"
    )
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(
        distances
    )

    k_counter = 0
    closest_strings = []
    for i in indices_of_nearest_neighbors:
        if query == strings[i]:
            continue
        if k_counter >= k_nearest_neighbors:
            break
        k_counter += 1
        s = f"""
        --- Recommendation #{k_counter} (nearest neighbor {k_counter} of {k_nearest_neighbors}) ---
        String: {strings[i]}
        Distance: {distances[i]:0.3f}"""
        closest_strings.append(s)

    return indices_of_nearest_neighbors, closest_strings


def fragrance_recommender(frag_description: str, review_df, n_recs: int = 5):
    """Given a description of what type of fragrance you want,
    returns a number of fragrances based on similarity of
    description to reviews of various fragrances.
    """
    query = " ".join(frag_description.split())
    reviews = list(review_df.cleaned_review)
    indices, reviews = get_recommendations_from_strings(reviews, query, n_recs)
    return review_df[["brand", "name", "cleaned_review"]].iloc[indices[:n_recs]]


def fragrance_recommender_alt(
    frag_description: str, fragrance_embeddings, n_recs: int = 5
):
    """Alternate method. Takes mean of review embeddings for each fragrance
    to find fragrance embedding.
    Also returns most relevant review for those recommended fragrances.
    Probably returns more accurate fragrance recommendations, but will look less impressive
    to users, as the reviews won't match as closely.
    """
    query = " ".join(frag_description.split())
    query_embedding = embedding(query)

    distances = distances_from_embeddings(
        query_embedding, fragrance_embeddings, distance_metric="cosine"
    )
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(
        distances
    )
    out = []
    for brand, name in fragrance_embeddings.index[
        indices_of_nearest_neighbors[:n_recs]
    ]:
        df = df_reviews_and_embeddings
        df = df[(df.brand == brand) & (df.name == name)]
        distances = distances_from_embeddings(
            query_embedding, df.babbage_similarity, distance_metric="cosine",
        )
        nearest_neighbor = indices_of_nearest_neighbors_from_distances(distances)[0]
        out.append(
            tuple([f"{brand}: {name}", df.cleaned_review.iloc[nearest_neighbor],])
        )
    return pd.DataFrame(out, columns=["Fragrance", "Relevant review"])
