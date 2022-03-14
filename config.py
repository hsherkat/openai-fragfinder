import os

from dotenv import load_dotenv

import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# scraper
REVIEW_DATA_PATH = "reviews.csv"
NOTE_DATA_PATH = "frag_notes.csv"

# embeddings
CLEAN_REVIEW_PATH = "reviews_cleaned.csv"


# embeddings, recommendation, and app
EMBEDDING_TRAIN_CACHE_PATH = "embeddings.pkl"
EMBEDDING_QUERY_CACHE_PATH = "embeddings_query.pkl"
TESTING_BOOL = False  # If subsampling for testing, set to True.
SAMPLE_RATE = 0.03
SEED = 40
REVIEWS_AND_EMBEDDINGS_PATH = (
    "reviews_final.csv" if not TESTING_BOOL else "sampled_reviews.csv"
)
