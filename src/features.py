"""Feature extraction helpers."""

from __future__ import annotations

from typing import Tuple

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def create_vectorizer(
    vectorizer_name: str,
    ngram_range: Tuple[int, int],
    min_df: int,
):
    """Create a configured CountVectorizer or TfidfVectorizer."""
    if vectorizer_name == "count":
        return CountVectorizer(ngram_range=ngram_range, min_df=min_df)
    if vectorizer_name == "tfidf":
        return TfidfVectorizer(ngram_range=ngram_range, min_df=min_df)

    raise ValueError(f"Unsupported vectorizer_name: {vectorizer_name}")
