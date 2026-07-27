"""TF-IDF publication representation."""

from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer


def fit_tfidf(documents: list[str], preprocessing: dict[str, Any]) -> TfidfVectorizer:
    """Fit and return a TF-IDF vectorizer using shared preprocessing settings."""
    ngram_range = tuple(preprocessing["ngram_range"])
    vectorizer = TfidfVectorizer(
        max_features=preprocessing["max_features"],
        ngram_range=ngram_range,
        min_df=preprocessing["min_df"],
        max_df=preprocessing["max_df"],
    )
    vectorizer.fit(documents)
    return vectorizer
