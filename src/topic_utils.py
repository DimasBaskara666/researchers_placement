"""Shared topic-quality and topic-export helpers."""

from __future__ import annotations

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel


def topics_frame(topic_lists: list[list[str]]) -> pd.DataFrame:
    """Create topic rows containing an ID and its top keywords."""
    return pd.DataFrame(
        {"topic_id": range(len(topic_lists)), "top_keywords": [" | ".join(words) for words in topic_lists]}
    )


def coherence_cv(topic_lists: list[list[str]], documents: list[str]) -> float:
    """Calculate C_v coherence using the training publication texts."""
    texts = [document.split() for document in documents]
    dictionary = Dictionary(texts)
    coherence = CoherenceModel(
        topics=topic_lists,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v",
        processes=1,
    )
    return float(coherence.get_coherence())
