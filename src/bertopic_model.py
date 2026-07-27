"""BERTopic publication representation using SPECTER embeddings."""

from __future__ import annotations

from typing import Any

import numpy as np
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from preprocessing import bag_of_words_stopwords


def fit_bertopic(
    documents: list[str],
    parameters: dict[str, Any],
    random_seed: int,
    embedding_model: SentenceTransformer | None = None,
    embeddings: np.ndarray | None = None,
) -> BERTopic:
    """Fit BERTopic with SPECTER embeddings or supplied precomputed embeddings."""
    if embedding_model is None:
        embedding_model = SentenceTransformer(parameters["embedding_model"])
    umap_model = UMAP(
        n_neighbors=parameters["umap_n_neighbors"],
        n_components=parameters["umap_n_components"],
        min_dist=parameters["umap_min_dist"],
        metric="cosine",
        random_state=random_seed,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=parameters["min_topic_size"],
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(stop_words=sorted(bag_of_words_stopwords()))
    model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
        verbose=False,
    )
    if embeddings is None:
        model.fit(documents)
    else:
        model.fit(documents, embeddings=embeddings)
    return model


def topic_distributions(model: BERTopic, documents: list[str]) -> np.ndarray:
    """Return BERTopic document-topic distributions for the supplied documents."""
    distributions, _ = model.approximate_distribution(documents)
    return np.asarray(distributions, dtype=float)


def topic_keywords(model: BERTopic, top_n: int = 10) -> list[tuple[int, list[str]]]:
    """Return non-noise BERTopic IDs and their top keywords."""
    return [
        (topic_id, [word for word, _ in words[:top_n]])
        for topic_id, words in sorted(model.get_topics().items())
        if topic_id != -1
    ]


def noise_statistics(model: BERTopic) -> tuple[int, float]:
    """Return the number and percentage of training documents assigned to noise."""
    topics = np.asarray(model.topics_)
    noise_count = int(np.count_nonzero(topics == -1))
    noise_percentage = 100 * noise_count / len(topics) if len(topics) else 0.0
    return noise_count, noise_percentage
