"""LDA document-topic representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


@dataclass
class LdaModel:
    """A fitted LDA model and its training-document vectorizer."""

    model: LatentDirichletAllocation
    vectorizer: CountVectorizer


def fit_lda(
    documents: list[str],
    parameters: dict[str, Any],
    preprocessing: dict[str, Any],
    random_seed: int,
) -> LdaModel:
    """Fit LDA to count vectors from the training publications."""
    vectorizer = CountVectorizer(
        max_features=preprocessing["max_features"],
        min_df=preprocessing["min_df"],
        max_df=preprocessing["max_df"],
    )
    matrix = vectorizer.fit_transform(documents)
    model = LatentDirichletAllocation(
        n_components=parameters["num_topics"],
        max_iter=parameters["max_iter"],
        learning_method="batch",
        random_state=random_seed,
    )
    model.fit(matrix)
    return LdaModel(model=model, vectorizer=vectorizer)


def topic_vectors(model: LdaModel, documents: list[str]) -> np.ndarray:
    """Transform documents into fitted LDA topic vectors."""
    return model.model.transform(model.vectorizer.transform(documents))


def topic_keywords(model: LdaModel, top_n: int = 10) -> list[list[str]]:
    """Return the top keywords for each LDA topic."""
    vocabulary = model.vectorizer.get_feature_names_out()
    return [
        vocabulary[np.argsort(component)[-top_n:][::-1]].tolist()
        for component in model.model.components_
    ]
