"""NMF document-topic representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class NmfModel:
    """A fitted NMF model and its training-document vectorizer."""

    model: NMF
    vectorizer: TfidfVectorizer


def fit_nmf(
    documents: list[str],
    parameters: dict[str, Any],
    preprocessing: dict[str, Any],
    random_seed: int,
) -> NmfModel:
    """Fit NMF to TF-IDF vectors from the training publications."""
    vectorizer = TfidfVectorizer(
        max_features=preprocessing["max_features"],
        min_df=preprocessing["min_df"],
        max_df=preprocessing["max_df"],
    )
    matrix = vectorizer.fit_transform(documents)
    model = NMF(
        n_components=parameters["num_topics"],
        init="nndsvda",
        max_iter=parameters["max_iter"],
        random_state=random_seed,
    )
    model.fit(matrix)
    return NmfModel(model=model, vectorizer=vectorizer)


def topic_vectors(model: NmfModel, documents: list[str]) -> np.ndarray:
    """Transform documents into fitted NMF topic vectors."""
    return model.model.transform(model.vectorizer.transform(documents))


def topic_keywords(model: NmfModel, top_n: int = 10) -> list[list[str]]:
    """Return the top keywords for each NMF topic."""
    vocabulary = model.vectorizer.get_feature_names_out()
    return [
        vocabulary[np.argsort(component)[-top_n:][::-1]].tolist()
        for component in model.model.components_
    ]
