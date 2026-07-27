"""Cosine-similarity ranking of research groups."""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def rank_groups(
    researcher_profiles: dict[str, np.ndarray], group_profiles: dict[str, np.ndarray]
) -> dict[str, list[tuple[str, float]]]:
    """Rank every group for each researcher by descending cosine similarity."""
    group_names = sorted(group_profiles)
    group_matrix = np.vstack([group_profiles[name] for name in group_names])
    rankings: dict[str, list[tuple[str, float]]] = {}
    for researcher, profile in researcher_profiles.items():
        scores = cosine_similarity(profile.reshape(1, -1), group_matrix).ravel()
        ordered = sorted(zip(group_names, scores, strict=True), key=lambda item: (-item[1], item[0]))
        rankings[researcher] = [(group, float(score)) for group, score in ordered]
    return rankings
