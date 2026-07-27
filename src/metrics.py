"""Ranking metrics for multi-label research-group recommendations."""

from __future__ import annotations

import math


def precision_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    """Return the fraction of the first k ranked groups that are relevant."""
    return sum(group in relevant for group in ranking[:k]) / k


def recall_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    """Return the fraction of all relevant groups retrieved in the first k ranks."""
    return sum(group in relevant for group in ranking[:k]) / len(relevant) if relevant else 0.0


def f1_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    """Return the harmonic mean of Precision@k and Recall@k."""
    precision = precision_at_k(ranking, relevant, k)
    recall = recall_at_k(ranking, relevant, k)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def reciprocal_rank(ranking: list[str], relevant: set[str]) -> float:
    """Return reciprocal rank of the first relevant group, or zero if absent."""
    for position, group in enumerate(ranking, start=1):
        if group in relevant:
            return 1 / position
    return 0.0


def ndcg_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    """Return binary-relevance normalized discounted cumulative gain at k."""
    dcg = sum(
        1 / math.log2(position + 1)
        for position, group in enumerate(ranking[:k], start=1)
        if group in relevant
    )
    ideal_count = min(len(relevant), k)
    ideal_dcg = sum(1 / math.log2(position + 1) for position in range(1, ideal_count + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0
