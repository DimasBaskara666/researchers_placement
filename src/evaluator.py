"""Evaluation and result-table creation."""

from __future__ import annotations

from typing import Callable

import pandas as pd

from metrics import f1_at_k, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank


def evaluate_rankings(
    rankings: dict[str, list[tuple[str, float]]], truth: dict[str, set[str]]
) -> dict[str, float]:
    """Compute mean ranking metrics across all ranked testing researchers."""
    if not rankings:
        raise ValueError("No rankings were supplied for evaluation.")
    metric_functions: dict[str, Callable[[list[str], set[str], int], float]] = {
        "precision": precision_at_k,
        "recall": recall_at_k,
        "f1": f1_at_k,
        "ndcg": ndcg_at_k,
    }
    values: dict[str, list[float]] = {f"{name}@{k}": [] for name in metric_functions for k in (1, 3, 5)}
    values["mrr"] = []
    for researcher, scored_groups in rankings.items():
        relevant = truth[researcher]
        ranking = [group for group, _ in scored_groups]
        for name, function in metric_functions.items():
            for k in (1, 3, 5):
                values[f"{name}@{k}"].append(function(ranking, relevant, k))
        values["mrr"].append(reciprocal_rank(ranking, relevant))
    return {name: sum(scores) / len(scores) for name, scores in values.items()}


def recommendations_frame(
    model: str, rankings: dict[str, list[tuple[str, float]]], truth: dict[str, set[str]]
) -> pd.DataFrame:
    """Create one recommendation-output row per researcher and model."""
    rows: list[dict[str, object]] = []
    for researcher, ranking in rankings.items():
        row: dict[str, object] = {
            "model": model,
            "researcher_name": researcher,
            "ground_truth_groups": " | ".join(sorted(truth[researcher])),
        }
        for k in (1, 3, 5):
            top_k = ranking[:k]
            row[f"top_{k}_groups"] = " | ".join(group for group, _ in top_k)
            row[f"top_{k}_scores"] = " | ".join(f"{score:.6f}" for _, score in top_k)
        rows.append(row)
    return pd.DataFrame(rows)


def rankings_frame(
    model: str, rankings: dict[str, list[tuple[str, float]]], truth: dict[str, set[str]]
) -> pd.DataFrame:
    """Create one complete-ranking row per researcher, group, and model."""
    rows: list[dict[str, object]] = []
    for researcher, ranking in rankings.items():
        for rank, (group, score) in enumerate(ranking, start=1):
            rows.append(
                {
                    "model": model,
                    "researcher_name": researcher,
                    "rank": rank,
                    "group_name": group,
                    "similarity": score,
                    "is_relevant": group in truth[researcher],
                }
            )
    return pd.DataFrame(rows)
