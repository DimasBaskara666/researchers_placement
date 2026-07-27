"""Researcher-level train/test splitting."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def split_researchers(
    researchers: list[str], train_ratio: float, random_seed: int
) -> tuple[set[str], set[str]]:
    """Return disjoint researcher-level training and testing sets."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be strictly between 0 and 1.")
    unique_researchers = sorted(set(researchers))
    if len(unique_researchers) < 2:
        raise ValueError("At least two researchers are required for a split.")
    train, test = train_test_split(
        unique_researchers, train_size=train_ratio, random_state=random_seed
    )
    return set(train), set(test)


def publications_for_researchers(
    publications: pd.DataFrame, researchers: set[str]
) -> pd.DataFrame:
    """Return publications belonging to a supplied set of researchers."""
    return publications[publications["researcher_name"].isin(researchers)].copy()
