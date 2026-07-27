"""Construction of researcher and research-group profiles."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import issparse


def average_researcher_profiles(
    publications: pd.DataFrame, publication_vectors: object
) -> dict[str, np.ndarray]:
    """Average each researcher's publication vectors into one profile."""
    if len(publications) != publication_vectors.shape[0]:
        raise ValueError("Publication data and vectors must have equal row counts.")
    profiles: dict[str, np.ndarray] = {}
    grouped_indices = publications.groupby("researcher_name", sort=True).indices
    for researcher, indices in grouped_indices.items():
        rows = publication_vectors[indices]
        mean = rows.mean(axis=0)
        profiles[researcher] = np.asarray(mean).ravel() if issparse(rows) else np.asarray(mean).ravel()
    return profiles


def build_group_profiles(
    researcher_profiles: dict[str, np.ndarray], groups: pd.DataFrame
) -> dict[str, np.ndarray]:
    """Average available member profiles to construct each group profile."""
    result: dict[str, np.ndarray] = {}
    for group_name, members in groups.groupby("group_name", sort=True)["researcher_name"]:
        member_profiles = [researcher_profiles[name] for name in members if name in researcher_profiles]
        if member_profiles:
            result[group_name] = np.mean(np.vstack(member_profiles), axis=0)
    if not result:
        raise ValueError("No group profiles could be built from training researchers.")
    return result


def ground_truth_groups(groups: pd.DataFrame) -> dict[str, set[str]]:
    """Map each researcher to its set of known research-group memberships."""
    return groups.groupby("researcher_name")["group_name"].agg(set).to_dict()
