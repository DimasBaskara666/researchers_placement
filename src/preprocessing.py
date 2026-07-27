"""Loading and minimal text preparation for the input datasets."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

import pandas as pd
from nltk.corpus import stopwords


REQUIRED_PUBLICATION_COLUMNS = {"Title", "Abstract", "Authors"}
REQUIRED_GROUP_COLUMNS = {"researcher_name", "group_name"}
RESEARCH_JARGON_STOPWORDS = {
    "abstract",
    "paper",
    "article",
    "proceedings",
    "conference",
    "journal",
    "author",
    "contributors",
    "manuscript",
    "review",
}


def clean_text(value: object) -> str:
    """Collapse whitespace while preserving the original text for embeddings."""
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


@lru_cache(maxsize=1)
def bag_of_words_stopwords() -> frozenset[str]:
    """Return NLTK English stopwords combined with non-domain research jargon."""
    return frozenset(stopwords.words("english")) | RESEARCH_JARGON_STOPWORDS


def preprocess_bag_of_words(document: str) -> str:
    """Lowercase and remove punctuation, non-alphanumeric tokens, and configured stopwords."""
    tokens = re.sub(r"[^a-z0-9\s]", " ", document.lower()).split()
    stopword_set = bag_of_words_stopwords()
    return " ".join(token for token in tokens if token not in stopword_set)


def load_datasets(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load data and prepare original-text and bag-of-words publication views.

    The ``document`` column preserves original title-plus-abstract text for
    transformer embeddings. ``bow_document`` applies conventional bag-of-words
    cleaning for TF-IDF, LDA, and NMF. Authors are split and limited to labelled
    researchers in the group dataset.
    """
    publications_path = data_dir / "publications_2015_2025_10k_plus.csv"
    groups_path = data_dir / "groups_2015_2025_10k_plus.csv"
    publications = pd.read_csv(publications_path)
    groups = pd.read_csv(groups_path)

    missing_publication = REQUIRED_PUBLICATION_COLUMNS - set(publications.columns)
    missing_group = REQUIRED_GROUP_COLUMNS - set(groups.columns)
    if missing_publication:
        raise ValueError(f"Publication dataset is missing columns: {sorted(missing_publication)}")
    if missing_group:
        raise ValueError(f"Group dataset is missing columns: {sorted(missing_group)}")

    groups = groups.copy()
    groups["researcher_name"] = groups["researcher_name"].astype(str).str.strip()
    groups["group_name"] = groups["group_name"].astype(str).str.strip()
    groups = groups[(groups["researcher_name"] != "") & (groups["group_name"] != "")]
    groups = groups.drop_duplicates().reset_index(drop=True)
    known_researchers = set(groups["researcher_name"])

    records: list[dict[str, str]] = []
    for row in publications[["Title", "Abstract", "Authors"]].itertuples(index=False):
        document = " ".join(part for part in (clean_text(row.Title), clean_text(row.Abstract)) if part)
        bow_document = preprocess_bag_of_words(document)
        if not document or not bow_document:
            continue
        if pd.isna(row.Authors):
            continue
        for author in str(row.Authors).split(";"):
            researcher = author.strip()
            if researcher in known_researchers:
                records.append(
                    {"researcher_name": researcher, "document": document, "bow_document": bow_document}
                )

    prepared_publications = pd.DataFrame(records).reset_index(drop=True)
    if prepared_publications.empty:
        raise ValueError("No usable publications match researchers in the group dataset.")
    return prepared_publications, groups
