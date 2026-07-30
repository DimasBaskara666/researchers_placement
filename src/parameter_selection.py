"""Select LDA, NMF, and BERTopic parameters using training-corpus C_v coherence."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import time

import pandas as pd
import yaml

from lda_model import fit_lda, topic_keywords as lda_keywords
from nmf_model import fit_nmf, topic_keywords as nmf_keywords
from preprocessing import load_datasets
from splitter import publications_for_researchers, split_researchers
from topic_utils import coherence_cv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LDA_TOPICS = [19, 21, 29, 31]
NMF_TOPICS = [9, 11, 12, 15]
BERTOPIC_MIN_TOPIC_SIZES = [40, 45, 50]
BERTOPIC_NEIGHBORS = [50, 55, 60]


def load_config(path: Path) -> dict:
    """Load selection settings and default model parameters from YAML."""
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def training_publications(config: dict) -> pd.DataFrame:
    """Return only the researcher-level training publications used for selection."""
    publications, groups = load_datasets(PROJECT_ROOT / "data")
    researchers = sorted(set(publications["researcher_name"]) & set(groups["researcher_name"]))
    train_researchers, _ = split_researchers(
        researchers, config["train_ratio"], config["random_seed"]
    )
    return publications_for_researchers(publications, train_researchers)


def search_lda(documents: list[str], config: dict) -> pd.DataFrame:
    """Evaluate each LDA topic-count candidate using C_v coherence."""
    rows: list[dict[str, object]] = []
    for num_topics in LDA_TOPICS:
        started = time.perf_counter()
        try:
            parameters = {**config["lda"], "num_topics": num_topics}
            model = fit_lda(documents, parameters, config["preprocessing"], config["random_seed"])
            coherence = coherence_cv(lda_keywords(model), documents)
            error = ""
        except Exception as exception:
            coherence = float("nan")
            error = str(exception)
        rows.append(
            {
                "num_topics": num_topics,
                "coherence_cv": coherence,
                "runtime_seconds": time.perf_counter() - started,
                "error": error,
            }
        )
    return pd.DataFrame(rows)


def search_nmf(documents: list[str], config: dict) -> pd.DataFrame:
    """Evaluate each NMF topic-count candidate using C_v coherence."""
    rows: list[dict[str, object]] = []
    for num_topics in NMF_TOPICS:
        started = time.perf_counter()
        try:
            parameters = {**config["nmf"], "num_topics": num_topics}
            model = fit_nmf(documents, parameters, config["preprocessing"], config["random_seed"])
            coherence = coherence_cv(nmf_keywords(model), documents)
            error = ""
        except Exception as exception:
            coherence = float("nan")
            error = str(exception)
        rows.append(
            {
                "num_topics": num_topics,
                "coherence_cv": coherence,
                "runtime_seconds": time.perf_counter() - started,
                "error": error,
            }
        )
    return pd.DataFrame(rows)


def search_bertopic(
    embedding_documents: list[str], coherence_documents: list[str], config: dict
) -> pd.DataFrame:
    """Evaluate BERTopic candidates using one shared SPECTER embedding matrix."""
    from bertopic_model import fit_bertopic, noise_statistics, topic_keywords
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer(config["bertopic"]["embedding_model"])
    embeddings = embedding_model.encode(embedding_documents, show_progress_bar=False)

    rows: list[dict[str, object]] = []
    for min_topic_size in BERTOPIC_MIN_TOPIC_SIZES:
        for umap_n_neighbors in BERTOPIC_NEIGHBORS:
            started = time.perf_counter()
            try:
                parameters = {
                    **config["bertopic"],
                    "min_topic_size": min_topic_size,
                    "umap_n_neighbors": umap_n_neighbors,
                }
                model = fit_bertopic(
                    embedding_documents,
                    parameters,
                    config["random_seed"],
                    embedding_model=embedding_model,
                    embeddings=embeddings,
                )
                words = [keywords for _, keywords in topic_keywords(model)]
                coherence = coherence_cv(words, coherence_documents)
                noise_docs, _ = noise_statistics(model)
                actual_topics = len(words)
                error = ""
            except Exception as exception:
                coherence = float("nan")
                actual_topics = float("nan")
                noise_docs = float("nan")
                error = str(exception)
            rows.append(
                {
                    "min_topic_size": min_topic_size,
                    "umap_n_neighbors": umap_n_neighbors,
                    "actual_topics": actual_topics,
                    "noise_docs": noise_docs,
                    "coherence_cv": coherence,
                    "runtime_seconds": time.perf_counter() - started,
                    "error": error,
                }
            )
    return pd.DataFrame(rows)


def rank_results(results: pd.DataFrame) -> pd.DataFrame:
    """Sort candidates by coherence and add a one-based rank column."""
    ranked = results.sort_values("coherence_cv", ascending=False, na_position="last").reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def best_row(results: pd.DataFrame, model_name: str) -> pd.Series:
    """Return the highest-coherence valid candidate or fail with an explicit error."""
    valid = results.dropna(subset=["coherence_cv"])
    if valid.empty:
        raise RuntimeError(f"No valid {model_name} candidates. Inspect its search CSV for errors.")
    return valid.loc[valid["coherence_cv"].idxmax()]


def save_results(
    model_name: str, results: pd.DataFrame, config: dict, output_dir: Path
) -> None:
    """Save one model's search results and update only its best-config values."""
    results.to_csv(output_dir / f"{model_name}_search.csv", index=False)
    config_path = output_dir / "best_config.yaml"
    best_config = load_config(config_path) if config_path.exists() else deepcopy(config)
    best = best_row(results, model_name.upper())
    if model_name == "lda":
        best_config["lda"]["num_topics"] = int(best["num_topics"])
    elif model_name == "nmf":
        best_config["nmf"]["num_topics"] = int(best["num_topics"])
    else:
        best_config["bertopic"]["min_topic_size"] = int(best["min_topic_size"])
        best_config["bertopic"]["umap_n_neighbors"] = int(best["umap_n_neighbors"])
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(best_config, handle, sort_keys=False)


def main(config_path: Path, selected_model: str) -> None:
    """Run only the requested parameter searches and update their outputs."""
    config = load_config(config_path)
    publications = training_publications(config)
    output_dir = PROJECT_ROOT / config["output_dir"] / "parameter_selection"
    output_dir.mkdir(parents=True, exist_ok=True)
    if selected_model == "all":
        with (output_dir / "best_config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(deepcopy(config), handle, sort_keys=False)

    models = ("lda", "nmf", "bertopic") if selected_model == "all" else (selected_model,)
    for model_name in models:
        if model_name == "lda":
            results = search_lda(publications["bow_document"].tolist(), config)
        elif model_name == "nmf":
            results = search_nmf(publications["bow_document"].tolist(), config)
        else:
            results = search_bertopic(
                publications["document"].tolist(), publications["bow_document"].tolist(), config
            )
        save_results(model_name, rank_results(results), config, output_dir)
    print(f"Saved {selected_model} parameter-selection results to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--model", choices=("lda", "nmf", "bertopic", "all"), default="all")
    arguments = parser.parse_args()
    main(arguments.config, arguments.model)
