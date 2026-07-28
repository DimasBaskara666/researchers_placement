"""Run the document-representation comparison experiment."""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bertopic_model import fit_bertopic, noise_statistics, topic_distributions, topic_keywords as bertopic_keywords
from evaluator import evaluate_rankings, rankings_frame, recommendations_frame
from lda_model import fit_lda, topic_keywords as lda_keywords, topic_vectors as lda_vectors
from nmf_model import fit_nmf, topic_keywords as nmf_keywords, topic_vectors as nmf_vectors
from preprocessing import load_datasets
from profile_builder import average_researcher_profiles, build_group_profiles, ground_truth_groups
from recommender import rank_groups
from splitter import publications_for_researchers, split_researchers
from topic_utils import coherence_cv, topics_frame


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ModelRun:
    """Outputs from fitting one representation and ranking testing researchers."""

    rankings: dict[str, list[tuple[str, float]]]
    statistics: dict[str, object]
    topics: pd.DataFrame | None
    runtime: dict[str, float | str]


def load_config(path: Path) -> dict:
    """Load the experiment configuration from YAML."""
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_random_seed(seed: int) -> None:
    """Set random seeds used by Python, NumPy, and Sentence Transformers' PyTorch backend."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def dataset_statistics(
    publications: pd.DataFrame,
    groups: pd.DataFrame,
    train_researchers: set[str],
    test_researchers: set[str],
) -> pd.DataFrame:
    """Summarize the prepared dataset and researcher-level split."""
    publications_per_researcher = publications.groupby("researcher_name").size()
    researchers_per_group = groups.groupby("group_name")["researcher_name"].nunique()
    values = {
        "number_of_publications": publications["document"].nunique(),
        "number_of_researchers": publications_per_researcher.size,
        "number_of_research_groups": groups["group_name"].nunique(),
        "training_researchers": len(train_researchers),
        "testing_researchers": len(test_researchers),
        "average_publications_per_researcher": publications_per_researcher.mean(),
        "minimum_publications_per_researcher": publications_per_researcher.min(),
        "maximum_publications_per_researcher": publications_per_researcher.max(),
        "average_researchers_per_group": researchers_per_group.mean(),
    }
    return pd.DataFrame({"statistic": values.keys(), "value": values.values()})


def coherence_with_note(topic_lists: list[list[str]], documents: list[str]) -> tuple[float, str]:
    """Calculate C_v coherence or record why the requested metric is unavailable."""
    try:
        return coherence_cv(topic_lists, documents), ""
    except (ValueError, ZeroDivisionError) as error:
        return float("nan"), f"C_v coherence unavailable: {error}"


def run_model(
    model_name: str,
    train_publications: pd.DataFrame,
    test_publications: pd.DataFrame,
    groups: pd.DataFrame,
    config: dict,
) -> ModelRun:
    """Fit one representation, create profiles, and rank testing researchers."""
    train_documents = train_publications["document"].tolist()
    test_documents = test_publications["document"].tolist()
    train_bow_documents = train_publications["bow_document"].tolist()
    test_bow_documents = test_publications["bow_document"].tolist()
    started = time.perf_counter()
    topics: pd.DataFrame | None = None
    statistics: dict[str, object] = {"model": model_name}

    if model_name in {"lda", "nmf"}:
        if model_name == "lda":
            model = fit_lda(
                train_bow_documents, config["lda"], config["preprocessing"], config["random_seed"]
            )
            train_vectors = lda_vectors(model, train_bow_documents)
            words = lda_keywords(model)
        else:
            model = fit_nmf(
                train_bow_documents, config["nmf"], config["preprocessing"], config["random_seed"]
            )
            train_vectors = nmf_vectors(model, train_bow_documents)
            words = nmf_keywords(model)
        coherence = coherence_cv(words, train_bow_documents)
        topics = topics_frame(words)
        statistics.update(number_of_topics=len(words), coherence_cv=coherence, coherence_note="")
    elif model_name == "bertopic":
        model = fit_bertopic(train_documents, config["bertopic"], config["random_seed"])
        train_vectors = topic_distributions(model, train_documents)
        topic_items = bertopic_keywords(model)
        words = [keywords for _, keywords in topic_items]
        coherence, note = coherence_with_note(words, train_bow_documents)
        topics = pd.DataFrame(
            {
                "topic_id": [topic_id for topic_id, _ in topic_items],
                "top_keywords": [" | ".join(keywords) for keywords in words],
            }
        )
        noise_count, noise_percentage = noise_statistics(model)
        statistics.update(
            number_of_topics=len(topic_items),
            coherence_cv=coherence,
            coherence_note=note,
            noise_documents=noise_count,
            noise_percentage=noise_percentage,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    train_profiles = average_researcher_profiles(train_publications, train_vectors)
    group_profiles = build_group_profiles(train_profiles, groups)
    training_time = time.perf_counter() - started

    started = time.perf_counter()
    if model_name == "lda":
        test_vectors = lda_vectors(model, test_bow_documents)
    elif model_name == "nmf":
        test_vectors = nmf_vectors(model, test_bow_documents)
    else:
        test_vectors = topic_distributions(model, test_documents)
    test_profiles = average_researcher_profiles(test_publications, test_vectors)
    rankings = rank_groups(test_profiles, group_profiles)
    inference_time = time.perf_counter() - started
    return ModelRun(
        rankings=rankings,
        statistics=statistics,
        topics=topics,
        runtime={
            "model": model_name,
            "training_time_seconds": training_time,
            "recommendation_inference_time_seconds": inference_time,
            "total_execution_time_seconds": training_time + inference_time,
        },
    )


def main(config_path: Path) -> None:
    """Execute all models and save evaluation, topic, and runtime outputs."""
    config = load_config(config_path)
    set_random_seed(config["random_seed"])
    publications, groups = load_datasets(PROJECT_ROOT / "data")
    researchers = sorted(set(publications["researcher_name"]) & set(groups["researcher_name"]))
    train_researchers, test_researchers = split_researchers(
        researchers, config["train_ratio"], config["random_seed"]
    )
    train_publications = publications_for_researchers(publications, train_researchers)
    test_publications = publications_for_researchers(publications, test_researchers)
    truth = ground_truth_groups(groups)

    metrics_rows: list[dict[str, object]] = []
    model_statistics_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, float | str]] = []
    recommendation_tables: list[pd.DataFrame] = []
    ranking_tables: list[pd.DataFrame] = []
    topic_tables: dict[str, pd.DataFrame] = {}
    for model_name in ("lda", "nmf", "bertopic"):
        print(f"Running {model_name}...")
        run = run_model(model_name, train_publications, test_publications, groups, config)
        metrics_rows.append({"model": model_name, **evaluate_rankings(run.rankings, truth)})
        model_statistics_rows.append(run.statistics)
        runtime_rows.append(run.runtime)
        recommendation_tables.append(recommendations_frame(model_name, run.rankings, truth))
        ranking_tables.append(rankings_frame(model_name, run.rankings, truth))
        if run.topics is not None:
            topic_tables[model_name] = run.topics

    output_dir = PROJECT_ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_statistics(publications, groups, train_researchers, test_researchers).to_csv(
        output_dir / "dataset_statistics.csv", index=False
    )
    pd.DataFrame(model_statistics_rows).to_csv(output_dir / "model_statistics.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(output_dir / "runtime_statistics.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(output_dir / "metrics.csv", index=False)
    pd.concat(recommendation_tables, ignore_index=True).to_csv(
        output_dir / "recommendations.csv", index=False
    )
    pd.concat(ranking_tables, ignore_index=True).to_csv(output_dir / "rankings.csv", index=False)
    topics_dir = output_dir / "topics"
    topics_dir.mkdir(parents=True, exist_ok=True)
    for model_name, topic_table in topic_tables.items():
        topic_table.to_csv(topics_dir / f"{model_name}_topics.csv", index=False)
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    arguments = parser.parse_args()
    try:
        main(arguments.config)
    except Exception as error:
        print(f"Experiment failed: {error}", file=sys.stderr)
        raise
