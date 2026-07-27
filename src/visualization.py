"""Create the two paper figures supported by experiment metrics."""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRIC_FAMILIES = {"Precision": "precision", "Recall": "recall", "F1-score": "f1"}
MODEL_COLOURS = {"tfidf": "#3B82B6", "lda": "#9A6FB0", "nmf": "#4E9A7A", "bertopic": "#D2695E"}


def configure_style() -> None:
    """Apply a consistent minimalist Matplotlib style for paper figures."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def display_model(model: str) -> str:
    """Return a publication-ready label for a model identifier."""
    return {"tfidf": "TF-IDF", "lda": "LDA", "nmf": "NMF", "bertopic": "BERTopic"}.get(
        model.lower(), model
    )


def model_colours(models: list[str]) -> list[str]:
    """Return stable colours for the supplied models."""
    fallback = plt.get_cmap("tab10")
    return [MODEL_COLOURS.get(model.lower(), fallback(index % 10)) for index, model in enumerate(models)]


def metric_columns(frame: pd.DataFrame, prefix: str) -> list[tuple[int, str]]:
    """Return columns matching a metric prefix, ordered by cutoff k."""
    pattern = re.compile(rf"^{re.escape(prefix)}@(\d+)$", re.IGNORECASE)
    return sorted(
        (int(match.group(1)), column) for column in frame.columns if (match := pattern.match(column))
    )


def save_figure(figure: plt.Figure, name: str) -> None:
    """Save one 300-DPI PNG to the configured figures directory."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_retrieval_performance(metrics: pd.DataFrame) -> bool:
    """Plot available Precision, Recall, and F1 comparisons across cutoffs."""
    families = [(label, metric_columns(metrics, prefix)) for label, prefix in METRIC_FAMILIES.items()]
    families = [(label, columns) for label, columns in families if columns]
    if not families:
        return False
    models = metrics["model"].astype(str).tolist()
    figure, axes = plt.subplots(1, len(families), figsize=(4.0 * len(families), 3.6), squeeze=False)
    for axis, (label, columns) in zip(axes[0], families, strict=True):
        positions = np.arange(len(columns))
        width = 0.76 / len(models)
        for index, (model, colour) in enumerate(zip(models, model_colours(models), strict=True)):
            values = pd.to_numeric(metrics.loc[metrics.index[index], [column for _, column in columns]])
            axis.bar(
                positions + (index - (len(models) - 1) / 2) * width,
                values,
                width=width,
                color=colour,
                label=display_model(model),
            )
        axis.set_title(label)
        axis.set_xticks(positions, [f"@{cutoff}" for cutoff, _ in columns])
        axis.set_ylim(0, 1)
        axis.set_ylabel("Score")
        axis.grid(axis="x", visible=False)
    axes[0][0].legend(frameon=False, loc="lower left")
    figure.suptitle("Retrieval Performance by Cutoff", y=1.02, fontsize=13, fontweight="semibold")
    figure.tight_layout()
    save_figure(figure, "retrieval_performance")
    return True


def plot_ranking_quality(metrics: pd.DataFrame) -> bool:
    """Plot MRR and available NDCG values for model comparison."""
    columns: list[tuple[str, str]] = []
    mrr = next((column for column in metrics if column.lower() == "mrr"), None)
    if mrr:
        columns.append(("MRR", mrr))
    columns.extend((f"NDCG@{cutoff}", column) for cutoff, column in metric_columns(metrics, "ndcg"))
    if not columns:
        return False
    models = metrics["model"].astype(str).tolist()
    positions = np.arange(len(columns))
    width = 0.76 / len(models)
    figure, axis = plt.subplots(figsize=(5.4, 3.8))
    for index, (model, colour) in enumerate(zip(models, model_colours(models), strict=True)):
        values = pd.to_numeric(metrics.loc[metrics.index[index], [column for _, column in columns]])
        axis.bar(
            positions + (index - (len(models) - 1) / 2) * width,
            values,
            width=width,
            color=colour,
            label=display_model(model),
        )
    axis.set_xticks(positions, [label for label, _ in columns])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Ranking Quality")
    axis.grid(axis="x", visible=False)
    axis.legend(frameon=False, loc="lower left")
    figure.tight_layout()
    save_figure(figure, "ranking_quality")
    return True


def main() -> None:
    """Generate only the performance figures supported by metrics.csv."""
    metrics_path = RESULTS_DIR / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing required result file: {metrics_path}")
    metrics = pd.read_csv(metrics_path)
    if "model" not in metrics:
        raise ValueError("metrics.csv must contain a 'model' column.")
    configure_style()
    generated = [
        name
        for name, created in (
            ("retrieval_performance", plot_retrieval_performance(metrics)),
            ("ranking_quality", plot_ranking_quality(metrics)),
        )
        if created
    ]
    if not generated:
        raise ValueError("metrics.csv does not contain supported retrieval or ranking metrics.")
    print(f"Generated figures: {', '.join(generated)}")


if __name__ == "__main__":
    main()
