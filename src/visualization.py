"""Visualization module for dataset, topic modeling, evaluation, and system analysis."""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRIC_FAMILIES = {
    "Precision": "precision",
    "Recall": "recall",
    "F1-score": "f1",
    "NDCG": "ndcg",
}
MODEL_ORDER = ["lda", "nmf", "bertopic"]
MODEL_COLOURS = {"lda": "#0072B2", "nmf": "#E69F00", "bertopic": "#009E73"}
MODEL_MARKERS = {"lda": "o", "nmf": "s", "bertopic": "^"}
MODEL_LINESTYLES = {"lda": "-", "nmf": "--", "bertopic": "-."}


def configure_style() -> None:
    """Apply a consistent minimalist Matplotlib style for paper figures."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": "#D6DCE2",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.7,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def display_model(model: str) -> str:
    """Return a publication-ready label for a model identifier."""
    return {"lda": "LDA", "nmf": "NMF", "bertopic": "BERTopic"}.get(
        model.lower(), model
    )


def model_colours(models: list[str]) -> list[str]:
    """Return stable colours for the supplied models."""
    fallback = plt.get_cmap("tab10")
    return [MODEL_COLOURS.get(model.lower(), fallback(index % 10)) for index, model in enumerate(models)]


def available_models(values: pd.Series) -> list[str]:
    """Return the supported models present in a result-table model column."""
    present = set(values.astype(str).str.lower())
    return [model for model in MODEL_ORDER if model in present]


def metric_columns(frame: pd.DataFrame, prefix: str) -> list[tuple[int, str]]:
    """Return columns matching a metric prefix, ordered by cutoff k."""
    pattern = re.compile(rf"^{re.escape(prefix)}@(\d+)$", re.IGNORECASE)
    return sorted(
        (int(match.group(1)), column) for column in frame.columns if (match := pattern.match(column))
    )


def save_figure(figure: plt.Figure, name: str) -> None:
    """Save 300-DPI PNG and vector PDF files to the configured figures directory."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    figure.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


# --- Stage 1: Dataset Figures ---

def plot_dataset_characterization(data_dir: Path = PROJECT_ROOT / "data") -> bool:
    """Plot group sizes, publication counts, and affiliation counts in one figure."""
    groups_path = data_dir / "groups_2015_2025_10k_plus.csv"
    if not groups_path.exists():
        return False

    from preprocessing import load_datasets

    publications, groups = load_datasets(data_dir)
    group_sizes = (
        groups.groupby("group_name")["researcher_name"].nunique().sort_values(ascending=True)
    )
    publication_counts = publications.groupby("researcher_name").size()
    affiliations = (
        groups.groupby("researcher_name")["group_name"].nunique().value_counts().sort_index()
    )

    figure = plt.figure(figsize=(10.2, 5.4))
    layout = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.35, 1.0),
        height_ratios=(1.0, 1.0),
        wspace=0.34,
        hspace=0.42,
    )
    group_axis = figure.add_subplot(layout[:, 0])
    publication_axis = figure.add_subplot(layout[0, 1])
    affiliation_axis = figure.add_subplot(layout[1, 1])

    bars = group_axis.barh(
        group_sizes.index, group_sizes.values, color="#56B4E9", height=0.62
    )
    for bar, count in zip(bars, group_sizes.values, strict=True):
        group_axis.text(
            count + group_sizes.max() * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}",
            ha="left",
            va="center",
            fontsize=7.5,
        )
    group_axis.set_xlim(0, group_sizes.max() * 1.15)
    group_axis.set_xlabel("Researchers")
    group_axis.set_title("(a) Research group membership", loc="left")
    group_axis.grid(axis="x")

    bins = np.arange(0.5, publication_counts.max() + 2.5, 2)
    publication_axis.hist(
        publication_counts,
        bins=bins,
        color="#66C2A5",
        edgecolor="white",
        linewidth=0.5,
    )
    publication_axis.axvline(
        publication_counts.mean(),
        color="#D55E00",
        linestyle="--",
        linewidth=1.4,
        label=f"Mean {publication_counts.mean():.1f}",
    )
    publication_axis.axvline(
        publication_counts.median(),
        color="#4D4D4D",
        linestyle=":",
        linewidth=1.4,
        label=f"Median {publication_counts.median():.1f}",
    )
    publication_axis.set_yscale("log")
    publication_axis.set_xlabel("Publications per researcher")
    publication_axis.set_ylabel("Researchers (log scale)")
    publication_axis.set_title("(b) Publication profile size", loc="left")
    publication_axis.legend(frameon=False, loc="upper right", ncol=2)
    publication_axis.grid(axis="y")

    affiliation_bars = affiliation_axis.bar(
        affiliations.index,
        affiliations.values,
        color="#CC79A7",
        width=0.62,
    )
    total = affiliations.sum()
    for bar, count in zip(affiliation_bars, affiliations.values, strict=True):
        affiliation_axis.text(
            bar.get_x() + bar.get_width() / 2,
            count + affiliations.max() * 0.025,
            f"{count / total:.0%}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    affiliation_axis.set_ylim(0, affiliations.max() * 1.18)
    affiliation_axis.set_xticks(affiliations.index)
    affiliation_axis.set_xlabel("Affiliated research groups")
    affiliation_axis.set_ylabel("Researchers")
    affiliation_axis.set_title("(c) Multi-group affiliation", loc="left")
    affiliation_axis.grid(axis="y")

    figure.subplots_adjust(left=0.22, right=0.98, bottom=0.11, top=0.95)
    save_figure(figure, "dataset_characterization")
    return True


# --- Stage 2 & 3: Parameter Selection & Topic Coherence ---

def plot_hyperparameter_search(param_dir: Path = RESULTS_DIR / "parameter_selection") -> bool:
    """Plot LDA line search, NMF line search, and BERTopic search heatmap."""
    lda_path = param_dir / "lda_search.csv"
    nmf_path = param_dir / "nmf_search.csv"
    bertopic_path = param_dir / "bertopic_search.csv"

    if not (lda_path.exists() and nmf_path.exists() and bertopic_path.exists()):
        return False

    lda_df = pd.read_csv(lda_path)
    nmf_df = pd.read_csv(nmf_path)
    bertopic_df = pd.read_csv(bertopic_path)

    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.35))

    # LDA Subplot
    ax_lda = axes[0]
    lda_sorted = lda_df.dropna(subset=["coherence_cv"]).sort_values("num_topics")
    ax_lda.scatter(
        lda_sorted["num_topics"],
        lda_sorted["coherence_cv"],
        color=MODEL_COLOURS["lda"],
        s=42,
        zorder=3,
    )
    if not lda_sorted.empty:
        best_lda_idx = lda_sorted["coherence_cv"].idxmax()
        best_k = lda_sorted.loc[best_lda_idx, "num_topics"]
        best_c = lda_sorted.loc[best_lda_idx, "coherence_cv"]
        ax_lda.scatter(
            [best_k],
            [best_c],
            s=92,
            facecolors="none",
            edgecolors="#222222",
            linewidths=1.2,
            zorder=4,
        )
    for _, row in lda_sorted.iterrows():
        ax_lda.annotate(
            f"{row['coherence_cv']:.3f}",
            (row["num_topics"], row["coherence_cv"]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )
    ax_lda.set_title("(a) LDA")
    ax_lda.set_xlabel("Number of Topics ($k$)")
    ax_lda.set_ylabel("Topic Coherence ($C_v$)")
    ax_lda.grid(axis="y")

    # NMF Subplot
    ax_nmf = axes[1]
    nmf_sorted = nmf_df.dropna(subset=["coherence_cv"]).sort_values("num_topics")
    ax_nmf.scatter(
        nmf_sorted["num_topics"],
        nmf_sorted["coherence_cv"],
        color=MODEL_COLOURS["nmf"],
        s=42,
        zorder=3,
    )
    if not nmf_sorted.empty:
        best_nmf_idx = nmf_sorted["coherence_cv"].idxmax()
        best_k = nmf_sorted.loc[best_nmf_idx, "num_topics"]
        best_c = nmf_sorted.loc[best_nmf_idx, "coherence_cv"]
        ax_nmf.scatter(
            [best_k],
            [best_c],
            s=92,
            facecolors="none",
            edgecolors="#222222",
            linewidths=1.2,
            zorder=4,
        )
    for _, row in nmf_sorted.iterrows():
        ax_nmf.annotate(
            f"{row['coherence_cv']:.3f}",
            (row["num_topics"], row["coherence_cv"]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )
    ax_nmf.set_title("(b) NMF")
    ax_nmf.set_xlabel("Number of Topics ($k$)")
    ax_nmf.set_ylabel("Topic Coherence ($C_v$)")
    ax_nmf.grid(axis="y")

    # BERTopic Subplot (Heatmap)
    ax_bert = axes[2]
    pivot = bertopic_df.pivot(
        index="min_topic_size", columns="umap_n_neighbors", values="coherence_cv"
    ).sort_index(ascending=True)

    min_sizes = pivot.index.tolist()
    neighbors = pivot.columns.tolist()
    matrix = pivot.to_numpy()

    colour_map = plt.get_cmap("YlGnBu").copy()
    colour_map.set_bad("#EEF1F4")
    ax_bert.imshow(matrix, cmap=colour_map, aspect="auto")

    best_val = -1.0
    best_pos = (-1, -1)
    for i in range(len(min_sizes)):
        for j in range(len(neighbors)):
            val = matrix[i, j]
            if pd.isna(val):
                ax_bert.text(
                    j, i, "Failed", ha="center", va="center", fontsize=7, color="#687078"
                )
            else:
                if val > best_val:
                    best_val = val
                    best_pos = (i, j)
                ax_bert.text(
                    j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color="#222222"
                )

    if best_pos != (-1, -1):
        bi, bj = best_pos
        rect = plt.Rectangle(
            (bj - 0.45, bi - 0.45),
            0.9,
            0.9,
            fill=False,
            edgecolor="#D55E00",
            linewidth=2,
            linestyle="--",
        )
        ax_bert.add_patch(rect)

    ax_bert.set_xticks(np.arange(len(neighbors)))
    ax_bert.set_xticklabels(neighbors)
    ax_bert.set_yticks(np.arange(len(min_sizes)))
    ax_bert.set_yticklabels(min_sizes)
    ax_bert.set_xlabel("UMAP Neighbors")
    ax_bert.set_ylabel("Min Topic Size")
    ax_bert.set_title("(c) BERTopic")
    ax_bert.grid(False)

    figure.subplots_adjust(left=0.07, right=0.99, bottom=0.18, top=0.90, wspace=0.30)
    save_figure(figure, "hyperparameter_search")
    return True


def plot_topic_coherence_comparison(
    model_stats_path: Path = RESULTS_DIR / "model_statistics.csv",
    param_dir: Path = RESULTS_DIR / "parameter_selection",
) -> bool:
    """Plot a grouped bar chart comparing the final Topic Coherence of LDA, NMF, and BERTopic."""
    coherence_data: dict[str, float] = {}

    if model_stats_path.exists():
        df = pd.read_csv(model_stats_path)
        for _, row in df.iterrows():
            model = str(row.get("model", "")).lower()
            if model in {"lda", "nmf", "bertopic"}:
                val = row.get("coherence_cv")
                if pd.notna(val):
                    coherence_data[model] = float(val)

    models_needed = {"lda", "nmf", "bertopic"} - set(coherence_data.keys())
    for model in models_needed:
        path = param_dir / f"{model}_search.csv"
        if path.exists():
            df = pd.read_csv(path)
            valid = df.dropna(subset=["coherence_cv"])
            if not valid.empty:
                coherence_data[model] = float(valid["coherence_cv"].max())

    if not coherence_data:
        return False

    models = [m for m in ["lda", "nmf", "bertopic"] if m in coherence_data]
    scores = [coherence_data[m] for m in models]
    labels = [display_model(m) for m in models]
    colors = model_colours(models)

    positions = np.arange(len(models))
    figure, axis = plt.subplots(figsize=(5.2, 2.7))
    axis.hlines(positions, 0, scores, color="#D8DEE5", linewidth=1.2, zorder=1)
    axis.scatter(scores, positions, color=colors, s=72, zorder=3)

    for position, score in zip(positions, scores, strict=True):
        axis.text(
            score + 0.018,
            position,
            f"{score:.3f}",
            ha="left",
            va="center",
            fontsize=8,
            fontweight="semibold",
        )

    axis.set_yticks(positions, labels)
    axis.set_xlim(0, 1)
    axis.set_xlabel("Topic Coherence ($C_v$)")
    axis.grid(axis="x")

    figure.tight_layout()
    save_figure(figure, "topic_coherence_comparison")
    return True


# --- Stage 4 & 5: Recommendation & Evaluation Figures ---

def plot_recommendation_performance(metrics: pd.DataFrame) -> bool:
    """Plot Precision, Recall, F1, and NDCG curves across available cutoffs."""
    metrics = metrics.copy()
    metrics["model"] = metrics["model"].astype(str).str.lower()
    metrics = metrics[metrics["model"].isin(MODEL_ORDER)].set_index("model")
    models = [model for model in MODEL_ORDER if model in metrics.index]
    if not models:
        return False

    families = [(label, metric_columns(metrics, prefix)) for label, prefix in METRIC_FAMILIES.items()]
    families = [(label, columns) for label, columns in families if columns]
    if not families:
        return False

    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True, sharey=True)
    for axis, (label, columns) in zip(axes.ravel(), families, strict=True):
        cutoffs = [cutoff for cutoff, _ in columns]
        metric_names = [column for _, column in columns]
        for model in models:
            values = pd.to_numeric(metrics.loc[model, metric_names]).to_numpy(dtype=float)
            axis.plot(
                cutoffs,
                values,
                color=MODEL_COLOURS[model],
                marker=MODEL_MARKERS[model],
                linestyle=MODEL_LINESTYLES[model],
                label=display_model(model),
                markerfacecolor="white",
                markeredgewidth=1.2,
            )
        axis.set_title(label)
        axis.set_xticks(cutoffs)
        axis.set_ylim(0, 1)
        axis.grid(axis="y")
    axes[1, 0].set_xlabel("Cutoff $K$")
    axes[1, 1].set_xlabel("Cutoff $K$")
    axes[0, 0].set_ylabel("Score")
    axes[1, 0].set_ylabel("Score")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(models),
    )
    figure.subplots_adjust(left=0.09, right=0.99, bottom=0.10, top=0.88, wspace=0.16, hspace=0.28)
    save_figure(figure, "recommendation_performance")
    return True


def plot_per_group_performance(rankings_path: Path = RESULTS_DIR / "rankings.csv") -> bool:
    """Plot top-three recommendation precision per group and model as a heatmap."""
    if not rankings_path.exists():
        return False

    df = pd.read_csv(rankings_path)
    df["model"] = df["model"].astype(str).str.lower()
    df = df[df["model"].isin(MODEL_ORDER)]
    if df.empty:
        return False

    top3 = df[df["rank"] <= 3].copy()
    grouped = top3.groupby(["group_name", "model"])["is_relevant"].mean().unstack("model")
    models = [model for model in MODEL_ORDER if model in grouped.columns]
    grouped = grouped.reindex(columns=models)

    support = (
        df[df["is_relevant"] == True]  # noqa: E712
        .drop_duplicates(["researcher_name", "group_name"])
        .groupby("group_name")["researcher_name"]
        .nunique()
    )
    group_order = support.reindex(grouped.index).fillna(0).sort_values(ascending=False).index
    grouped = grouped.reindex(group_order)
    support = support.reindex(group_order).fillna(0).astype(int)
    row_labels = [f"{group}  ($n$={support[group]:,})" for group in grouped.index]

    matrix = grouped.to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    image = axis.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            if pd.isna(value):
                label = "—"
                color = "#6B7280"
            else:
                label = f"{value:.2f}"
                color = "white" if value >= 0.58 else "#1F2933"
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=7.5,
                color=color,
            )

    axis.set_xticks(np.arange(len(models)), [display_model(model) for model in models])
    axis.set_yticks(np.arange(len(grouped)), row_labels)
    axis.tick_params(length=0)
    axis.grid(False)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.025)
    colorbar.set_label("Top-3 precision")
    colorbar.outline.set_visible(False)
    figure.subplots_adjust(left=0.42, right=0.92, bottom=0.09, top=0.98)
    save_figure(figure, "per_group_performance")
    return True


def plot_rank_distribution(rankings_path: Path = RESULTS_DIR / "rankings.csv") -> bool:
    """Plot Cumulative Relevant Group Rank Distribution (CDF) across models."""
    if not rankings_path.exists():
        return False

    df = pd.read_csv(rankings_path)
    df["model"] = df["model"].astype(str).str.lower()
    df = df[df["model"].isin(MODEL_ORDER)]
    relevant = df[df["is_relevant"] == True]  # noqa: E712

    models = available_models(df["model"])

    figure, axis = plt.subplots(figsize=(5.8, 3.6))

    for model in models:
        m_rel = relevant[relevant["model"] == model]
        total_relevant = len(m_rel)
        if total_relevant == 0:
            continue

        counts = [(m_rel["rank"] <= k).sum() / total_relevant for k in range(1, 13)]
        axis.plot(
            range(1, 13),
            counts,
            marker=MODEL_MARKERS[model],
            color=MODEL_COLOURS[model],
            linestyle=MODEL_LINESTYLES[model],
            markerfacecolor="white",
            markeredgewidth=1.1,
            label=display_model(model),
        )

    axis.set_xticks(range(1, 13))
    axis.set_xlabel("Rank Cutoff ($K$)")
    axis.set_ylabel("Cumulative Relevant Group Fraction")
    axis.set_ylim(0, 1.02)
    axis.grid(axis="y")
    axis.legend(frameon=False, loc="lower right")

    figure.tight_layout()
    save_figure(figure, "rank_distribution")
    return True


# --- Stage 6: Computational Figures ---

def plot_computational_runtime(runtime_path: Path = RESULTS_DIR / "runtime_statistics.csv") -> bool:
    """Plot separate training and inference runtimes on a logarithmic scale."""
    if not runtime_path.exists():
        return False

    df = pd.read_csv(runtime_path)
    df["model"] = df["model"].astype(str).str.lower()
    df = df[df["model"].isin(MODEL_ORDER)].set_index("model")
    models = [model for model in MODEL_ORDER if model in df.index]
    if not models:
        return False
    df = df.reindex(models)

    train_times = df["training_time_seconds"].to_numpy(dtype=float)
    inference_times = df["recommendation_inference_time_seconds"].to_numpy(dtype=float)
    positions = np.arange(len(models))
    model_colours = [MODEL_COLOURS[model] for model in models]
    height = 0.30

    figure, axis = plt.subplots(figsize=(5.8, 3.0))
    training_bars = axis.barh(
        positions + height / 2,
        train_times,
        height=height,
        color=model_colours,
    )
    inference_bars = axis.barh(
        positions - height / 2,
        inference_times,
        height=height,
        facecolor="white",
        edgecolor=model_colours,
        hatch="///",
        linewidth=1.2,
    )

    for bars, values in ((training_bars, train_times), (inference_bars, inference_times)):
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                value * 1.08,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f} s",
                ha="left",
                va="center",
                fontsize=7.5,
            )

    axis.set_xscale("log")
    axis.set_yticks(positions, [display_model(model) for model in models])
    axis.set_xlabel("Execution Time (Seconds, Log Scale)")
    axis.grid(axis="x", which="major")
    axis.legend(
        handles=[
            Patch(facecolor="#777777", label="Training"),
            Patch(facecolor="white", edgecolor="#777777", hatch="///", label="Inference"),
        ],
        frameon=False,
        loc="lower right",
    )
    upper_limit = max(train_times.max(), inference_times.max()) * 2.2
    lower_limit = min(train_times.min(), inference_times.min()) / 1.8
    axis.set_xlim(lower_limit, upper_limit)

    figure.tight_layout()
    save_figure(figure, "computational_runtime")
    return True


def main() -> None:
    """Generate all figures supported by experiment metrics and parameter selection results."""
    configure_style()
    data_dir = PROJECT_ROOT / "data"
    metrics_path = RESULTS_DIR / "metrics.csv"
    model_stats_path = RESULTS_DIR / "model_statistics.csv"
    runtime_path = RESULTS_DIR / "runtime_statistics.csv"
    rankings_path = RESULTS_DIR / "rankings.csv"
    param_dir = RESULTS_DIR / "parameter_selection"

    generated: list[str] = []

    # Dataset figure
    if plot_dataset_characterization(data_dir):
        generated.append("dataset_characterization")

    # Hyperparameter & Topic figures
    if plot_hyperparameter_search(param_dir):
        generated.append("hyperparameter_search")
    if plot_topic_coherence_comparison(model_stats_path, param_dir):
        generated.append("topic_coherence_comparison")

    # Performance & Recommendation figures
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        if "model" in metrics:
            if plot_recommendation_performance(metrics):
                generated.append("recommendation_performance")

    if plot_per_group_performance(rankings_path):
        generated.append("per_group_performance")
    if plot_rank_distribution(rankings_path):
        generated.append("rank_distribution")

    # Runtime figure
    if plot_computational_runtime(runtime_path):
        generated.append("computational_runtime")

    if not generated:
        raise ValueError("No figures could be generated. Check result CSV files.")

    print(f"Generated figures: {', '.join(generated)}")


if __name__ == "__main__":
    main()
