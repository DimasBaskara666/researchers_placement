"""Visualization module for dataset, topic modeling, evaluation, and system analysis."""

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
    """Save 300-DPI PNG and vector PDF files to the configured figures directory."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    figure.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


# --- Stage 1: Dataset Figures ---

def plot_dataset_group_size_distribution(data_dir: Path = PROJECT_ROOT / "data") -> bool:
    """Plot horizontal bar chart of researcher counts per research group."""
    groups_path = data_dir / "groups_2015_2025_10k_plus.csv"
    if not groups_path.exists():
        return False

    df = pd.read_csv(groups_path)
    counts = df.groupby("group_name")["researcher_name"].nunique().sort_values(ascending=True)

    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    bars = axis.barh(counts.index, counts.values, color="#3B82B6", height=0.65)

    for bar, count in zip(bars, counts.values, strict=True):
        axis.text(
            count + max(counts.values) * 0.01,
            bar.get_y() + bar.get_height() / 2.0,
            f"{count:,}",
            ha="left",
            va="center",
            fontsize=8,
        )

    axis.set_xlabel("Number of Researchers")
    axis.set_title("Research Group Member Count")
    axis.grid(axis="y", visible=False)
    axis.set_xlim(0, max(counts.values) * 1.12)

    figure.tight_layout()
    save_figure(figure, "dataset_group_size_distribution")
    return True


def plot_dataset_publications_per_researcher(data_dir: Path = PROJECT_ROOT / "data") -> bool:
    """Plot histogram of publication counts per researcher."""
    pub_path = data_dir / "publications_2015_2025_10k_plus.csv"
    if not pub_path.exists():
        return False

    try:
        from preprocessing import load_datasets
        publications, _ = load_datasets(data_dir)
        counts = publications.groupby("researcher_name").size()
    except Exception:
        df = pd.read_csv(pub_path)
        if "Authors" in df.columns:
            authors_series = df["Authors"].dropna().str.split(";").explode().str.strip()
            counts = authors_series[authors_series != ""].value_counts()
        else:
            return False

    mean_val = counts.mean()
    median_val = counts.median()

    figure, axis = plt.subplots(figsize=(5.8, 3.8))
    axis.hist(counts, bins=30, color="#4E9A7A", edgecolor="white", alpha=0.85)

    axis.axvline(mean_val, color="#D2695E", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.1f}")
    axis.axvline(median_val, color="#333333", linestyle=":", linewidth=1.5, label=f"Median: {median_val:.1f}")

    axis.set_xlabel("Number of Publications per Researcher")
    axis.set_ylabel("Researcher Count")
    axis.set_title("Publications per Researcher Distribution")
    axis.legend(frameon=False, loc="upper right")
    axis.grid(axis="x", visible=False)

    figure.tight_layout()
    save_figure(figure, "dataset_publications_per_researcher")
    return True


def plot_dataset_groups_per_researcher(data_dir: Path = PROJECT_ROOT / "data") -> bool:
    """Plot bar chart of research group affiliations per researcher."""
    groups_path = data_dir / "groups_2015_2025_10k_plus.csv"
    if not groups_path.exists():
        return False

    df = pd.read_csv(groups_path)
    affiliations = df.groupby("researcher_name")["group_name"].nunique().value_counts().sort_index()

    figure, axis = plt.subplots(figsize=(5.4, 3.8))
    bars = axis.bar(affiliations.index.astype(str), affiliations.values, color="#9A6FB0", width=0.5)

    total = affiliations.sum()
    for bar, count in zip(bars, affiliations.values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            count + max(affiliations.values) * 0.015,
            f"{count:,}\n({count/total*100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    axis.set_xlabel("Number of Affiliated Research Groups")
    axis.set_ylabel("Researcher Count")
    axis.set_title("Multi-Group Affiliation Distribution")
    axis.set_ylim(0, max(affiliations.values) * 1.18)
    axis.grid(axis="x", visible=False)

    figure.tight_layout()
    save_figure(figure, "dataset_groups_per_researcher")
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

    figure, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))

    # LDA Subplot
    ax_lda = axes[0]
    lda_sorted = lda_df.dropna(subset=["coherence_cv"]).sort_values("num_topics")
    ax_lda.plot(
        lda_sorted["num_topics"],
        lda_sorted["coherence_cv"],
        marker="o",
        color=MODEL_COLOURS["lda"],
        linewidth=1.8,
        markersize=6,
    )
    if not lda_sorted.empty:
        best_lda_idx = lda_sorted["coherence_cv"].idxmax()
        best_k = lda_sorted.loc[best_lda_idx, "num_topics"]
        best_c = lda_sorted.loc[best_lda_idx, "coherence_cv"]
        ax_lda.annotate(
            f"Best: k={int(best_k)}\n({best_c:.3f})",
            xy=(best_k, best_c),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", lw=0.8, color="#333333"),
        )
    ax_lda.set_title("LDA Parameter Search")
    ax_lda.set_xlabel("Number of Topics ($k$)")
    ax_lda.set_ylabel("Topic Coherence ($C_v$)")

    # NMF Subplot
    ax_nmf = axes[1]
    nmf_sorted = nmf_df.dropna(subset=["coherence_cv"]).sort_values("num_topics")
    ax_nmf.plot(
        nmf_sorted["num_topics"],
        nmf_sorted["coherence_cv"],
        marker="o",
        color=MODEL_COLOURS["nmf"],
        linewidth=1.8,
        markersize=6,
    )
    if not nmf_sorted.empty:
        best_nmf_idx = nmf_sorted["coherence_cv"].idxmax()
        best_k = nmf_sorted.loc[best_nmf_idx, "num_topics"]
        best_c = nmf_sorted.loc[best_nmf_idx, "coherence_cv"]
        ax_nmf.annotate(
            f"Best: k={int(best_k)}\n({best_c:.3f})",
            xy=(best_k, best_c),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", lw=0.8, color="#333333"),
        )
    ax_nmf.set_title("NMF Parameter Search")
    ax_nmf.set_xlabel("Number of Topics ($k$)")
    ax_nmf.set_ylabel("Topic Coherence ($C_v$)")

    # BERTopic Subplot (Heatmap)
    ax_bert = axes[2]
    pivot = bertopic_df.pivot(
        index="min_topic_size", columns="umap_n_neighbors", values="coherence_cv"
    ).sort_index(ascending=True)

    min_sizes = pivot.index.tolist()
    neighbors = pivot.columns.tolist()
    matrix = pivot.to_numpy()

    ax_bert.imshow(matrix, cmap="YlGnBu", aspect="auto")

    best_val = -1.0
    best_pos = (-1, -1)
    for i in range(len(min_sizes)):
        for j in range(len(neighbors)):
            val = matrix[i, j]
            if pd.isna(val):
                ax_bert.text(
                    j, i, "N/A", ha="center", va="center", fontsize=8, color="#777777"
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
            edgecolor=MODEL_COLOURS["bertopic"],
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
    ax_bert.set_title("BERTopic Parameter Search ($C_v$)")
    ax_bert.grid(False)

    figure.tight_layout()
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

    figure, axis = plt.subplots(figsize=(5.4, 3.8))
    bars = axis.bar(labels, scores, color=colors, width=0.55)

    for bar, score in zip(bars, scores, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            score + 0.015,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="semibold",
        )

    axis.set_ylabel("Topic Coherence ($C_v$)")
    axis.set_title("Topic Coherence Comparison")
    axis.set_ylim(0, max(scores) * 1.15)
    axis.grid(axis="x", visible=False)

    figure.tight_layout()
    save_figure(figure, "topic_coherence_comparison")
    return True


# --- Stage 4 & 5: Recommendation & Evaluation Figures ---

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


def plot_per_group_performance(rankings_path: Path = RESULTS_DIR / "rankings.csv") -> bool:
    """Plot Precision@3 per research group across models."""
    if not rankings_path.exists():
        return False

    df = pd.read_csv(rankings_path)
    top3 = df[df["rank"] <= 3].copy()
    grouped = top3.groupby(["group_name", "model"])["is_relevant"].mean().unstack("model")

    models = [m for m in ["tfidf", "lda", "nmf", "bertopic"] if m in grouped.columns]
    grouped = grouped[models]

    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    positions = np.arange(len(grouped))
    height = 0.75 / len(models)

    for idx, (model, color) in enumerate(zip(models, model_colours(models), strict=True)):
        scores = grouped[model].values
        axis.barh(
            positions + (idx - (len(models) - 1) / 2) * height,
            scores,
            height=height,
            color=color,
            label=display_model(model),
        )

    axis.set_yticks(positions)
    axis.set_yticklabels(grouped.index)
    axis.set_xlabel("Precision@3 Score")
    axis.set_title("Per-Group Recommendation Precision@3")
    axis.set_xlim(0, 1.0)
    axis.grid(axis="y", visible=False)
    axis.legend(frameon=False, loc="lower right")

    figure.tight_layout()
    save_figure(figure, "per_group_performance")
    return True


def plot_rank_distribution(rankings_path: Path = RESULTS_DIR / "rankings.csv") -> bool:
    """Plot Cumulative Relevant Group Rank Distribution (CDF) across models."""
    if not rankings_path.exists():
        return False

    df = pd.read_csv(rankings_path)
    relevant = df[df["is_relevant"] == True]  # noqa: E712

    models = [m for m in ["tfidf", "lda", "nmf", "bertopic"] if m in df["model"].unique()]

    figure, axis = plt.subplots(figsize=(6.0, 4.0))
    markers = ["o", "s", "^", "D"]

    for model, color, marker in zip(models, model_colours(models), markers, strict=True):
        m_rel = relevant[relevant["model"] == model]
        total_relevant = len(m_rel)
        if total_relevant == 0:
            continue

        counts = [(m_rel["rank"] <= k).sum() / total_relevant for k in range(1, 13)]
        axis.plot(
            range(1, 13),
            counts,
            marker=marker,
            color=color,
            linewidth=1.8,
            markersize=5,
            label=display_model(model),
        )

    axis.set_xticks(range(1, 13))
    axis.set_xlabel("Rank Cutoff ($k$)")
    axis.set_ylabel("Cumulative Relevant Group Fraction")
    axis.set_title("Relevant Group Cumulative Rank Distribution")
    axis.set_ylim(0, 1.02)
    axis.legend(frameon=False, loc="lower right")

    figure.tight_layout()
    save_figure(figure, "rank_distribution")
    return True


def plot_similarity_distribution(rankings_path: Path = RESULTS_DIR / "rankings.csv") -> bool:
    """Plot similarity score distribution for relevant vs. non-relevant group pairs."""
    if not rankings_path.exists():
        return False

    df = pd.read_csv(rankings_path)
    models = [m for m in ["tfidf", "lda", "nmf", "bertopic"] if m in df["model"].unique()]

    figure, axis = plt.subplots(figsize=(7.5, 4.0))

    positions = np.arange(len(models))
    width = 0.35

    for idx, model in enumerate(models):
        m_df = df[df["model"] == model]
        rel_sims = m_df[m_df["is_relevant"] == True]["similarity"].dropna()  # noqa: E712
        non_rel_sims = m_df[m_df["is_relevant"] == False]["similarity"].dropna()  # noqa: E712

        axis.boxplot(
            rel_sims,
            positions=[positions[idx] - width / 2.0],
            widths=width * 0.8,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor="#4E9A7A", alpha=0.7, edgecolor="#222222"),
            medianprops=dict(color="#222222", linewidth=1.5),
        )
        axis.boxplot(
            non_rel_sims,
            positions=[positions[idx] + width / 2.0],
            widths=width * 0.8,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor="#D9D9D9", alpha=0.7, edgecolor="#666666"),
            medianprops=dict(color="#444444", linewidth=1.5),
        )

    # Custom legend
    axis.plot([], [], color="#4E9A7A", marker="s", linestyle="None", label="Relevant Pairs")
    axis.plot([], [], color="#D9D9D9", marker="s", linestyle="None", label="Non-Relevant Pairs")


    axis.set_xticks(positions)
    axis.set_xticklabels([display_model(m) for m in models])
    axis.set_ylabel("Cosine Similarity Score")
    axis.set_title("Cosine Similarity Score Distribution")
    axis.legend(frameon=False, loc="upper right")
    axis.grid(axis="x", visible=False)

    figure.tight_layout()
    save_figure(figure, "similarity_distribution")
    return True


# --- Stage 6 & 7: Computational & Discussion Figures ---

def plot_computational_runtime(runtime_path: Path = RESULTS_DIR / "runtime_statistics.csv") -> bool:
    """Plot computational runtime comparison across training and inference stages."""
    if not runtime_path.exists():
        return False

    df = pd.read_csv(runtime_path)
    models = [m for m in ["tfidf", "lda", "nmf", "bertopic"] if m in df["model"].str.lower().tolist()]

    df["model_lower"] = df["model"].str.lower()
    df = df.set_index("model_lower").reindex(models)

    train_times = df["training_time_seconds"].values
    inf_times = df["recommendation_inference_time_seconds"].values
    labels = [display_model(m) for m in models]

    figure, axis = plt.subplots(figsize=(6.0, 4.0))

    bars_train = axis.bar(labels, train_times, color="#3B82B6", width=0.5, label="Training Time")
    bars_inf = axis.bar(labels, inf_times, bottom=train_times, color="#D2695E", width=0.5, label="Inference Time")

    for bar_t, bar_i, t_val, i_val in zip(bars_train, bars_inf, train_times, inf_times, strict=True):
        total = t_val + i_val
        axis.text(
            bar_t.get_x() + bar_t.get_width() / 2.0,
            total + max(train_times + inf_times) * 0.02,
            f"{total:.1f}s",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="semibold",
        )

    axis.set_ylabel("Execution Time (Seconds)")
    axis.set_title("Computational Runtime Comparison")
    axis.set_ylim(0, max(train_times + inf_times) * 1.15)
    axis.grid(axis="x", visible=False)
    axis.legend(frameon=False, loc="upper left")

    figure.tight_layout()
    save_figure(figure, "computational_runtime")
    return True


def plot_coherence_vs_performance(
    model_stats_path: Path = RESULTS_DIR / "model_statistics.csv",
    metrics_path: Path = RESULTS_DIR / "metrics.csv",
) -> bool:
    """Plot relationship between topic coherence (C_v) and downstream ranking quality (NDCG@3)."""
    if not (model_stats_path.exists() and metrics_path.exists()):
        return False

    m_stats = pd.read_csv(model_stats_path)
    metrics = pd.read_csv(metrics_path)

    merged = pd.merge(m_stats, metrics, on="model").dropna(subset=["coherence_cv", "ndcg@3"])
    if merged.empty:
        return False

    figure, axis = plt.subplots(figsize=(5.8, 4.0))

    for _, row in merged.iterrows():
        model = str(row["model"])
        c_v = float(row["coherence_cv"])
        ndcg = float(row["ndcg@3"])
        color = MODEL_COLOURS.get(model.lower(), "#333333")

        axis.scatter(c_v, ndcg, color=color, s=90, label=display_model(model), zorder=3)
        axis.annotate(
            display_model(model),
            xy=(c_v, ndcg),
            xytext=(6, -2),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )

    axis.set_xlabel("Topic Coherence ($C_v$)")
    axis.set_ylabel("Ranking Quality (NDCG@3)")
    axis.set_title("Topic Coherence vs. Downstream Performance")
    axis.grid(True)

    figure.tight_layout()
    save_figure(figure, "coherence_vs_performance")
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

    # Dataset figures
    if plot_dataset_group_size_distribution(data_dir):
        generated.append("dataset_group_size_distribution")
    if plot_dataset_publications_per_researcher(data_dir):
        generated.append("dataset_publications_per_researcher")
    if plot_dataset_groups_per_researcher(data_dir):
        generated.append("dataset_groups_per_researcher")

    # Hyperparameter & Topic figures
    if plot_hyperparameter_search(param_dir):
        generated.append("hyperparameter_search")
    if plot_topic_coherence_comparison(model_stats_path, param_dir):
        generated.append("topic_coherence_comparison")

    # Performance & Recommendation figures
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        if "model" in metrics:
            if plot_retrieval_performance(metrics):
                generated.append("retrieval_performance")
            if plot_ranking_quality(metrics):
                generated.append("ranking_quality")

    if plot_per_group_performance(rankings_path):
        generated.append("per_group_performance")
    if plot_rank_distribution(rankings_path):
        generated.append("rank_distribution")
    if plot_similarity_distribution(rankings_path):
        generated.append("similarity_distribution")

    # Runtime & Discussion figures
    if plot_computational_runtime(runtime_path):
        generated.append("computational_runtime")
    if plot_coherence_vs_performance(model_stats_path, metrics_path):
        generated.append("coherence_vs_performance")

    if not generated:
        raise ValueError("No figures could be generated. Check result CSV files.")

    print(f"Generated figures: {', '.join(generated)}")


if __name__ == "__main__":
    main()
