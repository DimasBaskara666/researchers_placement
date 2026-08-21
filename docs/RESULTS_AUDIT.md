# Results Audit

## Audit basis and reporting conventions

This audit reports the checked-in outputs at repository commit `aaf2fad`. No model, parameter search, data-acquisition process, experiment, or figure-generation command was rerun.

Unless otherwise stated:

- values are read from files under `results/`;
- displayed values are rounded to six decimal places while source CSVs retain higher precision;
- rankings use higher-is-better ordering for recommendation and coherence metrics;
- runtime ordering uses lower-is-shorter;
- no causal explanation or statistical interpretation is made.

Read-only consistency calculations were performed on the existing CSV files. They did not write or alter any experiment artifact.

## 1. Results Overview

The repository contains these result categories:

- six final experiment CSVs under `results/`;
- three topic-keyword CSVs under `results/topics/`;
- nine parameter-search CSVs covering current, archived coarse, and archived fine searches;
- three parameter-selected YAML files, one for each search round;
- six publication figures, each exported as PDF and PNG.

Final evaluation covers LDA, NMF, and BERTopic. The available outputs include:

- Precision@1/@3/@5;
- Recall@1/@3/@5;
- F1@1/@3/@5;
- NDCG@1/@3/@5;
- MRR;
- C_v topic coherence;
- model topic counts and BERTopic noise statistics;
- training, inference/recommendation, and total runtimes;
- top-1/top-3/top-5 recommendations;
- complete 12-group rankings for every test researcher;
- per-topic keyword exports;
- dataset/split statistics;
- parameter-search candidates, scores, runtimes, and errors.

The authoritative final numerical files are:

- `results/metrics.csv`
- `results/model_statistics.csv`
- `results/runtime_statistics.csv`
- `results/dataset_statistics.csv`
- `results/recommendations.csv`
- `results/rankings.csv`

## 2. Final Recommendation Performance

Source: `results/metrics.csv`

Figure: `results/figures/recommendation_performance.pdf` and `.png`

### Precision and Recall

| Model | Precision@1 | Precision@3 | Precision@5 | Recall@1 | Recall@3 | Recall@5 |
|---|---:|---:|---:|---:|---:|---:|
| LDA | 0.659509 | 0.565951 | 0.496626 | 0.242137 | 0.548994 | 0.764742 |
| NMF | 0.628834 | 0.510736 | 0.430061 | 0.218528 | 0.498701 | 0.676990 |
| BERTopic | 0.687117 | 0.578732 | 0.507975 | 0.251288 | 0.565841 | 0.782423 |

### F1, NDCG, and MRR

| Model | F1@1 | F1@3 | F1@5 | NDCG@1 | NDCG@3 | NDCG@5 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| LDA | 0.328309 | 0.518832 | 0.565168 | 0.659509 | 0.662546 | 0.719909 | 0.793238 |
| NMF | 0.302684 | 0.467906 | 0.492299 | 0.628834 | 0.604125 | 0.640513 | 0.774992 |
| BERTopic | 0.342141 | 0.532229 | 0.578973 | 0.687117 | 0.678037 | 0.736175 | 0.809277 |

The performance figure plots Precision, Recall, F1, and NDCG against K and presents MRR as a separate scalar panel. It uses the same values as `metrics.csv`.

## 3. Model Ranking Summary

Source: `results/metrics.csv`

All 13 reported recommendation metric columns have the same model ordering:

| Metric | Highest | Second | Third |
|---|---|---|---|
| Precision@1 | BERTopic | LDA | NMF |
| Precision@3 | BERTopic | LDA | NMF |
| Precision@5 | BERTopic | LDA | NMF |
| Recall@1 | BERTopic | LDA | NMF |
| Recall@3 | BERTopic | LDA | NMF |
| Recall@5 | BERTopic | LDA | NMF |
| F1@1 | BERTopic | LDA | NMF |
| F1@3 | BERTopic | LDA | NMF |
| F1@5 | BERTopic | LDA | NMF |
| NDCG@1 | BERTopic | LDA | NMF |
| NDCG@3 | BERTopic | LDA | NMF |
| NDCG@5 | BERTopic | LDA | NMF |
| MRR | BERTopic | LDA | NMF |

This table is a direct sort of each column in `metrics.csv`; it does not introduce a combined or weighted model score.

## 4. Topic Quality Results

Source: `results/model_statistics.csv`

Figure: topic-coherence panel of `results/figures/model_quality_and_runtime.pdf` and `.png`

The only exported topic-quality metric is C_v coherence.

| Coherence rank | Model | Number of topics | C_v coherence | Coherence note |
|---:|---|---:|---:|---|
| 1 | NMF | 11 | 0.686782 | blank |
| 2 | BERTopic | 43 | 0.659480 | blank |
| 3 | LDA | 29 | 0.521625 | blank |

`model_statistics.csv` also contains BERTopic-specific noise fields:

| Model | Noise documents | Noise percentage |
|---|---:|---:|
| LDA | blank | blank |
| NMF | blank | blank |
| BERTopic | 0 | 0.0% |

The topic keyword files contain exactly the same number of rows as the topic counts:

- `results/topics/lda_topics.csv`: 29 topic rows.
- `results/topics/nmf_topics.csv`: 11 topic rows.
- `results/topics/bertopic_topics.csv`: 43 topic rows.

## 5. Hyperparameter Selection Results

### Final/current selected configurations

Sources:

- `results/parameter_selection/lda_search.csv`
- `results/parameter_selection/nmf_search.csv`
- `results/parameter_selection/bertopic_search.csv`
- `results/parameter_selection/best_config.yaml`

| Model | Selected configuration | Selection C_v |
|---|---|---:|
| LDA | `num_topics=29` | 0.521625 |
| NMF | `num_topics=11` | 0.686782 |
| BERTopic | `min_topic_size=45`, `umap_n_neighbors=50` | 0.659480 |

The current `best_config.yaml` also records:

- LDA `max_iter=20`;
- NMF `max_iter=800`;
- BERTopic `embedding_model=allenai/specter`;
- BERTopic UMAP components 5 and minimum distance 0.0;
- training ratio 0.8 and random seed 42.

The root `config.yaml` contains the same selected model values.

### Current LDA search

Source: `results/parameter_selection/lda_search.csv`

| Rank | Topics | C_v | Runtime (s) | Error |
|---:|---:|---:|---:|---|
| 1 | 29 | 0.521625 | 284.739925 | blank |
| 2 | 19 | 0.518153 | 272.027967 | blank |
| 3 | 21 | 0.516896 | 263.063478 | blank |
| 4 | 31 | 0.512700 | 283.994076 | blank |

### Current NMF search

Source: `results/parameter_selection/nmf_search.csv`

| Rank | Topics | C_v | Runtime (s) | Error |
|---:|---:|---:|---:|---|
| 1 | 11 | 0.686782 | 25.326236 | blank |
| 2 | 9 | 0.683174 | 21.515288 | blank |
| 3 | 12 | 0.672344 | 31.566944 | blank |
| 4 | 15 | 0.671161 | 27.566503 | blank |

### Current BERTopic search

Source: `results/parameter_selection/bertopic_search.csv`

| Rank | Min. topic size | UMAP neighbors | Actual topics | Noise docs | C_v | Runtime (s) | Error |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 45 | 50 | 43 | 0 | 0.659480 | 47.560791 | blank |
| 2 | 45 | 55 | 41 | 0 | 0.658543 | 50.125586 | blank |
| 3 | 50 | 60 | 38 | 0 | 0.656804 | 54.382509 | blank |
| 4 | 40 | 55 | 46 | 0 | 0.654319 | 70.229787 | blank |
| 5 | 50 | 55 | 38 | 0 | 0.651622 | 53.016731 | blank |
| 6 | 50 | 50 | 39 | 0 | 0.650780 | 49.846476 | blank |
| 7 | 40 | 50 | 44 | 0 | 0.646736 | 113.017592 | blank |
| 8 | 40 | 60 | 5 | 0 | 0.533384 | 50.009339 | blank |
| 9 | 45 | 60 | 5 | 0 | 0.533384 | 47.909066 | blank |

### Archived search-round outcomes

The archived directories contain additional search outputs:

| Round | Model | Highest-ranked configuration | C_v | Source |
|---|---|---|---:|---|
| Coarse | LDA | 20 topics | 0.528668 | `coarse_search/lda_search.csv` |
| Coarse | NMF | 10 topics | 0.681803 | `coarse_search/nmf_search.csv` |
| Coarse | BERTopic | size 20, neighbors 30 | 0.550283 | `coarse_search/bertopic_search.csv` |
| Fine | LDA | 20 topics | 0.528668 | `fine_search/lda_search.csv` |
| Fine | NMF | 10 topics | 0.681803 | `fine_search/nmf_search.csv` |
| Fine | BERTopic | size 40, neighbors 50 | 0.645842 | `fine_search/bertopic_search.csv` |

Each archived round has its own `best_config.yaml`. The current hyperparameter-search figure uses only the three current search CSVs, not the archived directories.

## 6. Runtime Results

Source: `results/runtime_statistics.csv`

Figure: runtime panel of `results/figures/model_quality_and_runtime.pdf` and `.png`

| Runtime rank (total) | Model | Training (s) | Recommendation/inference (s) | Total (s) |
|---:|---|---:|---:|---:|
| 1 | NMF | 18.064461 | 0.721571 | 18.786032 |
| 2 | LDA | 178.496110 | 1.762246 | 180.258356 |
| 3 | BERTopic | 966.536182 | 5.240865 | 971.777047 |

NMF has the shortest recorded value in all three runtime columns, followed by LDA and BERTopic. The figure displays training and inference separately on a logarithmic horizontal axis.

The total column equals training plus inference within floating-point precision for all models.

Parameter-search runtimes are separate values stored in each search CSV. They should not be substituted for the final runtimes above.

## 7. Dataset Statistics

### Exported statistics

Source: `results/dataset_statistics.csv`

| Statistic | Value |
|---|---:|
| Number of unique prepared publication documents | 11,111 |
| Number of researchers | 3,257 |
| Number of research groups | 12 |
| Training researchers | 2,605 |
| Testing researchers | 652 |
| Average prepared publication rows per researcher | 6.131409 |
| Minimum prepared publication rows per researcher | 1 |
| Maximum prepared publication rows per researcher | 59 |
| Average researchers per group | 952.5 |

The source publication input contains 11,681 raw CSV rows. After preprocessing and author expansion there are 19,970 author-publication rows representing 11,111 unique document strings. These quantities have different definitions and are not interchangeable.

### Group sizes shown in dataset characterization

Source: `data/groups_2015_2025_10k_plus.csv`, after the cleaning used by `load_datasets()`

Figure: group-membership panel of `dataset_characterization`

| Research group | Researchers |
|---|---:|
| Machine Learning | 2,480 |
| Artificial Intelligence | 2,090 |
| Computer Vision | 1,748 |
| Natural Language Processing | 1,333 |
| Robotics | 868 |
| The Web & Information Retrieval | 831 |
| Data Mining | 711 |
| Databases | 423 |
| Computer Graphics | 358 |
| Human-Computer Interaction | 343 |
| Software Engineering | 127 |
| Visualization | 118 |

### Publications and group affiliations

The dataset-characterization figure uses a mean of 6.131409 and median of 2 prepared publication rows per researcher. The group-membership-count distribution is:

| Groups per researcher | Researchers |
|---:|---:|
| 1 | 421 |
| 2 | 587 |
| 3 | 703 |
| 4 | 649 |
| 5 | 465 |
| 6 | 267 |
| 7 | 126 |
| 8 | 30 |
| 9 | 9 |

These figure-derived statistics are not stored as additional standalone result CSVs.

## 8. Per-Group Results

Source: `results/rankings.csv`

Figure: `results/figures/per_group_performance.pdf` and `.png`

The per-group heatmap reports mean `is_relevant` among cases where a group appears in ranks 1–3, grouped by research group and model. The figure labels this quantity “Top-3 precision.” Support is the number of distinct testing researchers for whom the group is relevant.

| Research group | Support | LDA | NMF | BERTopic |
|---|---:|---:|---:|---:|
| Machine Learning | 483 | 0.912134 | 0.865385 | 0.910314 |
| Artificial Intelligence | 395 | 0.663968 | 0.627329 | 0.666667 |
| Computer Vision | 347 | 0.855372 | 0.893204 | 0.879032 |
| Natural Language Processing | 264 | 0.736527 | 0.718919 | 0.709184 |
| The Web & Information Retrieval | 173 | 0.545455 | 0.543956 | 0.588235 |
| Robotics | 167 | 0.527094 | 0.603896 | 0.510000 |
| Data Mining | 139 | 0.354497 | 0.438356 | 0.393258 |
| Databases | 81 | 0.505882 | 0.333333 | 0.536842 |
| Human-Computer Interaction | 73 | 0.296703 | 0.176101 | 0.438356 |
| Computer Graphics | 67 | 0.335821 | 0.308176 | 0.324841 |
| Visualization | 28 | 0.139535 | 0.071429 | 0.129870 |
| Software Engineering | 24 | 0.084034 | 0.074074 | 0.096000 |

No standalone per-group summary CSV is exported. The values are calculated by `visualization.plot_per_group_performance()` from `rankings.csv`.

## 9. Ranking Outputs

### Complete rankings

`results/rankings.csv` contains 23,472 rows with:

- `model`
- `researcher_name`
- `rank`
- `group_name`
- `similarity`
- `is_relevant`

It covers three models, 652 testing researchers, and 12 candidate groups. Every model/researcher pair has a complete rank sequence from 1 through 12.

Each model has 2,241 relevant researcher-group rows. These rows represent relevant group memberships, not distinct researchers.

### Compact recommendations

`results/recommendations.csv` contains 1,956 rows: one row for each of 652 researchers and three models. Columns contain:

- pipe-separated ground-truth groups;
- top-1, top-3, and top-5 group names;
- corresponding similarity scores rounded to six decimals.

The exported top-K names and scores match the corresponding prefixes of `rankings.csv`.

### Cumulative relevant-rank distribution

Source: `results/rankings.csv`

Figure: `results/figures/rank_distribution.pdf` and `.png`

The figure reports the cumulative fraction of all relevant researcher-group rows at or above each rank cutoff:

| Model | K=1 | K=2 | K=3 | K=4 | K=5 | K=6 |
|---|---:|---:|---:|---:|---:|---:|
| LDA | 0.191879 | 0.356983 | 0.493976 | 0.618474 | 0.722445 | 0.797412 |
| NMF | 0.182954 | 0.323516 | 0.445783 | 0.527443 | 0.625614 | 0.721553 |
| BERTopic | 0.199911 | 0.354752 | 0.505132 | 0.630522 | 0.738956 | 0.818831 |

| Model | K=7 | K=8 | K=9 | K=10 | K=11 | K=12 |
|---|---:|---:|---:|---:|---:|---:|
| LDA | 0.861669 | 0.915216 | 0.949576 | 0.979920 | 0.994645 | 1.000000 |
| NMF | 0.811691 | 0.879518 | 0.925926 | 0.962963 | 0.991075 | 1.000000 |
| BERTopic | 0.878626 | 0.921464 | 0.957608 | 0.981705 | 0.991075 | 1.000000 |

These cumulative values are visualization-derived and are not stored in a separate ranking-statistics CSV.

## 10. Generated Figures

All figures are produced by `src/visualization.py` and exported in PDF and PNG.

| Figure basename | Purpose | Visualization | Direct sources |
|---|---|---|---|
| `dataset_characterization` | Dataset composition and distributions | Multi-panel horizontal bars, histogram, discrete bars | Publication and group input CSVs |
| `hyperparameter_search` | Current parameter-search outcomes | LDA/NMF annotated dots; BERTopic heatmap | Current three search CSVs |
| `model_quality_and_runtime` | Final C_v and runtime values | Coherence dot/lollipop plus grouped log-runtime bars | `model_statistics.csv`, `runtime_statistics.csv`; search CSV coherence fallback |
| `recommendation_performance` | All final recommendation metrics | Four metric line panels plus scalar MRR panel | `metrics.csv` |
| `per_group_performance` | Group-specific top-3 precision | Annotated heatmap | `rankings.csv` |
| `rank_distribution` | Cumulative relevant-group ranks | Cumulative line plot | `rankings.csv` |

Each basename has:

- `results/figures/<basename>.pdf`
- `results/figures/<basename>.png`

## 11. Generated CSV Files

### Final experiment CSVs

| File | Rows | Purpose | Primary columns | Producer | Downstream use |
|---|---:|---|---|---|---|
| `results/dataset_statistics.csv` | 9 | Dataset/split statistics | `statistic`, `value` | `experiment.py` | Reporting |
| `results/model_statistics.csv` | 3 | Topic/coherence/noise summary | model, topics, C_v, noise | `experiment.py` | Quality/runtime figure |
| `results/runtime_statistics.csv` | 3 | Final timing | model, training, inference, total | `experiment.py` | Quality/runtime figure |
| `results/metrics.csv` | 3 | Aggregate recommendation metrics | model and 13 metric columns | `experiment.py` | Performance figure |
| `results/recommendations.csv` | 1,956 | Compact top-K output | model, researcher, truth, top-K names/scores | `experiment.py` | Inspection/reporting |
| `results/rankings.csv` | 23,472 | Complete candidate rankings | model, researcher, rank, group, similarity, relevance | `experiment.py` | Per-group and rank figures |

### Topic CSVs

| File | Rows | Columns | Producer |
|---|---:|---|---|
| `results/topics/lda_topics.csv` | 29 | `topic_id`, `top_keywords` | `experiment.py` |
| `results/topics/nmf_topics.csv` | 11 | `topic_id`, `top_keywords` | `experiment.py` |
| `results/topics/bertopic_topics.csv` | 43 | `topic_id`, `top_keywords` | `experiment.py` |

### Parameter-search CSVs

| File | Rows | Purpose |
|---|---:|---|
| `parameter_selection/lda_search.csv` | 4 | Current LDA candidates |
| `parameter_selection/nmf_search.csv` | 4 | Current NMF candidates |
| `parameter_selection/bertopic_search.csv` | 9 | Current BERTopic candidates |
| `parameter_selection/coarse_search/lda_search.csv` | 5 | Archived coarse LDA candidates |
| `parameter_selection/coarse_search/nmf_search.csv` | 5 | Archived coarse NMF candidates |
| `parameter_selection/coarse_search/bertopic_search.csv` | 12 | Archived coarse BERTopic candidates |
| `parameter_selection/fine_search/lda_search.csv` | 7 | Archived fine LDA candidates |
| `parameter_selection/fine_search/nmf_search.csv` | 6 | Archived fine NMF candidates |
| `parameter_selection/fine_search/bertopic_search.csv` | 15 | Archived fine BERTopic candidates |

LDA/NMF search columns are rank, topic count, C_v, runtime, and error. BERTopic additionally records minimum topic size, UMAP neighbors, actual topics, and noise documents.

The two CSV files under `data/` are experiment inputs rather than outputs from `experiment.py`.

## 12. Best Observed Results

### Highest value in every final recommendation column

Source: `results/metrics.csv`

| Metric | Model | Highest value |
|---|---|---:|
| Precision@1 | BERTopic | 0.687117 |
| Precision@3 | BERTopic | 0.578732 |
| Precision@5 | BERTopic | 0.507975 |
| Recall@1 | BERTopic | 0.251288 |
| Recall@3 | BERTopic | 0.565841 |
| Recall@5 | BERTopic | 0.782423 |
| F1@1 | BERTopic | 0.342141 |
| F1@3 | BERTopic | 0.532229 |
| F1@5 | BERTopic | 0.578973 |
| NDCG@1 | BERTopic | 0.687117 |
| NDCG@3 | BERTopic | 0.678037 |
| NDCG@5 | BERTopic | 0.736175 |
| MRR | BERTopic | 0.809277 |

Other strongest observed values by result category:

| Result category | Observed result | Source |
|---|---|---|
| Highest final topic coherence | NMF, 0.686782 | `model_statistics.csv` |
| Shortest final training time | NMF, 18.064461 s | `runtime_statistics.csv` |
| Shortest final inference time | NMF, 0.721571 s | `runtime_statistics.csv` |
| Shortest final total time | NMF, 18.786032 s | `runtime_statistics.csv` |
| Highest per-group top-3 precision cell | LDA on Machine Learning, 0.912134 | derived from `rankings.csv` |
| Current selected LDA | 29 topics | `lda_search.csv`, `best_config.yaml` |
| Current selected NMF | 11 topics | `nmf_search.csv`, `best_config.yaml` |
| Current selected BERTopic | size 45, neighbors 50 | `bertopic_search.csv`, `best_config.yaml` |

These entries summarize maxima/minima within their own reported categories; they are not a cross-metric composite score.

## 13. Result Consistency Check

### Checks that passed

1. **Metrics versus rankings:** all 13 metric columns were independently recomputed from `rankings.csv` using the implemented definitions. The maximum absolute difference from `metrics.csv` was approximately `1.33 × 10^-15`, consistent with floating-point rounding.
2. **Recommendations versus rankings:** all top-1, top-3, and top-5 names and six-decimal scores in `recommendations.csv` match the corresponding prefixes in `rankings.csv`.
3. **Row dimensions:** `recommendations.csv` has 652 researchers × 3 models = 1,956 rows. `rankings.csv` has 652 researchers × 3 models × 12 groups = 23,472 rows.
4. **Topic counts:** each topic CSV row count matches `number_of_topics` in `model_statistics.csv`.
5. **Selected parameters:** current rank-1 search rows match `results/parameter_selection/best_config.yaml`.
6. **Final configuration:** root `config.yaml` matches the current selected model configuration.
7. **Final coherence:** model-statistics C_v values match the selected current search scores for all three models.
8. **Runtime totals:** every total runtime equals training plus inference within floating-point precision.
9. **Figure sourcing:** all six expected PDF/PNG pairs exist, and their plotting functions read the documented CSV/input sources.
10. **Ground truth and rank completeness:** every recommendation row's ground-truth set matches the relevant groups marked in `rankings.csv`, and every model/researcher ranking contains ranks 1 through 12 exactly once.

### Documented differences or cautions

1. **Archived BERTopic noise:** coarse/fine BERTopic search artifacts contain nonzero noise-document counts, whereas all current search rows and the final model statistics contain zero. These are different search-round artifacts; the repository does not include version metadata sufficient to reconstruct every implementation change between them.
2. **Publication counts use different units:** the raw publication CSV has 11,681 rows, `dataset_statistics.csv` reports 11,111 unique prepared documents, and preprocessing yields 19,970 author-publication rows. These are consistent with different counting definitions, but paper text should name the unit explicitly.
3. **Per-group and cumulative-rank values are not separately exported:** they are reproducible from `rankings.csv` and encoded in figures, but no dedicated summary CSV exists.
4. **README figure names are stale:** README refers to `retrieval_performance.png` and `ranking_quality.png`; the current generated suite instead uses `recommendation_performance` and `model_quality_and_runtime` among its six basenames.

No final metric, topic count, runtime total, recommendation prefix, or selected current parameter mismatch was found.

## 14. Result Traceability

### Recommendation metrics

```text
Complete model rankings
results/rankings.csv
        ↓ evaluation aggregation
results/metrics.csv
        ↓
recommendation_performance.pdf/.png
        ↓
Final Precision, Recall, F1, NDCG, and MRR values
```

`metrics.csv` is the authoritative source for final aggregate numerical reporting. `rankings.csv` is the authoritative detailed source from which those metrics can be reproduced.

### Topic quality

```text
Training-corpus topic keywords and C_v calculation
        ↓
results/model_statistics.csv
        ├──→ results/topics/<model>_topics.csv
        └──→ model_quality_and_runtime.pdf/.png
```

Current parameter-search coherence is separately traceable through each current search CSV and `best_config.yaml`.

### Runtime

```text
Per-model timing in experiment.run_model()
        ↓
results/runtime_statistics.csv
        ↓
model_quality_and_runtime.pdf/.png
```

### Parameter selection

```text
Candidate model fits on training publications
        ↓ C_v scoring
<model>_search.csv
        ↓ highest-ranked valid row
best_config.yaml
        ↓ explicit experiment configuration
Final model statistics and evaluation outputs
```

### Per-group and rank-distribution results

```text
Complete rankings with relevance labels
results/rankings.csv
        ├──→ per_group_performance.pdf/.png
        └──→ rank_distribution.pdf/.png
```

### Dataset characterization

```text
Publication and group input CSVs
        ├──→ results/dataset_statistics.csv
        └──→ dataset_characterization.pdf/.png
```

## 15. Results Inventory

| Category | Authoritative artifacts | Derived figure/artifact |
|---|---|---|
| Recommendation metrics | `metrics.csv`, reproducible from `rankings.csv` | `recommendation_performance` |
| Compact recommendations | `recommendations.csv` | None |
| Complete rankings | `rankings.csv` | `per_group_performance`, `rank_distribution` |
| Topic quality/count/noise | `model_statistics.csv` | `model_quality_and_runtime` |
| Topic keywords | three files under `results/topics/` | None |
| Runtime | `runtime_statistics.csv` | `model_quality_and_runtime` |
| Dataset/split statistics | `dataset_statistics.csv` | `dataset_characterization` |
| Current parameter selection | three current search CSVs, current `best_config.yaml` | `hyperparameter_search` |
| Archived coarse selection | three coarse CSVs, coarse `best_config.yaml` | No current automatic figure |
| Archived fine selection | three fine CSVs, fine `best_config.yaml` | No current automatic figure |
| Final figures | six PDF/PNG pairs | Produced by `visualization.py` |

Total generated CSV inventory:

- 6 final root result CSVs;
- 3 topic CSVs;
- 9 parameter-search CSVs;
- 18 generated result CSVs in total.

Additional non-CSV result artifacts:

- 3 parameter-selected YAML files;
- 6 PDF figures;
- 6 PNG figures.

## 16. Notes for Paper Writing

- Use `results/metrics.csv` for every final Precision, Recall, F1, NDCG, and MRR value.
- Use `results/model_statistics.csv` for final topic counts, C_v coherence, and BERTopic noise.
- Use `results/runtime_statistics.csv` for final runtime values; do not substitute candidate-search runtimes.
- Use the current search CSVs and current `best_config.yaml` when reporting final parameter selection.
- Label coarse and fine search files explicitly as archived earlier rounds if they are discussed.
- State whether “publications” means 11,681 raw CSV rows, 11,111 unique prepared documents, or 19,970 author-publication modeling rows.
- The performance figure summarizes all 13 final metric values, including scalar MRR.
- The combined quality/runtime figure summarizes two different authoritative CSV files.
- The per-group heatmap and rank-distribution figure are derived from `rankings.csv`; their numerical summaries are not separately exported.
- `recommendations.csv` is suitable for reporting examples of top-K output, while `rankings.csv` is the authoritative complete-ranking artifact.
- No uncertainty intervals or statistical significance outputs exist, so none can be reported from the current artifacts.
- Numerical values in paper tables should preserve enough digits to distinguish models while acknowledging that source files store greater precision than the six-decimal presentation used in this audit.
