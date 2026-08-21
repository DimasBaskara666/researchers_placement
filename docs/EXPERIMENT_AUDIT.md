# Experimental Audit

## Audit basis and limitations

This audit describes the experiment code, configuration, and checked-in artifacts at repository commit `aaf2fad`. No experiment, parameter search, model fit, data acquisition process, or visualization generation was executed for this audit.

The following provenance limits are visible in the repository:

- Search results exist for coarse, fine, and current rounds, but the current `src/parameter_selection.py` contains only the current-round candidate constants. The exact earlier source revisions or commands that produced the archived coarse/fine outputs are not checked in as separate scripts.
- `src/experiment.py` does not automatically consume `results/parameter_selection/best_config.yaml`; the selected YAML must be supplied with `--config`. The root `config.yaml`, current `best_config.yaml`, and final model-statistics topic counts are consistent, but the repository does not store the exact command invocation that produced the final result CSVs.
- Output files have no run identifier, timestamp, environment manifest, hardware description, or software-version record. Runtime values can be inspected, but their execution environment cannot be established from the artifacts.

## 1. Experimental Overview

The implemented experiment compares LDA, NMF, and BERTopic as publication-level topic representations for researcher-to-research-group recommendation. The three representations share the same researcher split, profile aggregation, group-profile construction, cosine ranking, and evaluation code.

The implemented workflow is:

1. Optionally search parameters on the training partition using C_v coherence.
2. Explicitly choose a YAML configuration for the experiment.
3. Load and preprocess the two input datasets.
4. split researchers into training and testing partitions.
5. Fit LDA, NMF, and BERTopic sequentially on training publications.
6. Aggregate training publication representations into researcher and group profiles.
7. Represent testing publications and aggregate them into testing researcher profiles.
8. Rank all available group profiles by cosine similarity for each testing researcher.
9. Evaluate rankings with Precision@K, Recall@K, F1@K, NDCG@K, and MRR.
10. Export result tables and topic keywords.
11. Generate publication figures from the data and result artifacts.

Parameter selection, the final experiment, and visualization are separate command-line executions rather than one orchestrated command.

## 2. Experiment Pipeline

### Entry points

- `src/parameter_selection.py`: parameter-search entry point.
- `src/experiment.py`: final experiment entry point.
- `src/visualization.py`: figure-generation entry point.

The README documents this nominal sequence:

```powershell
python src/parameter_selection.py
python src/experiment.py --config results/parameter_selection/best_config.yaml
python src/visualization.py
```

Running `experiment.py` without `--config` instead uses the root `config.yaml`.

### Final experiment execution

`src/experiment.py:main()`:

1. Loads YAML through `load_config()`.
2. Calls `set_random_seed()`.
3. Calls `preprocessing.load_datasets()`.
4. Intersects publication researchers with labelled group researchers.
5. Calls `splitter.split_researchers()`.
6. Filters publication rows into training and testing frames.
7. Builds the full researcher-to-group truth map.
8. Calls `run_model()` for `lda`, `nmf`, and `bertopic`, in that order.
9. Evaluates each model and constructs recommendation/ranking/topic tables.
10. Writes all result CSVs after all three model runs complete.

Within each `run_model()` call:

```text
Fit model on training publications
        ↓
Transform training publications
        ↓
Extract topics and compute training-corpus coherence
        ↓
Average publication vectors into training researcher profiles
        ↓
Average training researcher profiles into group profiles
        ↓
Transform testing publications
        ↓
Average publication vectors into testing researcher profiles
        ↓
Rank group profiles by cosine similarity
```

Metrics and export-table construction occur in `main()` after `run_model()` returns.

### Module dependencies

- `preprocessing.py`: input validation and text preparation.
- `splitter.py`: researcher-level split.
- `lda_model.py`, `nmf_model.py`, `bertopic_model.py`: fitted representations.
- `topic_utils.py`: C_v coherence and topic-table creation.
- `profile_builder.py`: researcher profiles, group profiles, and truth sets.
- `recommender.py`: cosine ranking.
- `metrics.py`: individual metric definitions.
- `evaluator.py`: per-researcher evaluation aggregation and output tables.
- `visualization.py`: post-experiment plots.

## 3. Dataset Used in Experiments

### Input files

`preprocessing.load_datasets()` reads fixed filenames:

- `data/publications_2015_2025_10k_plus.csv`
- `data/groups_2015_2025_10k_plus.csv`

The checked-in publication CSV has 11,681 rows and columns:

`Title, Abstract, Authors, Year, Venue, DOI`

The checked-in group CSV has 11,430 rows and columns:

`researcher_name, group_name`

Only `Title`, `Abstract`, and `Authors` are required and used from the publication file. Only `researcher_name` and `group_name` are required from the group file.

### Verified prepared-dataset statistics

`results/dataset_statistics.csv` records:

- 11,111 unique prepared publication documents;
- 3,257 researchers;
- 12 research groups;
- 2,605 training researchers;
- 652 testing researchers;
- mean 6.1314 prepared publication rows per researcher;
- minimum 1 and maximum 59 prepared publication rows per researcher;
- mean 952.5 labelled researchers per group.

These values are generated by `experiment.dataset_statistics()`. `number_of_publications` counts unique prepared `document` strings, not raw CSV rows.

### Pre-experiment filtering

`load_datasets()`:

- strips group and researcher labels;
- drops blank group/researcher values and duplicate membership rows;
- concatenates whitespace-normalized title and abstract;
- creates a lowercased alphanumeric bag-of-words version;
- removes NLTK English stopwords and ten configured research-jargon terms;
- skips publications with unusable text or missing authors;
- splits `Authors` on semicolons;
- retains only author names exactly present in the group file.

One prepared row is created for every matched author-publication association. A publication can therefore appear more than once in the modeling frame when multiple labelled coauthors are present.

The experiment does not filter on the `Year`, `Venue`, or `DOI` columns.

## 4. Experimental Protocol

### Split

Researchers—not publications—are split with scikit-learn `train_test_split()`. Names are deduplicated and sorted before splitting. The configured values are:

- training ratio: 0.8;
- random seed: 42;
- no group stratification;
- no cross-validation or repeated splits.

All prepared publication rows for one researcher remain in the same partition.

### Training stage

Each model is fitted only on training-researcher publications. LDA and NMF receive `bow_document`; BERTopic receives the original-text `document`.

After fitting, training publication representations are averaged per researcher. Group profiles are then averages of available training-member researcher profiles. Although the complete group-membership frame is supplied, test researchers do not contribute because they have no profile in the training-profile dictionary.

Groups with no available training member profile are omitted from the candidate set.

### Testing and recommendation stage

The fitted representation transforms testing publications. Publication representations are averaged by testing researcher. `recommender.rank_groups()` computes cosine similarity from each testing researcher profile to every available group profile and sorts descending, with group name as the deterministic secondary key.

### Evaluation and output stage

The complete rankings are compared with each testing researcher's full set of group labels. Evaluation is multi-label. After all three model runs, the experiment writes aggregate metrics, detailed recommendations, complete rankings, topic tables, model statistics, dataset statistics, and runtime statistics.

## 5. Hyperparameter Selection

### Shared selection procedure

`src/parameter_selection.py` reconstructs the same seeded researcher split as the final experiment and retains only training-researcher publication rows.

Every candidate is evaluated by training-corpus C_v coherence from `topic_utils.coherence_cv()`. Candidate tables are sorted by descending `coherence_cv`; failed/NaN candidates appear last. Runtime and exception text are stored for every candidate.

Selection does not use Precision, Recall, F1, NDCG, MRR, or any test data.

### Current executable search space

The current module-level constants are:

| Model | Parameter(s) | Candidates |
|---|---|---|
| LDA | `num_topics` | 19, 21, 29, 31 |
| NMF | `num_topics` | 9, 11, 12, 15 |
| BERTopic | `min_topic_size` | 40, 45, 50 |
| BERTopic | `umap_n_neighbors` | 50, 55, 60 |

BERTopic evaluates the Cartesian product, giving nine candidates. Its fixed embedding model is loaded once and the training documents are encoded once. The resulting SPECTER embedding matrix is reused across all candidate fits.

The current search files are:

- `results/parameter_selection/lda_search.csv`
- `results/parameter_selection/nmf_search.csv`
- `results/parameter_selection/bertopic_search.csv`
- `results/parameter_selection/best_config.yaml`

### Current selected configuration

The highest-coherence rows in the current artifacts, and the values stored in `best_config.yaml`, are:

| Model | Selected values |
|---|---|
| LDA | `num_topics=29` |
| NMF | `num_topics=11` |
| BERTopic | `min_topic_size=45`, `umap_n_neighbors=50` |

Other fixed values in the selected configuration are LDA `max_iter=20`, NMF `max_iter=800`, BERTopic `embedding_model=allenai/specter`, UMAP components 5, and UMAP minimum distance 0.0.

The root `config.yaml` currently contains the same final model values.

### Archived coarse search

`results/parameter_selection/coarse_search/` contains an earlier search artifact set:

| Model | Candidate space visible in CSV | Highest-ranked archived configuration |
|---|---|---|
| LDA | 10, 20, 30, 40, 50 topics | 20 topics |
| NMF | 10, 20, 30, 40, 50 topics | 10 topics |
| BERTopic | minimum topic size 5, 10, 15, 20 × neighbors 10, 15, 30 | size 20, neighbors 30 |

The directory includes its own `best_config.yaml`.

### Archived fine search

`results/parameter_selection/fine_search/` contains:

| Model | Candidate space visible in CSV | Highest-ranked archived configuration |
|---|---|---|
| LDA | 18, 20, 22, 24, 26, 28, 30 topics | 20 topics |
| NMF | 5, 7, 10, 12, 15, 20 topics | 10 topics |
| BERTopic | minimum topic size 20, 25, 30, 35, 40 × neighbors 30, 40, 50 | size 40, neighbors 50 |

This directory also includes its own `best_config.yaml`.

The current code does not create the `coarse_search/` or `fine_search/` directories and does not contain these earlier candidate constants. These CSVs verify what was evaluated, but not the exact source revision or invocation that generated them.

### BERTopic artifact-version caution

The current BERTopic search uses fixed embedding-based `reduce_outliers()` followed by `update_topics()`. Current BERTopic search rows report zero noise documents. Archived coarse/fine BERTopic rows report thousands of noise documents. This verifies that the artifact behavior differs, but the repository does not contain enough version metadata to establish every implementation difference between those runs.

## 6. Model Training

### LDA

`lda_model.fit_lda()`:

1. Fits `CountVectorizer` to training `bow_document`.
2. Uses configured `max_features=20000`, `min_df=2`, and `max_df=0.8`.
3. Fits scikit-learn batch LDA with the selected topic count, 20 maximum iterations, and seed 42.
4. Transforms training documents into document-topic vectors.
5. Extracts ten highest-weight terms per topic.

The fitted vectorizer and LDA estimator exist only in memory inside `run_model()`.

### NMF

`nmf_model.fit_nmf()`:

1. Fits `TfidfVectorizer` to training `bow_document`.
2. Uses the same feature/document-frequency limits.
3. Fits NMF with the selected topic count, `init="nndsvda"`, 800 maximum iterations, and seed 42.
4. Transforms training documents into NMF component vectors.
5. Extracts ten highest-weight terms per topic.

TF-IDF is internal NMF preprocessing; it is not an evaluated standalone model. The fitted vectorizer and estimator are not serialized.

### BERTopic

`bertopic_model.fit_bertopic()`:

1. Loads `allenai/specter`.
2. Builds UMAP with cosine distance and seed 42.
3. Builds HDBSCAN with the configured minimum cluster size and `prediction_data=True`.
4. Builds a CountVectorizer using the shared stopword set.
5. Computes SPECTER embeddings once when embeddings are not supplied.
6. Fits BERTopic with probabilities disabled.
7. Calls `reduce_outliers(strategy="embeddings")` with the same embedding matrix.
8. Calls `update_topics()` with the reassigned topics and CountVectorizer.
9. Calls `approximate_distribution()` to obtain training publication-topic distributions.

Topic keywords and post-fit noise counts are exported. The fitted BERTopic model, SPECTER embeddings, and UMAP/HDBSCAN objects are not saved.

### Stored training outputs

The experiment stores only derived tables:

- topic keywords;
- topic count/coherence/noise statistics;
- runtimes;
- aggregate metrics;
- recommendations;
- complete rankings.

No model checkpoint, vectorizer, document vector, researcher profile, group profile, or embedding matrix is persisted.

## 7. Recommendation Experiment

Inputs to the recommendation stage are:

- fitted model-specific publication representations;
- training and testing publication frames;
- full group-membership table.

`profile_builder.average_researcher_profiles()` computes an arithmetic mean of all publication vectors belonging to each researcher.

`profile_builder.build_group_profiles()` computes an arithmetic mean of available training-researcher profiles in each group.

`recommender.rank_groups()` stacks group profiles, computes scikit-learn cosine similarity, and produces a complete ordered list for every test researcher.

The verified final artifacts contain:

- 652 distinct testing researchers;
- 12 ranked group candidates;
- three model identifiers (`lda`, `nmf`, `bertopic`);
- 1,956 recommendation rows (652 × 3);
- 23,472 ranking rows (652 × 12 × 3).

`recommendations.csv` retains top-1, top-3, and top-5 outputs. `rankings.csv` retains the complete ranking and is the richer reusable evaluation artifact.

## 8. Evaluation Procedure

`evaluator.evaluate_rankings()` iterates over every model's testing-researcher rankings. For each researcher:

1. Retrieve the set of relevant groups from `ground_truth_groups()`.
2. Remove similarity values to form an ordered group-name list.
3. Compute Precision, Recall, F1, and NDCG at K = 1, 3, and 5.
4. Compute reciprocal rank of the first relevant group.

Metric definitions in `metrics.py` are:

- Precision@K: relevant items among the first K divided by K.
- Recall@K: relevant items among the first K divided by the number of relevant groups.
- F1@K: harmonic mean of that researcher's Precision@K and Recall@K.
- NDCG@K: binary-relevance DCG normalized by the ideal DCG up to K.
- Reciprocal rank: inverse position of the first relevant group, or zero.

The evaluator then takes the arithmetic mean of each per-researcher metric list. Thus aggregation is macro averaging over the 652 test researchers.

No confidence interval, standard deviation, statistical significance test, repeated split, or per-run variance is calculated.

Exports:

- `metrics.csv`: one aggregate row per model.
- `recommendations.csv`: one top-K recommendation row per model/researcher.
- `rankings.csv`: one row per model/researcher/candidate group.

## 9. Output Files

### Final experiment tables

| Artifact | Generated by | Purpose / later use |
|---|---|---|
| `results/dataset_statistics.csv` | `experiment.main()` | Prepared-dataset and split summary; used as an audit/reporting artifact |
| `results/model_statistics.csv` | `experiment.main()` | Topic count, C_v coherence, BERTopic noise fields; used by quality/runtime figure |
| `results/runtime_statistics.csv` | `experiment.main()` | Training, inference/recommendation, and total time by model; used by quality/runtime figure |
| `results/metrics.csv` | `experiment.main()` | Aggregate Precision/Recall/F1/NDCG/MRR; used by recommendation-performance figure |
| `results/recommendations.csv` | `experiment.main()` | Human-readable top-1/top-3/top-5 group outputs and scores |
| `results/rankings.csv` | `experiment.main()` | Complete ranked candidate list with similarity and relevance; used by per-group and rank-distribution figures |

### Topic outputs

| Artifact | Rows | Purpose |
|---|---:|---|
| `results/topics/lda_topics.csv` | 29 | LDA topic IDs and ten representative keywords |
| `results/topics/nmf_topics.csv` | 11 | NMF topic IDs and ten representative keywords |
| `results/topics/bertopic_topics.csv` | 43 | Non-noise BERTopic IDs and representative keywords |

### Parameter-selection outputs

Current, coarse, and fine directories each contain LDA, NMF, and BERTopic search CSVs plus a `best_config.yaml`. Search CSVs store one row per candidate, rank, coherence, runtime, and error text; BERTopic additionally stores actual topic and noise counts.

### Figures

`results/figures/` contains six figures in both PDF and PNG:

- `dataset_characterization`
- `hyperparameter_search`
- `model_quality_and_runtime`
- `recommendation_performance`
- `per_group_performance`
- `rank_distribution`

No fitted-model or profile artifacts are generated.

## 10. Runtime Measurement

### Parameter-search runtime

Each candidate search records `runtime_seconds` from immediately before the candidate's `try` block until after fitting/coherence or failure. For BERTopic, SPECTER model loading and the one-time shared embedding computation occur before the candidate loop and are therefore excluded from every candidate runtime.

Search-table runtime includes:

- candidate model fitting;
- keyword extraction;
- C_v coherence calculation;
- BERTopic noise/topic-count extraction when applicable;
- exception handling overhead.

### Final experiment runtime

`experiment.run_model()` records two intervals with `time.perf_counter()`.

`training_time_seconds` begins before model fitting and ends after training researcher profiles and group profiles are built. It includes:

- model/vectorizer fitting;
- training-document transformation;
- topic extraction;
- C_v coherence;
- BERTopic outlier reduction/topic updating/noise statistics;
- training researcher-profile aggregation;
- group-profile aggregation.

`recommendation_inference_time_seconds` begins before testing-document transformation and ends after complete group rankings are generated. It includes:

- testing-document transformation;
- testing researcher-profile aggregation;
- cosine similarity and sorting.

`total_execution_time_seconds` is the sum of those two measured intervals.

The timers exclude initial data loading, splitting, global seeding, metric evaluation, result-table construction, CSV writing, and visualization. Because the three models run sequentially, each runtime row represents one model invocation, not parallel execution.

## 11. Experiment Configuration

### Root and selected YAML

`config.yaml` and the current `results/parameter_selection/best_config.yaml` both specify:

```yaml
random_seed: 42
train_ratio: 0.8
preprocessing:
  max_features: 20000
  ngram_range: [1, 1]
  min_df: 2
  max_df: 0.8
lda:
  num_topics: 29
  max_iter: 20
nmf:
  num_topics: 11
  max_iter: 800
bertopic:
  embedding_model: allenai/specter
  min_topic_size: 45
  umap_n_neighbors: 50
  umap_n_components: 5
  umap_min_dist: 0.0
output_dir: results
```

Both entry points load YAML with `yaml.safe_load()` and directly index required keys. There is no schema validation or automatic default merge.

`preprocessing.ngram_range` is present but is not passed to the LDA CountVectorizer, NMF TfidfVectorizer, or BERTopic CountVectorizer in the current implementation.

### Command-line options

`parameter_selection.py`:

- `--config`, default root `config.yaml`;
- `--model`, one of `lda`, `nmf`, `bertopic`, `all`, default `all`.

`experiment.py`:

- `--config`, default root `config.yaml`.

`visualization.py`:

- no command-line options.

### Hard-coded experimental constants

- model execution order: LDA, NMF, BERTopic;
- evaluation cutoffs: 1, 3, 5;
- top recommendation exports: 1, 3, 5;
- topic keywords: ten by default;
- parameter candidate spaces: module constants in `parameter_selection.py`;
- visualization model order and figure cutoffs: constants/code in `visualization.py`.

## 12. Reproducibility

Implemented mechanisms:

- fixed split/model seed 42;
- sorted researcher names before splitting;
- Python and NumPy seeding in the final experiment;
- PyTorch CPU/CUDA seeding when PyTorch is available;
- LDA and NMF `random_state`;
- UMAP `random_state`;
- deterministic alphabetical secondary sort for tied similarity scores;
- one checked-in root configuration and three round-specific selected configurations;
- checked-in input CSVs and generated result artifacts;
- BERTopic embedding reuse within the current parameter search;
- parameter-search error columns and candidate runtimes.

Limitations visible in the repository:

- dependency versions are not pinned or recorded;
- hardware, operating environment, and accelerator details are not stored;
- training/test researcher IDs are not exported, although the split can be reconstructed from the data, algorithm, ratio, seed, and compatible library behavior;
- final command lines and run logs are not persisted;
- outputs are overwritten at fixed paths and have no run identifier;
- fitted models, embeddings, vectors, and profiles are not saved;
- the parameter-selection entry point does not call `experiment.set_random_seed()`, although individual LDA/NMF/UMAP objects receive the configured seed;
- archived coarse/fine candidate spaces are not represented in the current source constants;
- exact acquisition provenance for the current named dataset files is not fully encoded in the experiment workflow.

The existing artifacts support result reuse and inspection, but exact runtime replication cannot be guaranteed from the repository alone.

## 13. Generated Figures

All figures are generated by `src/visualization.py` using Matplotlib's `Agg` backend. `save_figure()` exports PDF and 300-DPI PNG.

| Figure basename | Direct input artifacts |
|---|---|
| `dataset_characterization` | `data/publications_2015_2025_10k_plus.csv`, `data/groups_2015_2025_10k_plus.csv` through `load_datasets()` |
| `hyperparameter_search` | current `lda_search.csv`, `nmf_search.csv`, `bertopic_search.csv` |
| `model_quality_and_runtime` | `model_statistics.csv`, `runtime_statistics.csv`; parameter-search CSVs are fallback coherence sources |
| `recommendation_performance` | `metrics.csv` |
| `per_group_performance` | `rankings.csv` |
| `rank_distribution` | `rankings.csv` |

Each basename has `.pdf` and `.png` output files under `results/figures/`.

The plotting script filters model rows to LDA, NMF, and BERTopic. It does not modify the source result CSVs.

## 14. Experiment Dependency Flow

The implemented flow, including its manual configuration boundary, is:

```text
Input publication and group CSVs
        ↓
Preprocessing and researcher-level split
        ├──────────────────────────────────────┐
        ↓                                      ↓
Training-only parameter selection        Final experiment
        ↓                                      ↑
C_v-ranked search CSVs                         │
        ↓                                      │
best_config.yaml ── explicit --config choice ──┘
                                               ↓
                            LDA / NMF / BERTopic training
                                               ↓
                              Training researcher profiles
                                               ↓
                                    Group profiles
                                               ↓
                               Testing researcher profiles
                                               ↓
                              Cosine-similarity rankings
                                               ↓
                      Precision / Recall / F1 / NDCG / MRR
                                               ↓
                 Metrics + recommendations + rankings + topics
                                               ↓
                                 Visualization script
                                               ↓
                                     PDF/PNG figures
```

Parameter selection is optional from the perspective of `experiment.py`; any compatible YAML can be passed directly.

## 15. Experiment Artifacts Summary

### Configuration artifacts

- `config.yaml`: default/current final experiment configuration.
- `results/parameter_selection/best_config.yaml`: current coherence-selected configuration.
- `results/parameter_selection/coarse_search/best_config.yaml`: archived coarse-round selection.
- `results/parameter_selection/fine_search/best_config.yaml`: archived fine-round selection.

### Search artifacts

- Three current search CSVs.
- Three archived coarse search CSVs.
- Three archived fine search CSVs.

Each set covers LDA, NMF, and BERTopic.

### Final tabular artifacts

- dataset statistics;
- model/topic/noise statistics;
- runtime statistics;
- aggregate recommendation metrics;
- compact top-K recommendations;
- complete rankings;
- three topic-keyword tables.

### Figure artifacts

- six PDF files;
- six corresponding PNG files.

### Artifacts not present

- fitted model files;
- serialized vectorizers;
- SPECTER embedding arrays;
- UMAP/HDBSCAN state;
- publication vectors;
- researcher/group profile vectors;
- explicit split membership files;
- environment or dependency manifest;
- final execution log or run metadata.

## 16. Experimental Notes

- All three models use the same researcher split, profile aggregation, candidate groups, cosine-ranking code, and evaluation functions.
- Parameter selection uses only training publications and only C_v coherence.
- Final evaluation is performed once on one fixed researcher-level test partition.
- The recommendation task is multi-label; every known group membership is relevant.
- Group profiles contain available training members only.
- Metric aggregation gives each test researcher equal weight.
- LDA and NMF coherence and fitting use cleaned bag-of-words text. BERTopic fitting uses original title-plus-abstract text, while BERTopic coherence is evaluated against cleaned training text.
- NMF uses TF-IDF internally; there is no standalone TF-IDF experimental model.
- BERTopic embedding-based outlier reduction is fixed in the current shared fitting function rather than searched.
- Current BERTopic parameter search excludes the one-time SPECTER encoding cost from per-candidate runtimes; final BERTopic training time includes embedding computation.
- Final training runtimes include coherence and profile construction, so they are broader than estimator `fit()` time.
- No model persistence is implemented; reproducing recommendations requires fitting again.
- `config.yaml` currently matches the current selected configuration, but the exact final invocation is not logged.
- Archived coarse/fine artifacts are useful audit evidence, but their generating source state cannot be fully reconstructed from the current parameter-selection constants.
