# Implementation Audit

## Audit basis and boundaries

This document describes the repository at commit `aaf2fad` by tracing the checked-in Python and YAML implementation. It does not reconstruct behavior from the reported results when the code does not establish that behavior. The two acquisition utilities are treated as historical/audit context; they were not executed during this audit.

Two repository inconsistencies affect the traceability of data acquisition:

- The acquisition utilities are physically stored as `src/fetch_csrankings_data.py` and `src/regenerate_group_labels.py`, although their documentation and imports refer to a `scripts` package.
- The acquisition code writes `publications.csv` and `groups.csv` under a configurable output directory. The experiment reads `data/publications_2015_2025_10k_plus.csv` and `data/groups_2015_2025_10k_plus.csv`. No checked-in function performs the rename or copy between these names.

Consequently, the acquisition logic can be documented, but the exact operation that produced the two current experiment CSVs cannot be verified solely from this repository.

## 1. Project Overview

The implemented objective is to recommend one or more research groups for a researcher from that researcher's publication titles and abstracts. The experiment compares three topic representations—LDA, NMF, and BERTopic—inside the same profile-construction, cosine-ranking, and evaluation pipeline.

Important directories and files:

- `data/` contains the publication and researcher-to-group input CSVs.
- `src/` contains acquisition utilities, preprocessing, splitting, model wrappers, profile construction, ranking, evaluation, experiment orchestration, parameter selection, and visualization.
- `results/` contains parameter-selection tables, experiment tables, topic exports, and figures.
- `config.yaml` contains the main experiment settings.
- `README.md` describes dependencies and command-line entry points. Its final figure-name examples are stale relative to the current visualization implementation.

The main runtime entry points are:

- `src/parameter_selection.py`: searches model parameters using training-corpus topic coherence.
- `src/experiment.py`: runs the three-model recommendation experiment.
- `src/visualization.py`: creates figures from data and result CSVs.
- `src/fetch_csrankings_data.py` and `src/regenerate_group_labels.py`: historical acquisition/label-generation utilities, separate from the current experiment execution.

No automated test suite or dependency lock/requirements file is present in the inspected repository.

## 2. Dataset Acquisition

### 2.1 Historical acquisition pipeline

`src/fetch_csrankings_data.py` implements a seven-stage acquisition pipeline in `run_pilot()`:

1. Load CSRankings `generated-author-info.csv` with `load_generated_author_info()`.
2. Select researchers with `select_researchers()`.
3. Fetch their publication records from DBLP with `fetch_all_dblp_papers()` and `fetch_dblp_papers()`.
4. Deduplicate records with `deduplicate_papers()`.
5. retrieve abstracts from Semantic Scholar with `fetch_s2_abstracts()`.
6. Merge records and remove records without a sufficiently long abstract using `merge_and_filter()`.
7. Write publication data, group labels, and an acquisition log.

The default CLI downloads `generated-author-info.csv` from the CSRankings GitHub repository when it is absent. `load_generated_author_info()` expects fields including `name`, `area`, `count`, `adjustedcount`, `year`, and optionally `dept`.

### 2.2 Researcher and area selection

`AREA_VENUES` defines 12 named research areas and their CSRankings venue codes. `VENUE_TO_AREA` reverses that mapping. `select_researchers()`:

- filters author-information rows to the requested year range and target venues;
- sums `adjustedcount` by researcher and mapped research area;
- retains researchers whose summed adjusted count is at least `min_papers`;
- sorts descending by total score; and
- optionally limits the list to `max_researchers`.

The defaults are a 2015–2025 range, all defined target areas, at most 500 researchers, and a minimum adjusted count of 5. These are acquisition-script defaults, not evidence of the exact settings used for the checked-in experiment CSVs.

### 2.3 DBLP collection

`fetch_dblp_papers()` queries the DBLP publication-search API once per selected researcher. It:

- removes a trailing four-digit DBLP disambiguation suffix from the query name;
- requests up to 1,000 records;
- filters by the configured year interval;
- maps returned venue text to a target venue code through `_match_venue()`;
- extracts title, semicolon-separated authors, year, venue, DOI, and DBLP URL.

`fetch_all_dblp_papers()` checkpoints `dblp_papers.json` and `dblp_progress.json`, resumes completed researchers, and preserves failed researchers for later retry. It checkpoints every ten researchers and at completion.

`deduplicate_papers()` uses lowercased DOI when present; otherwise it uses normalized title plus year. DOI-bearing records and DOI-less records are tracked in separate key sets, so the code does not prove that a DOI-bearing record and an otherwise identical DOI-less record will be merged.

### 2.4 Semantic Scholar enrichment

`fetch_s2_abstracts()`:

- prefers Semantic Scholar DOI batch lookup;
- uses individual title searches for records without a DOI;
- requests only the fields needed for title/abstract retrieval;
- supports an explicit API key or the `S2_API_KEY` / `SEMANTIC_SCHOLAR_API_KEY` environment variables;
- applies request delays, jitter, retry handling, and rate-limit tracking; and
- checkpoints responses in `s2_abstracts.json`.

`merge_and_filter()` discards records with no abstract or an abstract shorter than `MIN_ABSTRACT_LENGTH` (100 characters by default). Accepted records become `EnrichedPaper` objects.

### 2.5 Acquisition outputs and labels

`generate_publications_csv()` writes:

`Title, Abstract, Authors, Year, Venue, DOI`

`generate_groups_csv()` writes:

`researcher_name, group_name`

Group names are CSRankings area labels. A researcher can receive multiple rows/labels. Only selected researchers who appear among authors of retained enriched publications are written. `run_pilot()` also writes `acquisition_log.json` containing counts, errors, rate-limit events, group distribution, timestamps, and duration.

The CLI defaults its output to `data/pilot_2021_2025`, despite its default year parameters being 2015–2025. The directory name is therefore not a reliable statement of the actual configured year range.

### 2.6 Group-label regeneration utility

`src/regenerate_group_labels.py` is intended to rebuild group labels without reacquiring publications. `publication_authors()` extracts all semicolon-separated authors from an existing publication CSV. `main()` then reloads CSRankings author information and calls `generate_groups_csv()` for those authors and the requested year/area settings.

As checked in, this utility imports `scripts.fetch_csrankings_data` and `utils.config`, but neither a `scripts` package nor `utils/config.py` exists in this repository. Therefore, its intended logic is visible, but the current file is not directly runnable in this repository without external or missing modules. No execution was attempted.

### 2.7 Boundary with the experiment

The current experiment inputs have the same column schemas as the acquisition outputs but different names:

- `data/publications_2015_2025_10k_plus.csv`
- `data/groups_2015_2025_10k_plus.csv`

There is no checked-in orchestrator that invokes acquisition before the experiment. Acquisition, preprocessing, and experiment execution are separate operations.

## 3. Data Loading

`src/preprocessing.py:load_datasets()` is the experiment loader. It reads the two fixed filenames above with pandas.

Required publication columns are defined by `REQUIRED_PUBLICATION_COLUMNS`:

- `Title`
- `Abstract`
- `Authors`

Required group columns are defined by `REQUIRED_GROUP_COLUMNS`:

- `researcher_name`
- `group_name`

Missing required columns raise `ValueError`. Although the current publication CSV also contains `Year`, `Venue`, and `DOI`, the experiment loader does not use or validate them.

Group values are converted to strings, stripped, filtered when empty, deduplicated across the two columns, and reindexed. Publication authors are accepted only when their stripped name exactly matches a cleaned `researcher_name` in the group table.

Missing titles or abstracts are converted to an empty string independently. A record remains usable if the resulting title-plus-abstract document and its bag-of-words version are both nonempty. Rows with missing `Authors` are skipped. If no usable author-publication records remain, the loader raises `ValueError`.

## 4. Data Preprocessing

`load_datasets()` creates two text views for every matched author-publication association:

- `document`: whitespace-normalized original-case title plus abstract.
- `bow_document`: the bag-of-words view used by LDA, NMF, and coherence calculation.

`clean_text()` collapses all whitespace. `preprocess_bag_of_words()`:

- lowercases;
- replaces characters outside `[a-z0-9 whitespace]` with spaces;
- tokenizes by whitespace;
- removes NLTK English stopwords and `RESEARCH_JARGON_STOPWORDS`; and
- rejoins retained tokens with spaces.

The explicit research-jargon list contains `abstract`, `paper`, `article`, `proceedings`, `conference`, `journal`, `author`, `contributors`, `manuscript`, and `review`. The stopword set is cached with `lru_cache`.

The implementation does not perform stemming, lemmatization, phrase detection, or spelling normalization. Digits are retained.

The semicolon-separated `Authors` field is exploded: one prepared row is created for each matched known researcher. Consequently, one publication document can occur multiple times when multiple labelled coauthors occur in its author list. The returned publication frame has only `researcher_name`, `document`, and `bow_document`.

## 5. Dataset Splitting

`src/splitter.py:split_researchers()` performs a researcher-level random split:

- names are deduplicated and sorted;
- `train_test_split()` receives `train_size=train_ratio`;
- `random_state=random_seed` controls the split;
- outputs are converted to disjoint sets.

It rejects ratios outside `(0, 1)` and datasets with fewer than two unique researchers. There is no stratification by group, publication count, or multi-label membership.

In both `src/experiment.py:main()` and `src/parameter_selection.py:training_publications()`, the split population is the intersection of researchers present in the prepared publication frame and group frame. With `config.yaml`, the split is 80% training and 20% testing with seed 42.

`publications_for_researchers()` filters all exploded publication rows by researcher membership, so all rows associated with one researcher remain in the same split.

Parameter selection reconstructs this same seeded split and uses only training-researcher publications. The final experiment performs the same split again.

## 6. Topic Modeling Pipeline

### 6.1 LDA

`src/lda_model.py:fit_lda()` fits a scikit-learn `CountVectorizer` on training `bow_document` values. It uses `max_features`, `min_df`, and `max_df` from the preprocessing configuration.

The count matrix is passed to `LatentDirichletAllocation` with:

- `n_components=config["lda"]["num_topics"]`;
- `max_iter=config["lda"]["max_iter"]`;
- `learning_method="batch"`; and
- `random_state=config["random_seed"]`.

`topic_vectors()` applies the fitted vectorizer and LDA transform to documents. `topic_keywords()` returns the ten highest-weight vocabulary terms per component by default. `LdaModel` stores the fitted estimator and vectorizer.

### 6.2 NMF

`src/nmf_model.py:fit_nmf()` fits a scikit-learn `TfidfVectorizer` on training `bow_document` values, using the same three preprocessing limits as LDA.

The TF-IDF matrix is passed to `NMF` with:

- `n_components=config["nmf"]["num_topics"]`;
- `init="nndsvda"`;
- `max_iter=config["nmf"]["max_iter"]`; and
- `random_state=config["random_seed"]`.

`topic_vectors()` transforms documents through the fitted TF-IDF vectorizer and NMF model. `topic_keywords()` returns the ten highest-weight terms per component. The TF-IDF operation here is internal NMF preprocessing, not a standalone experimental baseline.

### 6.3 BERTopic

`src/bertopic_model.py:fit_bertopic()` constructs:

- a `SentenceTransformer` from `embedding_model` unless one is supplied;
- UMAP with configured neighbors, components, minimum distance, cosine metric, and random seed;
- HDBSCAN with `min_cluster_size=min_topic_size` and `prediction_data=True`;
- a `CountVectorizer` using the sorted shared stopword list.

These components are supplied to `BERTopic` with `calculate_probabilities=False`.

If embeddings are not supplied, SPECTER embeddings are computed once for the input documents. The model is fitted with `model.fit(documents, embeddings=embeddings)`. The fixed post-fit sequence is:

1. `reduce_outliers(..., strategy="embeddings", embeddings=embeddings)`;
2. `update_topics(..., topics=reduced_topics, vectorizer_model=vectorizer_model)`.

Thus outlier reduction is not a searched parameter. The same embedding matrix is reused for fitting and embedding-based reassignment.

`topic_distributions()` calls BERTopic's `approximate_distribution()` and returns its distribution matrix as floating-point NumPy data. This method is used for both training and testing publication representations after fitting. The repository delegates the distribution algorithm to BERTopic; it does not reimplement it.

`topic_keywords()` returns non-noise topic IDs and up to ten terms from `model.get_topics()`. `noise_statistics()` counts `-1` values in `model.topics_`.

### 6.4 Topic coherence

`src/topic_utils.py:coherence_cv()` splits each preprocessed training document on whitespace, builds a gensim `Dictionary`, and constructs `CoherenceModel` with `coherence="c_v"` and `processes=1`.

LDA and NMF keywords and documents are both derived from the bag-of-words text. BERTopic keywords are evaluated against the same bag-of-words training documents, although BERTopic fitting receives the original-text `document` values.

`config.yaml` contains `preprocessing.ngram_range`, but none of the checked-in vectorizer constructors passes this value. It therefore does not affect the current implementation.

## 7. Researcher Profile Construction

`src/profile_builder.py:average_researcher_profiles()` groups publication-row indices by `researcher_name`. For each researcher, it takes the arithmetic mean of all corresponding publication vectors along the document axis and stores a flattened NumPy array in:

`dict[str, numpy.ndarray]`

The function supports dense and SciPy sparse input and verifies that the publication row count equals the vector-matrix row count.

Training researcher profiles are created from training publication vectors. Testing researcher profiles are created separately from test publication vectors after the model has been fitted on training data.

## 8. Research Group Profile Construction

`src/profile_builder.py:build_group_profiles()` groups the full cleaned membership table by `group_name`. For each group, it collects profiles only for names present in the supplied researcher-profile dictionary and averages them with `numpy.mean`.

During the experiment, the supplied profiles are training-researcher profiles. Therefore, each group profile is the mean of its available training members, even though the function receives the full membership table. A group with no training member profile is omitted. If every group is omitted, the function raises `ValueError`.

The result is:

`dict[group_name, numpy.ndarray]`

`ground_truth_groups()` independently maps every researcher in the cleaned membership table to a set of all known group memberships.

## 9. Recommendation Pipeline

`src/recommender.py:rank_groups()` receives testing researcher profiles and training-derived group profiles.

The implementation sequence is:

1. Sort group names alphabetically.
2. Stack group vectors into one matrix.
3. For each testing researcher, calculate scikit-learn cosine similarity between the researcher vector and every group vector.
4. Sort pairs by descending similarity.
5. Resolve equal scores by ascending group name.
6. Store the complete ordered list of `(group_name, float_score)`.

The return value is:

`dict[researcher_name, list[tuple[group_name, similarity]]]`

`src/evaluator.py:recommendations_frame()` converts this to one row per model/researcher with ground-truth groups and top-1, top-3, and top-5 group names and scores. Names and scores are pipe-separated strings; scores are formatted to six decimal places.

`rankings_frame()` exports every candidate group for every model/researcher, with one-based rank, raw similarity, and a Boolean relevance label.

## 10. Evaluation Pipeline

`src/evaluator.py:evaluate_rankings()` evaluates each testing researcher's complete ranking against `ground_truth_groups()`.

Metrics from `src/metrics.py` are:

- Precision@K: relevant groups in the first K divided by K.
- Recall@K: relevant groups in the first K divided by the number of relevant groups.
- F1@K: harmonic mean of Precision@K and Recall@K.
- MRR: reciprocal rank of the first relevant group, or zero.
- NDCG@K: binary-relevance DCG divided by the ideal DCG for `min(number of relevant groups, K)`.

Precision, Recall, F1, and NDCG are computed at K = 1, 3, and 5. Each metric is averaged arithmetically across ranked testing researchers, giving each researcher equal weight. No confidence intervals, significance tests, or repeated splits are implemented.

Evaluation occurs after a model has produced all test rankings. It does not influence model fitting or parameter selection.

## 11. Experiment Pipeline

`src/experiment.py` is the primary experiment entry point. Its CLI accepts only `--config`, defaulting to the root `config.yaml`.

`main()` performs:

1. Load YAML with `load_config()`.
2. Seed Python, NumPy, and, when installed, PyTorch CPU/CUDA RNGs with `set_random_seed()`.
3. Load and prepare datasets.
4. Construct and perform the researcher-level split.
5. Build the complete researcher-to-group truth map.
6. Run `run_model()` sequentially for `lda`, `nmf`, and `bertopic`.
7. Evaluate rankings and construct output tables.
8. Write result CSVs.

Within `run_model()`, the measured training phase includes model fitting, training-document transformation, keyword/topic extraction, coherence calculation, researcher-profile construction, and group-profile construction. The measured recommendation/inference phase includes test-document transformation, test-profile construction, and cosine ranking. Data loading, splitting, metric evaluation, and CSV writing occur outside these timers.

Outputs written under `config["output_dir"]` are:

- `dataset_statistics.csv`
- `model_statistics.csv`
- `runtime_statistics.csv`
- `metrics.csv`
- `recommendations.csv`
- `rankings.csv`
- `topics/lda_topics.csv`
- `topics/nmf_topics.csv`
- `topics/bertopic_topics.csv`

The topic files contain `topic_id` and pipe-separated `top_keywords`.

### Parameter selection

`src/parameter_selection.py` is a separate entry point with `--config` and `--model {lda,nmf,bertopic,all}`.

Current hard-coded candidate sets are:

- LDA topics: 19, 21, 29, 31.
- NMF topics: 9, 11, 12, 15.
- BERTopic minimum topic size: 40, 45, 50.
- BERTopic UMAP neighbors: 50, 55, 60.

Each candidate is fitted only to training-researcher publications and ranked by C_v coherence. Runtime and exception text are recorded. BERTopic loads SPECTER and encodes the training original-text documents once, then reuses the same embeddings and embedding model for all nine candidate combinations.

`rank_results()` orders valid candidates by decreasing coherence, with missing values last. `save_results()` writes `<model>_search.csv` and updates only the selected model fields in `best_config.yaml`.

When `--model all` is used, `best_config.yaml` is first reset from the input config. A selected configuration is not automatically used by `experiment.py`; it must be passed explicitly, for example with `--config results/parameter_selection/best_config.yaml`.

The `coarse_search/` and `fine_search/` result directories are stored artifacts. The current parameter-selection entry point writes only directly to `results/parameter_selection/` and does not create those historical subdirectories.

## 12. Visualization Pipeline

`src/visualization.py` uses the noninteractive Matplotlib `Agg` backend. `main()` calls `configure_style()` and conditionally generates figures when required source files exist. `save_figure()` writes both PDF and 300-DPI PNG to `results/figures/`.

Current figures and sources are:

- `dataset_characterization`: calls `load_datasets()` and shows group member counts, prepared publication rows per researcher, and groups per researcher.
- `hyperparameter_search`: reads the three current `results/parameter_selection/*_search.csv` files.
- `model_quality_and_runtime`: reads `model_statistics.csv` and `runtime_statistics.csv`; missing coherence values can fall back to the maximum valid coherence in a model search CSV.
- `recommendation_performance`: reads `metrics.csv` and plots Precision, Recall, F1, and NDCG across K plus scalar MRR.
- `per_group_performance`: reads `rankings.csv`, computes mean relevance among ranks 1–3 by group/model, and sorts rows by relevant-researcher support.
- `rank_distribution`: reads `rankings.csv` and plots the cumulative fraction of relevant ranking rows at cutoffs 1 through 12.

Only LDA, NMF, and BERTopic rows are retained in plots. Model colors, markers, and line styles are defined centrally. The current automatic figure set has six PDF/PNG pairs.

## 13. Configuration

### Root YAML

`config.yaml` defines:

- `random_seed: 42`
- `train_ratio: 0.8`
- preprocessing: `max_features=20000`, `ngram_range=[1,1]`, `min_df=2`, `max_df=0.8`
- LDA: 29 topics, 20 maximum iterations
- NMF: 11 topics, 800 maximum iterations
- BERTopic: `allenai/specter`, minimum topic size 45, UMAP neighbors 50, components 5, minimum distance 0.0
- `output_dir: results`

The experiment and parameter-selection modules load YAML independently with `yaml.safe_load()`. They do not apply a schema or merge defaults; missing keys will fail when accessed.

### Constants

Preprocessing stopwords, parameter-search candidates, acquisition areas/venues, acquisition rate limits, year ranges, batch sizes, abstract-length threshold, model plotting order, and plotting styles are module-level constants.

### CLI and environment configuration

- `experiment.py`: `--config`.
- `parameter_selection.py`: `--config`, `--model`.
- `fetch_csrankings_data.py`: acquisition stage, directories, areas, researcher limit, minimum papers, year bounds, Semantic Scholar credentials/rate controls, batch size, and cache clearing.
- `regenerate_group_labels.py`: publication CSV, author-info CSV, output CSV, year bounds, and target areas, although its current imports are unresolved in this repository.
- Semantic Scholar credentials can also come from `S2_API_KEY` or `SEMANTIC_SCHOLAR_API_KEY`.
- `visualization.py` has no CLI arguments.

## 14. File Dependency Map

| File | Primary responsibility | Depends on | Used by |
|---|---|---|---|
| `src/fetch_csrankings_data.py` | Historical CSRankings/DBLP/Semantic Scholar acquisition | Python standard library, external HTTP services | Intended group-label utility; not used by experiment |
| `src/regenerate_group_labels.py` | Rebuild group labels from author metadata | Intended `scripts.fetch_csrankings_data`, missing `utils.config` | Standalone historical utility |
| `src/preprocessing.py` | Load, validate, clean, and explode input data | pandas, NLTK | Experiment, parameter selection, visualization |
| `src/splitter.py` | Researcher-level train/test split and row filtering | pandas, scikit-learn | Experiment, parameter selection |
| `src/lda_model.py` | Count-vector LDA fit/transform/keywords | NumPy, scikit-learn | Experiment, parameter selection |
| `src/nmf_model.py` | TF-IDF/NMF fit/transform/keywords | NumPy, scikit-learn | Experiment, parameter selection |
| `src/bertopic_model.py` | SPECTER/UMAP/HDBSCAN/BERTopic and outlier reduction | BERTopic stack, preprocessing stopwords | Experiment; imported lazily by BERTopic selection |
| `src/topic_utils.py` | Topic export table and C_v coherence | pandas, gensim | Experiment, parameter selection |
| `src/profile_builder.py` | Researcher profiles, group profiles, truth sets | NumPy, pandas, SciPy | Experiment |
| `src/recommender.py` | Cosine ranking | NumPy, scikit-learn | Experiment |
| `src/metrics.py` | Individual ranking metrics | Python standard library | Evaluator |
| `src/evaluator.py` | Metric aggregation and output-table construction | pandas, metrics | Experiment |
| `src/parameter_selection.py` | Training-only coherence search and best-config export | Models, preprocessing, splitter, topic utilities | Standalone entry point; outputs consumed manually by experiment and by visualization |
| `src/experiment.py` | End-to-end three-model experiment orchestration | All core modeling/recommendation modules, YAML | Standalone entry point; produces visualization inputs |
| `src/visualization.py` | Figure generation | pandas, NumPy, Matplotlib, preprocessing | Standalone final stage |

## 15. End-to-End Execution Flow

The verified current flow is:

```text
Historical acquisition utility (separate, not called by experiment)
CSRankings generated-author-info.csv
        ↓
Researcher/area selection
        ↓
DBLP publication metadata
        ↓
Deduplication
        ↓
Semantic Scholar abstract enrichment
        ↓
publications.csv + groups.csv + acquisition_log.json
        ↓
[unimplemented/undocumented rename or transfer step]
        ↓
data/publications_2015_2025_10k_plus.csv
data/groups_2015_2025_10k_plus.csv
        ↓
load_datasets(): validation, author matching, dual text views
        ↓
Researcher-level 80/20 split
        ├──────────────→ Parameter selection on training publications
        │                 ↓
        │               coherence-ranked search CSVs + best_config.yaml
        │                 ↓
        │               explicit --config selection by user
        ↓
LDA / NMF / BERTopic fitting on training publications
        ↓
Training publication vectors
        ↓
Mean training researcher profiles
        ↓
Mean group profiles from available training members
        ↓
Testing publication vectors
        ↓
Mean testing researcher profiles
        ↓
Cosine similarity to every available group profile
        ↓
Complete ranked group lists
        ↓
Precision/Recall/F1/NDCG at 1,3,5 + MRR
        ↓
Result CSVs and topic tables
        ↓
Visualization reads CSVs and writes PDF/PNG figures
```

## 16. Implementation Notes

- The same researcher split procedure and seed are used by parameter selection and the final experiment.
- Parameter selection is based only on training-corpus C_v coherence; recommendation metrics are calculated only by the experiment.
- Group profiles use only training researchers, while relevance labels come from the full cleaned membership table.
- Multi-label memberships are preserved as sets and used by every metric.
- A publication may be duplicated in the modeling corpus across matched coauthors because preprocessing creates author-publication rows.
- LDA and NMF use the cleaned bag-of-words view; BERTopic embeddings use original title-plus-abstract text. BERTopic's topic-word vectorizer uses the shared stopwords.
- NMF's TF-IDF vectorizer is an internal part of NMF, not an experimental TF-IDF baseline.
- BERTopic outlier reduction is a fixed embedding-based post-fit step and is used during both candidate fitting and final experiment fitting through the shared `fit_bertopic()` function.
- BERTopic parameter selection reuses one SPECTER embedding matrix across candidates. Final experiment fitting computes an embedding matrix inside `fit_bertopic()` because none is supplied by `experiment.py`.
- The root `ngram_range` setting is currently unused by the model constructors.
- Coherence computation is included in the recorded training time; metric evaluation and file writing are not.
- The experiment does not serialize fitted models or profile vectors.
- There are no repeated random splits, cross-validation, significance tests, confidence intervals, or explicit group-stratification mechanisms in the checked-in code.
- Current acquisition-to-experiment provenance is incomplete at the filename/transfer boundary, and the group-label regeneration utility has unresolved imports in its checked-in location. These are traceability observations, not claims about how the dataset was actually produced outside the repository.
