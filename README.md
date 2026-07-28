# Researcher-to-Research-Group Recommendation

This project compares LDA, NMF, and BERTopic using `allenai/specter`
embeddings to recommend research groups from publication titles and abstracts.

## Method

- `Authors` in the publications dataset is split on semicolons.
- Only authors listed in `groups_2015_2025_10k_plus.csv` are modelled because
  they have known group memberships.
- Researchers, rather than individual publications, are split 80/20. All of a
  researcher's publications remain in one split.
- Every researcher profile is the mean of that researcher's publication vectors.
  Every group profile is the mean of its training members' profiles.
- All models use cosine similarity. Researchers may have multiple relevant groups.
- LDA and NMF use lowercase alphanumeric text with NLTK English and
  research-jargon stopwords removed. BERTopic receives original title-plus-abstract
  text for SPECTER embeddings and applies the same stopwords only to c-TF-IDF topics.

## Run

Use Python 3.11+ with `pandas`, `numpy`, `scikit-learn`, `gensim`,
`nltk` (with the English stopword corpus), `sentence-transformers`, `bertopic`,
`scipy`, `pyyaml`, and `matplotlib`.

```powershell
python src/parameter_selection.py
python src/experiment.py
python src/visualization.py
```

The first BERTopic run may download `allenai/specter` if it is not already cached.

Parameter selection uses only the training corpus and writes coherence-based search
results to `results/parameter_selection/`. To use the selected settings in the
main experiment, run:

```powershell
python src/experiment.py --config results/parameter_selection/best_config.yaml
```

## Outputs

- `results/dataset_statistics.csv` — dataset and researcher-split summary.
- `results/model_statistics.csv` — topic counts, C_v coherence,
  and BERTopic noise statistics.
- `results/runtime_statistics.csv` — training, recommendation/inference, and total
  runtime per representation.
- `results/metrics.csv` — Precision@k, Recall@k, F1@k, MRR, and NDCG@k.
- `results/recommendations.csv` — top-1, top-3, and top-5 recommendations and scores.
- `results/rankings.csv` — complete group rankings for each testing researcher.
- `results/topics/` — ten representative keywords per LDA, NMF, and BERTopic topic.
- `results/figures/retrieval_performance.png` and `ranking_quality.png` —
  performance-focused publication figures.

Configure random seed, split ratio, and representation parameters in `config.yaml`.
