## 3 Methodology

This study developed a BERTopic-based framework for researcher-to-research-group recommendation and compared it with LDA and NMF under one experimental pipeline. As shown in Fig. 1, the pipeline covered data collection, data preprocessing, hyperparameter optimization, topic modeling, profile construction, group recommendation, and evaluation. Publication-level topic vectors were averaged into researcher profiles, while the profiles of training researchers formed the research group profiles. Cosine similarity between these profiles produced the ranked recommendations.

### 3.1 Data Sources

The researchers and research group labels were obtained from CSRankings. Publication metadata were collected from DBLP, and abstracts were retrieved from Semantic Scholar. Each record contained a title, abstract, author list, publication year, venue, and Digital Object Identifier when available. The CSRankings research areas defined 12 target research groups. After data validation and preprocessing, the dataset contained 11,111 publication documents from 3,257 researchers. The researchers were divided into 2,605 training researchers and 652 test researchers using an 80/20 researcher-level split with random seed 42. This split kept every publication by one researcher in the same partition.

### 3.2 Data Preprocessing

Each publication document combined its title and abstract. Records without usable text or a labeled researcher were excluded. Two text representations were prepared. LDA and NMF used bag-of-words text that was converted to lowercase and stripped of punctuation, non-alphanumeric tokens, common English stopwords, and generic publication terms. BERTopic used the original title-and-abstract text with normalized whitespace to preserve the information required by the embedding model. The vectorizers were fitted only on training publications. Their vocabulary was limited to 20,000 features, with minimum and maximum document frequencies of 2 and 0.8, respectively.

### 3.3 Hyperparameter Optimization

Hyperparameters were selected using only the training corpus. Candidate configurations were ranked by $C_v$ topic coherence, and the highest-scoring configuration for each model was used in the final experiment. The search varied the number of topics for LDA and NMF. For BERTopic, it evaluated combinations of the HDBSCAN minimum topic size and the number of UMAP neighbors. The selected settings were 29 topics and 20 iterations for LDA, and 11 topics and 800 iterations for NMF. BERTopic used a minimum topic size of 45, 50 UMAP neighbors, five UMAP components, and a minimum distance of 0.0. The optimization criterion was used only for model selection; the final recommendation metrics were calculated on the held-out researchers. All models used random seed 42 where supported.

### 3.4 Topic Modeling

LDA, NMF, and BERTopic were fitted to the same training partition. Each model produced a topic vector for every training and test publication.

LDA fitted a probabilistic topic model to document-term count vectors with batch learning. Its output represented each publication as a probability distribution over 29 latent topics. NMF decomposed a TF-IDF document-term matrix into non-negative document-topic and topic-term matrices. It used NNDSVDa initialization, and the resulting 11-dimensional document-topic vector represented each publication. The fitted training vectorizers transformed the test publications without refitting.

BERTopic encoded the original publication text with the pretrained `allenai/specter` model. UMAP reduced the SPECTER embeddings using cosine distance, and HDBSCAN formed the topic clusters. The model then reassigned outliers through embedding similarity and updated the topics with the same vectorizer to keep their representations consistent. BERTopic's `approximate_distribution()` function generated a document-topic probability distribution for each publication. The recommendation stage used these distributions rather than single topic labels.

### 3.5 Profile Construction

All profiles were constructed in the topic space produced by the corresponding model. Let $D_r=\{d_1,d_2,\ldots,d_n\}$ denote the publications of researcher $r$, and let $v_d$ be the topic vector of publication $d$. The researcher profile was the arithmetic mean of the publication vectors:

$$
R_r=\frac{1}{|D_r|}\sum_{d\in D_r}v_d. \tag{1}
$$

This operation produced one topic profile for each researcher. Research group profiles were built only from training researchers. For a research group with training members $G=\{r_1,r_2,\ldots,r_m\}$, the group profile was

$$
P_G=\frac{1}{|G|}\sum_{r\in G}R_r, \tag{2}
$$

where $R_r$ was the profile of member $r$. Restricting this aggregation to training members prevented test researcher profiles from entering the group representations.

### 3.6 Research Group Recommendation

Each test researcher profile was compared with all 12 research group profiles using cosine similarity. The groups were sorted by decreasing similarity, with an alphabetical secondary order for equal scores. The same ranking procedure was applied to LDA, NMF, and BERTopic profiles, so the topic representation was the only model-dependent part of the recommendation pipeline. The system returned a ranked list rather than a single group because a researcher could have more than one ground-truth research group.

### 3.7 Evaluation

The ranked groups for each of the 652 test researchers were compared with that researcher's known CSRankings group memberships. Precision@$K$, Recall@$K$, F1-score@$K$, and normalized discounted cumulative gain (NDCG@$K$) were calculated at $K=1$, 3, and 5. Precision measured the proportion of the first $K$ recommendations that were relevant. Recall measured the proportion of a researcher's relevant groups retrieved within the first $K$ positions, and F1-score was their harmonic mean. NDCG assigned greater value to relevant groups placed near the top of the ranking. Mean reciprocal rank (MRR) measured the inverse rank of the first relevant group. Each reported metric was the arithmetic mean of the per-researcher scores. This protocol used the same split, group candidates, profile construction, ranking method, and metrics for all three topic models.
