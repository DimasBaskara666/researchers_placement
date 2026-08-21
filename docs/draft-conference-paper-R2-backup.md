A Comparative Evaluation of Topic Modeling Methods for Researcher-to-Research-Group Recommendation

*Note: Sub-titles are not captured in Xplore and should not be used

***Abstract—*** **Matching researchers with suitable research groups can support collaboration and researcher placement, but manual matching is difficult to apply at scale. Publication text provides evidence of research expertise, yet the relative value of classical and contextual topic models for this recommendation task remains unclear. This study evaluated a BERTopic-based researcher-to-research-group recommendation framework against Latent Dirichlet Allocation (LDA) and Non-negative Matrix Factorization (NMF) under the same experimental pipeline. The dataset contained 11,111 publication documents from 3,257 researchers across 12 computer science research groups. Topic vectors from publication titles and abstracts were averaged to form researcher and group profiles, and cosine similarity was used to rank candidate groups. The models were evaluated on 652 test researchers using top-** ***K*** **ranking metrics. BERTopic ranked first in all 13 recommendation metrics, with a normalized discounted cumulative gain at five recommendations of 0.736175 and a mean reciprocal rank of 0.809277. NMF obtained the highest topic coherence score of 0.686782 but ranked last on every recommendation metric. It required only 18.8 s of total execution time, compared with 971.8 s for BERTopic. The findings show that topic coherence does not reliably predict recommendation quality. BERTopic is preferable when ranking accuracy has priority, while LDA and NMF provide lower-cost options for resource-limited or frequently updated systems.**

**Keywords—BERTopic, scholarly recommender systems, researcher profiling, scientific document analysis, ranking evaluation**

## 1 Introduction

Research groups support scientific collaboration, mentoring, and the sharing of expertise and resources [1]. Placing researchers in groups that match their research interests can support collaboration and institutional planning [1], [2]. This matching is often performed manually, which makes the process difficult to scale and sensitive to subjective judgment [3]. Scholarly recommender systems have addressed related tasks through content-based and interaction-based methods [3], [4]. Interaction data, such as co-authorship and citation links, may be sparse for new researchers. Publication titles and abstracts provide a direct source of evidence about current research interests and can support recommendations without requiring prior interaction records [4], [5].

Topic modeling has been used to convert publication text into compact thematic representations. LDA models documents as probability distributions over latent topics [6], [7], while NMF derives document-topic factors from non-negative text matrices [8]. BERTopic combines contextual document embeddings, dimensionality reduction, density-based clustering, and class-based TF-IDF [9]. These models offer different ways to represent researcher expertise. Previous studies have compared topic modeling methods on social media, online communities, news, and other short-text collections [10], [11], [12]. Most comparisons, however, have assessed topic interpretability or clustering behavior rather than the value of topic vectors inside the same downstream recommendation task.

This limitation creates two related gaps. First, existing studies have not established how LDA, NMF, and BERTopic compare when they use identical profile construction, similarity ranking, and researcher-level evaluation for researcher-to-research-group recommendation. Second, model selection has often relied on intrinsic topic coherence, although prior work has questioned whether higher coherence predicts better downstream performance [13], [14]. The relationship between coherence and research group ranking quality remains unclear because the two objectives measure different properties of a topic representation.

This study addresses these gaps through a comparative evaluation of LDA, NMF, and BERTopic for publication-based researcher-to-research-group recommendation. It evaluates 11,111 prepared publication documents from 3,257 researchers across 12 research groups. The framework uses a researcher-level train-test split and constructs group profiles only from training researchers. The main contributions are:

1. A comparative evaluation of probabilistic, matrix-factorization, and contextual topic representations under the same profile construction, cosine-similarity ranking, and test protocol.
2. A standardized researcher-level evaluation procedure in which test researchers remain separate from model fitting and group-profile construction.
3. An analysis of the relationship between  $C_{v}$ topic coherence and 13 recommendation metrics, including Precision, Recall, F1-score, NDCG, and MRR.
## 2 Related Works

Scholarly recommendation has covered papers, experts, collaborators, reviewers, venues, and research groups. Expert recommendation studies have combined researcher profiles with multilevel information mining and large-scale processing [1], [2]. Surveys of scholarly and paper recommender systems have grouped existing methods into content-based, collaborative, graph-based, and hybrid approaches [3], [4], [5]. Broader recommender-system research has reported the same dependence on user-item interactions and content representations [15]. Interaction-based methods can represent collaboration patterns, but they are less useful when historical links are unavailable. Publication-based profiling avoids this dependency by representing expertise from the researcher's own work. Semantic Academic Profiler, for example, has used semantic topic modeling to assess researcher expertise from scholarly content [16]. These studies support publication-based researcher representation, but they have not directly compared several topic models for ranking research groups under one evaluation pipeline.

Topic modeling provides one family of content representations for scholarly text. LDA has remained a common probabilistic model for discovering latent themes [6], [7], while NMF has represented documents through non-negative matrix decomposition [8]. Contextual language models have extended text representation beyond word co-occurrence [17], and BERTopic has combined such embeddings with UMAP, HDBSCAN, and class-based TF-IDF [9]. Comparative studies have evaluated LDA, NMF, BERTopic, and related models on Twitter data, online communities, and news headlines [10], [11], [12]. Reviews have also examined topic modeling for short social media text [18]. Together, this literature shows that model behavior depends on the corpus and evaluation objective. It does not establish which representation is most useful after publication vectors are aggregated into researcher and research group profiles.

Topic coherence has often been used to select and compare topic models. Measures such as  $C_{v}$ estimate whether the leading topic words occur in a semantically consistent context [13], [19]. Empirical work has found that strong intrinsic topic quality does not necessarily correspond to strong performance on downstream tasks [14]. Recommendation evaluation addresses a different objective by measuring whether relevant items appear near the top of a ranked list. Precision, Recall, F1-score, NDCG, and MRR provide complementary views of that ranking quality [20]. The literature has studied topic quality and recommendation quality largely as separate concerns. This study connects them by comparing three topic representations within the same researcher-to-research-group recommendation task and by examining whether coherence agrees with the resulting rankings.

## 3 Methodology

This study developed a BERTopic-based framework for researcher-to-research-group recommendation and compared it with LDA and NMF under one experimental pipeline. As shown in Fig. 1, the pipeline covered data collection, data preprocessing, hyperparameter optimization, topic modeling, profile construction, group recommendation, and evaluation. Publication-level topic vectors were averaged into researcher profiles, while the profiles of training researchers formed the research group profiles. Cosine similarity between these profiles produced the ranked recommendations.

### 3.1 Data Source

Researchers and research group data were obtained from CSRankings. Publication metadata were collected from DBLP, and abstracts were retrieved from Semantic Scholar. Each record contained a title, abstract, author list, publication year, venue, and Digital Object Identifier when available. The CSRankings research areas defined 12 target research groups. After data validation and preprocessing, the dataset contained 11,111 publication documents from 3,257 researchers. The researchers were divided into 2,605 training researchers and 652 test researchers using an 80/20 researcher-level split with random seed 42. This split kept every publication by one researcher in the same partition.

### 3.2 Data Preprocessing

Each publication document combined its title and abstract. Records without usable text or a labeled researcher were excluded. Two text representations were prepared. LDA and NMF used bag-of-words text that was converted to lowercase and stripped of punctuation, non-alphanumeric tokens, common English stopwords, and generic publication terms. BERTopic used the original title-and-abstract text with normalized whitespace to preserve the information required by the embedding model. The vectorizers were fitted only on training publications. Their vocabulary was limited to 20,000 features, with minimum and maximum document frequencies of 2 and 0.8, respectively.

### 3.3 Hyperparameter Optimization

Hyperparameters were selected using only the training corpus. Candidate configurations were ranked by topic coherence, and the highest-scoring configuration for each model was used in the final experiment. The search varied the number of topics for LDA and NMF. For BERTopic, it evaluated combinations of the HDBSCAN minimum topic size and the number of UMAP neighbors. The selected settings were 29 topics and 20 iterations for LDA, and 11 topics and 800 iterations for NMF. BERTopic used a minimum topic size of 45, 50 UMAP neighbors, five UMAP components, and a minimum distance of 0.0. The optimization criterion was used only for model selection; the final recommendation metrics were calculated on the held-out researchers. All models used random seed 42 where supported.

### 3.4 Topic Modeling

LDA, NMF, and BERTopic were fitted to the same training partition. Each model produced a topic vector for every training and test publication.

LDA fitted a probabilistic topic model to document-term count vectors with batch learning. Its output represented each publication as a probability distribution over 29 latent topics. NMF decomposed a TF-IDF document-term matrix into non-negative document-topic and topic-term matrices. It used NNDSVDa initialization, and the resulting 11-dimensional document-topic vector represented each publication. The fitted training vectorizers transformed the test publications without refitting.

BERTopic encoded the original publication text with the pretrained `allenai/specter` model. UMAP reduced the SPECTER embeddings using cosine distance, and HDBSCAN formed the topic clusters. The model then reassigned outliers through embedding similarity and updated the topics with the same vectorizer to keep their representations consistent. BERTopic's `approximate\_distribution()` function generated a document-topic probability distribution for each publication. The recommendation stage used these distributions rather than single topic labels.

### 3.5 Profile Construction

All profiles were constructed in the topic space produced by the corresponding model. Let  $D_{r}=\{\{}d_{1},d_{2}, \text{ \textellipsis } ,d_{n}{\}\}$  denote the publications of researcher r, and let  $v_{d}$ be the topic vector of publication d. The researcher profile was the arithmetic mean of the publication vectors:

$$R_{r}=\frac{1}{\left|D_{r}\right|}\sum_{d \in D_{r}}^{}v_{d} \# \left(1\right)$$

This operation produced one topic profile for each researcher. Research group profiles were built only from training researchers. For a research group with training members  $G=\{\{}r_{1},r_{2}, \text{ \textellipsis } ,r_{m}{\}\}$ , the group profile was

$$P_{G}=\frac{1}{\left|G\right|}\sum_{r \in G}^{}R_{r} \# \left(2\right)$$

where  $R_{r}$  was the profile of member  $r$ . Restricting this aggregation to training members prevented test researcher profiles from entering the group representations.

### 3.6 Research Group Recommendation

Each test researcher profile was compared with all 12 research group profiles using cosine similarity. The groups were sorted by decreasing similarity, with an alphabetical secondary order for equal scores. The same ranking procedure was applied to LDA, NMF, and BERTopic profiles, so the topic representation was the only model-dependent part of the recommendation pipeline. The system returned a ranked list rather than a single group because a researcher could have more than one ground-truth research group.

### 3.7 Evalution

The ranked groups for each of the 652 test researchers were compared with that researcher's known CSRankings group memberships. Precision@ *K* , Recall@ *K* , F1-score@ *K* , and normalized discounted cumulative gain (NDCG@ *K* ) were calculated at *K* = 1, 3, and 5. Precision measured the proportion of the first *K* recommendations that were relevant. Recall measured the proportion of a researcher's relevant groups retrieved within the first *K* positions, and F1-score was their harmonic mean. NDCG assigned greater value to relevant groups placed near the top of the ranking. Mean reciprocal rank (MRR) measured the inverse rank of the first relevant group. Each reported metric was the arithmetic mean of the per-researcher scores. This protocol used the same split, group candidates, profile construction, ranking method, and metrics for all three topic models.

## 4 Results and Discussion

### 4.1 Dataset

**The prepared dataset contained 11,111 unique publication documents from 3,257 researchers. The researcher-level split assigned 2,605 researchers to training and 652 to testing. Fig. 2(a) showed a large difference in research group membership. Machine Learning was the largest group with 2,480 researchers, followed by Artificial Intelligence with 2,090 and Computer Vision with 1,748. Visualization and Software Engineering were the smallest groups, with 118 and 127 researchers, respectively. The counts exceeded the number of unique researchers because one researcher could belong to several groups.**

**Publication profile size was also uneven. Fig. 2(b) showed a mean of 6.1 publications per researcher and a median of 2. The observed profiles ranged from 1 to 59 publications. The distance between the mean and median, together with the long right tail, showed that most researchers had short publication profiles while a smaller number had many publications.**

**Fig. 2(c) described the multi-label structure of the recommendation task. Only 421 researchers, or about 13%, belonged to one group. Membership in three groups was the most common case, covering 703 researchers or about 22% of the dataset. Researchers with two and four groups accounted for about 18% and 20%, respectively. The remaining researchers belonged to five or more groups, including nine researchers with nine group labels. This distribution represented researcher placement as a ranked multi-group problem rather than a single-label classification task** .

### 4.2 Hyperparamter Optimization

Fig. 3 reported the training-corpus  $C_{v}$ coherence obtained during the current parameter search. Four topic counts were evaluated for LDA. The highest coherence, 0.521625, was obtained with 29 topics. The other tested counts produced values between 0.512700 and 0.518153. NMF was also evaluated at four topic counts. Its best configuration used 11 topics and reached 0.686782, compared with 0.683174 for 9 topics and lower values for 12 and 15 topics.

**The BERTopic search covered nine combinations of minimum topic size and UMAP neighbors. A minimum topic size of 45 with 50 neighbors ranked first at 0.659480 and produced 43 topics. The configuration with 45 and 55 neighbors was close at 0.658543. Two configurations using 60 neighbors produced only five topics and had the lowest coherence, 0.533384. All current BERTopic search configurations reported zero remaining noise documents after outlier reassignment. The rank-one configuration for each model was used in the final experiment.**

### 4.3 Recommendation Performance

**Table I summarized the main test results at** ***K*** **=5 and the model-level MRR. BERTopic obtained the highest value in every displayed column. It reached an NDCG@5 of 0.736 and an MRR of 0.809. LDA ranked second at 0.719 and 0.793, while NMF obtained 0.640 and 0.774. BERTopic also recorded the highest Precision@5, Recall@5, and F1-score@5. The difference between BERTopic and LDA was smaller than the difference between BERTopic and NMF.**

- TABLE 1.  Recommendation Performance Test Researchers
| Model    | Metrics   | Metrics   | Metrics   | Metrics   | Metrics   |
|----------|-----------|-----------|-----------|-----------|-----------|
| Model    | P@5       | ***R@5*** | F1@5      | NDCG@5    | MRR       |
| LDA      | 0.4966    | 0.7647    | 0.5652    | 0.7199    | 0.7932    |
| NMF      | 0.4301    | 0.6770    | 0.4923    | 0.6405    | 0.7750    |
| BERTopic | 0.5080    | 0.7824    | 0.5790    | 0.7362    | 0.8093    |

**Fig. 4 extended the comparison to Precision, Recall, F1-score, and NDCG at** ***K*** **=1, 3, and 5. BERTopic ranked first in all 12 cutoff-based metrics. At** ***K*** **=1, its Precision and NDCG were both 0.687117, compared with 0.659509 for LDA and 0.628834 for NMF. Its Recall@1 was 0.251288, which increased to 0.565841 at** ***K*** **=3 and 0.782423 at** ***K*** **=5. The corresponding F1-score rose from 0.342141 to 0.578973 as the cutoff increased.**

**The three models followed the same cutoff pattern. Precision decreased as more groups were returned, while Recall increased. F1-score increased from** ***K*** **=1 to** ***K*** **=5 because the gain in Recall was larger than the loss in Precision. NDCG remained above 0.60 for every model and cutoff. BERTopic obtained the highest MRR at 0.809277, followed by LDA at 0.793238 and NMF at 0.774992. Across all 13 reported recommendation metrics, BERTopic ranked first, LDA ranked second, and NMF ranked third.**

**BERTopic's advantage may be related to its SPECTER embeddings, which represent scientific publications in a contextual semantic space. This representation can keep related publications close even when they use different terms, which may produce researcher profiles that are easier to match with group profiles. The experiment did not isolate the embedding, dimensionality reduction, clustering, and outlier-reassignment components, so the performance difference cannot be assigned to one component alone.**

### 4.4 Topic Quality and Computational Runtime

Fig. 5(a) compared the final topic coherence values. NMF produced the highest  $C_{v}$ score of 0.686782. BERTopic followed with 0.659480, and LDA had the lowest value at 0.521625. This order differed from the recommendation results: NMF had the highest coherence but the lowest score in every recommendation metric, while BERTopic had the best recommendations with the second-highest coherence.

This mismatch suggests that intrinsic topic coherence and downstream recommendation quality measure different properties.  $C_{v}$ evaluates the co-occurrence of leading words, whereas recommendation depends on whether the complete topic vectors separate researcher and group profiles after aggregation. A model can produce coherent topic words without producing the most useful ranking representation. This interpretation is consistent with prior evidence that intrinsic topic metrics do not always predict downstream performance [14].

**The runtime measurements in Fig. 5(b) also differed substantially. NMF had the shortest training time at 18.1 s and the shortest recommendation inference time at 0.7 s. LDA required 178.5 s for training and 1.8 s for inference. BERTopic required 966.5 s for training and 5.2 s for inference. Total execution time was 18.8 s for NMF, 180.3 s for LDA, and 971.8 s for BERTopic. Training accounted for most of the measured time for every model.**

**The runtime pattern shows a clear trade-off between recommendation quality and computational cost. BERTopic is preferable when ranking accuracy has priority and training can run offline, while NMF is more practical when frequent retraining or limited computing resources matter. LDA occupies the middle position, with recommendation quality close to BERTopic and a lower training cost. The recorded times describe this implementation and run rather than a hardware-independent benchmark.**

### 4.5 Performance Across Research Groups

**Fig. 6 reported Top-3 precision for each research group. Performance varied more across groups than across models within the same group. Machine Learning and Computer Vision had the highest values. Machine Learning reached approximately 0.91 with LDA and BERTopic and 0.87 with NMF. Computer Vision ranged from 0.86 to 0.89. Artificial Intelligence and Natural Language Processing formed the next group of results, with values from 0.63 to 0.74.**

**The middle and lower rows showed a wider separation among models. BERTopic obtained 0.59 for The Web and Information Retrieval, 0.54 for Databases, and 0.44 for Human-Computer Interaction. NMF was strongest for Robotics at 0.60 and Data Mining at 0.44. LDA obtained the highest value for Natural Language Processing at 0.74. No model ranked first for every research group.**

**The smallest test groups recorded the lowest Top-3 precision. Visualization had 28 relevant test researchers and scores between 0.07 and 0.14. Software Engineering had 24 and scores between 0.07 and 0.10. Computer Graphics also remained below 0.34 for all models despite having 67 relevant test researchers. By contrast, Databases reached 0.54 with BERTopic from 81 relevant researchers, so test-group size alone did not fully describe the observed differences.**

## 5 Conclusion

**This study answers the research question by showing that publication-based topic representations can support researcher-to-research-group recommendation. Among the compared models, BERTopic produced the best ranking performance, while LDA remained competitive and NMF provided the most efficient execution. The main contribution is a BERTopic-based recommendation framework that represents publications, researchers, and research groups in a shared topic space. Its novelty lies in the controlled comparison of LDA, NMF, and BERTopic within the same recommendation setting, rather than in proposing a new topic modeling algorithm. This comparison clarifies how the choice of topic representation affects the quality of research group rankings.**

**The study also shows that topic coherence is not a reliable substitute for downstream evaluation. NMF generated the most coherent topics but did not produce the strongest recommendations. BERTopic offered better ranking quality at a much higher computational cost, while LDA provided a more moderate balance between accuracy and runtime. In practice, model selection should reflect the intended use. BERTopic is suitable when recommendation quality has priority and training can run offline, whereas LDA or NMF may be preferable when frequent updates or limited computing resources matter.**

**The evaluation is limited to CSRankings data from 12 computer science research groups, one researcher-level split, and profiles derived only from publication titles and abstracts. Uneven group sizes and publication counts may also affect how well some researcher and group profiles are represented. These boundaries limit the generalization of the observed model ordering beyond the current dataset. Future work should evaluate repeated splits and additional institutions or disciplines, report uncertainty and statistical significance, and examine performance by profile size. It should also test citation, collaboration, and temporal information, together with alternative profile aggregation and component-level ablation, to determine when each representation is most useful.**

###### References

[1]	C. Yang, J. Ma, T. Silva, X. Liu, and Z. Hua, “A multilevel information mining approach for expert recommendation in online scientific communities,” *Computer Journal* , vol. 58, no. 9, pp. 1921–1936, Sep. 2014, doi: 10.1093/comjnl/bxu033.

[2]	J. Sun, W. Xu, J. Ma, and J. Sun, “Leverage RAF to find domain experts on research social network services: A big data analytics methodology with MapReduce framework,” *Int. J. Prod. Econ.* , vol. 165, pp. 185–193, Jul. 2015, doi: 10.1016/j.ijpe.2014.12.038.

[3]	Z. Zhang *et al.* , “Scholarly recommendation systems: a literature survey,” Nov. 01, 2023, *Springer Science and Business Media Deutschland GmbH* . doi: 10.1007/s10115-023-01901-x.

[4]	X. Bai, M. Wang, I. Lee, Z. Yang, X. Kong, and F. Xia, “Scientific paper recommendation: A survey,” 2019, *Institute of Electrical and Electronics Engineers Inc.* doi: 10.1109/ACCESS.2018.2890388.

[5]	J. Beel, B. Gipp, S. Langer, and C. Breitinger, “Research-paper recommender systems: a literature survey,” *International Journal on Digital Libraries* , vol. 17, no. 4, pp. 305–338, Nov. 2016, doi: 10.1007/s00799-015-0156-0.

[6]	H. Jelodar *et al.* , “Latent Dirichlet allocation (LDA) and topic modeling: models, applications, a survey,” *Multimed. Tools Appl.* , vol. 78, no. 11, pp. 15169–15211, Jun. 2019, doi: 10.1007/s11042-018-6894-4.

[7]	D. M. Blei, A. Y. Ng, and J. B. Edu, “Latent Dirichlet Allocation Michael I. Jordan,” 2003.

[8]	D. D. Lee and H. S. Seung, “Algorithms for Non-negative Matrix Factorization.”

[9]	M. Grootendorst, “BERTopic: Neural topic modeling with a class-based TF-IDF procedure,” Mar. 2022, [Online]. Available: http://arxiv.org/abs/2203.05794

[10]	R. Egger and J. Yu, “A Topic Modeling Comparison Between LDA, NMF, Top2Vec, and BERTopic to Demystify Twitter Posts,” *Frontiers in Sociology* , vol. 7, May 2022, doi: 10.3389/fsoc.2022.886498.

[11]	A. Kaur and J. R. Wallace, “Moving Beyond LDA: A Comparison of Unsupervised Topic Modelling Techniques for Qualitative Data Analysis of Online Communities,” Dec. 2024, [Online]. Available: http://arxiv.org/abs/2412.14486

[12]	O. Babalola, B. Ojokoh, and O. Boyinbode, “Comprehensive Evaluation of LDA, NMF, and BERTopic’s Performance on News Headline Topic Modeling,” *Journal of Computing Theories and Applications* , vol. 2, no. 2, pp. 268–289, Nov. 2024, doi: 10.62411/jcta.11635.

[13]	M. Röder, A. Both, and A. Hinneburg, “Exploring the space of topic coherence measures,” in *WSDM 2015 - Proceedings of the 8th ACM International Conference on Web Search and Data Mining* , Association for Computing Machinery, Feb. 2015, pp. 399–408. doi: 10.1145/2684822.2685324.

[14]	M. Rüdiger, D. Antons, A. M. Joshi, and T. O. Salge, “Topic modeling revisited: New evidence on algorithm performance and quality metrics,” *PLoS One* , vol. 17, no. 4 April, Apr. 2022, doi: 10.1371/journal.pone.0266325.

[15]	M. Marcuzzo, A. Zangari, A. Albarelli, and A. Gasparetto, “Recommendation Systems: An Insight Into Current Development and Future Research Challenges,” *IEEE Access* , vol. 10, pp. 86578–86623, 2022, doi: 10.1109/ACCESS.2022.3194536.

[16]	F. Viegas *et al.* , “Semantic Academic Profiler (SAP): a framework for researcher assessment based on semantic topic modeling,” *Scientometrics* , vol. 127, no. 8, pp. 5005–5026, Aug. 2022, doi: 10.1007/s11192-022-04449-9.

[17]	A. Vaswani *et al.* , “Attention Is All You Need,” Aug. 2023, [Online]. Available: http://arxiv.org/abs/1706.03762

[18]	C. D. P. Laureate, W. Buntine, and H. Linger, “A systematic review of the use of topic models for short text social media analysis,” *Artif. Intell. Rev.* , vol. 56, no. 12, pp. 14223–14255, Dec. 2023, doi: 10.1007/s10462-023-10471-x.

[19]	D. Mimno, H. M. Wallach, E. Talley, M. Leenders, and A. Mccallum, “Optimizing Semantic Coherence in Topic Models,” Association for Computational Linguistics.

[20]	D. Valcarce, A. Bellogín, J. Parapar, and P. Castells, “Assessing ranking metrics in top-N recommendation,” *Information Retrieval Journal* , vol. 23, no. 4, pp. 411–448, Aug. 2020, doi: 10.1007/s10791-020-09377-x.