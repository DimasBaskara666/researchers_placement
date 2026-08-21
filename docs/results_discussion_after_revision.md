# IV. RESULTS AND DISCUSSION

The experiment compared the three topic representations on the same 652 test researchers and 12 candidate research groups. The results are presented in the order of dataset characteristics, parameter selection, recommendation accuracy, topic quality and runtime, and group-level performance.

## A. Dataset characteristics

The prepared dataset contained 11,111 unique publication documents from 3,257 researchers. The researcher-level split assigned 2,605 researchers to training and 652 to testing. Fig. 2(a) showed a large difference in research group membership. Machine Learning was the largest group with 2,480 researchers, followed by Artificial Intelligence with 2,090 and Computer Vision with 1,748. Visualization and Software Engineering were the smallest groups, with 118 and 127 researchers, respectively. The counts exceeded the number of unique researchers because one researcher could belong to several groups.

Publication profile size was also uneven. Fig. 2(b) showed a mean of 6.1 publications per researcher and a median of 2. The observed profiles ranged from 1 to 59 publications. The distance between the mean and median, together with the long right tail, showed that most researchers had short publication profiles while a smaller number had many publications.

Fig. 2(c) described the multi-label structure of the recommendation task. Only 421 researchers, or about 13%, belonged to one group. Membership in three groups was the most common case, covering 703 researchers or about 22% of the dataset. Researchers with two and four groups accounted for about 18% and 20%, respectively. The remaining researchers belonged to five or more groups, including nine researchers with nine group labels. This distribution represented researcher placement as a ranked multi-group problem rather than a single-label classification task.

*Fig. 2. Dataset characteristics: (a) researcher membership in each group, (b) the distribution of publication profile sizes, and (c) the number of research group affiliations per researcher.*

## B. Hyperparameter selection

Fig. 3 reported the training-corpus $C_v$ coherence obtained during the current parameter search. Four topic counts were evaluated for LDA. The highest coherence, 0.521625, was obtained with 29 topics. The other tested counts produced values between 0.512700 and 0.518153. NMF was also evaluated at four topic counts. Its best configuration used 11 topics and reached 0.686782, compared with 0.683174 for 9 topics and lower values for 12 and 15 topics.

The BERTopic search covered nine combinations of minimum topic size and UMAP neighbors. A minimum topic size of 45 with 50 neighbors ranked first at 0.659480 and produced 43 topics. The configuration with 45 and 55 neighbors was close at 0.658543. Two configurations using 60 neighbors produced only five topics and had the lowest coherence, 0.533384. All current BERTopic search configurations reported zero remaining noise documents after outlier reassignment. The rank-one configuration for each model was used in the final experiment.

*Fig. 3. Training-corpus hyperparameter search based on $C_v$ coherence for LDA topic count, NMF topic count, and the BERTopic combinations of minimum topic size and UMAP neighbors. The marked settings were selected for the final experiment.*

## C. Recommendation performance

Table I summarized the main test results at $K=5$ and the model-level MRR. BERTopic obtained the highest value in every displayed column. It reached an NDCG@5 of 0.736 and an MRR of 0.809. LDA ranked second at 0.719 and 0.793, while NMF obtained 0.640 and 0.774. BERTopic also recorded the highest Precision@5, Recall@5, and F1-score@5. The difference between BERTopic and LDA was smaller than the difference between BERTopic and NMF.

**TABLE I. RECOMMENDATION PERFORMANCE ON 652 TEST RESEARCHERS**

| Model | P@5 | R@5 | F1@5 | NDCG@5 | MRR |
|---|---:|---:|---:|---:|---:|
| LDA | 0.496 | 0.764 | 0.565 | 0.719 | 0.793 |
| NMF | 0.430 | 0.676 | 0.492 | 0.640 | 0.774 |
| BERTopic | 0.507 | 0.782 | 0.578 | 0.736 | 0.809 |

Fig. 4 extended the comparison to Precision, Recall, F1-score, and NDCG at $K=1$, 3, and 5. BERTopic ranked first in all 12 cutoff-based metrics. At $K=1$, its Precision and NDCG were both 0.687117, compared with 0.659509 for LDA and 0.628834 for NMF. Its Recall@1 was 0.251288, which increased to 0.565841 at $K=3$ and 0.782423 at $K=5. The corresponding F1-score rose from 0.342141 to 0.578973 as the cutoff increased.

The three models followed the same cutoff pattern. Precision decreased as more groups were returned, while Recall increased. F1-score increased from $K=1$ to $K=5$ because the gain in Recall was larger than the loss in Precision. NDCG remained above 0.60 for every model and cutoff. BERTopic obtained the highest MRR at 0.809277, followed by LDA at 0.793238 and NMF at 0.774992. Across all 13 reported recommendation metrics, BERTopic ranked first, LDA ranked second, and NMF ranked third.

BERTopic's advantage may be related to its SPECTER embeddings, which represent scientific publications in a contextual semantic space. This representation can keep related publications close even when they use different terms, which may produce researcher profiles that are easier to match with group profiles. The experiment did not isolate the embedding, dimensionality reduction, clustering, and outlier-reassignment components, so the performance difference cannot be assigned to one component alone.

*Fig. 4. Recommendation performance of LDA, NMF, and BERTopic for Precision, Recall, F1-score, and NDCG at $K=1$, 3, and 5, with MRR reported as a model-level ranking measure.*

## D. Topic quality and computational runtime

Fig. 5(a) compared the final topic coherence values. NMF produced the highest $C_v$ score of 0.686782. BERTopic followed with 0.659480, and LDA had the lowest value at 0.521625. This order differed from the recommendation results: NMF had the highest coherence but the lowest score in every recommendation metric, while BERTopic had the best recommendations with the second-highest coherence.

This mismatch suggests that intrinsic topic coherence and downstream recommendation quality measure different properties. $C_v$ evaluates the co-occurrence of leading words, whereas recommendation depends on whether the complete topic vectors separate researcher and group profiles after aggregation. A model can produce coherent topic words without producing the most useful ranking representation. This interpretation is consistent with prior evidence that intrinsic topic metrics do not always predict downstream performance [14].

The runtime measurements in Fig. 5(b) also differed substantially. NMF had the shortest training time at 18.1 s and the shortest recommendation inference time at 0.7 s. LDA required 178.5 s for training and 1.8 s for inference. BERTopic required 966.5 s for training and 5.2 s for inference. Total execution time was 18.8 s for NMF, 180.3 s for LDA, and 971.8 s for BERTopic. Training accounted for most of the measured time for every model.

The runtime pattern shows a clear trade-off between recommendation quality and computational cost. BERTopic is preferable when ranking accuracy has priority and training can run offline, while NMF is more practical when frequent retraining or limited computing resources matter. LDA occupies the middle position, with recommendation quality close to BERTopic and a lower training cost. The recorded times describe this implementation and run rather than a hardware-independent benchmark.

*Fig. 5. Final model characteristics: (a) training-corpus $C_v$ topic coherence and (b) training and recommendation inference time on a logarithmic time scale.*

## E. Performance across research groups

Fig. 6 reported Top-3 precision for each research group. Performance varied more across groups than across models within the same group. Machine Learning and Computer Vision had the highest values. Machine Learning reached approximately 0.91 with LDA and BERTopic and 0.87 with NMF. Computer Vision ranged from 0.86 to 0.89. Artificial Intelligence and Natural Language Processing formed the next group of results, with values from 0.63 to 0.74.

The middle and lower rows showed a wider separation among models. BERTopic obtained 0.59 for The Web and Information Retrieval, 0.54 for Databases, and 0.44 for Human-Computer Interaction. NMF was strongest for Robotics at 0.60 and Data Mining at 0.44. LDA obtained the highest value for Natural Language Processing at 0.74. No model ranked first for every research group.

The smallest test groups recorded the lowest Top-3 precision. Visualization had 28 relevant test researchers and scores between 0.07 and 0.14. Software Engineering had 24 and scores between 0.07 and 0.10. Computer Graphics also remained below 0.34 for all models despite having 67 relevant test researchers. By contrast, Databases reached 0.54 with BERTopic from 81 relevant researchers, so test-group size alone did not fully describe the observed differences.

*Fig. 6. Top-3 precision by research group for LDA, NMF, and BERTopic. The value of $n$ denotes the number of test researchers for whom each group was relevant.*
