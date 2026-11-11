# Final Report and Presentation Content

This file contains the report and slide content. The result fields were filled in from the five-seed runs regenerated on 2026-11-10, after the NetMF feature-scaling fix (see `OBSERVATIONS.md`, section 0).

## The story

**One-sentence thesis:** NetMF, GCN, and GraphSAGE make different mistakes, so choosing the right expert for each node could add 9 to 12 points of accuracy, but the practical ways of combining them we tested gain at most about one point over a plain GCN, because the signals available to a router do not reveal which expert is right.

The argument has four steps, and each one is supported by a table or plot below:

1. **There is room to gain.** For 9 to 10 percent of Cora test nodes (9 to 12 percent on CiteSeer), GCN is wrong but at least one other expert is correct. An oracle that always picks a correct expert reaches 86.5 percent on Cora versus 78.0 percent for GCN.
2. **Simple combinations tie GCN or beat it by about one point.** Once its features are scaled correctly, NetMF plus GCN matches GCN on Cora and CiteSeer, including with missing edges, and beats it by about one point on PubMed (+1.3 points at 20 labels, p = 0.02 to 0.04 over ten seeds). Equal averaging ties GCN on Cora and PubMed and loses 5 points on CiteSeer, where NetMF is weak.
3. **The learned router fails to exploit the headroom.** It learns to give NetMF almost no weight and stays slightly below GCN. Our original hypothesis that NetMF helps low-degree nodes is wrong: NetMF is weakest on exactly those nodes.
4. **Why the router fails:** its inputs (degree, confidence, entropy, and disagreement) flag where the experts disagree but barely tell which one is right. Among nodes where NetMF and GCN disagree, a classifier trained on these inputs separates the two cases with an AUC of only 0.58 to 0.65 (0.5 is chance), and NetMF is right in only 10 to 23 percent of those disagreements while GCN is right in 56 to 63 percent. A switching rule tuned on validation data therefore gains nothing (between -1.6 and +0.2 points).

**What the project contributes:** a careful negative result with clean controls (disjoint validation subsets, the same damaged graph for every method, five seeds), a measured upper bound on routing, and a practical warning that embedding concatenation needs feature scaling. Without scaling, our own baseline first appeared 14 to 24 points worse than GCN.

**Claims to avoid:** "NetMF improves robustness to missing edges" (with ten seeds the lead is +0.1 points, p = 0.81), "the router is competitive" without saying it never wins, and any statement about GPU memory (it was not measured on MPS).

### Draft abstract

We study whether combining NetMF embeddings with message-passing graph neural networks improves semi-supervised node classification on Cora, CiteSeer, and PubMed with 5, 10, and 20 labeled nodes per class and with up to 40 percent of edges removed. We compare a GCN that receives NetMF embeddings as extra features, equal prediction averaging, a fixed degree-based router, and a learned per-node router over NetMF, GCN, and GraphSAGE. An oracle that selects a correct expert whenever one exists reaches 86.5 percent accuracy on Cora versus 78.0 percent for GCN, which shows that the experts make complementary errors. However, practical fusion gains little: NetMF concatenation matches GCN on Cora and CiteSeer once the embedding is rescaled and beats it by about one point on PubMed, averaging ties or underperforms it, and the learned router assigns NetMF almost no weight and never beats GCN. A diagnostic shows why: among nodes where the experts disagree, the router's inputs predict which expert is correct with an AUC of only 0.54 to 0.65, and GCN is usually the correct one. We also find that unscaled concatenation lowers accuracy by 14 to 24 points, a pitfall that would make the embedding look harmful.

## Research question

Does adaptive prediction fusion improve node classification compared with a standard GCN and the simpler approach of adding NetMF embeddings to GCN input features?

## Why this comparison matters

NetMF summarizes multi-hop graph structure through matrix factorization. GCN and GraphSAGE combine node features with local message passing. These representations may make different mistakes, but using three complete models also increases computation. We therefore compare simple feature concatenation, fixed averaging, and a learned router before deciding whether adaptive fusion is justified.

## Experimental protocol

- Datasets: Cora, CiteSeer, and PubMed (PubMed uses the eigen approximation of NetMF; see Methods).
- Label budgets: 5, 10, and 20 labeled nodes per class.
- Random seeds: 42, 43, 44, 45, and 46.
- Validation split: 50% for base-model early stopping, 30% for router training, and 20% for router early stopping.
- Edge removal: 0%, 20%, and 40% at twenty labels per class.
- Metrics: test accuracy, macro-F1, runtime, and NVIDIA peak memory when available.

## Methods

The individual baselines are MLP, NetMF with logistic regression, GCN, and GraphSAGE. NetMF plus GCN concatenates the NetMF embedding with the original node features before message passing. Because the unit-norm NetMF rows are about four to five times larger than the row-normalized word features, the NetMF block is rescaled to 0.1, 0.3, or 1.0 times the feature norm, with the scale chosen on validation accuracy. Without rescaling, the NetMF columns dominate and accuracy drops by 14 points on Cora and 24 points on CiteSeer. Equal averaging gives the three graph experts the same weight. The fixed degree router uses manually chosen weights. The learned router predicts three weights from degree, confidence, entropy, and pairwise disagreement. The oracle checks whether any expert was correct and provides an upper bound rather than a usable model.

## Results

Test accuracy (%), mean ± standard deviation over five seeds. Runtime is the mean training time per run at 20 labels on Apple MPS. For the fusion methods it includes training all three experts. For NetMF plus GCN it includes all three scale candidates and excludes NetMF preprocessing (under about 2 seconds). GPU memory could not be measured on MPS and is omitted.

**Cora**

| Method | 5 labels | 10 labels | 20 labels | Runtime |
|---|---:|---:|---:|---:|
| GCN | 72.48 ± 2.1 | 74.54 ± 3.6 | 78.02 ± 1.6 | 4.1 s |
| NetMF plus GCN | 73.12 ± 3.5 | 75.02 ± 2.3 | 78.16 ± 1.4 | 9.3 s |
| Equal average | 72.88 ± 3.2 | 74.50 ± 2.7 | 77.82 ± 0.6 | 22.7 s |
| Learned router | 71.66 ± 3.0 | 73.02 ± 2.8 | 77.20 ± 1.8 | 24.3 s |
| Oracle (upper bound) | 82.32 | 83.76 | 86.52 | — |

**CiteSeer**

| Method | 5 labels | 10 labels | 20 labels | Runtime |
|---|---:|---:|---:|---:|
| GCN | 60.30 ± 4.9 | 64.82 ± 2.1 | 70.68 ± 1.1 | 3.2 s |
| NetMF plus GCN | 59.66 ± 5.5 | 64.34 ± 4.2 | 70.86 ± 1.8 | 9.9 s |
| Equal average | 56.14 ± 7.1 | 59.66 ± 6.3 | 65.66 ± 3.3 | 22.3 s |
| Learned router | 58.88 ± 5.4 | 64.16 ± 3.1 | 70.20 ± 1.0 | 23.4 s |
| Oracle (upper bound) | 72.72 | 75.68 | 79.82 | — |

**PubMed (ten seeds, 42 to 51)**

| Method | 5 labels | 10 labels | 20 labels |
|---|---:|---:|---:|
| GCN | 67.13 ± 5.7 | 74.09 ± 2.4 | 76.35 ± 2.0 |
| NetMF plus GCN | 68.51 ± 4.5 | 74.53 ± 2.6 | 77.67 ± 1.8 |
| Equal average | 67.17 ± 5.2 | 74.31 ± 2.4 | 76.92 ± 2.3 |
| Learned router | 67.01 ± 5.6 | 74.08 ± 2.8 | 76.41 ± 1.6 |
| Oracle (upper bound) | 79.61 | 85.30 | 85.35 |

NetMF plus GCN minus GCN, paired over seeds: +1.38 points at 5 labels (p = 0.16), +0.44 at 10 (p = 0.30), +1.32 at 20 (t-test p = 0.04, Wilcoxon p = 0.02, ahead on 7 of 10 seeds). PubMed's NetMF embedding uses the eigen approximation with 256 eigenpairs, which on Cora and CiteSeer gives a better NetMF embedding than the exact version.

**Cora, 20 labels, edges removed**

| Edges removed | GCN | NetMF plus GCN | Equal average | Learned router |
|---:|---:|---:|---:|---:|
| 0% | 78.02 ± 1.6 | 78.16 ± 1.4 | 77.82 ± 0.6 | 77.20 ± 1.8 |
| 20% | 76.30 ± 1.8 | 76.76 ± 1.3 | 76.52 ± 1.7 | 76.04 ± 1.6 |
| 40% | 72.86 ± 2.9 | 73.72 ± 1.7 | 72.54 ± 3.5 | 70.82 ± 4.0 |


A ten-seed rerun (seeds 42–51, `results_missing_edges_10seeds/`) checks whether NetMF + GCN's lead at 20% and 40% edge removal is real. It is not:

| Edges removed | GCN | NetMF + GCN | Equal average | Learned router |
|---:|---:|---:|---:|---:|
| 0% | 78.60 ± 1.8 | 78.53 ± 3.0 | 78.87 ± 1.6 | 78.17 ± 2.0 |
| 20% | 77.64 ± 2.0 | 77.64 ± 1.4 | 77.69 ± 2.1 | 76.69 ± 1.9 |
| 40% | 74.43 ± 2.7 | 74.57 ± 1.5 | 74.40 ± 3.3 | 73.12 ± 3.7 |

Paired over seeds, NetMF + GCN and equal averaging differ from GCN by at most 0.3 points (all p > 0.5). The learned router is worse than GCN at 20% removal (−0.95 points, p = 0.03) and at 40% removal (−1.31 points, p = 0.03, lower on all 10 seeds).

**Degree groups (Cora, 20 labels, test accuracy %)**

| Method | Low degree | Medium | High |
|---|---:|---:|---:|
| GCN | 74.3 | 79.7 | 81.7 |
| GraphSAGE | 70.7 | 77.9 | 80.8 |
| Learned router | 72.8 | 78.9 | 81.9 |
| NetMF | 59.4 | 72.6 | 76.8 |

## Conclusion

Applying the interpretation rules below:

- NetMF plus GCN matches GCN on Cora and CiteSeer and beats it by about one point on PubMed, while running a single model at prediction time. Under the rules, it is the preferred fusion method. Its PubMed gain is small and significant only at 20 labels and when all budgets are pooled.
- The learned router does not improve overall accuracy, missing-edge robustness, or any degree group. It assigns NetMF almost no weight on Cora and at most 0.20 on CiteSeer.
- The oracle is 9–12 points above GCN, so the experts make complementary mistakes. The current router inputs do not identify which expert is correct.
- The degree hypothesis is not supported: NetMF is least accurate on low-degree nodes.
- GCN remains the strongest single model. Prediction-level fusion (averaging and routing) does not justify running three models; feature-level fusion (NetMF plus GCN) is cheap and gives at most about one point.

## Interpretation rules

- Prefer NetMF plus GCN if it matches or beats the router because it uses fewer models at prediction time.
- Treat the learned router as useful only if it improves overall accuracy, missing-edge robustness, or a clearly defined degree group.
- Use the oracle only to describe remaining routing headroom.
- If GCN stays strongest, conclude that fusion did not justify its added complexity on the tested data.

## Limitations

The study uses small citation networks and a transductive setting. NetMF also depends on graph connectivity, so low degree does not guarantee that it will perform well. Five seeds measure variation but do not establish broad statistical significance. The explicit NetMF implementation is not intended for very large graphs.

## Slide sequence

Each slide title states its takeaway. The supporting figure is given in brackets.

1. **Predicting a paper's topic from its words and citations.** The task, Cora, and CiteSeer.
2. **NetMF and GNNs see the graph differently.** Multi-hop factorization versus local message passing.
3. **Question: does combining them beat a plain GCN?** Includes the simpler alternative of adding NetMF features to the GCN.
4. **Clean protocol.** Disjoint validation subsets, five seeds, and the same damaged graph for every method.
5. **Four ways to combine the experts.** Concatenation, equal averaging, fixed degree router, learned router, plus the oracle as an upper bound.
6. **The experts make different mistakes: the oracle is 9 to 12 points above GCN.** [`model_comparison.png`]
7. **Pitfall: unscaled NetMF features cost 14 to 24 points.** The NetMF_GCN_Unscaled vs NetMF_GCN bars, with the norm comparison 1.0 vs 0.25.
8. **No combination reliably beats GCN on Cora or CiteSeer; NetMF plus GCN gains about one point on PubMed.** [`accuracy_vs_labels.png` for all three datasets]
9. **Missing edges: NetMF adds no robustness, and the learned router degrades fastest.** [`accuracy_vs_edge_removal.png`]
10. **NetMF is weakest on low-degree nodes, the opposite of our hypothesis.** [`accuracy_by_degree.png`]
11. **The learned router learns to ignore NetMF.** [`router_weights.png`, plus the AUC 0.54 to 0.65 result]
12. **Conclusion: use GCN, or NetMF plus GCN at a similar prediction cost; better routing signals are future work.** Limitations.

## Speaker notes for result slides

**Limited-label results:** We compare all methods using the same training nodes for each seed. The important comparison is whether the learned router beats both GCN and NetMF plus GCN. It does not: GCN, NetMF plus GCN, and the learned router are within about one point of each other, which is inside the seed-to-seed variation. The dashed oracle line is only an upper bound, and its 9 to 12 point gap shows that better routing could still help.

**NetMF scaling (suggested extra slide):** Our first NetMF plus GCN results were 14 to 24 points below GCN. The cause was feature scale: NetMF rows are about four to five times larger than the word-feature rows, so the GCN mostly learned from NetMF. After rescaling, chosen on validation data, NetMF plus GCN matches GCN on Cora and CiteSeer and slightly beats it on PubMed.

**Missing-edge results:** We remove the same edges for every method within a seed and recompute NetMF on that damaged graph. This tests robustness without giving any method cleaner graph information. With five seeds, NetMF plus GCN looked slightly ahead at 20% and 40% removal. A ten-seed rerun removes that lead (+0.0 and +0.1 points, p > 0.8), so there is no robustness benefit. The learned router falls behind GCN as edges are removed (−1.3 points at 40%, p = 0.03, lower on all ten seeds).

**Degree-group results:** These results test our original degree hypothesis. We do not assume that NetMF helps low-degree nodes. We report what the experiment shows for low-, medium-, and high-degree test nodes. NetMF is weakest on low-degree nodes (59% vs 74% for GCN on Cora), which contradicts the hypothesis behind the fixed degree router.
