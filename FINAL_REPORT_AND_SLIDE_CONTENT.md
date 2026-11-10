# Final Report and Presentation Content

This file contains the sections that remain stable before the final experiments. Replace bracketed result fields only after the five-seed runs finish.

## Research question

Does adaptive prediction fusion improve node classification compared with a standard GCN and the simpler approach of adding NetMF embeddings to GCN input features?

## Why this comparison matters

NetMF summarizes multi-hop graph structure through matrix factorization. GCN and GraphSAGE combine node features with local message passing. These representations may make different mistakes, but using three complete models also increases computation. We therefore compare simple feature concatenation, fixed averaging, and a learned router before deciding whether adaptive fusion is justified.

## Experimental protocol

- Datasets: Cora and, if completed, CiteSeer.
- Label budgets: 5, 10, and 20 labeled nodes per class.
- Random seeds: 42, 43, 44, 45, and 46.
- Validation split: 50% for base-model early stopping, 30% for router training, and 20% for router early stopping.
- Edge removal: 0%, 20%, and 40% at twenty labels per class.
- Metrics: test accuracy, macro-F1, runtime, and NVIDIA peak memory when available.

## Methods

The individual baselines are MLP, NetMF with logistic regression, GCN, and GraphSAGE. NetMF plus GCN concatenates the NetMF embedding with the original node features before message passing. Equal averaging gives the three graph experts the same weight. The fixed degree router uses manually chosen weights. The learned router predicts three weights from degree, confidence, entropy, and pairwise disagreement. The oracle checks whether any expert was correct and provides an upper bound rather than a usable model.

## Result table template

| Method | 5 labels | 10 labels | 20 labels | Runtime | GPU memory |
|---|---:|---:|---:|---:|---:|
| GCN | [run] | [run] | [run] | [run] | [run] |
| NetMF plus GCN | [run] | [run] | [run] | [run] | [run] |
| Equal average | [run] | [run] | [run] | [run] | [run] |
| Learned router | [run] | [run] | [run] | [run] | [run] |

## Interpretation rules

- Prefer NetMF plus GCN if it matches or beats the router because it uses fewer models at prediction time.
- Treat the learned router as useful only if it improves overall accuracy, missing-edge robustness, or a clearly defined degree group.
- Use the oracle only to describe remaining routing headroom.
- If GCN stays strongest, conclude that fusion did not justify its added complexity on the tested data.

## Limitations

The study uses small citation networks and a transductive setting. NetMF also depends on graph connectivity, so low degree does not guarantee that it will perform well. Five seeds measure variation but do not establish broad statistical significance. The explicit NetMF implementation is not intended for very large graphs.

## Slide sequence

1. Problem and prediction task
2. Why compare NetMF with message-passing GNNs
3. Research question
4. Dataset and clean data split
5. Methods and NetMF plus GCN baseline
6. Learned router
7. Oracle and ablation logic
8. Limited-label results placeholder
9. Missing-edge results placeholder
10. Degree-group results placeholder
11. Accuracy versus complexity decision
12. Conclusion and limitations

## Speaker notes for result slides

**Limited-label results:** We compare all methods using the same training nodes for each seed. The important comparison is whether the learned router beats both GCN and NetMF plus GCN. The dashed oracle line is only an upper bound.

**Missing-edge results:** We remove the same edges for every method within a seed and recompute NetMF on that damaged graph. This tests robustness without giving any method cleaner graph information.

**Degree-group results:** These results test our original degree hypothesis. We do not assume that NetMF helps low-degree nodes. We report what the experiment shows for low-, medium-, and high-degree test nodes.
