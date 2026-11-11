# Easy versus difficult nodes — PubMed

Five cached runs (seeds 42, 43, 44, 45, 46,
20 labels per class) are analyzed without retraining. Within each seed, test nodes are ranked by
GCN maximum predicted probability and split into three nearly equal groups. Labels are not used
to create the groups. Accuracy is retrospective test evaluation.

| Confidence group | Model | Nodes | Accuracy (%) | Difference from GCN (pp) |
|---|---|---:|---:|---:|
| Low | GCN | 334.0 ± 0.0 | 61.44 ± 2.19 | +0.00 ± 0.00 |
| Low | NetMF | 334.0 ± 0.0 | 54.37 ± 5.78 | -7.07 ± 5.66 |
| Low | GraphSAGE | 334.0 ± 0.0 | 56.17 ± 2.52 | -5.27 ± 2.10 |
| Low | Weighted GCN + NetMF | 334.0 ± 0.0 | 62.16 ± 4.13 | +0.72 ± 2.97 |
| Low | LearnedRouter | 334.0 ± 0.0 | 60.30 ± 1.46 | -1.14 ± 1.33 |
| Medium | GCN | 333.0 ± 0.0 | 81.86 ± 2.73 | +0.00 ± 0.00 |
| Medium | NetMF | 333.0 ± 0.0 | 75.62 ± 0.39 | -6.25 ± 2.78 |
| Medium | GraphSAGE | 333.0 ± 0.0 | 80.12 ± 2.76 | -1.74 ± 0.83 |
| Medium | Weighted GCN + NetMF | 333.0 ± 0.0 | 82.04 ± 2.53 | +0.18 ± 0.27 |
| Medium | LearnedRouter | 333.0 ± 0.0 | 80.90 ± 3.41 | -0.96 ± 0.68 |
| High | GCN | 333.0 ± 0.0 | 89.85 ± 1.23 | +0.00 ± 0.00 |
| High | NetMF | 333.0 ± 0.0 | 84.50 ± 1.60 | -5.35 ± 1.62 |
| High | GraphSAGE | 333.0 ± 0.0 | 90.03 ± 1.32 | +0.18 ± 0.46 |
| High | Weighted GCN + NetMF | 333.0 ± 0.0 | 89.97 ± 0.96 | +0.12 ± 0.27 |
| High | LearnedRouter | 333.0 ± 0.0 | 89.97 ± 1.76 | +0.12 ± 0.66 |

## Validation-selected fusion weights

These Task 3 weights were selected on each cache's validation mask and verified by reproducing
the complete 0.0–1.0 search. Test labels did not affect selection.

| Seed | GCN weight | NetMF weight |
|---:|---:|---:|
| 42 | 0.7 | 0.3 |
| 43 | 0.6 | 0.4 |
| 44 | 0.7 | 0.3 |
| 45 | 0.5 | 0.5 |
| 46 | 0.5 | 0.5 |

Values are mean ± sample standard deviation across five seeds. Confidence groups contain 334,
333, and 333 nodes per seed (low to high). Group membership can differ between seeds because each
seed has different cached GCN probabilities.
