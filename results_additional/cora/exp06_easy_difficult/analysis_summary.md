# Easy versus difficult nodes — Cora

Five cached runs (seeds 42, 43, 44, 45, 46,
20 labels per class) are analyzed without retraining. Within each seed, test nodes are ranked by
GCN maximum predicted probability and split into three nearly equal groups. Labels are not used
to create the groups. Accuracy is retrospective test evaluation.

| Confidence group | Model | Nodes | Accuracy (%) | Difference from GCN (pp) |
|---|---|---:|---:|---:|
| Low | GCN | 334.0 ± 0.0 | 58.50 ± 3.69 | +0.00 ± 0.00 |
| Low | NetMF | 334.0 ± 0.0 | 48.74 ± 3.40 | -9.76 ± 2.60 |
| Low | GraphSAGE | 334.0 ± 0.0 | 53.59 ± 4.37 | -4.91 ± 2.71 |
| Low | Weighted GCN + NetMF | 334.0 ± 0.0 | 58.92 ± 4.17 | +0.42 ± 0.62 |
| Low | LearnedRouter | 334.0 ± 0.0 | 56.35 ± 2.14 | -2.16 ± 2.11 |
| Medium | GCN | 333.0 ± 0.0 | 83.48 ± 1.63 | +0.00 ± 0.00 |
| Medium | NetMF | 333.0 ± 0.0 | 73.09 ± 2.67 | -10.39 ± 3.62 |
| Medium | GraphSAGE | 333.0 ± 0.0 | 81.98 ± 2.05 | -1.50 ± 1.02 |
| Medium | Weighted GCN + NetMF | 333.0 ± 0.0 | 83.30 ± 1.68 | -0.18 ± 0.81 |
| Medium | LearnedRouter | 333.0 ± 0.0 | 82.10 ± 2.15 | -1.38 ± 1.28 |
| High | GCN | 333.0 ± 0.0 | 92.31 ± 3.82 | +0.00 ± 0.00 |
| High | NetMF | 333.0 ± 0.0 | 83.42 ± 2.02 | -8.89 ± 2.54 |
| High | GraphSAGE | 333.0 ± 0.0 | 92.79 ± 2.16 | +0.48 ± 2.11 |
| High | Weighted GCN + NetMF | 333.0 ± 0.0 | 92.43 ± 3.69 | +0.12 ± 0.16 |
| High | LearnedRouter | 333.0 ± 0.0 | 92.85 ± 2.45 | +0.54 ± 2.15 |

## Validation-selected fusion weights

These Task 3 weights were selected on each cache's validation mask and verified by reproducing
the complete 0.0–1.0 search. Test labels did not affect selection.

| Seed | GCN weight | NetMF weight |
|---:|---:|---:|
| 42 | 0.4 | 0.6 |
| 43 | 0.5 | 0.5 |
| 44 | 0.5 | 0.5 |
| 45 | 0.8 | 0.2 |
| 46 | 0.9 | 0.1 |

Values are mean ± sample standard deviation across five seeds. Confidence groups contain 334,
333, and 333 nodes per seed (low to high). Group membership can differ between seeds because each
seed has different cached GCN probabilities.
