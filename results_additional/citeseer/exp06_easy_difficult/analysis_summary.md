# Easy versus difficult nodes — CiteSeer

Five cached runs (seeds 42, 43, 44, 45, 46,
20 labels per class) are analyzed without retraining. Within each seed, test nodes are ranked by
GCN maximum predicted probability and split into three nearly equal groups. Labels are not used
to create the groups. Accuracy is retrospective test evaluation.

| Confidence group | Model | Nodes | Accuracy (%) | Difference from GCN (pp) |
|---|---|---:|---:|---:|
| Low | GCN | 334.0 ± 0.0 | 52.10 ± 4.33 | +0.00 ± 0.00 |
| Low | NetMF | 334.0 ± 0.0 | 26.17 ± 2.72 | -25.93 ± 4.96 |
| Low | GraphSAGE | 334.0 ± 0.0 | 46.95 ± 3.09 | -5.15 ± 3.92 |
| Low | Weighted GCN + NetMF | 334.0 ± 0.0 | 51.38 ± 4.67 | -0.72 ± 1.07 |
| Low | LearnedRouter | 334.0 ± 0.0 | 50.90 ± 5.18 | -1.20 ± 2.68 |
| Medium | GCN | 333.0 ± 0.0 | 76.16 ± 2.36 | +0.00 ± 0.00 |
| Medium | NetMF | 333.0 ± 0.0 | 43.48 ± 3.34 | -32.67 ± 2.95 |
| Medium | GraphSAGE | 333.0 ± 0.0 | 73.09 ± 2.52 | -3.06 ± 3.26 |
| Medium | Weighted GCN + NetMF | 333.0 ± 0.0 | 76.04 ± 2.31 | -0.12 ± 0.27 |
| Medium | LearnedRouter | 333.0 ± 0.0 | 76.52 ± 1.97 | +0.36 ± 0.81 |
| High | GCN | 333.0 ± 0.0 | 85.89 ± 1.62 | +0.00 ± 0.00 |
| High | NetMF | 333.0 ± 0.0 | 59.16 ± 1.50 | -26.73 ± 1.06 |
| High | GraphSAGE | 333.0 ± 0.0 | 85.41 ± 1.37 | -0.48 ± 1.42 |
| High | Weighted GCN + NetMF | 333.0 ± 0.0 | 85.89 ± 1.62 | +0.00 ± 0.00 |
| High | LearnedRouter | 333.0 ± 0.0 | 86.13 ± 1.35 | +0.24 ± 0.39 |

## Validation-selected fusion weights

These Task 3 weights were selected on each cache's validation mask and verified by reproducing
the complete 0.0–1.0 search. Test labels did not affect selection.

| Seed | GCN weight | NetMF weight |
|---:|---:|---:|
| 42 | 0.9 | 0.1 |
| 43 | 0.8 | 0.2 |
| 44 | 1.0 | 0.0 |
| 45 | 1.0 | 0.0 |
| 46 | 1.0 | 0.0 |

Values are mean ± sample standard deviation across five seeds. Confidence groups contain 334,
333, and 333 nodes per seed (low to high). Group membership can differ between seeds because each
seed has different cached GCN probabilities.
