# GCN error overlap — Cora

This retrospective analysis uses five cached runs (seeds 42–46, 20 labels per class) and does
not retrain any model. A rescue is a test node that GCN predicts incorrectly and the comparator
predicts correctly.

| Comparator | Both correct (%) | Only GCN (%) | Rescue (%) | Both wrong (%) | Rescued nodes | Rescue rate among GCN mistakes (%) |
|---|---:|---:|---:|---:|---:|---:|
| NetMF | 62.18 ± 1.75 | 15.90 ± 2.01 | 6.22 ± 0.62 | 15.70 ± 0.84 | 62.20 ± 6.18 | 28.39 ± 2.85 |
| GraphSAGE | 71.94 ± 1.91 | 6.14 ± 1.72 | 4.16 ± 2.06 | 17.76 ± 2.11 | 41.60 ± 20.62 | 18.98 ± 9.32 |

## Rescue location by GCN confidence

| Comparator | Confidence | Rescued nodes | Share of comparator rescues (%) | GCN mistakes rescued in group (%) |
|---|---|---:|---:|---:|
| NetMF | Low | 42.00 ± 5.48 | 67.51 ± 6.28 | 30.66 ± 6.15 |
| NetMF | Medium | 15.00 ± 2.55 | 24.33 ± 5.07 | 27.23 ± 3.72 |
| NetMF | High | 5.20 ± 3.49 | 8.15 ± 4.71 | 19.26 ± 3.66 |
| GraphSAGE | Low | 27.60 ± 6.11 | 72.33 ± 16.51 | 20.35 ± 6.52 |
| GraphSAGE | Medium | 8.00 ± 5.00 | 18.36 ± 4.50 | 14.59 ± 8.79 |
| GraphSAGE | High | 6.00 ± 10.75 | 9.31 ± 13.64 | 14.78 ± 22.31 |

Values are mean ± sample standard deviation across five seeds. Confidence groups are formed by
ranking GCN maximum probability on test nodes, without labels. Each seed has 334 low-confidence,
333 medium-confidence, and 333 high-confidence nodes. True labels are used only afterward to
measure retrospective correctness and rescue counts.
