# GCN error overlap — CiteSeer

This retrospective analysis uses five cached runs (seeds 42–46, 20 labels per class) and does
not retrain any model. A rescue is a test node that GCN predicts incorrectly and the comparator
predicts correctly.

| Comparator | Both correct (%) | Only GCN (%) | Rescue (%) | Both wrong (%) | Rescued nodes | Rescue rate among GCN mistakes (%) |
|---|---:|---:|---:|---:|---:|---:|
| NetMF | 37.60 ± 1.76 | 33.76 ± 1.65 | 5.32 ± 0.85 | 23.32 ± 1.27 | 53.20 ± 8.53 | 18.52 ± 2.19 |
| GraphSAGE | 64.34 ± 2.12 | 7.02 ± 2.59 | 4.12 ± 1.34 | 24.52 ± 2.00 | 41.20 ± 13.42 | 14.39 ± 4.46 |

## Rescue location by GCN confidence

| Comparator | Confidence | Rescued nodes | Share of comparator rescues (%) | GCN mistakes rescued in group (%) |
|---|---|---:|---:|---:|
| NetMF | Low | 31.60 ± 5.86 | 59.35 ± 5.27 | 19.76 ± 3.29 |
| NetMF | Medium | 15.60 ± 4.39 | 29.38 ± 6.66 | 19.71 ± 5.34 |
| NetMF | High | 6.00 ± 1.87 | 11.27 ± 3.37 | 12.92 ± 4.58 |
| GraphSAGE | Low | 30.80 ± 7.63 | 76.53 ± 10.57 | 19.37 ± 4.94 |
| GraphSAGE | Medium | 8.40 ± 5.81 | 18.79 ± 7.12 | 10.45 ± 6.59 |
| GraphSAGE | High | 2.00 ± 1.87 | 4.68 ± 4.62 | 4.20 ± 3.81 |

Values are mean ± sample standard deviation across five seeds. Confidence groups are formed by
ranking GCN maximum probability on test nodes, without labels. Each seed has 334 low-confidence,
333 medium-confidence, and 333 high-confidence nodes. True labels are used only afterward to
measure retrospective correctness and rescue counts.
