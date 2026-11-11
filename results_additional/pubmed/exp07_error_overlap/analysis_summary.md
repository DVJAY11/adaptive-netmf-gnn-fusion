# GCN error overlap — PubMed

This retrospective analysis uses five cached runs (seeds 42–46, 20 labels per class) and does
not retrain any model. A rescue is a test node that GCN predicts incorrectly and the comparator
predicts correctly.

| Comparator | Both correct (%) | Only GCN (%) | Rescue (%) | Both wrong (%) | Rescued nodes | Rescue rate among GCN mistakes (%) |
|---|---:|---:|---:|---:|---:|---:|
| NetMF | 65.24 ± 0.27 | 12.46 ± 1.58 | 6.24 ± 2.04 | 16.06 ± 1.85 | 62.40 ± 20.35 | 27.85 ± 8.62 |
| GraphSAGE | 71.44 ± 1.37 | 6.26 ± 0.67 | 3.98 ± 0.63 | 18.32 ± 1.14 | 39.80 ± 6.30 | 17.81 ± 2.17 |

## Rescue location by GCN confidence

| Comparator | Confidence | Rescued nodes | Share of comparator rescues (%) | GCN mistakes rescued in group (%) |
|---|---|---:|---:|---:|
| NetMF | Low | 42.40 ± 10.76 | 70.02 ± 8.06 | 33.01 ± 8.61 |
| NetMF | Medium | 13.20 ± 6.42 | 19.76 ± 5.59 | 21.38 ± 9.33 |
| NetMF | High | 6.80 ± 3.70 | 10.22 ± 3.24 | 19.63 ± 9.61 |
| GraphSAGE | Low | 32.00 ± 5.70 | 80.67 ± 10.26 | 24.88 ± 4.40 |
| GraphSAGE | Medium | 4.80 ± 2.68 | 11.76 ± 6.18 | 7.54 ± 3.64 |
| GraphSAGE | High | 3.00 ± 2.55 | 7.57 ± 6.75 | 8.65 ± 7.03 |

Values are mean ± sample standard deviation across five seeds. Confidence groups are formed by
ranking GCN maximum probability on test nodes, without labels. Each seed has 334 low-confidence,
333 medium-confidence, and 333 high-confidence nodes. True labels are used only afterward to
measure retrospective correctness and rescue counts.
