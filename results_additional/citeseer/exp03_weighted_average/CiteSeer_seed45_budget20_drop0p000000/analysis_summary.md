# Weighted prediction averaging

This is a **single-seed experiment** on CiteSeer (seed 45,
20 labels per class, edge drop 0.0). It uses cached
probabilities only; no model was retrained. Candidate fixed weights range from 0.0 to 1.0 in
steps of 0.1. Pair weights sum to one, and the three-model search covers all integer-tenths
weight triples that sum to one.

## Selected weights and held-out test results

| Method | Selected weights | Validation accuracy | Test accuracy |
|---|---|---:|---:|
| Weighted GCN + NetMF | GCN=1.0, NetMF=0.0, GraphSAGE=0.0 | 71.80% | 71.20% |
| Weighted GCN + GraphSAGE | GCN=0.9, NetMF=0.0, GraphSAGE=0.1 | 71.80% | 71.20% |
| Weighted GCN + NetMF + GraphSAGE | GCN=0.9, NetMF=0.0, GraphSAGE=0.1 | 71.80% | 71.20% |

Ties in validation correct-node count are resolved by choosing the candidate closest to equal
weighting, then deterministically by its weight tuple. Test accuracy never participates in
selection. The complete 88-candidate validation search is in `weight_search.csv`.

## Cached baselines

| Method | Reported validation accuracy | Test accuracy |
|---|---:|---:|
| GCN | 71.80% | 71.20% |
| EqualAverage | 59.60% | 59.80% |
| LearnedRouter | 71.80% | 71.30% |

The strongest selected weighted average on this test split is **Weighted GCN + GraphSAGE** at
**71.20%** (GCN=0.9, NetMF=0.0, GraphSAGE=0.1). This result comes from one seed and
does not establish a general improvement.

## Learned-router comparability warning

The learned router used 151 nodes from `router_train_mask` to fit its parameters
and 100 nodes from `router_stop_mask` for early stopping. All 251
of those nodes are included in the 500-node saved validation mask used to report its
validation accuracy. Therefore its reported validation accuracy is **not directly comparable** to
the validation score used to select fixed averaging weights. Held-out test accuracy is the cleaner
comparison. The fixed-weight search itself also reuses its validation score for model selection,
so that score should not be interpreted as performance on an untouched split.

## Validation checks

- Cached GCN, EqualAverage, and LearnedRouter scores match `/Users/rishitharamesh/Documents/GitHub/adaptive-netmf-gnn-fusion/results_additional/citeseer/raw_results.csv`.
- All required probability arrays have matching node and class dimensions and finite values.
- Validation and test masks are nonempty, aligned to the arrays, and disjoint.
- The cached equal average matches the mean of NetMF, GCN, and GraphSAGE probabilities.
- Weight selection receives only validation labels and predictions; test evaluation occurs after
  selected candidates are fixed.
- `weight_search.csv` contains validation results only, preventing candidate selection from test scores.
- Existing artifacts are overwritten only for the same cache hash, or after an explicit notice
  when the same cache path and dataset/seed/budget/edge-drop configuration are detected.

Prediction cache: `/Users/rishitharamesh/Documents/GitHub/adaptive-netmf-gnn-fusion/results_additional/citeseer/predictions/CiteSeer_seed45_budget20_drop0p000000.npz`
