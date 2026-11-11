# Pairwise oracle analysis

This analysis uses cached predictions only. No models were trained or changed. The oracle is a
retrospective upper bound: a test node counts as correct when at least one included model predicts
its true label. It cannot be used for prediction because it depends on the true test labels.

## Configuration

- Dataset: CiteSeer
- Seed: 45
- Labels per class: 20
- Edge drop: 0.0
- Prediction cache: `/Users/rishitharamesh/Documents/GitHub/adaptive-netmf-gnn-fusion/results_additional/citeseer/predictions/CiteSeer_seed45_budget20_drop0p000000.npz`
- Test nodes: 1000

## Individual accuracy

| Model | Test accuracy |
|---|---:|
| GCN | 71.20% |
| NetMF | 42.90% |
| GraphSAGE | 71.20% |

## Oracle results

| Combination | Oracle accuracy | Best individual | Gain (points) | Rescued nodes | All wrong |
|---|---:|---:|---:|---:|---:|
| GCN + NetMF | 76.60% | 71.20% | 5.40 | 54 | 234 |
| GCN + GraphSAGE | 74.60% | 71.20% | 3.40 | 34 | 254 |
| NetMF + GraphSAGE | 76.00% | 71.20% | 4.80 | 48 | 240 |
| GCN + NetMF + GraphSAGE | 78.70% | 71.20% | 7.50 | 75 | 213 |

Among the pairs, **GCN + NetMF** has the highest oracle accuracy at
**76.60%**. Its gain means that the two models make some different
errors: one model correctly classifies nodes missed by the stronger individual model. This shows
theoretical headroom, not that a practical fusion method will necessarily achieve that accuracy.

Using all three models raises the retrospective upper bound to **78.70%**.

## Pairwise error overlap

“Only A” and “Only B” follow the model order shown in the combination column.

| Combination | Both correct | Only A | Only B | Both wrong |
|---|---:|---:|---:|---:|
| GCN + NetMF | 375 | 337 | 54 | 234 |
| GCN + GraphSAGE | 678 | 34 | 34 | 254 |
| NetMF + GraphSAGE | 381 | 48 | 331 | 240 |

## Validation

- Individual accuracies match `/Users/rishitharamesh/Documents/GitHub/adaptive-netmf-gnn-fusion/results_additional/citeseer/raw_results.csv` within 0.00005 percentage points.
- Three-model oracle matches `/Users/rishitharamesh/Documents/GitHub/adaptive-netmf-gnn-fusion/results_additional/citeseer/oracle_analysis.csv` within 0.00005 percentage points.
- All cached model arrays have matching node and class dimensions and contain finite values.
- Accuracy uses only the saved test mask (1000 nodes).
- Every oracle accuracy is at least its included individual accuracies; all gains are nonnegative.
- Output reuse is allowed only when the existing result records the same prediction-file SHA-256.
