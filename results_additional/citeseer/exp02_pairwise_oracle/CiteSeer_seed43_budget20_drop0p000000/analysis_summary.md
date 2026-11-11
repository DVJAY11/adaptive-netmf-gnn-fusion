# Pairwise oracle analysis

This analysis uses cached predictions only. No models were trained or changed. The oracle is a
retrospective upper bound: a test node counts as correct when at least one included model predicts
its true label. It cannot be used for prediction because it depends on the true test labels.

## Configuration

- Dataset: CiteSeer
- Seed: 43
- Labels per class: 20
- Edge drop: 0.0
- Prediction cache: `/Users/rishitharamesh/Documents/GitHub/adaptive-netmf-gnn-fusion/results_additional/citeseer/predictions/CiteSeer_seed43_budget20_drop0p000000.npz`
- Test nodes: 1000

## Individual accuracy

| Model | Test accuracy |
|---|---:|
| GCN | 69.70% |
| NetMF | 43.50% |
| GraphSAGE | 66.90% |

## Oracle results

| Combination | Oracle accuracy | Best individual | Gain (points) | Rescued nodes | All wrong |
|---|---:|---:|---:|---:|---:|
| GCN + NetMF | 74.80% | 69.70% | 5.10 | 51 | 252 |
| GCN + GraphSAGE | 72.60% | 69.70% | 2.90 | 29 | 274 |
| NetMF + GraphSAGE | 73.60% | 66.90% | 6.70 | 67 | 264 |
| GCN + NetMF + GraphSAGE | 77.00% | 69.70% | 7.30 | 73 | 230 |

Among the pairs, **GCN + NetMF** has the highest oracle accuracy at
**74.80%**. Its gain means that the two models make some different
errors: one model correctly classifies nodes missed by the stronger individual model. This shows
theoretical headroom, not that a practical fusion method will necessarily achieve that accuracy.

Using all three models raises the retrospective upper bound to **77.00%**.

## Pairwise error overlap

“Only A” and “Only B” follow the model order shown in the combination column.

| Combination | Both correct | Only A | Only B | Both wrong |
|---|---:|---:|---:|---:|
| GCN + NetMF | 384 | 313 | 51 | 252 |
| GCN + GraphSAGE | 640 | 57 | 29 | 274 |
| NetMF + GraphSAGE | 368 | 67 | 301 | 264 |

## Validation

- Individual accuracies match `/Users/rishitharamesh/Documents/GitHub/adaptive-netmf-gnn-fusion/results_additional/citeseer/raw_results.csv` within 0.00005 percentage points.
- Three-model oracle matches `/Users/rishitharamesh/Documents/GitHub/adaptive-netmf-gnn-fusion/results_additional/citeseer/oracle_analysis.csv` within 0.00005 percentage points.
- All cached model arrays have matching node and class dimensions and contain finite values.
- Accuracy uses only the saved test mask (1000 nodes).
- Every oracle accuracy is at least its included individual accuracies; all gains are nonnegative.
- Output reuse is allowed only when the existing result records the same prediction-file SHA-256.
