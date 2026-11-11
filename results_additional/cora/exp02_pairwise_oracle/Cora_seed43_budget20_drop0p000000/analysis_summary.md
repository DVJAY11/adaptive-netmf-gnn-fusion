# Pairwise oracle analysis

This analysis uses cached predictions only. No models were trained or changed. The oracle is a
retrospective upper bound: a test node counts as correct when at least one included model predicts
its true label. It cannot be used for prediction because it depends on the true test labels.

## Configuration

- Dataset: Cora
- Seed: 43
- Labels per class: 20
- Edge drop: 0.0
- Prediction cache: `/Users/rishitharamesh/Documents/GitHub/adaptive-netmf-gnn-fusion/results_additional/cora/predictions/Cora_seed43_budget20_drop0p000000.npz`
- Test nodes: 1000

## Individual accuracy

| Model | Test accuracy |
|---|---:|
| GCN | 79.00% |
| NetMF | 68.70% |
| GraphSAGE | 76.70% |

## Oracle results

| Combination | Oracle accuracy | Best individual | Gain (points) | Rescued nodes | All wrong |
|---|---:|---:|---:|---:|---:|
| GCN + NetMF | 84.90% | 79.00% | 5.90 | 59 | 151 |
| GCN + GraphSAGE | 82.90% | 79.00% | 3.90 | 39 | 171 |
| NetMF + GraphSAGE | 83.70% | 76.70% | 7.00 | 70 | 163 |
| GCN + NetMF + GraphSAGE | 86.90% | 79.00% | 7.90 | 79 | 131 |

Among the pairs, **GCN + NetMF** has the highest oracle accuracy at
**84.90%**. Its gain means that the two models make some different
errors: one model correctly classifies nodes missed by the stronger individual model. This shows
theoretical headroom, not that a practical fusion method will necessarily achieve that accuracy.

Using all three models raises the retrospective upper bound to **86.90%**.

## Pairwise error overlap

“Only A” and “Only B” follow the model order shown in the combination column.

| Combination | Both correct | Only A | Only B | Both wrong |
|---|---:|---:|---:|---:|
| GCN + NetMF | 628 | 162 | 59 | 151 |
| GCN + GraphSAGE | 728 | 62 | 39 | 171 |
| NetMF + GraphSAGE | 617 | 70 | 150 | 163 |

## Validation

- Individual accuracies match `/Users/rishitharamesh/Documents/GitHub/adaptive-netmf-gnn-fusion/results_additional/cora/raw_results.csv` within 0.00005 percentage points.
- Three-model oracle matches `/Users/rishitharamesh/Documents/GitHub/adaptive-netmf-gnn-fusion/results_additional/cora/oracle_analysis.csv` within 0.00005 percentage points.
- All cached model arrays have matching node and class dimensions and contain finite values.
- Accuracy uses only the saved test mask (1000 nodes).
- Every oracle accuracy is at least its included individual accuracies; all gains are nonnegative.
- Output reuse is allowed only when the existing result records the same prediction-file SHA-256.
