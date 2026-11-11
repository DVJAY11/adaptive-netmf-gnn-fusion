# Pairwise oracle analysis

This analysis uses cached predictions only. No models were trained or changed. The oracle is a
retrospective upper bound: a test node counts as correct when at least one included model predicts
its true label. It cannot be used for prediction because it depends on the true test labels.

## Configuration

- Dataset: PubMed
- Seed: 42
- Labels per class: 20
- Edge drop: 0.0
- Prediction cache: `/Users/rishitharamesh/Documents/GitHub/adaptive-netmf-gnn-fusion/results_additional/pubmed/predictions/PubMed_seed42_budget20_drop0p000000.npz`
- Test nodes: 1000

## Individual accuracy

| Model | Test accuracy |
|---|---:|
| GCN | 79.80% |
| NetMF | 71.80% |
| GraphSAGE | 76.30% |

## Oracle results

| Combination | Oracle accuracy | Best individual | Gain (points) | Rescued nodes | All wrong |
|---|---:|---:|---:|---:|---:|
| GCN + NetMF | 86.40% | 79.80% | 6.60 | 66 | 136 |
| GCN + GraphSAGE | 83.40% | 79.80% | 3.60 | 36 | 166 |
| NetMF + GraphSAGE | 84.20% | 76.30% | 7.90 | 79 | 158 |
| GCN + NetMF + GraphSAGE | 88.00% | 79.80% | 8.20 | 82 | 120 |

Among the pairs, **GCN + NetMF** has the highest oracle accuracy at
**86.40%**. Its gain means that the two models make some different
errors: one model correctly classifies nodes missed by the stronger individual model. This shows
theoretical headroom, not that a practical fusion method will necessarily achieve that accuracy.

Using all three models raises the retrospective upper bound to **88.00%**.

## Pairwise error overlap

“Only A” and “Only B” follow the model order shown in the combination column.

| Combination | Both correct | Only A | Only B | Both wrong |
|---|---:|---:|---:|---:|
| GCN + NetMF | 652 | 146 | 66 | 136 |
| GCN + GraphSAGE | 727 | 71 | 36 | 166 |
| NetMF + GraphSAGE | 639 | 79 | 124 | 158 |

## Validation

- Individual accuracies match `/Users/rishitharamesh/Documents/GitHub/adaptive-netmf-gnn-fusion/results_additional/pubmed/raw_results.csv` within 0.00005 percentage points.
- Three-model oracle matches `/Users/rishitharamesh/Documents/GitHub/adaptive-netmf-gnn-fusion/results_additional/pubmed/oracle_analysis.csv` within 0.00005 percentage points.
- All cached model arrays have matching node and class dimensions and contain finite values.
- Accuracy uses only the saved test mask (1000 nodes).
- Every oracle accuracy is at least its included individual accuracies; all gains are nonnegative.
- Output reuse is allowed only when the existing result records the same prediction-file SHA-256.
