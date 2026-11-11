# Pairwise oracle analysis

This analysis uses cached predictions only. No models were trained or changed. The oracle is a
retrospective upper bound: a test node counts as correct when at least one included model predicts
its true label. It cannot be used for prediction because it depends on the true test labels.

## Configuration

- Dataset: PubMed
- Seed: 44
- Labels per class: 20
- Edge drop: 0.0
- Prediction cache: `/Users/rishitharamesh/Documents/GitHub/adaptive-netmf-gnn-fusion/results_additional/pubmed/predictions/PubMed_seed44_budget20_drop0p000000.npz`
- Test nodes: 1000

## Individual accuracy

| Model | Test accuracy |
|---|---:|
| GCN | 78.70% |
| NetMF | 67.90% |
| GraphSAGE | 76.40% |

## Oracle results

| Combination | Oracle accuracy | Best individual | Gain (points) | Rescued nodes | All wrong |
|---|---:|---:|---:|---:|---:|
| GCN + NetMF | 81.60% | 78.70% | 2.90 | 29 | 184 |
| GCN + GraphSAGE | 82.00% | 78.70% | 3.30 | 33 | 180 |
| NetMF + GraphSAGE | 80.90% | 76.40% | 4.50 | 45 | 191 |
| GCN + NetMF + GraphSAGE | 83.80% | 78.70% | 5.10 | 51 | 162 |

Among the pairs, **GCN + GraphSAGE** has the highest oracle accuracy at
**82.00%**. Its gain means that the two models make some different
errors: one model correctly classifies nodes missed by the stronger individual model. This shows
theoretical headroom, not that a practical fusion method will necessarily achieve that accuracy.

Using all three models raises the retrospective upper bound to **83.80%**.

## Pairwise error overlap

“Only A” and “Only B” follow the model order shown in the combination column.

| Combination | Both correct | Only A | Only B | Both wrong |
|---|---:|---:|---:|---:|
| GCN + NetMF | 650 | 137 | 29 | 184 |
| GCN + GraphSAGE | 731 | 56 | 33 | 180 |
| NetMF + GraphSAGE | 634 | 45 | 130 | 191 |

## Validation

- Individual accuracies match `/Users/rishitharamesh/Documents/GitHub/adaptive-netmf-gnn-fusion/results_additional/pubmed/raw_results.csv` within 0.00005 percentage points.
- Three-model oracle matches `/Users/rishitharamesh/Documents/GitHub/adaptive-netmf-gnn-fusion/results_additional/pubmed/oracle_analysis.csv` within 0.00005 percentage points.
- All cached model arrays have matching node and class dimensions and contain finite values.
- Accuracy uses only the saved test mask (1000 nodes).
- Every oracle accuracy is at least its included individual accuracies; all gains are nonnegative.
- Output reuse is allowed only when the existing result records the same prediction-file SHA-256.
