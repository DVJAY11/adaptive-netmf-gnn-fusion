# Final Project Direction and Suggestions

## Project title

Adaptive Fusion of NetMF and Graph Neural Networks for Node Classification

## What we are predicting

We are predicting the research topic of each paper in a citation network.

In Cora:

- Each node is a research paper.
- Each edge is a citation between two papers.
- Each node has 1,433 word features.
- Each node belongs to one of seven topic classes.
- The task is node classification.

## Revised problem statement

Different graph models use different information:

- NetMF uses multi-hop graph structure derived from random walks.
- GCN uses node features and local neighborhood information.
- GraphSAGE learns a separate transformation for the node and its neighborhood.

One model may not be best for every node or every graph condition. We want to study whether combining these representations improves node classification when labels are limited or graph edges are missing.

The revised research question is:

> Does NetMF provide useful complementary information to GNNs, and is a learned node-specific combination better than simpler fusion methods?

This wording is safer than assuming that NetMF is automatically better for sparse graphs or low-degree nodes.

## Important correction about sparse graphs

We should not claim that NetMF performs better on sparse graphs or low-degree nodes.

NetMF also depends on graph connectivity because it learns from random-walk paths. A node with few edges may have weak structural information. GCN can still use the node's word features even when it has few neighbors.

Our degree router was based on a hypothesis:

> Low-degree nodes may benefit from more NetMF weight because NetMF captures multi-hop structure.

This is not yet a result. We must test it using degree-wise accuracy and missing-edge experiments.

The correct claim is:

> NetMF captures structural information that differs from GNN message passing. We will test whether this complementary information helps when local neighborhoods are limited or damaged.

## What we have completed

We completed an intermediate baseline on Cora with:

- Label budgets of 5, 10, and 20 labeled nodes per class
- Random seeds 42, 43, and 44
- No edge removal
- Up to 200 epochs for the neural models
- Early stopping patience of 30 epochs
- Six evaluated methods

The six methods were:

1. MLP
2. NetMF
3. GCN
4. GraphSAGE
5. Equal averaging
6. Fixed degree router

The 54 recorded results come from:

```text
6 methods x 3 label budgets x 3 seeds = 54 result rows
```

Equal averaging and the degree router do not require separate model training. Therefore, 54 means 54 evaluated method results, not 54 independently trained neural networks.

## Current results

Mean test accuracy on Cora:

| Method | 5 labels | 10 labels | 20 labels |
|---|---:|---:|---:|
| MLP | 44.47% | 49.40% | 56.03% |
| NetMF | 51.37% | 61.67% | 68.03% |
| GCN | 72.23% | 76.20% | 78.53% |
| GraphSAGE | 71.63% | 74.53% | 75.67% |
| Equal average | 72.37% | 75.43% | 78.20% |
| Degree router | 71.70% | 75.20% | 78.17% |

### What the current results show

- Graph information is useful because GCN and GraphSAGE perform much better than the feature-only MLP.
- More labels improve every method.
- GCN is the strongest overall method.
- Equal averaging is slightly best at five labels, but the difference from GCN is small.
- Equal averaging and the degree router are close to GCN at twenty labels.
- The fixed degree router does not currently justify its added assumptions.
- We cannot claim statistical significance from only three seeds.

### What the current results do not show

- They do not show that NetMF is better for low-degree nodes.
- They do not show that the router improves GCN.
- They do not include missing-edge experiments.
- They do not include CiteSeer or PubMed.
- They do not include the learned router.
- They do not include the NetMF-plus-GCN feature-concatenation baseline.
- They do not include reliable NVIDIA GPU-memory measurements.

## Existing methods and our contribution

The following ideas already exist:

- NetMF
- GCN
- GraphSAGE
- Model ensembles and equal averaging
- Graph mixture-of-experts and node-wise routing

We should not claim that we invented graph mixture-of-experts.

Our defensible contribution is an applied comparison of heterogeneous graph experts:

- A matrix-factorization representation from NetMF
- A convolutional message-passing model from GCN
- A neighborhood aggregation model from GraphSAGE
- Simple early fusion
- Fixed late fusion
- Learned node-specific late fusion

We will test these methods under limited labels and missing edges and analyze which node groups benefit.

## Most important new baseline: NetMF plus GCN

The professor suggested computing NetMF embeddings and giving them directly to GCN. This is an important and simpler approach.

NetMF produces a 128-dimensional embedding for each node. We concatenate it with the original 1,433 node features:

```text
1,433 original features + 128 NetMF features = 1,561 GCN input features
```

GCN then uses:

- Original word features
- NetMF structural features
- Citation edges for message passing

This is feature-level or early fusion.

### Why we need this baseline

- It is simpler than running three models and a router.
- It allows GCN to learn which NetMF dimensions are useful.
- It adds only 8,192 weights to the first GCN layer when the hidden size is 64.
- It may combine global and local structure more directly.
- If it beats the router, we should prefer the simpler model.

The learned router is useful only if it performs better, improves robustness, or helps a specific group of difficult nodes.

## Fusion methods to compare

### GCN alone

This is the strongest current baseline.

Purpose:

- Establish the accuracy and cost that more complex methods must beat.

### NetMF plus GCN feature concatenation

Concatenate NetMF embeddings with the original node features before GCN.

Purpose:

- Test the simplest way to give global structural information to GCN.

### Equal averaging

Average the probabilities from NetMF, GCN, and GraphSAGE:

```text
33.3% NetMF + 33.3% GCN + 33.3% GraphSAGE
```

Purpose:

- Show whether a simple ensemble is enough.

### Fixed degree router

Current manually defined weights:

| Degree group | NetMF | GCN | GraphSAGE |
|---|---:|---:|---:|
| Low | 50% | 25% | 25% |
| Medium | 34% | 33% | 33% |
| High | 20% | 40% | 40% |

Purpose:

- Test the hypothesis that model usefulness changes with node degree.
- Provide an interpretable baseline.

Limitation:

- The weights are manually selected and not learned.
- The assumption that NetMF helps low-degree nodes is not established.

### Learned router

Freeze the trained NetMF, GCN, and GraphSAGE models. For every node, calculate router features such as:

- Normalized node degree
- Confidence of each model
- Entropy or uncertainty of each model
- Disagreement among model predictions

A small MLP outputs three weights:

```text
NetMF weight, GCN weight, GraphSAGE weight
```

Softmax makes the weights positive and ensures they add up to one. The final prediction is the weighted sum of the three probability vectors.

Purpose:

- Learn the combination from data instead of manually choosing weights.
- Test whether different nodes benefit from different experts.

## Learned-router data split

The current Cora experiment uses:

- A limited training set for the base models
- 500 validation nodes for early stopping
- 1,000 test nodes for final evaluation

The learned router needs separate data to avoid test leakage. A simple plan is:

1. Train the base models on the limited training set.
2. Divide the current validation set into router-training and router-validation subsets.
3. Train the router on router-training labels.
4. Use router-validation nodes for early stopping and router settings.
5. Evaluate only once on the untouched test set.

A more rigorous but more complex option is to use out-of-fold base-model predictions for router training.

## Ablation studies

An ablation removes one component to see whether it matters.

Router-input ablations:

- Degree only
- Confidence only
- Disagreement only
- Degree plus confidence
- Degree plus disagreement
- All router inputs

Model ablations:

- GCN and GraphSAGE without NetMF
- NetMF and GCN without GraphSAGE
- NetMF and GraphSAGE without GCN
- All three models

Fusion ablations:

- GCN alone
- NetMF plus GCN concatenation
- Equal averaging
- Fixed degree router
- Learned router

Purpose:

- Determine whether the learned router is needed.
- Determine whether NetMF contributes useful information.
- Identify which router inputs are useful.

## Oracle analysis

For each test node, check whether any of NetMF, GCN, or GraphSAGE predicted the correct class. This produces an oracle upper bound representing a perfect expert selector.

Purpose:

- Measure how complementary the base models are.
- Determine whether routing has enough potential to be worthwhile.

Interpretation:

- If oracle accuracy is only slightly above GCN, routing has little room to help.
- If oracle accuracy is much higher than GCN, better routing may provide a meaningful gain.

The oracle is an analysis tool, not a deployable model, because it uses the true answer to choose an expert.

## Degree-wise analysis

Report accuracy separately for:

- Low-degree nodes
- Medium-degree nodes
- High-degree nodes

Also report the average router weight assigned to each model in each degree group.

Purpose:

- Test the assumption behind the fixed degree router.
- Show whether any method helps a specific node group even when overall accuracy does not improve.

For the current Cora graph, the approximate degree groups used by the router are:

- Low: degree 1 or 2
- Medium: degree 3 or 4
- High: degree greater than 4

## Missing-edge experiments

Run each method with:

- 0% removed edges
- 20% removed edges
- 40% removed edges

Use the same edge-removal mask for all methods under a seed.

Important:

- Recompute NetMF embeddings from the damaged graph.
- Train GNNs using the same damaged graph.
- Do not compute NetMF from the full graph and GNNs from the damaged graph.

Purpose:

- Test robustness when citation information is incomplete.
- Determine whether one representation degrades more slowly than another.
- Test whether fusion helps when the graph is damaged.

## Final experimental plan

### Datasets

Required:

- Cora
- CiteSeer

Optional if time and memory permit:

- PubMed

### Label budgets

- 5 labels per class
- 10 labels per class
- 20 labels per class
- 40 labels per class

### Seeds

- Five seeds for final results

The current intermediate results use only three seeds. Presentation wording should clearly say:

> Intermediate results use three seeds. Final experiments will use five seeds.

### Edge-drop rates

- 0%
- 20%
- 40%

### Main methods

At minimum:

1. MLP
2. NetMF
3. GCN
4. GraphSAGE
5. NetMF plus GCN feature concatenation
6. Equal averaging
7. Fixed degree router
8. Learned router

If time is limited, prioritize GCN, NetMF plus GCN, equal averaging, and the learned router after verifying the individual base models.

## Evaluation metrics

Report:

- Test accuracy
- Macro-F1
- Mean across seeds
- Standard deviation across seeds
- Training time
- NetMF preprocessing time
- Peak GPU memory when measured on the L40S

Current memory values are not final. The intermediate run used Apple MPS, so it did not provide reliable NVIDIA GPU-memory measurements. Do not claim that memory evaluation is complete.

## Complexity and cost

Current trainable parameter counts:

- GCN: 92,231
- GraphSAGE: 184,391
- NetMF logistic-regression classifier: 903

The fixed degree router and equal averaging have no trainable parameters.

The learned router will add only a small number of parameters. The main cost comes from computing and running all three base models.

NetMF plus GCN is cheaper at prediction time than the three-model router because it uses one GCN after NetMF preprocessing.

The router must therefore provide enough accuracy, robustness, or interpretability to justify its additional cost.

## GraphSAGE clarification

GraphSAGE supports neighbor sampling in general. However, the current Cora implementation uses full-graph training with `SAGEConv` and does not use a neighbor-sampling loader.

Do not claim that the current intermediate experiment demonstrates sampling-based speed or scalability.

## How to position the project

Use this description:

> We compare simple and learned ways to combine a global matrix-factorization representation with local message-passing GNNs. We test whether the additional structural representation helps under limited labels and missing graph edges, and we analyze which nodes benefit.

Avoid this description:

> NetMF works better for sparse or low-degree nodes, so our router improves GCN.

The second statement is an untested assumption and is not supported by the current results.

## Decision rule for the final project

- If NetMF plus GCN clearly performs best, use it as the main method and present routing as a comparison.
- If the learned router performs best, use the router as the main contribution.
- If GCN remains best overall, analyze whether fusion helps under missing edges or for a specific node group.
- If no fusion method helps, present the result honestly and explain why NetMF and GNN predictions were not sufficiently complementary.

The goal is to answer the research question, not to force the router to win.

## Short explanation for the professor

> Our intermediate baseline shows that GCN is currently strongest and that simple fusion is close but does not improve it. We agree that assuming NetMF works better for sparse nodes is not justified. We will add the simpler NetMF-plus-GCN feature-concatenation baseline, a learned router using confidence and disagreement, degree-wise analysis, and missing-edge tests. The final project will determine whether adaptive fusion provides a real benefit or whether the simpler GCN-based model is preferable.

