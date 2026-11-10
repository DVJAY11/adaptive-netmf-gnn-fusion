# Adaptive NetMF-GNN Fusion

Course project for CSC 591/791: Machine Learning with Graphs at NC State University.

This project studies whether NetMF and graph neural networks make complementary node-classification predictions. We compare simple feature concatenation and prediction averaging with a learned router that assigns a different NetMF, GCN, and GraphSAGE weight to each node.

## Research question

Does adaptive prediction fusion improve node classification compared with a standard GCN and the simpler approach of adding NetMF embeddings to the GCN input features?

## Main findings

The experiments use five random seeds and report mean test accuracy.

| Dataset | Method | 5 labels | 10 labels | 20 labels |
|---|---|---:|---:|---:|
| Cora | GCN | **72.28%** | **74.54%** | 78.26% |
| Cora | Learned router | 71.52% | 72.32% | 77.42% |
| Cora | Equal average | 71.90% | 73.38% | **78.32%** |
| Cora | NetMF + GCN | 51.42% | 58.16% | 64.32% |
| CiteSeer | GCN | 58.44% | 64.06% | **70.94%** |
| CiteSeer | Learned router | 58.20% | 64.32% | 70.62% |
| CiteSeer | Equal average | 57.40% | 59.32% | 67.36% |
| CiteSeer | NetMF + GCN | 32.92% | 40.20% | 46.38% |

On Cora, GCN also remains strongest as graph edges are removed:

| Edges removed | GCN | Learned router | Equal average |
|---:|---:|---:|---:|
| 0% | 78.26% | 77.42% | **78.32%** |
| 20% | **76.80%** | 75.84% | 76.18% |
| 40% | **73.28%** | 72.98% | 72.68% |

The learned router is competitive, but it does not provide a consistent improvement over GCN. Equal averaging gives a very small gain on Cora with twenty labels per class, while NetMF feature concatenation performs substantially worse. The oracle analysis shows that the experts make complementary mistakes, but the current router features do not reliably identify the correct expert.

## Methods

The pipeline evaluates eleven methods or analysis rows per experimental configuration:

- MLP without graph structure
- NetMF with logistic regression
- GCN
- GraphSAGE
- NetMF embeddings concatenated with the original GCN features
- Equal probability averaging
- Fixed degree-based routing
- Learned routing using all router inputs
- Degree-only learned routing
- Confidence-and-entropy learned routing
- Oracle upper bound

The oracle uses the true label to select a correct expert whenever one exists. It measures routing headroom and is not a deployable model.

## Learned router

For each node, the router receives:

- Normalized log degree
- Maximum confidence from each expert
- Normalized prediction entropy from each expert
- Pairwise disagreement between expert predictions

A small MLP produces three softmax weights. The final prediction is the weighted sum of the NetMF, GCN, and GraphSAGE probability vectors.

## Experimental design

- Datasets: Cora and CiteSeer
- Training budgets: 5, 10, and 20 labeled nodes per class
- Seeds: 42, 43, 44, 45, and 46
- Robustness test: 0%, 20%, and 40% edge removal on Cora
- Metrics: test accuracy, macro-F1, training time, preprocessing time, and peak memory

The official validation nodes are divided by class into three disjoint subsets:

- 50% for base-model early stopping
- 30% for router training
- 20% for router early stopping

The test labels remain untouched until final evaluation. For missing-edge experiments, every method receives the same damaged graph within a seed, and NetMF is recomputed from that graph.

## Repository structure

```text
fusion.py                 Fusion methods, learned router, and oracle
models.py                 MLP, GCN, GraphSAGE, and training loop
netmf.py                  Explicit NetMF implementation for small graphs
run_experiments.py        Main experiment pipeline
plot_results.py           Result visualization
tests/                    Unit tests
results_final_cora/       Five-seed Cora results and plots
results_missing_edges/    Cora edge-removal results and plots
results_citeseer/         Five-seed CiteSeer results and plots
artifacts/                Report and presentation templates
```

Datasets, virtual environments, NetMF caches, and obsolete intermediate outputs are excluded from Git.

## Setup

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check the installation with a short smoke run:

```bash
./run_smoke.sh
```

## Reproduce the experiments

Run the five-seed Cora experiment:

```bash
./run_final_cora.sh
```

Run the Cora missing-edge experiment:

```bash
./run_missing_edges.sh
```

Run the five-seed CiteSeer experiment:

```bash
./run_citeseer.sh
```

The first run downloads the requested Planetoid dataset and computes its NetMF embedding. Later runs reuse the cached embedding. Each script automatically uses `.venv/bin/python` when the virtual environment exists.

## Output files

Each result directory contains:

- `raw_results.csv`: one row per method, seed, label budget, and edge setting
- `summary.csv`: means and standard deviations
- `degree_group_results.csv`: low-, medium-, and high-degree accuracy
- `oracle_analysis.csv`: oracle accuracy, agreement, and unique expert contributions
- `router_weights.csv`: average expert weights by degree group
- `plots/`: presentation-ready figures

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover probability normalization, oracle correctness, router feature ablations, and the disjoint validation split.

## Limitations

- The experiments use small transductive citation networks.
- The fixed degree router is a hypothesis-driven baseline; the results do not show that NetMF is better for low-degree nodes.
- Five seeds describe variation but do not establish broad statistical significance.
- The explicit NetMF matrix is intended for Cora and CiteSeer. PubMed requires a more memory-efficient implementation.
- Current runtime and memory files were produced on Apple MPS. NVIDIA memory should be measured separately before making hardware-efficiency claims.
