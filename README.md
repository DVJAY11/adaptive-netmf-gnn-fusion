# Adaptive NetMF-GNN Fusion

Course project for CSC 591/791: Machine Learning with Graphs at NC State University.

This project studies whether NetMF and graph neural networks make complementary node-classification predictions. We compare simple feature concatenation and prediction averaging with a learned router that assigns a different NetMF, GCN, and GraphSAGE weight to each node.

## Research question

Does adaptive prediction fusion improve node classification compared with a standard GCN and the simpler approach of adding NetMF embeddings to the GCN input features?

## Datasets

The experiments use Cora, CiteSeer and PubMed from the Planetoid citation-network collection. Each node is a research paper, each edge is a citation, the node features represent words in the paper, and the target is the paper's topic.

| Dataset | Nodes | Edges | Features | Classes | Status | Download |
|---|---:|---:|---:|---:|---|---|
| Cora | 2,708 | 10,556 | 1,433 | 7 | Completed | [Download Cora (`.tgz`)](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz) |
| CiteSeer | 3,327 | 9,104 | 3,703 | 6 | Completed | [Download CiteSeer (`.tgz`)](https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz) |
| PubMed | 19,717 | 88,648 | 500 | 3 | Completed (eigen NetMF) | [Download PubMed (`.tgz`)](https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz) |

The experiment scripts use [PyTorch Geometric's Planetoid loader](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.datasets.Planetoid.html), so manual downloading is not required. The requested dataset is downloaded automatically on the first run and stored under `data/`.

## Main findings

The experiments use five random seeds and report mean test accuracy (± standard deviation across seeds). Standard deviations are 0.5–3.6 points on Cora and 1.0–7.1 points on CiteSeer, so differences of about one point should be treated as ties.

| Dataset | Method | 5 labels | 10 labels | 20 labels |
|---|---|---:|---:|---:|
| Cora | GCN | 72.48 ± 2.1 | 74.54 ± 3.6 | 78.02 ± 1.6 |
| Cora | NetMF + GCN | 73.12 ± 3.5 | 75.02 ± 2.3 | 78.16 ± 1.4 |
| Cora | Equal average | 72.88 ± 3.2 | 74.50 ± 2.7 | 77.82 ± 0.6 |
| Cora | Learned router | 71.66 ± 3.0 | 73.02 ± 2.8 | 77.20 ± 1.8 |
| Cora | Oracle (upper bound) | 82.32 | 83.76 | 86.52 |
| CiteSeer | GCN | 60.30 ± 4.9 | 64.82 ± 2.1 | 70.68 ± 1.1 |
| CiteSeer | NetMF + GCN | 59.66 ± 5.5 | 64.34 ± 4.2 | 70.86 ± 1.8 |
| CiteSeer | Equal average | 56.14 ± 7.1 | 59.66 ± 6.3 | 65.66 ± 3.3 |
| CiteSeer | Learned router | 58.88 ± 5.4 | 64.16 ± 3.1 | 70.20 ± 1.0 |
| CiteSeer | Oracle (upper bound) | 72.72 | 75.68 | 79.82 |

Cora with twenty labels per class as graph edges are removed:

| Edges removed | GCN | NetMF + GCN | Equal average | Learned router |
|---:|---:|---:|---:|---:|
| 0% | 78.02 ± 1.6 | 78.16 ± 1.4 | 77.82 ± 0.6 | 77.20 ± 1.8 |
| 20% | 76.30 ± 1.8 | 76.76 ± 1.3 | 76.52 ± 1.7 | 76.04 ± 1.6 |
| 40% | 72.86 ± 2.9 | 73.72 ± 1.7 | 72.54 ± 3.5 | 70.82 ± 4.0 |


A ten-seed rerun (seeds 42–51, `results_missing_edges_10seeds/`) checks whether NetMF + GCN's lead at 20% and 40% edge removal is real. It is not:

| Edges removed | GCN | NetMF + GCN | Equal average | Learned router |
|---:|---:|---:|---:|---:|
| 0% | 78.60 ± 1.8 | 78.53 ± 3.0 | 78.87 ± 1.6 | 78.17 ± 2.0 |
| 20% | 77.64 ± 2.0 | 77.64 ± 1.4 | 77.69 ± 2.1 | 76.69 ± 1.9 |
| 40% | 74.43 ± 2.7 | 74.57 ± 1.5 | 74.40 ± 3.3 | 73.12 ± 3.7 |

Paired over seeds, NetMF + GCN and equal averaging differ from GCN by at most 0.3 points (all p > 0.5). The learned router is worse than GCN at 20% removal (−0.95 points, p = 0.03) and at 40% removal (−1.31 points, p = 0.03, lower on all 10 seeds).

PubMed, ten seeds (42–51), mean test accuracy (%) ± standard deviation:

| Method | 5 labels | 10 labels | 20 labels |
|---|---:|---:|---:|
| GCN | 67.13 ± 5.7 | 74.09 ± 2.4 | 76.35 ± 2.0 |
| NetMF + GCN | 68.51 ± 4.5 | 74.53 ± 2.6 | 77.67 ± 1.8 |
| Equal average | 67.17 ± 5.2 | 74.31 ± 2.4 | 76.92 ± 2.3 |
| Learned router | 67.01 ± 5.6 | 74.08 ± 2.8 | 76.41 ± 1.6 |
| NetMF alone | 60.44 ± 3.7 | 67.92 ± 2.1 | 70.84 ± 2.7 |
| Oracle (upper bound) | 79.61 | 85.30 | 85.35 |

On Cora and CiteSeer no method reliably improves on a standard GCN. On PubMed, NetMF + GCN gives a small gain:


- **NetMF + GCN matches GCN.** Its accuracy is within one point of GCN's in every setting. With ten seeds, its lead when edges are removed disappears (+0.14 points at 40% removal, p = 0.81). This requires rescaling the NetMF features (see [Methods](#methods)). Without rescaling, the concatenation loses 14 points on Cora and 24 points on CiteSeer.
- **On PubMed, NetMF + GCN is about one point better than GCN.** The gain is +1.32 points at 20 labels (paired t-test p = 0.04, Wilcoxon p = 0.02, ahead on 7 of 10 seeds), smaller and not significant at 5 and 10 labels, and +1.05 points pooled over all 30 runs (Wilcoxon p = 0.01). PubMed's NetMF embedding comes from the eigen approximation (see below), so this is not a like-for-like comparison with Cora and CiteSeer.
- **Equal averaging ties GCN on Cora and is 5 points worse on CiteSeer**, where NetMF alone is weak (43%).
- **The learned router is within about one point of GCN but rarely beats it.** In most settings it gives NetMF almost no weight.
- **The oracle is 8.5–12.4 points above GCN.** The experts therefore make complementary mistakes, but the router's inputs (degree, confidence, entropy and disagreement) do not identify which expert is correct for a given node.

## Methods

### NetMF variants

`netmf.py` provides two NetMF implementations, selected with `--netmf-method` (default `auto`):

- **exact**: sums the first T powers of the random-walk transition matrix. Used for Cora and CiteSeer.
- **eigen**: approximates those sums from the top 256 eigenpairs of the normalized adjacency matrix (`--netmf-rank`), the large-window method from the NetMF paper. `auto` uses it above 10,000 nodes. On PubMed it takes about 16 seconds and under 3 GB of memory.


The pipeline evaluates twelve methods or analysis rows per experimental configuration:

- MLP without graph structure
- NetMF with logistic regression
- GCN
- GraphSAGE
- NetMF + GCN: rescaled NetMF embeddings concatenated with the original GCN features
- NetMF + GCN (unscaled): the same concatenation without rescaling, kept for comparison
- Equal probability averaging
- Fixed degree-based routing
- Learned routing using all router inputs
- Degree-only learned routing
- Confidence-and-entropy learned routing
- Oracle upper bound

The oracle uses the true label to select a correct expert whenever one exists. It measures routing headroom and is not a deployable model.

## NetMF feature scaling

The node features are row-normalized bag-of-words vectors with a mean L2 norm of about 0.25 on Cora and 0.18 on CiteSeer, while every NetMF row has an L2 norm of 1. If the two are concatenated directly, the NetMF columns dominate the GCN's first layer, and the model performs about as poorly as a GCN that receives only NetMF.

NetMF + GCN therefore rescales the NetMF block so that its mean row norm is a chosen multiple of the original features' mean row norm. The multiple is chosen from `--netmf-concat-scales` (default 0.1, 0.3 and 1.0) by accuracy on the base-model early-stopping validation split, and it is recorded in the `netmf_scale` column of `raw_results.csv`. Validation selected 0.1 or 0.3 in every run and never 1.0. The reported training time for NetMF + GCN includes all candidate trainings.

## Learned router

For each node, the router receives:

- Normalized log degree
- Maximum confidence from each expert
- Normalized prediction entropy from each expert
- Pairwise disagreement between expert predictions

A small MLP produces three softmax weights. The final prediction is the weighted sum of the NetMF, GCN, and GraphSAGE probability vectors.

## Experimental design

- Datasets: Cora, CiteSeer and PubMed
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
run_experiments.py        Main experiment pipeline (one run per seed, edge-drop rate and label budget)
netmf.py                  NetMF embeddings: exact (small graphs) and eigen approximation (large graphs)
models.py                 MLP, GCN, GraphSAGE, and the shared early-stopping training loop
fusion.py                 Equal averaging, degree router, learned router, and oracle
plot_results.py           Figures for a results directory
run_*.sh                  Reproduction scripts (smoke test, Cora, CiteSeer, PubMed, missing edges)
tests/                    Unit tests
diagnostics/              Feature-scaling and router-predictability studies, with their results
results_final_cora/       Five-seed Cora results and plots
results_citeseer/         Five-seed CiteSeer results and plots
results_pubmed/           Five-seed PubMed results and plots
results_pubmed_10seeds/   Ten-seed PubMed results and plots
results_missing_edges/    Five-seed Cora edge-removal results and plots
results_missing_edges_10seeds/  Ten-seed rerun of the edge-removal experiment
docs/OBSERVATIONS.md      Reproduction notes and the full record of every experiment and diagnostic
docs/FINAL_REPORT_AND_SLIDE_CONTENT.md  Report text: story, abstract, results tables, conclusion
docs/PROJECT_DIRECTION_AND_SUGGESTIONS.md  Planning notes written before the final experiments
presentation/             Intermediate and final presentation decks
CLAUDE.md                 What was done with AI assistance (Claude), and how it was checked
requirements-lock.txt     Exact package versions used for the committed results
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

To use the exact package versions that produced the committed results (Python 3.13), install `requirements-lock.txt` instead. Each results folder's `config.json` also records the package versions used.

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

Run the five-seed PubMed experiment (about 8 minutes on Apple MPS; ten seeds are in `results_pubmed_10seeds/`):

```bash
./run_pubmed.sh
```

The first run downloads the requested Planetoid dataset and computes its NetMF embedding. Later runs reuse the cached embedding. Each script automatically uses `.venv/bin/python` when the virtual environment exists.

## Output files

Each result directory contains:

- `raw_results.csv`: one row per method, seed, label budget, and edge setting (`netmf_scale` gives the selected NetMF + GCN scale)
- `summary.csv`: means and standard deviations
- `degree_group_results.csv`: low-, medium-, and high-degree accuracy
- `oracle_analysis.csv`: oracle accuracy, agreement, and unique expert contributions
- `router_weights.csv`: average expert weights by degree group
- `plots/`: presentation-ready figures

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover probability normalization, oracle correctness, router feature ablations, the disjoint validation split, NetMF feature scaling, and the eigen NetMF embedding.

## Diagnostics

Two studies explain the main results. Run them from the repository root:

```bash
python -m diagnostics.scaling_check --dataset Cora --cache-dir cache_final_cora
python -m diagnostics.predictability_check --cache-dir cache_final_cora --output diagnostics/router_predictability.csv
```

- `scaling_check` trains a GCN on the word features plus NetMF at several NetMF scales. It shows that the unscaled concatenation, not NetMF itself, causes the 14–24 point drop.
- `predictability_check` tests whether the router's inputs can tell which expert is right. Among nodes where an expert disagrees with GCN, they predict the correct one with an AUC of only 0.54–0.65. This explains why the learned router cannot beat GCN.

Their results are recorded in `docs/OBSERVATIONS.md`.

## Limitations

- The experiments use small transductive citation networks.
- The fixed degree router is a hypothesis-driven baseline. The results contradict its assumption: NetMF is least accurate on low-degree nodes (59% vs 74% for GCN on Cora with twenty labels).
- On CiteSeer, validation usually selected 0.1, the smallest NetMF scale on the grid. Smaller scales were not tested.
- Five seeds describe variation but do not establish broad statistical significance.
- Cora and CiteSeer use the exact NetMF matrix; PubMed uses the eigen approximation, because the exact walk sums need far more memory at that size. On Cora and CiteSeer the eigen version gives a better NetMF embedding than the exact one (NetMF alone: 70.4% vs 68.4% on Cora, 48.4% vs 42.9% on CiteSeer, 20 labels). NetMF + GCN on Cora and CiteSeer has not been rerun with it.
- Current runtime and memory files were produced on Apple MPS. The `peak_memory_mb` column does not capture MPS allocations and is close to zero, so NVIDIA memory should be measured separately before making hardware-efficiency claims.
- Results are deterministic for a fixed environment but individual runs can change by several points across package versions or hardware.

## Reuse predictions for additional experiments (Task 1)

Add `--save-predictions` to an existing run command. For example:

```bash
python run_experiments.py --dataset Cora --budgets 20 --seeds 42 \
  --output-dir results_additional/cora --save-predictions
```

This option **does not change training or accuracy calculations**. It writes one
compressed file per `(dataset, seed, label budget, edge-drop rate)` to
`<output-dir>/predictions/`. Files include the full-node probability matrix for
all existing methods (`prob__GCN`, `prob__GraphSAGE`, `prob__NetMF`,
`prob__NetMF_GCN`, the routers, etc.), node labels and degrees, all three
standard train/validation/test masks, and the three validation submasks used for
base-model early stopping and router training/stopping. Each file also stores
its identifying parameters. Each setting's cache is written atomically after
that setting finishes. Choose a fresh output directory for each distinct
training configuration to avoid overwriting files with the same dataset/seed/
budget/edge-drop but different hyperparameters.

Example of reading saved predictions:

```python
from prediction_cache import load_prediction_cache
p = load_prediction_cache(
    "results_additional/cora/predictions/Cora_seed42_budget20_drop0p000000.npz"
)
mask = p["test_mask"]
gcn_predictions = p["prob__GCN"][mask].argmax(axis=1)
true_labels = p["labels"][mask]
```

**Evaluation hygiene:** Use validation data to tune any new fusion weights,
and only then evaluate on the test mask. `prob__OracleUpperBound` uses ground-
truth labels and is an *analysis-only ceiling*, not an inference-time model.
Caching all-node labels for offline evaluation does not authorize their use for
training or tuning a new predictor.

Existing results CSV files cannot reconstruct per-node probabilities, so
older runs need to be repeated once with `--save-predictions` if the original
probability arrays were not separately preserved.
