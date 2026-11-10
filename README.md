# Adaptive NetMF–GNN Fusion

This folder implements the experiment pipeline for the course project. It trains MLP, NetMF, GCN, and GraphSAGE baselines on a Planetoid citation network. It also tests a GCN that receives NetMF embeddings as extra features, two prediction routers, and an oracle upper bound.

## What the code produces

- Baselines: MLP, NetMF, GCN, GraphSAGE, and NetMF+GCN feature concatenation
- Fusion methods: equal averaging, a fixed degree router, and a learned router
- Oracle analysis: the accuracy possible if a perfect selector always chose a correct expert
- Router ablations: degree only, confidence and entropy only, and all inputs
- Limited-label settings: 5, 10, and 20 labeled nodes per class by default
- Optional missing-edge settings through `--edge-drop-rates`
- Metrics: validation/test accuracy, macro-F1, training time, preprocessing time, and peak memory
- Repeated runs with configurable random seeds
- Result files:
  - `results/raw_results.csv`
  - `results/summary.csv`
  - `results/router_weights.csv`
  - `results/oracle_analysis.csv`
  - `results/degree_group_results.csv`
  - `results/plots/accuracy_vs_labels.png`
  - `results/plots/model_comparison.png`
  - `results/plots/router_weights.png`

## 1. Create the environment

Run these commands from this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On the NCSU Ubuntu GPU machine, check that PyTorch can see the GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The experiments also run on CPU, but a CUDA GPU is faster.

## 2. Run a quick installation check

```bash
chmod +x run_smoke.sh run_intermediate.sh
./run_smoke.sh
```

The smoke run uses one seed, one label budget, five training epochs, and a smaller NetMF embedding. It checks the full pipeline without waiting for the complete experiments.
Its files are written to `results_smoke/`, so it does not overwrite the full intermediate results.

## 3. Generate the intermediate results

```bash
./run_intermediate.sh
```

Equivalent direct commands:

```bash
python run_experiments.py \
  --dataset Cora \
  --budgets 5 10 20 \
  --seeds 42 43 44 \
  --epochs 200 \
  --patience 30 \
  --device auto

python plot_results.py --results-dir results
```

This is the recommended intermediate-results run. It uses three random seeds to keep the turnaround reasonable.

## 4. Run the final five-seed version

```bash
./run_final_cora.sh
```

## 5. Add missing-edge experiments later

The dedicated script preserves the limited-label results:

```bash
./run_missing_edges.sh
```

## Notes

- The first run downloads Cora automatically and computes the NetMF embedding. Later runs reuse the file in `cache/`.
- The code uses the official Planetoid validation and test masks. It samples the requested number of training nodes per class from the remaining nodes.
- The degree-based router gives more NetMF weight to low-degree nodes and more GNN weight to high-degree nodes. This is an interpretable intermediate baseline, not the final learned router.
- The official validation set is divided by class: 50% stops the base models, 30% trains the router, and 20% stops the router. The three subsets never overlap.
- The learned router uses node degree, expert confidence, normalized entropy, and pairwise prediction disagreement. Degree-only and confidence-only variants provide ablations.
- `NetMF_GCN` concatenates the NetMF embedding with the original node features before the first GCN layer.
- `OracleUpperBound` uses the true labels to select a correct expert whenever possible. It is diagnostic only and cannot be used as a real prediction method.
- For PubMed, explicit NetMF can require much more memory. Start with Cora for the intermediate submission.
- `run_citeseer.sh` performs the same five-seed experiment on a second small dataset.
