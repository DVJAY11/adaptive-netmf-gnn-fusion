"""Does feature scale explain the weak unscaled NetMF + GCN baseline?

Trains a GCN on the word features concatenated with the NetMF embedding, with
the NetMF block rescaled to several multiples of the word features' mean row
L2 norm, and compares against a GCN on the word features alone.

Run from the repository root:

    python -m diagnostics.scaling_check --dataset Cora
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from models import GCN, seed_everything, train_model
from netmf import load_or_compute_netmf
from run_experiments import evaluate, make_limited_label_masks, split_validation_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=["Cora", "CiteSeer"], default="Cora")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--budgets", type=int, nargs="+", default=[5, 20])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--scales", type=float, nargs="+", default=[0.1, 0.3, 1.0],
                        help="NetMF block size relative to the word features' mean row norm")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, help="Optional CSV file for per-run results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = Planetoid(args.data_root, args.dataset, transform=NormalizeFeatures())
    graph = dataset[0]
    labels = graph.y.numpy()
    cache = args.cache_dir / f"{args.dataset.lower()}_drop0.00_seed0_w5_d128.npy"
    embedding, _, _ = load_or_compute_netmf(graph.edge_index, graph.num_nodes, cache, 128, 5, 0)

    x = graph.x.numpy()
    x_norm = np.linalg.norm(x, axis=1).mean()
    print(f"{args.dataset}: mean row L2 norm, word features {x_norm:.3f}, "
          f"NetMF {np.linalg.norm(embedding, axis=1).mean():.3f}")

    variants = {
        "words only (plain GCN)": x,
        "words + zeros (sanity check)": np.hstack([x, np.zeros_like(embedding)]),
        "NetMF only": embedding,
        "words + NetMF, unscaled": np.hstack([x, embedding]),
        **{f"words + NetMF x {s:g}": np.hstack([x, embedding * x_norm * s]) for s in args.scales},
    }

    device = torch.device(args.device)
    rows = []
    for name, features in variants.items():
        for budget in args.budgets:
            for seed in args.seeds:
                train, val, test = make_limited_label_masks(graph, budget, seed)
                base_stop, _, _ = split_validation_masks(labels, val, seed)
                variant = graph.clone()
                variant.x = torch.from_numpy(features.astype(np.float32))
                seed_everything(seed)
                model = GCN(variant.num_features, 64, dataset.num_classes, 0.5)
                result = train_model(model, variant, train, base_stop, device, 200, 30, 0.01, 5e-4)
                accuracy, _ = evaluate(result.probabilities, labels, test)
                rows.append({"variant": name, "budget": budget, "seed": seed, "test_accuracy": accuracy})

    results = pd.DataFrame(rows)
    if args.output:
        results.to_csv(args.output, index=False)
    table = results.groupby(["variant", "budget"], sort=False).test_accuracy.agg(["mean", "std"]).unstack("budget")
    print((table * 100).round(2).to_string())


if __name__ == "__main__":
    main()
