"""Can the router's inputs tell which expert is right?

For each dataset, label budget and seed, this rebuilds NetMF, GCN and GraphSAGE
exactly as run_experiments.py does and computes the learned router's full input
set (degree, confidence, entropy, disagreement). Logistic-regression detectors
are fit on the validation nodes the router may use and scored on the test nodes.

Targets:
  gcn_wrong                            GCN is wrong
  netmf_rescues / sage_rescues         that expert is right and GCN is wrong
  cond_<expert>_right_given_disagree   among nodes where the expert and GCN
                                       disagree, the expert is the one that is right

The rescue targets also report the best switching rule ("use the other expert
when the detector score exceeds t", t chosen on validation) as a change in test
accuracy over GCN. The conditional targets are the fair test: rescues can only
happen where experts disagree, and disagreement is itself a router input.

Run from the repository root:

    python -m diagnostics.predictability_check --output diagnostics/router_predictability.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import degree

from fusion import build_router_features
from models import GCN, GraphSAGE, resolve_device, seed_everything, train_model
from netmf import load_or_compute_netmf
from run_experiments import make_limited_label_masks, split_validation_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", choices=["Cora", "CiteSeer"], default=["Cora", "CiteSeer"])
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--budgets", type=int, nargs="+", default=[5, 20])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("router_predictability.csv"))
    return parser.parse_args()


def fit_detector(features: np.ndarray, target: np.ndarray, fit: np.ndarray) -> np.ndarray | None:
    """Return detector scores for every node, or None if a class is missing from the fit set."""
    if target[fit].sum() < 2 or (~target[fit]).sum() < 2:
        return None
    detector = LogisticRegression(max_iter=2000, class_weight="balanced")
    detector.fit(features[fit], target[fit])
    return detector.predict_proba(features)[:, 1]


def switching_gain(score, expert_right, gcn_right, fit, test) -> tuple[float, float]:
    """Gain in test accuracy from switching to the expert above a validation-tuned threshold."""
    best_threshold, best_accuracy = np.inf, gcn_right[fit].mean()
    for threshold in np.unique(score[fit]):
        accuracy = np.where(score[fit] >= threshold, expert_right[fit], gcn_right[fit]).mean()
        if accuracy > best_accuracy + 1e-12:
            best_threshold, best_accuracy = threshold, accuracy
    switched = score[test] >= best_threshold
    gain = np.where(switched, expert_right[test], gcn_right[test]).mean() - gcn_right[test].mean()
    return float(gain), float(switched.mean())


def analyse_setting(name, graph, labels, degrees, embedding, num_classes, budget, seed, device) -> list[dict]:
    train, val, test_mask = make_limited_label_masks(graph, budget, seed)
    base_stop, router_train, _ = split_validation_masks(labels, val, seed)

    probabilities = {}
    for model_name, cls in (("GCN", GCN), ("GraphSAGE", GraphSAGE)):
        seed_everything(seed)
        model = cls(graph.num_features, 64, num_classes, 0.5)
        trained = train_model(model, graph, train, base_stop, device, 200, 30, 0.01, 5e-4)
        probabilities[model_name] = trained.probabilities
    classifier = LogisticRegression(max_iter=2000, random_state=seed)
    classifier.fit(embedding[train.numpy()], labels[train.numpy()])
    probabilities["NetMF"] = classifier.predict_proba(embedding)

    experts = [probabilities["NetMF"], probabilities["GCN"], probabilities["GraphSAGE"]]
    features = build_router_features(experts, degrees, router_train, "all")
    predicted = {k: v.argmax(axis=1) for k, v in probabilities.items()}
    right = {k: v == labels for k, v in predicted.items()}
    fit = val.numpy() & ~base_stop.numpy()  # exactly the nodes the learned router may use
    test = test_mask.numpy()
    keys = {"dataset": name, "budget": budget, "seed": seed}
    rows = []

    targets = {"gcn_wrong": (~right["GCN"], None),
               "netmf_rescues": (right["NetMF"] & ~right["GCN"], "NetMF"),
               "sage_rescues": (right["GraphSAGE"] & ~right["GCN"], "GraphSAGE")}
    for target_name, (target, expert) in targets.items():
        row = {**keys, "target": target_name, "test_rate": target[test].mean()}
        score = fit_detector(features, target, fit)
        if score is not None:
            row["auc"] = roc_auc_score(target[test], score[test])
            if expert:
                row["switch_gain"], row["switch_rate"] = switching_gain(score, right[expert], right["GCN"], fit, test)
        rows.append(row)

    for expert in ("NetMF", "GraphSAGE"):
        disagree = predicted[expert] != predicted["GCN"]
        cond_test = test & disagree
        row = {**keys, "target": f"cond_{expert}_right_given_disagree", "test_rate": disagree[test].mean(),
               "expert_right_share": right[expert][cond_test].mean(),
               "gcn_right_share": right["GCN"][cond_test].mean(),
               "neither_share": (~right[expert][cond_test] & ~right["GCN"][cond_test]).mean()}
        target = right[expert]
        score = fit_detector(features, target, fit & disagree)
        if score is not None and 0 < target[cond_test].mean() < 1:
            row["auc"] = roc_auc_score(target[cond_test], score[cond_test])
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    rows = []
    for name in args.datasets:
        dataset = Planetoid(args.data_root, name, transform=NormalizeFeatures())
        graph = dataset[0]
        labels = graph.y.numpy()
        degrees = degree(graph.edge_index[0], num_nodes=graph.num_nodes).numpy()
        cache = args.cache_dir / f"{name.lower()}_drop0.00_seed0_w5_d128.npy"
        embedding, _, _ = load_or_compute_netmf(graph.edge_index, graph.num_nodes, cache, 128, 5, 0)
        for budget in args.budgets:
            for seed in args.seeds:
                rows.extend(analyse_setting(name, graph, labels, degrees, embedding, dataset.num_classes,
                                            budget, seed, device))
                print(f"{name} budget={budget} seed={seed} done", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(args.output, index=False)
    summary = results.groupby(["dataset", "budget", "target"]).agg(
        test_rate=("test_rate", "mean"), auc=("auc", "mean"), auc_std=("auc", "std"),
        switch_gain_points=("switch_gain", lambda s: 100 * s.mean()),
        expert_right_share=("expert_right_share", "mean"),
        gcn_right_share=("gcn_right_share", "mean"),
        neither_share=("neither_share", "mean"),
    ).round(3)
    print(summary.to_string())


if __name__ == "__main__":
    main()
