#!/usr/bin/env python3
"""Run intermediate NetMF, GNN, and fusion experiments on Planetoid data."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import degree, to_undirected

from fusion import (
    degree_router,
    equal_average,
    learned_router,
    oracle_best_expert,
    router_weight_summary,
)
from models import GCN, MLP, GraphSAGE, resolve_device, seed_everything, train_model
from netmf import load_or_compute_netmf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate intermediate results for adaptive NetMF-GNN fusion."
    )
    parser.add_argument("--dataset", choices=["Cora", "CiteSeer", "PubMed"], default="Cora")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--budgets", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--edge-drop-rates", type=float, nargs="+", default=[0.0])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--netmf-dim", type=int, default=128)
    parser.add_argument("--netmf-window", type=int, default=5)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--smoke", action="store_true", help="Run a short installation check")
    return parser.parse_args()


def make_limited_label_masks(data, budget: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = data.y.cpu().numpy()
    val_mask = data.val_mask.clone().bool()
    test_mask = data.test_mask.clone().bool()
    available = ~(val_mask | test_mask).cpu().numpy()
    rng = np.random.default_rng(seed)
    train_mask = np.zeros(data.num_nodes, dtype=bool)
    for class_id in range(int(labels.max()) + 1):
        candidates = np.flatnonzero((labels == class_id) & available)
        if len(candidates) < budget:
            raise ValueError(
                f"Class {class_id} has only {len(candidates)} available nodes; "
                f"cannot sample a budget of {budget}."
            )
        selected = rng.choice(candidates, size=budget, replace=False)
        train_mask[selected] = True
    return torch.from_numpy(train_mask), val_mask, test_mask


def drop_undirected_edges(edge_index: torch.Tensor, rate: float, seed: int) -> torch.Tensor:
    if rate <= 0:
        return edge_index.clone()
    if not 0 <= rate < 1:
        raise ValueError("edge-drop rates must be in [0, 1)")
    row, col = edge_index.cpu()
    keep_once = row < col
    pairs = edge_index[:, keep_once]
    generator = torch.Generator().manual_seed(seed)
    keep = torch.rand(pairs.size(1), generator=generator) >= rate
    return to_undirected(pairs[:, keep], num_nodes=int(edge_index.max()) + 1)


def evaluate(probabilities: np.ndarray, labels: np.ndarray, mask: torch.Tensor) -> tuple[float, float]:
    selected = mask.cpu().numpy().astype(bool)
    predictions = probabilities[selected].argmax(axis=1)
    truth = labels[selected]
    return float(accuracy_score(truth, predictions)), float(
        f1_score(truth, predictions, average="macro", zero_division=0)
    )


def split_validation_masks(
    labels: np.ndarray, val_mask: torch.Tensor, seed: int
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Create disjoint base-stop, router-train, and router-stop subsets."""
    rng = np.random.default_rng(seed + 10_000)
    available = val_mask.cpu().numpy().astype(bool)
    base_stop = np.zeros_like(available)
    router_train = np.zeros_like(available)
    router_stop = np.zeros_like(available)
    for class_id in np.unique(labels[available]):
        indices = np.flatnonzero(available & (labels == class_id))
        rng.shuffle(indices)
        if len(indices) < 3:
            raise ValueError("Each class needs at least three validation nodes")
        base_count = max(1, int(round(len(indices) * 0.50)))
        router_train_count = max(1, int(round(len(indices) * 0.30)))
        router_train_end = min(base_count + router_train_count, len(indices) - 1)
        base_stop[indices[:base_count]] = True
        router_train[indices[base_count:router_train_end]] = True
        router_stop[indices[router_train_end:]] = True
    masks = (base_stop, router_train, router_stop)
    if any(not np.any(mask) for mask in masks):
        raise ValueError("Validation set is too small for the three-way split")
    if np.any(base_stop & router_train) or np.any(base_stop & router_stop) or np.any(router_train & router_stop):
        raise RuntimeError("Validation subsets overlap")
    if not np.array_equal(base_stop | router_train | router_stop, available):
        raise RuntimeError("Validation subsets do not cover the official validation set")
    return torch.from_numpy(base_stop), router_train, router_stop


def degree_group_rows(
    *, dataset: str, seed: int, budget: int, edge_drop: float,
    probabilities: dict[str, np.ndarray], labels: np.ndarray,
    test_mask: torch.Tensor, degrees: np.ndarray,
) -> list[dict]:
    """Evaluate every method on low-, medium-, and high-degree test nodes."""
    low_threshold, high_threshold = np.quantile(degrees, [1.0 / 3.0, 2.0 / 3.0])
    test = test_mask.cpu().numpy().astype(bool)
    groups = {
        "low": degrees <= low_threshold,
        "medium": (degrees > low_threshold) & (degrees <= high_threshold),
        "high": degrees > high_threshold,
    }
    rows = []
    for model, values in probabilities.items():
        predictions = values.argmax(axis=1)
        for group, group_mask in groups.items():
            selected = test & group_mask
            truth = labels[selected]
            guessed = predictions[selected]
            rows.append({
                "dataset": dataset, "seed": seed, "budget": budget,
                "edge_drop": edge_drop, "model": model,
                "degree_group": group, "node_count": int(selected.sum()),
                "low_threshold": float(low_threshold),
                "high_threshold": float(high_threshold),
                "accuracy": float(accuracy_score(truth, guessed)),
                "macro_f1": float(f1_score(truth, guessed, average="macro", zero_division=0)),
            })
    return rows


def oracle_analysis_rows(
    *, dataset: str, seed: int, budget: int, edge_drop: float,
    probabilities: list[np.ndarray], labels: np.ndarray,
    val_mask: torch.Tensor, test_mask: torch.Tensor,
) -> list[dict]:
    """Measure oracle headroom and each expert's unique correct predictions."""
    names = ("NetMF", "GCN", "GraphSAGE")
    predictions = np.stack([values.argmax(axis=1) for values in probabilities], axis=1)
    correct = predictions == labels[:, None]
    rows = []
    for split, mask in (("validation", val_mask), ("test", test_mask)):
        selected = mask.cpu().numpy().astype(bool)
        split_correct = correct[selected]
        row = {
            "dataset": dataset,
            "seed": seed,
            "budget": budget,
            "edge_drop": edge_drop,
            "split": split,
            "oracle_accuracy": float(split_correct.any(axis=1).mean()),
            "all_experts_agree_rate": float(
                np.all(predictions[selected] == predictions[selected, :1], axis=1).mean()
            ),
        }
        for index, name in enumerate(names):
            others = np.delete(split_correct, index, axis=1)
            row[f"{name.lower()}_unique_correct_rate"] = float(
                (split_correct[:, index] & ~others.any(axis=1)).mean()
            )
        rows.append(row)
    return rows


def result_row(
    *, dataset: str, seed: int, budget: int, edge_drop: float, model: str,
    probabilities: np.ndarray, labels: np.ndarray, val_mask: torch.Tensor,
    test_mask: torch.Tensor, train_seconds: float, preprocess_seconds: float = 0.0,
    peak_memory_mb: float = 0.0, best_epoch: int = 0,
) -> dict:
    val_accuracy, val_f1 = evaluate(probabilities, labels, val_mask)
    test_accuracy, test_f1 = evaluate(probabilities, labels, test_mask)
    return {
        "dataset": dataset,
        "seed": seed,
        "budget": budget,
        "edge_drop": edge_drop,
        "model": model,
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_f1,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_f1,
        "train_seconds": train_seconds,
        "preprocess_seconds": preprocess_seconds,
        "peak_memory_mb": peak_memory_mb,
        "best_epoch": best_epoch,
    }


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "val_accuracy", "val_macro_f1", "test_accuracy", "test_macro_f1",
        "train_seconds", "preprocess_seconds", "peak_memory_mb", "best_epoch",
    ]
    summary = raw.groupby(["dataset", "budget", "edge_drop", "model"])[metrics].agg(
        ["mean", "std"]
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def main() -> None:
    args = parse_args()
    if args.smoke:
        if args.output_dir == Path("results"):
            args.output_dir = Path("results_smoke")
        args.budgets = [5]
        args.seeds = [42]
        args.edge_drop_rates = [0.0]
        args.epochs = 5
        args.patience = 2
        args.hidden_channels = 16
        args.netmf_dim = 32
        args.netmf_window = 2
    if any(budget < 1 for budget in args.budgets):
        raise ValueError("All label budgets must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Loading {args.dataset}; training device: {device}")
    dataset = Planetoid(args.data_root, args.dataset, transform=NormalizeFeatures())
    original = dataset[0]
    original.y = original.y.view(-1)
    labels = original.y.cpu().numpy()
    num_classes = dataset.num_classes
    raw_rows: list[dict] = []
    weight_rows: list[dict] = []
    oracle_rows: list[dict] = []
    degree_rows: list[dict] = []

    config = vars(args).copy()
    config.update({
        "data_root": str(args.data_root),
        "output_dir": str(args.output_dir),
        "cache_dir": str(args.cache_dir),
        "resolved_device": str(device),
        "num_nodes": original.num_nodes,
        "num_edges": original.num_edges,
        "num_features": original.num_features,
        "num_classes": num_classes,
    })
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    for seed in args.seeds:
        for edge_drop in args.edge_drop_rates:
            graph = original.clone()
            graph.edge_index = drop_undirected_edges(original.edge_index, edge_drop, seed)
            deg = degree(graph.edge_index[0], num_nodes=graph.num_nodes).cpu().numpy()
            cache_seed = seed if edge_drop > 0 else 0
            cache_name = (
                f"{args.dataset.lower()}_drop{edge_drop:.2f}_seed{cache_seed}_"
                f"w{args.netmf_window}_d{args.netmf_dim}.npy"
            )
            print(f"Preparing NetMF embedding for seed={seed}, edge_drop={edge_drop:.2f}")
            embedding, netmf_preprocess, cache_hit = load_or_compute_netmf(
                graph.edge_index,
                graph.num_nodes,
                args.cache_dir / cache_name,
                args.netmf_dim,
                args.netmf_window,
                cache_seed,
            )
            print(
                f"NetMF {'cache hit' if cache_hit else 'computed'}; "
                f"preprocessing={netmf_preprocess:.2f}s"
            )

            for budget in args.budgets:
                print(f"\nseed={seed} budget={budget} edge_drop={edge_drop:.2f}")
                train_mask, val_mask, test_mask = make_limited_label_masks(graph, budget, seed)
                base_stop_mask, router_train, router_stop = split_validation_masks(
                    labels, val_mask, seed
                )
                seed_everything(seed)
                probabilities: dict[str, np.ndarray] = {}

                model_builders = {
                    "MLP": lambda: MLP(graph.num_features, args.hidden_channels, num_classes, args.dropout),
                    "GCN": lambda: GCN(graph.num_features, args.hidden_channels, num_classes, args.dropout),
                    "GraphSAGE": lambda: GraphSAGE(graph.num_features, args.hidden_channels, num_classes, args.dropout),
                }
                for model_name, builder in model_builders.items():
                    seed_everything(seed)
                    trained = train_model(
                        builder(), graph, train_mask, base_stop_mask, device,
                        args.epochs, args.patience, args.lr, args.weight_decay,
                    )
                    probabilities[model_name] = trained.probabilities
                    row = result_row(
                        dataset=args.dataset, seed=seed, budget=budget, edge_drop=edge_drop,
                        model=model_name, probabilities=trained.probabilities, labels=labels,
                        val_mask=val_mask, test_mask=test_mask,
                        train_seconds=trained.train_seconds,
                        peak_memory_mb=trained.peak_memory_mb,
                        best_epoch=trained.best_epoch,
                    )
                    raw_rows.append(row)
                    print(
                        f"{model_name:12s} test_acc={row['test_accuracy']:.4f} "
                        f"macro_f1={row['test_macro_f1']:.4f}"
                    )

                concat_graph = graph.clone()
                netmf_features = torch.from_numpy(embedding).to(dtype=graph.x.dtype)
                concat_graph.x = torch.cat([graph.x, netmf_features], dim=1)
                seed_everything(seed)
                concat_trained = train_model(
                    GCN(concat_graph.num_features, args.hidden_channels, num_classes, args.dropout),
                    concat_graph, train_mask, base_stop_mask, device,
                    args.epochs, args.patience, args.lr, args.weight_decay,
                )
                probabilities["NetMF_GCN"] = concat_trained.probabilities
                concat_row = result_row(
                    dataset=args.dataset, seed=seed, budget=budget, edge_drop=edge_drop,
                    model="NetMF_GCN", probabilities=concat_trained.probabilities,
                    labels=labels, val_mask=val_mask, test_mask=test_mask,
                    train_seconds=concat_trained.train_seconds,
                    preprocess_seconds=netmf_preprocess,
                    peak_memory_mb=concat_trained.peak_memory_mb,
                    best_epoch=concat_trained.best_epoch,
                )
                raw_rows.append(concat_row)
                print(
                    f"{'NetMF_GCN':12s} test_acc={concat_row['test_accuracy']:.4f} "
                    f"macro_f1={concat_row['test_macro_f1']:.4f}"
                )

                classifier = LogisticRegression(max_iter=2000, random_state=seed)
                start = time.perf_counter()
                classifier.fit(embedding[train_mask.numpy()], labels[train_mask.numpy()])
                netmf_train = time.perf_counter() - start
                probabilities["NetMF"] = classifier.predict_proba(embedding)
                netmf_row = result_row(
                    dataset=args.dataset, seed=seed, budget=budget, edge_drop=edge_drop,
                    model="NetMF", probabilities=probabilities["NetMF"], labels=labels,
                    val_mask=val_mask, test_mask=test_mask, train_seconds=netmf_train,
                    preprocess_seconds=netmf_preprocess,
                )
                raw_rows.append(netmf_row)
                print(
                    f"{'NetMF':12s} test_acc={netmf_row['test_accuracy']:.4f} "
                    f"macro_f1={netmf_row['test_macro_f1']:.4f}"
                )

                fusion_inputs = [probabilities["NetMF"], probabilities["GCN"], probabilities["GraphSAGE"]]
                equal = equal_average(fusion_inputs)
                equal_row = result_row(
                    dataset=args.dataset, seed=seed, budget=budget, edge_drop=edge_drop,
                    model="EqualAverage", probabilities=equal, labels=labels,
                    val_mask=val_mask, test_mask=test_mask, train_seconds=0.0,
                )
                raw_rows.append(equal_row)

                routed = degree_router(fusion_inputs, deg)
                router_row = result_row(
                    dataset=args.dataset, seed=seed, budget=budget, edge_drop=edge_drop,
                    model="DegreeRouter", probabilities=routed.probabilities, labels=labels,
                    val_mask=val_mask, test_mask=test_mask, train_seconds=0.0,
                )
                raw_rows.append(router_row)

                learned_results = {}
                for feature_set, model_name in (
                    ("degree", "LearnedRouterDegree"),
                    ("confidence", "LearnedRouterConfidence"),
                    ("all", "LearnedRouter"),
                ):
                    learned = learned_router(
                        fusion_inputs, deg, labels, router_train, router_stop,
                        seed=seed, device=device, feature_set=feature_set,
                    )
                    learned_results[model_name] = learned
                    learned_row = result_row(
                        dataset=args.dataset, seed=seed, budget=budget, edge_drop=edge_drop,
                        model=model_name, probabilities=learned.probabilities,
                        labels=labels, val_mask=val_mask, test_mask=test_mask,
                        train_seconds=learned.train_seconds, best_epoch=learned.best_epoch,
                    )
                    raw_rows.append(learned_row)

                oracle = oracle_best_expert(fusion_inputs, labels)
                oracle_row = result_row(
                    dataset=args.dataset, seed=seed, budget=budget, edge_drop=edge_drop,
                    model="OracleUpperBound", probabilities=oracle.probabilities,
                    labels=labels, val_mask=val_mask, test_mask=test_mask,
                    train_seconds=0.0,
                )
                raw_rows.append(oracle_row)
                oracle_rows.extend(oracle_analysis_rows(
                    dataset=args.dataset, seed=seed, budget=budget,
                    edge_drop=edge_drop, probabilities=fusion_inputs, labels=labels,
                    val_mask=val_mask, test_mask=test_mask,
                ))
                print(
                    f"{'EqualAverage':12s} test_acc={equal_row['test_accuracy']:.4f}\n"
                    f"{'DegreeRouter':12s} test_acc={router_row['test_accuracy']:.4f}\n"
                    f"{'LearnedRouter':12s} test_acc="
                    f"{raw_rows[-2]['test_accuracy']:.4f}\n"
                    f"{'Oracle':12s} test_acc={oracle_row['test_accuracy']:.4f}"
                )
                for item in router_weight_summary(routed, deg):
                    weight_rows.append({
                        "dataset": args.dataset,
                        "seed": seed,
                        "budget": budget,
                        "edge_drop": edge_drop,
                        "router": "DegreeRouter",
                        **item,
                    })
                for model_name, learned_result in learned_results.items():
                    for item in router_weight_summary(learned_result, deg):
                        weight_rows.append({
                            "dataset": args.dataset, "seed": seed,
                            "budget": budget, "edge_drop": edge_drop,
                            "router": model_name, **item,
                        })

                evaluated = {
                    **probabilities,
                    "EqualAverage": equal,
                    "DegreeRouter": routed.probabilities,
                    **{name: result.probabilities for name, result in learned_results.items()},
                    "OracleUpperBound": oracle.probabilities,
                }
                degree_rows.extend(degree_group_rows(
                    dataset=args.dataset, seed=seed, budget=budget,
                    edge_drop=edge_drop, probabilities=evaluated, labels=labels,
                    test_mask=test_mask, degrees=deg,
                ))

                pd.DataFrame(raw_rows).to_csv(args.output_dir / "raw_results.csv", index=False)
                pd.DataFrame(weight_rows).to_csv(args.output_dir / "router_weights.csv", index=False)
                pd.DataFrame(oracle_rows).to_csv(args.output_dir / "oracle_analysis.csv", index=False)
                pd.DataFrame(degree_rows).to_csv(args.output_dir / "degree_group_results.csv", index=False)

    raw = pd.DataFrame(raw_rows)
    summary = summarize(raw)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    print(f"\nRaw results: {(args.output_dir / 'raw_results.csv').resolve()}")
    print(f"Summary:     {(args.output_dir / 'summary.csv').resolve()}")
    print("Run plot_results.py next to generate figures.")


if __name__ == "__main__":
    main()
