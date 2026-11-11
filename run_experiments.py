#!/usr/bin/env python3
"""Run intermediate NetMF, GNN, and fusion experiments on Planetoid data."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
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
from prediction_cache import save_prediction_cache


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
    parser.add_argument(
        "--netmf-method", choices=["auto", "exact", "eigen"], default="auto",
        help="exact sums walk powers (small graphs); eigen uses the top-rank "
             "eigenpairs (large graphs); auto picks eigen above 10,000 nodes",
    )
    parser.add_argument("--netmf-rank", type=int, default=256,
                        help="Eigenpairs used by the eigen NetMF method")
    parser.add_argument(
        "--netmf-concat-scales", type=float, nargs="+", default=[0.1, 0.3, 1.0],
        help="Candidate NetMF block scales for NetMF_GCN, relative to the mean row "
             "L2 norm of the original features; selected on validation accuracy",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--save-predictions", action="store_true",
                        help="Save all per-node prediction probabilities and split metadata as compressed NPZ files")
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


def concat_netmf_features(x: torch.Tensor, embedding: np.ndarray, scale: float | None) -> torch.Tensor:
    """Append NetMF columns to node features.

    With ``scale=None`` the embedding is appended unchanged. Otherwise it is
    multiplied so its mean row L2 norm equals ``scale`` times that of ``x``.
    Without rescaling, the unit-norm NetMF rows dominate the row-normalized
    bag-of-words features (row norm ~0.2 on Cora/CiteSeer).
    """
    netmf = torch.from_numpy(embedding).to(dtype=x.dtype)
    if scale is not None:
        x_norm = x.norm(dim=1).mean()
        netmf_norm = netmf.norm(dim=1).mean().clamp_min(1e-12)
        netmf = netmf * (scale * x_norm / netmf_norm)
    return torch.cat([x, netmf], dim=1)


def package_versions() -> dict[str, str]:
    import platform
    from importlib.metadata import PackageNotFoundError, version

    versions = {"python": platform.python_version()}
    for name in ("torch", "torch_geometric", "numpy", "scipy", "scikit-learn", "pandas"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not installed"
    return versions


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


@dataclass
class Setting:
    """One (seed, edge-drop rate, label budget) configuration and its data."""

    args: argparse.Namespace
    device: torch.device
    graph: object
    labels: np.ndarray
    num_classes: int
    degrees: np.ndarray
    embedding: np.ndarray
    netmf_seconds: float
    seed: int
    budget: int
    edge_drop: float
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    base_stop_mask: torch.Tensor
    router_train: np.ndarray
    router_stop: np.ndarray

    @property
    def keys(self) -> dict:
        return {"dataset": self.args.dataset, "seed": self.seed, "budget": self.budget, "edge_drop": self.edge_drop}

    def row(self, model: str, probabilities: np.ndarray, **extra) -> dict:
        return result_row(
            **self.keys, model=model, probabilities=probabilities, labels=self.labels,
            val_mask=self.val_mask, test_mask=self.test_mask, **extra,
        )

    def train(self, model: torch.nn.Module, graph=None):
        a = self.args
        return train_model(
            model, graph if graph is not None else self.graph, self.train_mask, self.base_stop_mask,
            self.device, a.epochs, a.patience, a.lr, a.weight_decay,
        )


@dataclass
class Outputs:
    raw: list = field(default_factory=list)
    weights: list = field(default_factory=list)
    oracle: list = field(default_factory=list)
    degree: list = field(default_factory=list)

    def save(self, output_dir: Path) -> None:
        pd.DataFrame(self.raw).to_csv(output_dir / "raw_results.csv", index=False)
        pd.DataFrame(self.weights).to_csv(output_dir / "router_weights.csv", index=False)
        pd.DataFrame(self.oracle).to_csv(output_dir / "oracle_analysis.csv", index=False)
        pd.DataFrame(self.degree).to_csv(output_dir / "degree_group_results.csv", index=False)


def print_row(row: dict, suffix: str = "") -> None:
    print(f"{row['model']:12s} test_acc={row['test_accuracy']:.4f} macro_f1={row['test_macro_f1']:.4f}{suffix}")


def prepare_graph(args: argparse.Namespace, original, seed: int, edge_drop: float):
    """Drop edges for this seed and load or compute the matching NetMF embedding."""
    graph = original.clone()
    graph.edge_index = drop_undirected_edges(original.edge_index, edge_drop, seed)
    degrees = degree(graph.edge_index[0], num_nodes=graph.num_nodes).cpu().numpy()
    cache_seed = seed if edge_drop > 0 else 0  # the undamaged graph shares one embedding
    suffix = f"_eig{args.netmf_rank}" if args.netmf_method == "eigen" else ""
    cache_name = (
        f"{args.dataset.lower()}_drop{edge_drop:.2f}_seed{cache_seed}_"
        f"w{args.netmf_window}_d{args.netmf_dim}{suffix}.npy"
    )
    print(f"Preparing NetMF embedding for seed={seed}, edge_drop={edge_drop:.2f}")
    embedding, seconds, cache_hit = load_or_compute_netmf(
        graph.edge_index, graph.num_nodes, args.cache_dir / cache_name,
        args.netmf_dim, args.netmf_window, cache_seed,
        method=args.netmf_method, rank=args.netmf_rank,
    )
    print(f"NetMF {'cache hit' if cache_hit else 'computed'}; preprocessing={seconds:.2f}s")
    return graph, degrees, embedding, seconds


def train_base_models(st: Setting, out: Outputs) -> dict[str, np.ndarray]:
    """Train MLP, GCN and GraphSAGE on the word features."""
    a, g = st.args, st.graph
    probabilities = {}
    for name, cls in (("MLP", MLP), ("GCN", GCN), ("GraphSAGE", GraphSAGE)):
        seed_everything(st.seed)
        trained = st.train(cls(g.num_features, a.hidden_channels, st.num_classes, a.dropout))
        probabilities[name] = trained.probabilities
        row = st.row(name, trained.probabilities, train_seconds=trained.train_seconds,
                     peak_memory_mb=trained.peak_memory_mb, best_epoch=trained.best_epoch)
        out.raw.append(row)
        print_row(row)
    return probabilities


def train_netmf_gcn(st: Setting, scale: float | None):
    a = st.args
    graph = st.graph.clone()
    graph.x = concat_netmf_features(st.graph.x, st.embedding, scale)
    seed_everything(st.seed)
    return st.train(GCN(graph.num_features, a.hidden_channels, st.num_classes, a.dropout), graph)


def train_feature_fusion(st: Setting, out: Outputs) -> dict[str, np.ndarray]:
    """NetMF + GCN concatenation: the unscaled original and the validation-tuned scaled version."""
    unscaled = train_netmf_gcn(st, None)
    unscaled_row = st.row("NetMF_GCN_Unscaled", unscaled.probabilities, train_seconds=unscaled.train_seconds,
                          preprocess_seconds=st.netmf_seconds, peak_memory_mb=unscaled.peak_memory_mb,
                          best_epoch=unscaled.best_epoch)
    out.raw.append(unscaled_row)

    best_scale, best_val, best, total_seconds = None, -1.0, None, 0.0
    for scale in st.args.netmf_concat_scales:
        candidate = train_netmf_gcn(st, scale)
        total_seconds += candidate.train_seconds
        score, _ = evaluate(candidate.probabilities, st.labels, st.base_stop_mask)
        if score > best_val + 1e-8:
            best_scale, best_val, best = scale, score, candidate
    row = st.row("NetMF_GCN", best.probabilities, train_seconds=total_seconds,
                 preprocess_seconds=st.netmf_seconds, peak_memory_mb=best.peak_memory_mb,
                 best_epoch=best.best_epoch)
    row["netmf_scale"] = best_scale
    out.raw.append(row)
    print_row(row, f" scale={best_scale} (unscaled test_acc={unscaled_row['test_accuracy']:.4f})")
    return {"NetMF_GCN_Unscaled": unscaled.probabilities, "NetMF_GCN": best.probabilities}


def train_netmf_classifier(st: Setting, out: Outputs) -> np.ndarray:
    train = st.train_mask.numpy()
    classifier = LogisticRegression(max_iter=2000, random_state=st.seed)
    start = time.perf_counter()
    classifier.fit(st.embedding[train], st.labels[train])
    seconds = time.perf_counter() - start
    probabilities = classifier.predict_proba(st.embedding)
    row = st.row("NetMF", probabilities, train_seconds=seconds, preprocess_seconds=st.netmf_seconds)
    out.raw.append(row)
    print_row(row)
    return probabilities


def fuse_predictions(st: Setting, out: Outputs, experts: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Equal average, degree router, the three learned routers and the oracle."""
    fused = {"EqualAverage": equal_average(experts)}
    routed = degree_router(experts, st.degrees)
    fused["DegreeRouter"] = routed.probabilities
    routers = {"DegreeRouter": routed}
    for feature_set, name in (("degree", "LearnedRouterDegree"), ("confidence", "LearnedRouterConfidence"),
                              ("all", "LearnedRouter")):
        routers[name] = learned_router(experts, st.degrees, st.labels, st.router_train, st.router_stop,
                                       seed=st.seed, device=st.device, feature_set=feature_set)
        fused[name] = routers[name].probabilities
    oracle = oracle_best_expert(experts, st.labels)
    fused["OracleUpperBound"] = oracle.probabilities

    for name, probabilities in fused.items():
        extra = {"train_seconds": 0.0}
        if name.startswith("LearnedRouter"):
            extra = {"train_seconds": routers[name].train_seconds, "best_epoch": routers[name].best_epoch}
        row = st.row(name, probabilities, **extra)
        out.raw.append(row)
        if name in ("EqualAverage", "DegreeRouter", "LearnedRouter", "OracleUpperBound"):
            print_row(row)

    out.oracle.extend(oracle_analysis_rows(**st.keys, probabilities=experts, labels=st.labels,
                                           val_mask=st.val_mask, test_mask=st.test_mask))
    for name, result in routers.items():
        out.weights.extend({**st.keys, "router": name, **item} for item in router_weight_summary(result, st.degrees))
    return fused


def run_setting(st: Setting, out: Outputs) -> None:
    probabilities = train_base_models(st, out)
    probabilities.update(train_feature_fusion(st, out))
    probabilities["NetMF"] = train_netmf_classifier(st, out)
    experts = [probabilities["NetMF"], probabilities["GCN"], probabilities["GraphSAGE"]]
    probabilities.update(fuse_predictions(st, out, experts))
    out.degree.extend(degree_group_rows(**st.keys, probabilities=probabilities, labels=st.labels,
                                        test_mask=st.test_mask, degrees=st.degrees))
    if st.args.save_predictions:
        path = save_prediction_cache(
            output_dir=st.args.output_dir, **st.keys,
            probabilities=probabilities, labels=st.labels, degrees=st.degrees,
            train_mask=st.train_mask.cpu().numpy(),
            val_mask=st.val_mask.cpu().numpy(), test_mask=st.test_mask.cpu().numpy(),
            base_stop_mask=st.base_stop_mask.cpu().numpy(),
            router_train_mask=st.router_train, router_stop_mask=st.router_stop,
        )
        print(f"Predictions: {path}")


def apply_smoke_settings(args: argparse.Namespace) -> None:
    if args.output_dir == Path("results"):
        args.output_dir = Path("results_smoke")
    args.budgets, args.seeds, args.edge_drop_rates = [5], [42], [0.0]
    args.epochs, args.patience, args.hidden_channels = 5, 2, 16
    args.netmf_dim, args.netmf_window = 32, 2


def write_config(args: argparse.Namespace, device: torch.device, data, num_classes: int) -> None:
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({
        "resolved_device": str(device),
        "num_nodes": data.num_nodes,
        "num_edges": data.num_edges,
        "num_features": data.num_features,
        "num_classes": num_classes,
        "package_versions": package_versions(),
    })
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    if args.smoke:
        apply_smoke_settings(args)
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
    if args.netmf_method == "auto":
        args.netmf_method = "eigen" if original.num_nodes > 10_000 else "exact"
    write_config(args, device, original, dataset.num_classes)

    out = Outputs()
    for seed in args.seeds:
        for edge_drop in args.edge_drop_rates:
            graph, degrees, embedding, netmf_seconds = prepare_graph(args, original, seed, edge_drop)
            for budget in args.budgets:
                print(f"\nseed={seed} budget={budget} edge_drop={edge_drop:.2f}")
                train_mask, val_mask, test_mask = make_limited_label_masks(graph, budget, seed)
                base_stop, router_train, router_stop = split_validation_masks(labels, val_mask, seed)
                setting = Setting(
                    args=args, device=device, graph=graph, labels=labels, num_classes=dataset.num_classes,
                    degrees=degrees, embedding=embedding, netmf_seconds=netmf_seconds,
                    seed=seed, budget=budget, edge_drop=edge_drop, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask, base_stop_mask=base_stop,
                    router_train=router_train, router_stop=router_stop,
                )
                run_setting(setting, out)
                out.save(args.output_dir)  # save after every setting so long runs can be inspected

    summarize(pd.DataFrame(out.raw)).to_csv(args.output_dir / "summary.csv", index=False)
    print(f"\nRaw results: {(args.output_dir / 'raw_results.csv').resolve()}")
    print(f"Summary:     {(args.output_dir / 'summary.csv').resolve()}")
    print("Run plot_results.py next to generate figures.")


if __name__ == "__main__":
    main()
