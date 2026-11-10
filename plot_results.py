#!/usr/bin/env python3
"""Create presentation-ready plots from experiment CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


COLORS = {
    "MLP": "#6B7280",
    "NetMF": "#2563EB",
    "GCN": "#E9A23B",
    "GraphSAGE": "#168C82",
    "EqualAverage": "#8B5CF6",
    "DegreeRouter": "#C84B31",
    "NetMF_GCN": "#0E7490",
    "LearnedRouter": "#DB2777",
    "LearnedRouterDegree": "#F97316",
    "LearnedRouterConfidence": "#EC4899",
    "OracleUpperBound": "#111827",
}


def accuracy_by_budget(raw: pd.DataFrame, output: Path) -> None:
    subset = raw[raw["edge_drop"] == raw["edge_drop"].min()]
    oracle = subset[subset["model"] == "OracleUpperBound"]
    subset = subset[subset["model"] != "OracleUpperBound"]
    grouped = subset.groupby(["model", "budget"], as_index=False)["test_accuracy"].agg(
        ["mean", "std"]
    ).reset_index()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for model, frame in grouped.groupby("model"):
        frame = frame.sort_values("budget")
        ax.errorbar(
            frame["budget"], frame["mean"], yerr=frame["std"].fillna(0),
            marker="o", linewidth=2, capsize=3, label=model,
            color=COLORS.get(model),
        )
    ax.set_xlabel("Labeled nodes per class")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy under limited supervision")
    if not oracle.empty:
        bound = oracle.groupby("budget")["test_accuracy"].mean().sort_index()
        ax.plot(bound.index, bound.values, linestyle="--", color="#111827", label="Oracle upper bound")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def model_comparison(raw: pd.DataFrame, output: Path) -> None:
    zero_drop = raw[raw["edge_drop"] == raw["edge_drop"].min()]
    subset = zero_drop[zero_drop["budget"] == zero_drop["budget"].max()]
    oracle = subset[subset["model"] == "OracleUpperBound"]["test_accuracy"].mean()
    subset = subset[subset["model"] != "OracleUpperBound"]
    grouped = subset.groupby("model")["test_accuracy"].agg(["mean", "std"])
    order = [model for model in COLORS if model in grouped.index]
    grouped = grouped.loc[order]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.bar(
        grouped.index,
        grouped["mean"],
        yerr=grouped["std"].fillna(0),
        capsize=4,
        color=[COLORS[name] for name in grouped.index],
    )
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_ylim(max(0, grouped["mean"].min() - 0.12), min(1.0, grouped["mean"].max() + 0.08))
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"Model comparison at {int(subset['budget'].max())} labels per class")
    if pd.notna(oracle):
        ax.axhline(oracle, linestyle="--", color="#111827", linewidth=1.5, label="Oracle upper bound")
        ax.legend(frameon=False)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def robustness_plot(raw: pd.DataFrame, output: Path) -> None:
    if raw["edge_drop"].nunique() < 2:
        return
    subset = raw[(raw["budget"] == raw["budget"].max()) & (raw["model"] != "OracleUpperBound")]
    grouped = subset.groupby(["model", "edge_drop"])["test_accuracy"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for model, frame in grouped.groupby("model"):
        ax.plot(frame["edge_drop"] * 100, frame["test_accuracy"], marker="o", linewidth=2,
                label=model, color=COLORS.get(model))
    ax.set_xlabel("Edges removed (%)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy after removing graph edges")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def degree_accuracy(results: pd.DataFrame, output: Path) -> None:
    subset = results[(results["edge_drop"] == results["edge_drop"].min()) &
                     (results["budget"] == results["budget"].max()) &
                     (results["model"] != "OracleUpperBound")]
    grouped = subset.groupby(["model", "degree_group"])["accuracy"].mean().unstack()
    grouped = grouped[[name for name in ("low", "medium", "high") if name in grouped]]
    order = [name for name in COLORS if name in grouped.index and name != "OracleUpperBound"]
    grouped = grouped.loc[order]
    ax = grouped.plot(kind="bar", figsize=(11, 5.5), color=["#93C5FD", "#3B82F6", "#1E3A8A"])
    ax.set_ylabel("Test accuracy")
    ax.set_xlabel("")
    ax.set_title("Accuracy by node-degree group")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def router_weights(weights: pd.DataFrame, output: Path) -> None:
    if weights.empty:
        return
    order = ["low", "medium", "high"]
    if "router" not in weights.columns:
        weights = weights.assign(router="DegreeRouter")
    routers = list(weights["router"].drop_duplicates())
    fig, axes = plt.subplots(1, len(routers), figsize=(7 * len(routers), 5), squeeze=False)
    for ax, router in zip(axes[0], routers):
        selected = weights[weights["router"] == router]
        grouped = selected.groupby(["degree_group", "base_model"])["weight"].mean().unstack()
        grouped = grouped.reindex(order).dropna(how="all")
        columns = [name for name in ("NetMF", "GCN", "GraphSAGE") if name in grouped]
        grouped[columns].plot(
            kind="bar", stacked=True, ax=ax,
            color=[COLORS[name] for name in columns],
        )
        ax.set_ylabel("Average router weight")
        ax.set_xlabel("Node-degree group")
        ax.set_title(f"{router} behavior")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(frameon=False, ncol=3, loc="upper center")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    raw_path = args.results_dir / "raw_results.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Run experiments first; missing {raw_path}")
    plots = args.results_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(raw_path)
    accuracy_by_budget(raw, plots / "accuracy_vs_labels.png")
    model_comparison(raw, plots / "model_comparison.png")
    robustness_plot(raw, plots / "accuracy_vs_edge_removal.png")
    weights_path = args.results_dir / "router_weights.csv"
    if weights_path.exists():
        router_weights(pd.read_csv(weights_path), plots / "router_weights.png")
    degree_path = args.results_dir / "degree_group_results.csv"
    if degree_path.exists():
        degree_accuracy(pd.read_csv(degree_path), plots / "accuracy_by_degree.png")
    print(f"Plots written to {plots.resolve()}")


if __name__ == "__main__":
    main()
