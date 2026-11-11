#!/usr/bin/env python3
"""Tune fixed prediction-averaging weights from a saved prediction cache.

Candidate weights are selected using validation accuracy only. Test labels are
used once, after selection, to evaluate the selected weights and cached baselines.
No model training is performed by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path

_plot_cache = Path(tempfile.gettempdir()) / "adaptive_netmf_plot_cache"
_plot_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_plot_cache / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_plot_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


BASE_MODELS = ("GCN", "NetMF", "GraphSAGE")
SEARCHES = (
    ("GCN + NetMF", ("GCN", "NetMF")),
    ("GCN + GraphSAGE", ("GCN", "GraphSAGE")),
    ("GCN + NetMF + GraphSAGE", BASE_MODELS),
)
OUTPUT_NAMES = (
    "weighted_average_results.csv",
    "weight_search.csv",
    "weighted_average_comparison.png",
    "analysis_summary.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune fixed probability-averaging weights using cached validation predictions."
    )
    parser.add_argument("--prediction-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def cache_scalar(cache: dict[str, np.ndarray], key: str):
    if key not in cache or cache[key].shape != ():
        raise ValueError(f"Cache key {key!r} must be a scalar")
    return cache[key].item()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(f"Prediction cache not found: {path}")
    with np.load(path, allow_pickle=False) as stored:
        cache = {key: stored[key] for key in stored.files}
    version = int(cache_scalar(cache, "format_version"))
    if version != 1:
        raise ValueError(f"Unsupported prediction-cache format_version {version}; expected 1")

    metadata = {
        "dataset": str(cache_scalar(cache, "dataset")),
        "seed": int(cache_scalar(cache, "seed")),
        "budget": int(cache_scalar(cache, "budget")),
        "edge_drop": float(cache_scalar(cache, "edge_drop")),
    }
    required_arrays = (
        "labels", "val_mask", "test_mask", "router_train_mask", "router_stop_mask",
        "prob__GCN", "prob__NetMF", "prob__GraphSAGE", "prob__EqualAverage",
        "prob__LearnedRouter",
    )
    missing = [key for key in required_arrays if key not in cache]
    if missing:
        raise KeyError(f"Prediction cache is missing required keys: {', '.join(missing)}")

    labels = np.asarray(cache["labels"])
    if labels.ndim != 1 or labels.size == 0 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be a nonempty one-dimensional integer array")
    n = labels.size
    for name in ("val_mask", "test_mask", "router_train_mask", "router_stop_mask"):
        mask = np.asarray(cache[name])
        if mask.shape != (n,) or mask.dtype != np.bool_:
            raise ValueError(f"{name} must be a boolean array with one entry per node")
        if not mask.any():
            raise ValueError(f"{name} must select at least one node")
    if np.any(cache["val_mask"] & cache["test_mask"]):
        raise ValueError("Validation and test masks overlap")
    if np.any(cache["router_train_mask"] & cache["router_stop_mask"]):
        raise ValueError("Router training and stopping masks overlap")
    router_used = cache["router_train_mask"] | cache["router_stop_mask"]
    if np.any(router_used & ~cache["val_mask"]):
        raise ValueError("Router training/stopping masks are not subsets of the validation mask")

    expected_shape = None
    for key in ("prob__GCN", "prob__NetMF", "prob__GraphSAGE",
                "prob__EqualAverage", "prob__LearnedRouter"):
        probabilities = np.asarray(cache[key])
        if probabilities.ndim != 2 or probabilities.shape[0] != n:
            raise ValueError(f"{key} has invalid shape {probabilities.shape}")
        if probabilities.shape[1] < 2 or not np.isfinite(probabilities).all():
            raise ValueError(f"{key} must contain finite scores for at least two classes")
        if expected_shape is None:
            expected_shape = probabilities.shape
        elif probabilities.shape != expected_shape:
            raise ValueError(
                f"Probability arrays must have matching dimensions, got {expected_shape} "
                f"and {probabilities.shape}"
            )
    assert expected_shape is not None
    if labels.min() < 0 or labels.max() >= expected_shape[1]:
        raise ValueError("Label values are incompatible with the prediction class dimension")

    recomputed_equal = np.mean(
        np.stack([cache[f"prob__{model}"] for model in ("NetMF", "GCN", "GraphSAGE")]),
        axis=0,
    )
    if not np.allclose(recomputed_equal, cache["prob__EqualAverage"], rtol=1e-6, atol=1e-7):
        raise AssertionError("Cached EqualAverage probabilities do not match the three experts")
    return metadata, cache


def weight_grid(models: tuple[str, ...]) -> list[dict[str, float]]:
    """Return the exact integer-tenths simplex for a pair or three models."""
    candidates = []
    if len(models) == 2:
        for first in range(11):
            units = (first, 10 - first)
            candidates.append(dict(zip(models, (value / 10.0 for value in units), strict=True)))
    elif len(models) == 3:
        for first in range(11):
            for second in range(11 - first):
                third = 10 - first - second
                units = (first, second, third)
                candidates.append(dict(zip(models, (value / 10.0 for value in units), strict=True)))
    else:
        raise ValueError("Only two- and three-model searches are supported")
    return candidates


def weighted_probabilities(
    probabilities: dict[str, np.ndarray], weights: dict[str, float]
) -> np.ndarray:
    if not np.isclose(sum(weights.values()), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("Weights must sum to one")
    return sum(weights[model] * probabilities[model] for model in weights)


def accuracy(probabilities: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> float:
    return float((probabilities[mask].argmax(axis=1) == labels[mask]).mean())


def choose_by_validation(rows: list[dict], models: tuple[str, ...]) -> int:
    """Choose by validation score, then closeness to equal weights, deterministically."""
    equal = 1.0 / len(models)
    return min(
        range(len(rows)),
        key=lambda index: (
            -rows[index]["validation_correct_nodes"],
            sum((rows[index][f"weight_{name.lower()}"] - equal) ** 2 for name in models),
            tuple(rows[index][f"weight_{name.lower()}"] for name in models),
        ),
    )


def search_validation_weights(
    metadata: dict, labels: np.ndarray, val_mask: np.ndarray,
    model_probabilities: dict[str, np.ndarray], source: Path, source_hash: str,
) -> tuple[pd.DataFrame, list[dict]]:
    """Search only the validation split; test labels are not accessed here."""
    common = {
        **metadata,
        "prediction_file": str(source.resolve()),
        "prediction_sha256": source_hash,
        "validation_nodes": int(val_mask.sum()),
    }
    rows = []
    selected = []
    for search_name, models in SEARCHES:
        search_rows = []
        for candidate_number, weights in enumerate(weight_grid(models), start=1):
            probabilities = weighted_probabilities(model_probabilities, weights)
            correct_nodes = int(
                (probabilities[val_mask].argmax(axis=1) == labels[val_mask]).sum()
            )
            row = {
                **common,
                "search": search_name,
                "candidate": candidate_number,
                "weight_gcn": weights.get("GCN", 0.0),
                "weight_netmf": weights.get("NetMF", 0.0),
                "weight_graphsage": weights.get("GraphSAGE", 0.0),
                "validation_correct_nodes": correct_nodes,
                "validation_accuracy_pct": round(100.0 * correct_nodes / int(val_mask.sum()), 10),
                "selected": False,
            }
            search_rows.append(row)
        selected_index = choose_by_validation(search_rows, models)
        search_rows[selected_index]["selected"] = True
        rows.extend(search_rows)
        selected.append(search_rows[selected_index])
    return pd.DataFrame(rows), selected


def evaluate_selected_and_baselines(
    metadata: dict, cache: dict[str, np.ndarray], source: Path, source_hash: str,
    selected: list[dict],
) -> pd.DataFrame:
    """Evaluate test labels only after validation-based selections are fixed."""
    labels, val_mask, test_mask = cache["labels"], cache["val_mask"], cache["test_mask"]
    model_probabilities = {model: cache[f"prob__{model}"] for model in BASE_MODELS}
    common = {
        **metadata,
        "prediction_file": str(source.resolve()),
        "prediction_sha256": source_hash,
        "validation_nodes": int(val_mask.sum()),
        "test_nodes": int(test_mask.sum()),
    }
    rows = []
    search_lookup = {name: models for name, models in SEARCHES}
    for selection in selected:
        models = search_lookup[selection["search"]]
        weights = {model: selection[f"weight_{model.lower()}"] for model in models}
        probabilities = weighted_probabilities(model_probabilities, weights)
        rows.append({
            **common,
            "method": f"Weighted {selection['search']}",
            "method_type": "validation_selected_weighted_average",
            "weight_gcn": selection["weight_gcn"],
            "weight_netmf": selection["weight_netmf"],
            "weight_graphsage": selection["weight_graphsage"],
            "validation_accuracy_pct": round(100.0 * accuracy(probabilities, labels, val_mask), 10),
            "test_accuracy_pct": round(100.0 * accuracy(probabilities, labels, test_mask), 10),
            "validation_comparison_note": "Weights selected on the full saved validation mask.",
        })

    baselines = (
        ("GCN", "prob__GCN", (1.0, 0.0, 0.0), "Cached individual baseline."),
        ("EqualAverage", "prob__EqualAverage", (1 / 3, 1 / 3, 1 / 3),
         "Cached equal average of NetMF, GCN, and GraphSAGE."),
        ("LearnedRouter", "prob__LearnedRouter", (np.nan, np.nan, np.nan),
         "Validation accuracy is not directly comparable: the router was fit and early-stopped "
         "on subsets of the reported validation mask."),
    )
    for method, key, weights, note in baselines:
        probabilities = cache[key]
        rows.append({
            **common,
            "method": method,
            "method_type": "cached_baseline",
            "weight_gcn": weights[0],
            "weight_netmf": weights[1],
            "weight_graphsage": weights[2],
            "validation_accuracy_pct": round(100.0 * accuracy(probabilities, labels, val_mask), 10),
            "test_accuracy_pct": round(100.0 * accuracy(probabilities, labels, test_mask), 10),
            "validation_comparison_note": note,
        })
    return pd.DataFrame(rows)


def matching_rows(frame: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    required = {"dataset", "seed", "budget", "edge_drop"}
    if not required.issubset(frame.columns):
        return frame.iloc[0:0]
    selected = (
        frame["dataset"].astype(str).eq(metadata["dataset"])
        & pd.to_numeric(frame["seed"], errors="coerce").eq(metadata["seed"])
        & pd.to_numeric(frame["budget"], errors="coerce").eq(metadata["budget"])
        & np.isclose(pd.to_numeric(frame["edge_drop"], errors="coerce"), metadata["edge_drop"])
    )
    return frame[selected]


def validate_baselines(
    prediction_file: Path, metadata: dict, results: pd.DataFrame
) -> list[str]:
    raw_path = prediction_file.parent.parent / "raw_results.csv"
    checks = []
    if not raw_path.is_file():
        return [f"Skipped baseline cross-check because `{raw_path}` was not found."]
    raw = matching_rows(pd.read_csv(raw_path), metadata)
    for model in ("GCN", "EqualAverage", "LearnedRouter"):
        expected = raw[raw["model"].eq(model)]
        actual = results[results["method"].eq(model)]
        if len(expected) != 1 or len(actual) != 1:
            raise ValueError(f"Expected one matching {model} row in both analysis and {raw_path}")
        for analysis_column, raw_column in (
            ("validation_accuracy_pct", "val_accuracy"),
            ("test_accuracy_pct", "test_accuracy"),
        ):
            analysis_value = float(actual.iloc[0][analysis_column]) / 100.0
            raw_value = float(expected.iloc[0][raw_column])
            if not np.isclose(analysis_value, raw_value, rtol=0.0, atol=5e-7):
                raise AssertionError(
                    f"{model} {analysis_column} {analysis_value:.12f} does not match "
                    f"{raw_path} ({raw_value:.12f})"
                )
    checks.append(f"Cached GCN, EqualAverage, and LearnedRouter scores match `{raw_path}`.")
    return checks


def guard_output_directory(
    output_dir: Path, source: Path, source_hash: str, metadata: dict
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = [output_dir / name for name in OUTPUT_NAMES if (output_dir / name).exists()]
    if not artifacts:
        return
    results_path = output_dir / OUTPUT_NAMES[0]
    if not results_path.is_file():
        raise FileExistsError(
            "Output directory contains analysis artifacts without a configuration-bearing "
            f"{results_path.name}; refusing to overwrite them"
        )
    prior = pd.read_csv(results_path)
    hashes = set(prior.get("prediction_sha256", pd.Series(dtype=str)).dropna().astype(str))
    if hashes == {source_hash}:
        print("Existing outputs are from the same prediction cache; regenerating them.")
        return
    same_source = (
        set(prior.get("prediction_file", pd.Series(dtype=str)).dropna().astype(str))
        == {str(source.resolve())}
    )
    prior_configuration = matching_rows(prior, metadata)
    if same_source and len(prior_configuration) == len(prior):
        print(
            "Prediction cache contents changed at the same path, but the saved dataset, seed, "
            "budget, and edge-drop configuration match; regenerating outputs explicitly."
        )
        return
    else:
        raise FileExistsError(
            "Output directory contains results from a different or unverifiable prediction cache; "
            "choose another --output-dir"
        )


def atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent,
        prefix=f".{path.stem}_", suffix=".csv", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def save_plot(results: pd.DataFrame, path: Path) -> None:
    baselines = results[results["method_type"].eq("cached_baseline")]
    selected = results[results["method_type"].eq("validation_selected_weighted_average")]
    plotted = pd.concat([baselines, selected], ignore_index=True)
    short_names = {
        "Weighted GCN + NetMF": "Weighted\nGCN + NetMF",
        "Weighted GCN + GraphSAGE": "Weighted\nGCN + GraphSAGE",
        "Weighted GCN + NetMF + GraphSAGE": "Weighted\nall three",
    }
    labels = [short_names.get(name, name) for name in plotted["method"]]
    values = plotted["test_accuracy_pct"].to_numpy()
    colors = [
        "#4C78A8" if kind == "cached_baseline" else "#F58518"
        for kind in plotted["method_type"]
    ]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    bars = ax.bar(range(len(labels)), values, color=colors)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Validation-selected weighted averaging on Cora (seed 42)")
    ax.set_xticks(range(len(labels)), labels, rotation=18, ha="right")
    ax.set_ylim(0, min(100, max(values) + 12))
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=3)
    ax.legend(
        handles=[
            Patch(facecolor="#4C78A8", label="Cached baseline"),
            Patch(facecolor="#F58518", label="Weights selected on validation"),
        ],
        loc="lower left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def format_weights(row: pd.Series) -> str:
    return (
        f"GCN={row['weight_gcn']:.1f}, NetMF={row['weight_netmf']:.1f}, "
        f"GraphSAGE={row['weight_graphsage']:.1f}"
    )


def save_summary(
    path: Path, metadata: dict, source: Path, results: pd.DataFrame,
    search: pd.DataFrame, cache: dict[str, np.ndarray], checks: list[str],
) -> None:
    selected = results[results["method_type"].eq("validation_selected_weighted_average")]
    baselines = results[results["method_type"].eq("cached_baseline")]
    selected_lines = "\n".join(
        f"| {row['method']} | {format_weights(row)} | {row['validation_accuracy_pct']:.2f}% | "
        f"{row['test_accuracy_pct']:.2f}% |"
        for _, row in selected.iterrows()
    )
    baseline_lines = "\n".join(
        f"| {row['method']} | {row['validation_accuracy_pct']:.2f}% | "
        f"{row['test_accuracy_pct']:.2f}% |"
        for _, row in baselines.iterrows()
    )
    best = selected.sort_values(["test_accuracy_pct", "method"], ascending=[False, True]).iloc[0]
    router_train_count = int(cache["router_train_mask"].sum())
    router_stop_count = int(cache["router_stop_mask"].sum())
    val_count = int(cache["val_mask"].sum())
    checks_text = "\n".join(f"- {check}" for check in checks)
    text = f"""# Weighted prediction averaging

This is a **single-seed experiment** on {metadata['dataset']} (seed {metadata['seed']},
{metadata['budget']} labels per class, edge drop {metadata['edge_drop']}). It uses cached
probabilities only; no model was retrained. Candidate fixed weights range from 0.0 to 1.0 in
steps of 0.1. Pair weights sum to one, and the three-model search covers all integer-tenths
weight triples that sum to one.

## Selected weights and held-out test results

| Method | Selected weights | Validation accuracy | Test accuracy |
|---|---|---:|---:|
{selected_lines}

Ties in validation correct-node count are resolved by choosing the candidate closest to equal
weighting, then deterministically by its weight tuple. Test accuracy never participates in
selection. The complete {len(search)}-candidate validation search is in `weight_search.csv`.

## Cached baselines

| Method | Reported validation accuracy | Test accuracy |
|---|---:|---:|
{baseline_lines}

The strongest selected weighted average on this test split is **{best['method']}** at
**{best['test_accuracy_pct']:.2f}%** ({format_weights(best)}). This result comes from one seed and
does not establish a general improvement.

## Learned-router comparability warning

The learned router used {router_train_count} nodes from `router_train_mask` to fit its parameters
and {router_stop_count} nodes from `router_stop_mask` for early stopping. All {router_train_count + router_stop_count}
of those nodes are included in the {val_count}-node saved validation mask used to report its
validation accuracy. Therefore its reported validation accuracy is **not directly comparable** to
the validation score used to select fixed averaging weights. Held-out test accuracy is the cleaner
comparison. The fixed-weight search itself also reuses its validation score for model selection,
so that score should not be interpreted as performance on an untouched split.

## Validation checks

{checks_text}
- All required probability arrays have matching node and class dimensions and finite values.
- Validation and test masks are nonempty, aligned to the arrays, and disjoint.
- The cached equal average matches the mean of NetMF, GCN, and GraphSAGE probabilities.
- Weight selection receives only validation labels and predictions; test evaluation occurs after
  selected candidates are fixed.
- `weight_search.csv` contains validation results only, preventing candidate selection from test scores.
- Existing artifacts are overwritten only for the same cache hash, or after an explicit notice
  when the same cache path and dataset/seed/budget/edge-drop configuration are detected.

Prediction cache: `{source}`
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    prediction_file = args.prediction_file.resolve()
    source_hash = sha256(prediction_file)
    metadata, cache = load_and_validate(prediction_file)

    # This function receives no test mask and therefore cannot inspect test labels.
    model_probabilities = {model: cache[f"prob__{model}"] for model in BASE_MODELS}
    search, selected = search_validation_weights(
        metadata, cache["labels"], cache["val_mask"], model_probabilities,
        prediction_file, source_hash,
    )
    results = evaluate_selected_and_baselines(
        metadata, cache, prediction_file, source_hash, selected
    )
    checks = validate_baselines(prediction_file, metadata, results)
    guard_output_directory(args.output_dir, prediction_file, source_hash, metadata)

    results_path = args.output_dir / OUTPUT_NAMES[0]
    search_path = args.output_dir / OUTPUT_NAMES[1]
    plot_path = args.output_dir / OUTPUT_NAMES[2]
    summary_path = args.output_dir / OUTPUT_NAMES[3]
    atomic_to_csv(results, results_path)
    atomic_to_csv(search, search_path)
    save_plot(results, plot_path)
    save_summary(summary_path, metadata, prediction_file, results, search, cache, checks)

    print(f"Searched {len(search)} weight combinations using {int(cache['val_mask'].sum())} validation nodes")
    print(results[[
        "method", "weight_gcn", "weight_netmf", "weight_graphsage",
        "validation_accuracy_pct", "test_accuracy_pct",
    ]].to_string(index=False))
    print(f"Saved analysis to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
