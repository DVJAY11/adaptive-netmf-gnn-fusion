#!/usr/bin/env python3
"""Analyze complementary errors in cached NetMF, GCN, and GraphSAGE outputs.

This is retrospective analysis only: an oracle may inspect the true label to
ask whether at least one model predicted a test node correctly.  It is not a
deployable classifier and this script never trains a model.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from itertools import combinations
from pathlib import Path

# Keep plotting caches in a writable temporary location on restricted systems.
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


MODELS = ("GCN", "NetMF", "GraphSAGE")
ORACLE_COMBINATIONS = (
    ("GCN", "NetMF"),
    ("GCN", "GraphSAGE"),
    ("NetMF", "GraphSAGE"),
    MODELS,
)
OUTPUT_NAMES = (
    "pairwise_oracle_results.csv",
    "pairwise_error_overlap.csv",
    "pairwise_oracle_comparison.png",
    "analysis_summary.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare pairwise and three-model oracle accuracy from a prediction cache."
    )
    parser.add_argument("--prediction-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def scalar(cache: dict[str, np.ndarray], key: str):
    if key not in cache or cache[key].shape != ():
        raise ValueError(f"Cache key {key!r} must be a scalar")
    return cache[key].item()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate(path: Path) -> tuple[dict, np.ndarray, dict[str, np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(f"Prediction cache not found: {path}")
    with np.load(path, allow_pickle=False) as stored:
        cache = {key: stored[key] for key in stored.files}

    version = int(scalar(cache, "format_version"))
    if version != 1:
        raise ValueError(f"Unsupported prediction-cache format_version {version}; expected 1")
    metadata = {
        "dataset": str(scalar(cache, "dataset")),
        "seed": int(scalar(cache, "seed")),
        "budget": int(scalar(cache, "budget")),
        "edge_drop": float(scalar(cache, "edge_drop")),
    }
    labels = np.asarray(cache.get("labels"))
    test_mask = np.asarray(cache.get("test_mask"))
    if labels.ndim != 1 or labels.size == 0 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be a nonempty one-dimensional integer array")
    if test_mask.shape != labels.shape or test_mask.dtype != np.bool_:
        raise ValueError("test_mask must be a boolean array with one entry per node")
    if not test_mask.any():
        raise ValueError("test_mask does not select any nodes")

    probabilities = {}
    expected_shape = None
    for model in MODELS:
        key = f"prob__{model}"
        if key not in cache:
            raise KeyError(f"Prediction cache is missing required key {key!r}")
        values = np.asarray(cache[key])
        if values.ndim != 2 or values.shape[0] != labels.size:
            raise ValueError(
                f"{key} has shape {values.shape}; expected ({labels.size}, number_of_classes)"
            )
        if values.shape[1] < 2 or not np.isfinite(values).all():
            raise ValueError(f"{key} must contain finite scores for at least two classes")
        if expected_shape is None:
            expected_shape = values.shape
        elif values.shape != expected_shape:
            raise ValueError(f"Model arrays have mismatched shapes: {expected_shape} and {values.shape}")
        probabilities[model] = values
    assert expected_shape is not None
    if labels.min() < 0 or labels.max() >= expected_shape[1]:
        raise ValueError(
            f"Labels range from {labels.min()} to {labels.max()}, but predictions have "
            f"{expected_shape[1]} classes"
        )
    return metadata, test_mask, probabilities | {"labels": labels}


def analyze(
    metadata: dict, test_mask: np.ndarray, arrays: dict[str, np.ndarray], source: Path,
    source_hash: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, np.ndarray]]:
    truth = arrays["labels"][test_mask]
    correct = {
        model: arrays[model][test_mask].argmax(axis=1) == truth for model in MODELS
    }
    accuracies = {model: float(values.mean()) for model, values in correct.items()}
    common = {
        **metadata,
        "prediction_file": str(source.resolve()),
        "prediction_sha256": source_hash,
        "test_nodes": int(test_mask.sum()),
    }

    result_rows = []
    for names in ORACLE_COMBINATIONS:
        oracle_correct = np.logical_or.reduce([correct[name] for name in names])
        best_accuracy = max(accuracies[name] for name in names)
        best_model = next(name for name in names if accuracies[name] == best_accuracy)
        oracle_count = int(oracle_correct.sum())
        best_count = int(correct[best_model].sum())
        gain = float(oracle_correct.mean() - best_accuracy)
        if oracle_correct.mean() + 1e-15 < best_accuracy or gain < -1e-15:
            raise AssertionError(f"Oracle invariant failed for {' + '.join(names)}")
        result_rows.append({
            **common,
            "combination": " + ".join(names),
            "models": ";".join(names),
            "model_count": len(names),
            "oracle_accuracy_pct": round(100.0 * float(oracle_correct.mean()), 10),
            "best_individual_model": best_model,
            "best_individual_accuracy_pct": round(100.0 * best_accuracy, 10),
            "oracle_gain_percentage_points": round(100.0 * max(0.0, gain), 10),
            "rescued_test_nodes": oracle_count - best_count,
            "all_models_wrong": int((~oracle_correct).sum()),
        })

    overlap_rows = []
    for model_a, model_b in combinations(MODELS, 2):
        a, b = correct[model_a], correct[model_b]
        counts = {
            "both_models_correct": int((a & b).sum()),
            "only_model_a_correct": int((a & ~b).sum()),
            "only_model_b_correct": int((~a & b).sum()),
            "both_models_wrong": int((~a & ~b).sum()),
        }
        if sum(counts.values()) != len(truth):
            raise AssertionError(f"Pairwise partition does not cover the test set for {model_a}, {model_b}")
        overlap_rows.append({
            **common,
            "combination": f"{model_a} + {model_b}",
            "model_a": model_a,
            "model_b": model_b,
            **counts,
        })
    return pd.DataFrame(result_rows), pd.DataFrame(overlap_rows), accuracies, correct


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


def validate_against_existing(
    prediction_file: Path, metadata: dict, accuracies: dict[str, float], oracle_accuracy: float,
) -> list[str]:
    """Cross-check the sibling experiment results when they are available."""
    checks = []
    results_dir = prediction_file.parent.parent
    raw_path = results_dir / "raw_results.csv"
    if raw_path.is_file():
        rows = matching_rows(pd.read_csv(raw_path), metadata)
        for model, accuracy in accuracies.items():
            model_rows = rows[rows["model"].eq(model)]
            if len(model_rows) != 1:
                raise ValueError(f"Expected one matching {model} row in {raw_path}, found {len(model_rows)}")
            recorded = float(model_rows.iloc[0]["test_accuracy"])
            if not np.isclose(accuracy, recorded, rtol=0.0, atol=5e-7):
                raise AssertionError(
                    f"{model} cache accuracy {accuracy:.12f} does not match {raw_path} ({recorded:.12f})"
                )
        checks.append(f"Individual accuracies match `{raw_path}` within 0.00005 percentage points.")
    else:
        checks.append(f"Skipped raw-results cross-check because `{raw_path}` was not found.")

    oracle_path = results_dir / "oracle_analysis.csv"
    if oracle_path.is_file():
        rows = matching_rows(pd.read_csv(oracle_path), metadata)
        rows = rows[rows["split"].eq("test")]
        if len(rows) != 1:
            raise ValueError(f"Expected one matching test row in {oracle_path}, found {len(rows)}")
        recorded = float(rows.iloc[0]["oracle_accuracy"])
        if not np.isclose(oracle_accuracy, recorded, rtol=0.0, atol=5e-7):
            raise AssertionError(
                f"Three-model oracle {oracle_accuracy:.12f} does not match {oracle_path} "
                f"({recorded:.12f})"
            )
        checks.append(f"Three-model oracle matches `{oracle_path}` within 0.00005 percentage points.")
    else:
        checks.append(f"Skipped prior-oracle cross-check because `{oracle_path}` was not found.")
    return checks


def guard_output_directory(output_dir: Path, source_hash: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / name for name in OUTPUT_NAMES if (output_dir / name).exists()]
    if not existing:
        return
    results_path = output_dir / OUTPUT_NAMES[0]
    if not results_path.is_file():
        raise FileExistsError(
            f"Output directory contains analysis artifacts but no configuration-bearing "
            f"{results_path.name}; refusing to overwrite them"
        )
    prior = pd.read_csv(results_path)
    hashes = set(prior.get("prediction_sha256", pd.Series(dtype=str)).dropna().astype(str))
    if hashes != {source_hash}:
        raise FileExistsError(
            "Output directory contains results from a different or unverifiable prediction cache; "
            "choose another --output-dir"
        )
    print("Existing outputs are from the same prediction cache; regenerating them.")


def atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent,
        prefix=f".{path.stem}_", suffix=".csv", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def save_plot(results: pd.DataFrame, accuracies: dict[str, float], path: Path) -> None:
    labels = list(MODELS) + results["combination"].tolist()
    values = [100.0 * accuracies[name] for name in MODELS] + results["oracle_accuracy_pct"].tolist()
    colors = ["#4C78A8"] * len(MODELS) + ["#F58518"] * (len(labels) - len(MODELS))
    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.bar(range(len(labels)), values, color=colors)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Individual and retrospective oracle accuracy")
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    ax.set_ylim(0, min(100, max(values) + 12))
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=3, fontsize=9)
    ax.legend(
        handles=[
            Patch(facecolor="#4C78A8", label="Individual model"),
            Patch(facecolor="#F58518", label="Oracle (any included model correct)"),
        ],
        loc="upper left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_summary(
    path: Path, metadata: dict, source: Path, results: pd.DataFrame,
    overlap: pd.DataFrame, accuracies: dict[str, float], checks: list[str],
) -> None:
    best_pair = results[results["model_count"].eq(2)].sort_values(
        ["oracle_accuracy_pct", "combination"], ascending=[False, True]
    ).iloc[0]
    triple = results[results["model_count"].eq(3)].iloc[0]
    individual_lines = "\n".join(
        f"| {model} | {100.0 * accuracies[model]:.2f}% |" for model in MODELS
    )
    result_lines = "\n".join(
        f"| {row.combination} | {row.oracle_accuracy_pct:.2f}% | "
        f"{row.best_individual_accuracy_pct:.2f}% | {row.oracle_gain_percentage_points:.2f} | "
        f"{int(row.rescued_test_nodes)} | {int(row.all_models_wrong)} |"
        for row in results.itertuples(index=False)
    )
    overlap_lines = "\n".join(
        f"| {row.combination} | {int(row.both_models_correct)} | "
        f"{int(row.only_model_a_correct)} | {int(row.only_model_b_correct)} | "
        f"{int(row.both_models_wrong)} |"
        for row in overlap.itertuples(index=False)
    )
    check_lines = "\n".join(f"- {check}" for check in checks)
    text = f"""# Pairwise oracle analysis

This analysis uses cached predictions only. No models were trained or changed. The oracle is a
retrospective upper bound: a test node counts as correct when at least one included model predicts
its true label. It cannot be used for prediction because it depends on the true test labels.

## Configuration

- Dataset: {metadata['dataset']}
- Seed: {metadata['seed']}
- Labels per class: {metadata['budget']}
- Edge drop: {metadata['edge_drop']}
- Prediction cache: `{source}`
- Test nodes: {int(results.iloc[0]['test_nodes'])}

## Individual accuracy

| Model | Test accuracy |
|---|---:|
{individual_lines}

## Oracle results

| Combination | Oracle accuracy | Best individual | Gain (points) | Rescued nodes | All wrong |
|---|---:|---:|---:|---:|---:|
{result_lines}

Among the pairs, **{best_pair['combination']}** has the highest oracle accuracy at
**{best_pair['oracle_accuracy_pct']:.2f}%**. Its gain means that the two models make some different
errors: one model correctly classifies nodes missed by the stronger individual model. This shows
theoretical headroom, not that a practical fusion method will necessarily achieve that accuracy.

Using all three models raises the retrospective upper bound to **{triple['oracle_accuracy_pct']:.2f}%**.

## Pairwise error overlap

“Only A” and “Only B” follow the model order shown in the combination column.

| Combination | Both correct | Only A | Only B | Both wrong |
|---|---:|---:|---:|---:|
{overlap_lines}

## Validation

{check_lines}
- All cached model arrays have matching node and class dimensions and contain finite values.
- Accuracy uses only the saved test mask ({int(results.iloc[0]['test_nodes'])} nodes).
- Every oracle accuracy is at least its included individual accuracies; all gains are nonnegative.
- Output reuse is allowed only when the existing result records the same prediction-file SHA-256.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    prediction_file = args.prediction_file.resolve()
    source_hash = file_sha256(prediction_file)
    metadata, test_mask, arrays = load_and_validate(prediction_file)
    results, overlap, accuracies, _ = analyze(
        metadata, test_mask, arrays, prediction_file, source_hash
    )
    triple_accuracy = float(
        results.loc[results["model_count"].eq(3), "oracle_accuracy_pct"].iloc[0] / 100.0
    )
    checks = validate_against_existing(prediction_file, metadata, accuracies, triple_accuracy)
    guard_output_directory(args.output_dir, source_hash)

    results_path = args.output_dir / OUTPUT_NAMES[0]
    overlap_path = args.output_dir / OUTPUT_NAMES[1]
    plot_path = args.output_dir / OUTPUT_NAMES[2]
    summary_path = args.output_dir / OUTPUT_NAMES[3]
    atomic_to_csv(results, results_path)
    atomic_to_csv(overlap, overlap_path)
    save_plot(results, accuracies, plot_path)
    save_summary(summary_path, metadata, prediction_file, results, overlap, accuracies, checks)

    print(f"Analyzed {int(test_mask.sum())} test nodes from {prediction_file}")
    print(results[[
        "combination", "oracle_accuracy_pct", "best_individual_accuracy_pct",
        "oracle_gain_percentage_points", "rescued_test_nodes", "all_models_wrong",
    ]].to_string(index=False))
    print(f"Saved analysis to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
