#!/usr/bin/env python3
"""Task 6: compare cached models across GCN-confidence test-node terciles."""

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
import numpy as np
import pandas as pd


GROUPS = ("low", "medium", "high")
MODELS = ("GCN", "NetMF", "GraphSAGE", "Weighted GCN + NetMF", "LearnedRouter")
OUTPUT_NAMES = (
    "easy_difficult_per_seed.csv",
    "easy_difficult_aggregate.csv",
    "accuracy_by_confidence.png",
    "difference_from_gcn.png",
    "analysis_summary.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument(
        "--weighted-results-dir", type=Path,
        help="Task 3 directory; defaults to a sibling exp03_weighted_average directory",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(cache: dict[str, np.ndarray], name: str):
    if name not in cache or cache[name].shape != ():
        raise ValueError(f"Cache key {name!r} must be a scalar")
    return cache[name].item()


def load_cache(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as stored:
        cache = {name: stored[name] for name in stored.files}
    if int(scalar(cache, "format_version")) != 1:
        raise ValueError(f"Unsupported cache format in {path}")
    metadata = {
        "dataset": str(scalar(cache, "dataset")),
        "seed": int(scalar(cache, "seed")),
        "budget": int(scalar(cache, "budget")),
        "edge_drop": float(scalar(cache, "edge_drop")),
    }
    required = ("labels", "val_mask", "test_mask", "prob__GCN", "prob__NetMF",
                "prob__GraphSAGE", "prob__LearnedRouter")
    missing = [name for name in required if name not in cache]
    if missing:
        raise KeyError(f"{path} is missing: {', '.join(missing)}")
    labels = np.asarray(cache["labels"])
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"Invalid labels in {path}")
    n = len(labels)
    for name in ("val_mask", "test_mask"):
        mask = cache[name]
        if mask.shape != (n,) or mask.dtype != np.bool_ or not mask.any():
            raise ValueError(f"Invalid {name} in {path}")
    if np.any(cache["val_mask"] & cache["test_mask"]):
        raise ValueError(f"Validation and test masks overlap in {path}")
    shape = None
    for name in ("prob__GCN", "prob__NetMF", "prob__GraphSAGE", "prob__LearnedRouter"):
        values = cache[name]
        if values.ndim != 2 or values.shape[0] != n or not np.isfinite(values).all():
            raise ValueError(f"Invalid probability array {name} in {path}: {values.shape}")
        if shape is None:
            shape = values.shape
        elif values.shape != shape:
            raise ValueError(f"Probability dimensions do not align in {path}")
    assert shape is not None
    if labels.min() < 0 or labels.max() >= shape[1]:
        raise ValueError(f"Labels and class dimension do not align in {path}")
    return metadata, cache


def validation_selected_pair(cache: dict[str, np.ndarray]) -> tuple[float, float, float]:
    """Reproduce Task 3's GCN+NetMF selection without receiving a test mask."""
    labels, val = cache["labels"], cache["val_mask"]
    gcn, netmf = cache["prob__GCN"], cache["prob__NetMF"]
    candidates = []
    for units in range(11):
        gcn_weight = units / 10.0
        netmf_weight = 1.0 - gcn_weight
        fused = gcn_weight * gcn + netmf_weight * netmf
        correct = int((fused[val].argmax(axis=1) == labels[val]).sum())
        candidates.append((correct, gcn_weight, netmf_weight))
    correct, gcn_weight, netmf_weight = min(
        candidates,
        key=lambda item: (-item[0], (item[1] - 0.5) ** 2 + (item[2] - 0.5) ** 2,
                          (item[1], item[2])),
    )
    return gcn_weight, netmf_weight, 100.0 * correct / int(val.sum())


def load_and_verify_task3_weights(
    cache_path: Path, cache_hash: str, cache: dict[str, np.ndarray], weighted_dir: Path,
) -> tuple[float, float, float]:
    results_path = weighted_dir / cache_path.stem / "weighted_average_results.csv"
    if not results_path.is_file():
        raise FileNotFoundError(f"Task 3 result not found: {results_path}")
    results = pd.read_csv(results_path, dtype={"prediction_sha256": str})
    selected = results[results["method"].eq("Weighted GCN + NetMF")]
    if len(selected) != 1:
        raise ValueError(f"Expected one Weighted GCN + NetMF row in {results_path}")
    row = selected.iloc[0]
    if str(row["prediction_sha256"]) != cache_hash:
        raise ValueError(f"Task 3 result hash does not match cache: {results_path}")
    verified = validation_selected_pair(cache)
    recorded = (float(row["weight_gcn"]), float(row["weight_netmf"]),
                float(row["validation_accuracy_pct"]))
    if not np.allclose(recorded, verified, rtol=0.0, atol=1e-10):
        raise AssertionError(f"Task 3 weights do not reproduce validation selection: {results_path}")
    return recorded


def confidence_terciles(gcn: np.ndarray, test_mask: np.ndarray) -> dict[str, np.ndarray]:
    """Split test indices by stable ascending GCN maximum-probability rank."""
    test_indices = np.flatnonzero(test_mask)
    confidence = gcn[test_indices].max(axis=1)
    ordered = test_indices[np.argsort(confidence, kind="stable")]
    pieces = np.array_split(ordered, 3)
    groups = {name: indices for name, indices in zip(GROUPS, pieces, strict=True)}
    if sum(len(indices) for indices in groups.values()) != len(test_indices):
        raise AssertionError("Confidence groups do not partition the test set")
    return groups


def analyze_cache(cache_path: Path, weighted_dir: Path) -> list[dict]:
    metadata, cache = load_cache(cache_path)
    cache_hash = file_hash(cache_path)
    gcn_weight, netmf_weight, selection_accuracy = load_and_verify_task3_weights(
        cache_path, cache_hash, cache, weighted_dir
    )
    probabilities = {
        "GCN": cache["prob__GCN"],
        "NetMF": cache["prob__NetMF"],
        "GraphSAGE": cache["prob__GraphSAGE"],
        "Weighted GCN + NetMF": (
            gcn_weight * cache["prob__GCN"] + netmf_weight * cache["prob__NetMF"]
        ),
        "LearnedRouter": cache["prob__LearnedRouter"],
    }
    predictions = {name: values.argmax(axis=1) for name, values in probabilities.items()}
    labels = cache["labels"]
    groups = confidence_terciles(cache["prob__GCN"], cache["test_mask"])
    rows = []
    for group_name, indices in groups.items():
        if len(indices) == 0:
            raise ValueError(f"Empty confidence group in {cache_path}")
        gcn_accuracy = float((predictions["GCN"][indices] == labels[indices]).mean())
        confidence = cache["prob__GCN"][indices].max(axis=1)
        for model in MODELS:
            model_accuracy = float((predictions[model][indices] == labels[indices]).mean())
            rows.append({
                **metadata,
                "prediction_file": str(cache_path.resolve()),
                "prediction_sha256": cache_hash,
                "confidence_group": group_name,
                "node_count": len(indices),
                "gcn_confidence_min": float(confidence.min()),
                "gcn_confidence_max": float(confidence.max()),
                "model": model,
                "accuracy_pct": 100.0 * model_accuracy,
                "gcn_accuracy_pct": 100.0 * gcn_accuracy,
                "difference_from_gcn_pp": 100.0 * (model_accuracy - gcn_accuracy),
                "weight_gcn": gcn_weight if model == "Weighted GCN + NetMF" else np.nan,
                "weight_netmf": netmf_weight if model == "Weighted GCN + NetMF" else np.nan,
                "weight_selection_validation_accuracy_pct": (
                    selection_accuracy if model == "Weighted GCN + NetMF" else np.nan
                ),
            })
    return rows


def aggregate(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in per_seed.groupby(["dataset", "confidence_group", "model"], sort=False):
        dataset, confidence_group, model = keys
        accuracy_std = float(group["accuracy_pct"].std(ddof=1))
        difference_std = float(group["difference_from_gcn_pp"].std(ddof=1))
        count_std = float(group["node_count"].std(ddof=1))
        rows.append({
            "dataset": dataset,
            "confidence_group": confidence_group,
            "model": model,
            "seeds": int(group["seed"].nunique()),
            "node_count_mean": float(group["node_count"].mean()),
            "node_count_std": count_std,
            "accuracy_mean_pct": float(group["accuracy_pct"].mean()),
            "accuracy_std_pct": accuracy_std,
            "accuracy_mean_std": f"{group['accuracy_pct'].mean():.2f} ± {accuracy_std:.2f}",
            "difference_from_gcn_mean_pp": float(group["difference_from_gcn_pp"].mean()),
            "difference_from_gcn_std_pp": difference_std,
            "difference_from_gcn_mean_std": (
                f"{group['difference_from_gcn_pp'].mean():+.2f} ± {difference_std:.2f}"
            ),
        })
    return pd.DataFrame(rows)


def guard_outputs(output_dir: Path, cache_hashes: set[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = output_dir / OUTPUT_NAMES[0]
    existing = [output_dir / name for name in OUTPUT_NAMES if (output_dir / name).exists()]
    if not existing:
        return
    if not result.is_file():
        raise FileExistsError(f"Cannot verify existing artifacts in {output_dir}")
    prior = pd.read_csv(result, dtype={"prediction_sha256": str})
    if set(prior["prediction_sha256"].astype(str)) != cache_hashes:
        raise FileExistsError(f"Existing outputs in {output_dir} use different caches")
    print("Existing Task 6 outputs use the same caches; regenerating them.")


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent,
        prefix=f".{path.stem}_", suffix=".csv", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def save_plots(aggregated: pd.DataFrame, output_dir: Path) -> None:
    colors = dict(zip(MODELS, plt.cm.tab10.colors[:len(MODELS)], strict=True))
    x = np.arange(len(GROUPS))
    width = 0.16
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for index, model in enumerate(MODELS):
        subset = aggregated[aggregated["model"].eq(model)].set_index("confidence_group").loc[list(GROUPS)]
        ax.bar(
            x + (index - 2) * width, subset["accuracy_mean_pct"], width,
            yerr=subset["accuracy_std_pct"], capsize=3, label=model, color=colors[model],
        )
    ax.set_xticks(x, [name.title() for name in GROUPS])
    ax.set_ylabel("Test accuracy across seeds (%)")
    ax.set_title(f"Accuracy by GCN confidence tercile — {aggregated.iloc[0]['dataset']}")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_by_confidence.png", dpi=180)
    plt.close(fig)

    comparison_models = [model for model in MODELS if model != "GCN"]
    width = 0.2
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for index, model in enumerate(comparison_models):
        subset = aggregated[aggregated["model"].eq(model)].set_index("confidence_group").loc[list(GROUPS)]
        ax.bar(
            x + (index - 1.5) * width, subset["difference_from_gcn_mean_pp"], width,
            yerr=subset["difference_from_gcn_std_pp"], capsize=3,
            label=model, color=colors[model],
        )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x, [name.title() for name in GROUPS])
    ax.set_ylabel("Accuracy difference from GCN (percentage points)")
    ax.set_title(f"Difference from GCN by confidence tercile — {aggregated.iloc[0]['dataset']}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "difference_from_gcn.png", dpi=180)
    plt.close(fig)


def save_summary(per_seed: pd.DataFrame, aggregated: pd.DataFrame, path: Path) -> None:
    dataset = str(per_seed.iloc[0]["dataset"])
    lines = []
    for group_name in GROUPS:
        for model in MODELS:
            row = aggregated[
                aggregated["confidence_group"].eq(group_name) & aggregated["model"].eq(model)
            ].iloc[0]
            lines.append(
                f"| {group_name.title()} | {model} | {row['node_count_mean']:.1f} ± "
                f"{row['node_count_std']:.1f} | {row['accuracy_mean_std']} | "
                f"{row['difference_from_gcn_mean_std']} |"
            )
    weights = per_seed[per_seed["model"].eq("Weighted GCN + NetMF")][
        ["seed", "weight_gcn", "weight_netmf"]
    ].drop_duplicates()
    weight_lines = "\n".join(
        f"| {int(row.seed)} | {row.weight_gcn:.1f} | {row.weight_netmf:.1f} |"
        for row in weights.itertuples(index=False)
    )
    text = f"""# Easy versus difficult nodes — {dataset}

Five cached runs (seeds {', '.join(str(value) for value in sorted(per_seed['seed'].unique()))},
20 labels per class) are analyzed without retraining. Within each seed, test nodes are ranked by
GCN maximum predicted probability and split into three nearly equal groups. Labels are not used
to create the groups. Accuracy is retrospective test evaluation.

| Confidence group | Model | Nodes | Accuracy (%) | Difference from GCN (pp) |
|---|---|---:|---:|---:|
{chr(10).join(lines)}

## Validation-selected fusion weights

These Task 3 weights were selected on each cache's validation mask and verified by reproducing
the complete 0.0–1.0 search. Test labels did not affect selection.

| Seed | GCN weight | NetMF weight |
|---:|---:|---:|
{weight_lines}

Values are mean ± sample standard deviation across five seeds. Confidence groups contain 334,
333, and 333 nodes per seed (low to high). Group membership can differ between seeds because each
seed has different cached GCN probabilities.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    prediction_dir = args.prediction_dir.resolve()
    weighted_dir = (
        args.weighted_results_dir.resolve() if args.weighted_results_dir
        else prediction_dir.parent / "exp03_weighted_average"
    )
    cache_paths = sorted(prediction_dir.glob("*.npz"))
    if not cache_paths:
        raise FileNotFoundError(f"No .npz caches found in {prediction_dir}")
    all_rows = []
    for cache_path in cache_paths:
        all_rows.extend(analyze_cache(cache_path, weighted_dir))
    per_seed = pd.DataFrame(all_rows)
    if per_seed["dataset"].nunique() != 1:
        raise ValueError("One invocation must contain caches from a single dataset")
    aggregated = aggregate(per_seed)
    cache_hashes = set(per_seed["prediction_sha256"])
    guard_outputs(args.output_dir, cache_hashes)
    atomic_csv(per_seed, args.output_dir / OUTPUT_NAMES[0])
    atomic_csv(aggregated, args.output_dir / OUTPUT_NAMES[1])
    save_plots(aggregated, args.output_dir)
    save_summary(per_seed, aggregated, args.output_dir / OUTPUT_NAMES[4])
    print(
        f"Task 6: analyzed {len(cache_paths)} {per_seed.iloc[0]['dataset']} caches; "
        f"saved {args.output_dir.resolve()}"
    )
    print(aggregated[[
        "confidence_group", "model", "accuracy_mean_std", "difference_from_gcn_mean_std"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
