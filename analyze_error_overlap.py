#!/usr/bin/env python3
"""Task 7: analyze GCN error overlap with NetMF and GraphSAGE caches."""

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
COMPARATORS = ("NetMF", "GraphSAGE")
CATEGORIES = ("both_correct", "only_gcn_correct", "only_comparator_correct", "both_wrong")
OUTPUT_NAMES = (
    "error_overlap_per_seed.csv",
    "rescues_by_confidence_per_seed.csv",
    "error_overlap_aggregate.csv",
    "rescues_by_confidence_aggregate.csv",
    "error_overlap.png",
    "rescues_by_confidence.png",
    "analysis_summary.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True)
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
    required = ("labels", "test_mask", "prob__GCN", "prob__NetMF", "prob__GraphSAGE")
    missing = [name for name in required if name not in cache]
    if missing:
        raise KeyError(f"{path} is missing: {', '.join(missing)}")
    labels, test = cache["labels"], cache["test_mask"]
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"Invalid labels in {path}")
    if test.shape != labels.shape or test.dtype != np.bool_ or not test.any():
        raise ValueError(f"Invalid test mask in {path}")
    shape = None
    for name in ("prob__GCN", "prob__NetMF", "prob__GraphSAGE"):
        values = cache[name]
        if values.ndim != 2 or values.shape[0] != len(labels) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {name} in {path}: {values.shape}")
        if shape is None:
            shape = values.shape
        elif values.shape != shape:
            raise ValueError(f"Probability arrays do not align in {path}")
    assert shape is not None
    if labels.min() < 0 or labels.max() >= shape[1]:
        raise ValueError(f"Labels and class dimension do not align in {path}")
    return metadata, cache


def confidence_terciles(gcn: np.ndarray, test_mask: np.ndarray) -> dict[str, np.ndarray]:
    test_indices = np.flatnonzero(test_mask)
    confidence = gcn[test_indices].max(axis=1)
    ordered = test_indices[np.argsort(confidence, kind="stable")]
    pieces = np.array_split(ordered, 3)
    groups = {name: indices for name, indices in zip(GROUPS, pieces, strict=True)}
    if sum(len(indices) for indices in groups.values()) != len(test_indices):
        raise AssertionError("Confidence groups do not partition the test set")
    return groups


def analyze_cache(cache_path: Path) -> tuple[list[dict], list[dict]]:
    metadata, cache = load_cache(cache_path)
    cache_hash = file_hash(cache_path)
    labels, test = cache["labels"], cache["test_mask"]
    predictions = {
        name: cache[f"prob__{name}"].argmax(axis=1) for name in ("GCN", *COMPARATORS)
    }
    correct = {name: predictions[name] == labels for name in predictions}
    test_nodes = int(test.sum())
    groups = confidence_terciles(cache["prob__GCN"], test)
    common = {
        **metadata,
        "prediction_file": str(cache_path.resolve()),
        "prediction_sha256": cache_hash,
        "test_nodes": test_nodes,
    }
    overlap_rows, rescue_rows = [], []
    for comparator in COMPARATORS:
        gcn_correct, comparator_correct = correct["GCN"][test], correct[comparator][test]
        category_masks = {
            "both_correct": gcn_correct & comparator_correct,
            "only_gcn_correct": gcn_correct & ~comparator_correct,
            "only_comparator_correct": ~gcn_correct & comparator_correct,
            "both_wrong": ~gcn_correct & ~comparator_correct,
        }
        counts = {name: int(mask.sum()) for name, mask in category_masks.items()}
        if sum(counts.values()) != test_nodes:
            raise AssertionError(f"Overlap categories do not partition test nodes in {cache_path}")
        gcn_wrong = counts["only_comparator_correct"] + counts["both_wrong"]
        rescues = counts["only_comparator_correct"]
        overlap_rows.append({
            **common,
            "comparator": comparator,
            **{f"{name}_count": counts[name] for name in CATEGORIES},
            **{f"{name}_pct": 100.0 * counts[name] / test_nodes for name in CATEGORIES},
            "gcn_wrong_count": gcn_wrong,
            "rescued_gcn_mistakes": rescues,
            "rescue_rate_among_gcn_mistakes_pct": 100.0 * rescues / gcn_wrong,
        })

        all_rescues = int((~correct["GCN"] & correct[comparator] & test).sum())
        grouped_rescues = 0
        for group_name, indices in groups.items():
            rescued = ~correct["GCN"][indices] & correct[comparator][indices]
            gcn_wrong_group = ~correct["GCN"][indices]
            rescue_count = int(rescued.sum())
            wrong_count = int(gcn_wrong_group.sum())
            grouped_rescues += rescue_count
            confidence = cache["prob__GCN"][indices].max(axis=1)
            rescue_rows.append({
                **common,
                "comparator": comparator,
                "confidence_group": group_name,
                "group_node_count": len(indices),
                "gcn_confidence_min": float(confidence.min()),
                "gcn_confidence_max": float(confidence.max()),
                "gcn_wrong_in_group": wrong_count,
                "rescued_in_group": rescue_count,
                "share_of_comparator_rescues_pct": (
                    100.0 * rescue_count / all_rescues if all_rescues else 0.0
                ),
                "rescue_rate_among_gcn_mistakes_in_group_pct": (
                    100.0 * rescue_count / wrong_count if wrong_count else 0.0
                ),
            })
        if grouped_rescues != all_rescues:
            raise AssertionError(f"Grouped rescues do not equal total rescues in {cache_path}")
    return overlap_rows, rescue_rows


def aggregate_overlap(per_seed: pd.DataFrame) -> pd.DataFrame:
    metrics = [f"{name}_{suffix}" for name in CATEGORIES for suffix in ("count", "pct")]
    metrics += ["gcn_wrong_count", "rescued_gcn_mistakes", "rescue_rate_among_gcn_mistakes_pct"]
    rows = []
    for (dataset, comparator), group in per_seed.groupby(["dataset", "comparator"], sort=False):
        row = {"dataset": dataset, "comparator": comparator, "seeds": group["seed"].nunique()}
        for metric in metrics:
            mean, std = float(group[metric].mean()), float(group[metric].std(ddof=1))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_mean_std"] = f"{mean:.2f} ± {std:.2f}"
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_rescues(per_seed: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "group_node_count", "gcn_wrong_in_group", "rescued_in_group",
        "share_of_comparator_rescues_pct", "rescue_rate_among_gcn_mistakes_in_group_pct",
    )
    rows = []
    for keys, group in per_seed.groupby(
        ["dataset", "comparator", "confidence_group"], sort=False
    ):
        dataset, comparator, confidence_group = keys
        row = {
            "dataset": dataset,
            "comparator": comparator,
            "confidence_group": confidence_group,
            "seeds": group["seed"].nunique(),
        }
        for metric in metrics:
            mean, std = float(group[metric].mean()), float(group[metric].std(ddof=1))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_mean_std"] = f"{mean:.2f} ± {std:.2f}"
        rows.append(row)
    return pd.DataFrame(rows)


def guard_outputs(output_dir: Path, hashes: set[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = output_dir / OUTPUT_NAMES[0]
    existing = [output_dir / name for name in OUTPUT_NAMES if (output_dir / name).exists()]
    if not existing:
        return
    if not result.is_file():
        raise FileExistsError(f"Cannot verify existing artifacts in {output_dir}")
    prior = pd.read_csv(result, dtype={"prediction_sha256": str})
    if set(prior["prediction_sha256"].astype(str)) != hashes:
        raise FileExistsError(f"Existing outputs in {output_dir} use different caches")
    print("Existing Task 7 outputs use the same caches; regenerating them.")


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent,
        prefix=f".{path.stem}_", suffix=".csv", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def save_plots(overlap: pd.DataFrame, rescues: pd.DataFrame, output_dir: Path) -> None:
    dataset = str(overlap.iloc[0]["dataset"])
    colors = {
        "both_correct": "#4C78A8",
        "only_gcn_correct": "#F58518",
        "only_comparator_correct": "#54A24B",
        "both_wrong": "#E45756",
    }
    labels = {
        "both_correct": "Both correct",
        "only_gcn_correct": "Only GCN correct",
        "only_comparator_correct": "Only comparator correct (rescue)",
        "both_wrong": "Both wrong",
    }
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    bottom = np.zeros(len(COMPARATORS))
    for category in CATEGORIES:
        values = np.array([
            overlap.loc[overlap["comparator"].eq(model), f"{category}_pct_mean"].iloc[0]
            for model in COMPARATORS
        ])
        ax.bar(COMPARATORS, values, bottom=bottom, label=labels[category], color=colors[category])
        bottom += values
    ax.set_ylabel("Mean share of test nodes (%)")
    ax.set_title(f"GCN error overlap across five seeds — {dataset}")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower center", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "error_overlap.png", dpi=180)
    plt.close(fig)

    x = np.arange(len(GROUPS))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    for index, comparator in enumerate(COMPARATORS):
        subset = rescues[rescues["comparator"].eq(comparator)].set_index("confidence_group").loc[list(GROUPS)]
        position = x + (index - 0.5) * width
        axes[0].bar(
            position, subset["share_of_comparator_rescues_pct_mean"], width,
            yerr=subset["share_of_comparator_rescues_pct_std"], capsize=3, label=comparator,
        )
        axes[1].bar(
            position, subset["rescue_rate_among_gcn_mistakes_in_group_pct_mean"], width,
            yerr=subset["rescue_rate_among_gcn_mistakes_in_group_pct_std"], capsize=3,
            label=comparator,
        )
    for ax in axes:
        ax.set_xticks(x, [name.title() for name in GROUPS])
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("Share of all rescues (%)")
    axes[0].set_title("Where rescues occur")
    axes[1].set_ylabel("GCN mistakes rescued in group (%)")
    axes[1].set_title("Rescue rate among GCN mistakes")
    fig.suptitle(f"Rescues by GCN confidence tercile — {dataset}")
    fig.tight_layout()
    fig.savefig(output_dir / "rescues_by_confidence.png", dpi=180)
    plt.close(fig)


def save_summary(
    per_seed_overlap: pd.DataFrame, overlap: pd.DataFrame, rescues: pd.DataFrame, path: Path,
) -> None:
    dataset = str(per_seed_overlap.iloc[0]["dataset"])
    overlap_lines = []
    for comparator in COMPARATORS:
        row = overlap[overlap["comparator"].eq(comparator)].iloc[0]
        overlap_lines.append(
            f"| {comparator} | {row['both_correct_pct_mean_std']} | "
            f"{row['only_gcn_correct_pct_mean_std']} | "
            f"{row['only_comparator_correct_pct_mean_std']} | "
            f"{row['both_wrong_pct_mean_std']} | "
            f"{row['rescued_gcn_mistakes_mean_std']} | "
            f"{row['rescue_rate_among_gcn_mistakes_pct_mean_std']} |"
        )
    rescue_lines = []
    for comparator in COMPARATORS:
        for confidence_group in GROUPS:
            row = rescues[
                rescues["comparator"].eq(comparator)
                & rescues["confidence_group"].eq(confidence_group)
            ].iloc[0]
            rescue_lines.append(
                f"| {comparator} | {confidence_group.title()} | "
                f"{row['rescued_in_group_mean_std']} | "
                f"{row['share_of_comparator_rescues_pct_mean_std']} | "
                f"{row['rescue_rate_among_gcn_mistakes_in_group_pct_mean_std']} |"
            )
    text = f"""# GCN error overlap — {dataset}

This retrospective analysis uses five cached runs (seeds 42–46, 20 labels per class) and does
not retrain any model. A rescue is a test node that GCN predicts incorrectly and the comparator
predicts correctly.

| Comparator | Both correct (%) | Only GCN (%) | Rescue (%) | Both wrong (%) | Rescued nodes | Rescue rate among GCN mistakes (%) |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(overlap_lines)}

## Rescue location by GCN confidence

| Comparator | Confidence | Rescued nodes | Share of comparator rescues (%) | GCN mistakes rescued in group (%) |
|---|---|---:|---:|---:|
{chr(10).join(rescue_lines)}

Values are mean ± sample standard deviation across five seeds. Confidence groups are formed by
ranking GCN maximum probability on test nodes, without labels. Each seed has 334 low-confidence,
333 medium-confidence, and 333 high-confidence nodes. True labels are used only afterward to
measure retrospective correctness and rescue counts.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    cache_paths = sorted(args.prediction_dir.resolve().glob("*.npz"))
    if not cache_paths:
        raise FileNotFoundError(f"No .npz caches found in {args.prediction_dir}")
    overlap_rows, rescue_rows = [], []
    for cache_path in cache_paths:
        cache_overlap, cache_rescues = analyze_cache(cache_path)
        overlap_rows.extend(cache_overlap)
        rescue_rows.extend(cache_rescues)
    per_seed_overlap, per_seed_rescues = pd.DataFrame(overlap_rows), pd.DataFrame(rescue_rows)
    if per_seed_overlap["dataset"].nunique() != 1:
        raise ValueError("One invocation must contain caches from a single dataset")
    overlap = aggregate_overlap(per_seed_overlap)
    rescues = aggregate_rescues(per_seed_rescues)
    hashes = set(per_seed_overlap["prediction_sha256"])
    guard_outputs(args.output_dir, hashes)
    atomic_csv(per_seed_overlap, args.output_dir / OUTPUT_NAMES[0])
    atomic_csv(per_seed_rescues, args.output_dir / OUTPUT_NAMES[1])
    atomic_csv(overlap, args.output_dir / OUTPUT_NAMES[2])
    atomic_csv(rescues, args.output_dir / OUTPUT_NAMES[3])
    save_plots(overlap, rescues, args.output_dir)
    save_summary(per_seed_overlap, overlap, rescues, args.output_dir / OUTPUT_NAMES[6])
    print(
        f"Task 7: analyzed {len(cache_paths)} {per_seed_overlap.iloc[0]['dataset']} caches; "
        f"saved {args.output_dir.resolve()}"
    )
    print(overlap[[
        "comparator", "only_comparator_correct_pct_mean_std",
        "rescued_gcn_mistakes_mean_std", "rescue_rate_among_gcn_mistakes_pct_mean_std",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
