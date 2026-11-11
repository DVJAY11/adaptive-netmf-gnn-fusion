"""Lossless per-node experiment outputs for follow-up analysis.

Only predictions and public graph/split metadata are cached; no model weights.
The oracle output uses true labels and must never be treated as a deployable model.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np


def cache_path(output_dir: Path, dataset: str, seed: int, budget: int, edge_drop: float) -> Path:
    """Build a unique file name for one experimental setting."""
    if not (0 <= edge_drop < 1):
        raise ValueError("edge_drop must be in [0, 1)")
    # 6 decimal places gives stable paths for normally specified drop fractions.
    # The full-precision rate is included in the payload for verification.
    rate = f"{edge_drop:.6f}".replace(".", "p")
    return Path(output_dir) / "predictions" / f"{dataset}_seed{seed}_budget{budget}_drop{rate}.npz"


def save_prediction_cache(*, output_dir: Path, dataset: str, seed: int, budget: int,
                          edge_drop: float, probabilities: dict[str, np.ndarray],
                          labels: np.ndarray, degrees: np.ndarray,
                          train_mask: np.ndarray, val_mask: np.ndarray,
                          test_mask: np.ndarray, base_stop_mask: np.ndarray,
                          router_train_mask: np.ndarray, router_stop_mask: np.ndarray) -> Path:
    """Write a compressed .npz atomically to avoid partial output on interruption.

    The ``prob__<model_name>`` arrays contain full-graph, class-ordered
    probabilities. Train/validation/test and router masks are stored explicitly.
    """
    labels = np.asarray(labels, dtype=np.int64)
    n = labels.size
    if labels.ndim != 1 or not probabilities:
        raise ValueError("Expected a 1-D labels array and at least one model")
    if any(not name.isidentifier() for name in probabilities):
        raise ValueError("Model names must be identifiers")
    payload = {
        "format_version": np.array(1, dtype=np.int32),
        "dataset": np.array(dataset),
        "seed": np.array(seed, dtype=np.int64),
        "budget": np.array(budget, dtype=np.int64),
        "edge_drop": np.array(edge_drop, dtype=np.float64),
        "labels": labels,
        "degrees": np.asarray(degrees, dtype=np.int64),
    }
    for name, mask in (
        ("train_mask", train_mask), ("val_mask", val_mask),
        ("test_mask", test_mask), ("base_stop_mask", base_stop_mask),
        ("router_train_mask", router_train_mask),
        ("router_stop_mask", router_stop_mask),
    ):
        array = np.asarray(mask, dtype=np.bool_)
        if array.shape != (n,):
            raise ValueError(f"{name} has shape {array.shape}, expected ({n},)")
        payload[name] = array
    if payload["degrees"].shape != (n,):
        raise ValueError("Degrees must have one entry per node")
    for name, probs in probabilities.items():
        arr = np.asarray(probs)
        if arr.ndim != 2 or arr.shape[0] != n or not np.isfinite(arr).all():
            raise ValueError(f"Invalid predictions for {name}: {arr.shape}")
        payload[f"prob__{name}"] = arr  # preserve original dtype and numerical precision
    path = cache_path(output_dir, dataset, seed, budget, edge_drop)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".prediction_", suffix=".npz", delete=False) as handle:
            temp_name = handle.name
            np.savez_compressed(handle, **payload)
        os.replace(temp_name, path)
    finally:
        if temp_name is not None and os.path.exists(temp_name):
            os.unlink(temp_name)
    return path


def load_prediction_cache(path: Path) -> dict[str, np.ndarray]:
    """Load cache arrays without enabling pickle object deserialization."""
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}
