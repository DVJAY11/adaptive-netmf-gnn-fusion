"""Prediction-fusion methods used by the project experiments."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RouterResult:
    probabilities: np.ndarray
    weights: np.ndarray
    low_threshold: float
    high_threshold: float


@dataclass(frozen=True)
class LearnedRouterResult:
    probabilities: np.ndarray
    weights: np.ndarray
    low_threshold: float
    high_threshold: float
    best_epoch: int
    train_seconds: float


@dataclass(frozen=True)
class OracleResult:
    probabilities: np.ndarray
    any_correct: np.ndarray
    selected_expert: np.ndarray


def _validate_probabilities(probabilities: list[np.ndarray]) -> tuple[int, int]:
    if len(probabilities) != 3:
        raise ValueError("NetMF, GCN, and GraphSAGE probabilities are required")
    shapes = {array.shape for array in probabilities}
    if len(shapes) != 1:
        raise ValueError(f"Probability arrays must share one shape, got {shapes}")
    rows, classes = probabilities[0].shape
    if rows == 0 or classes < 2:
        raise ValueError("Probability arrays must contain nodes and at least two classes")
    return rows, classes


def equal_average(probabilities: list[np.ndarray]) -> np.ndarray:
    """Average NetMF, GCN, and GraphSAGE probabilities equally."""
    _validate_probabilities(probabilities)
    return np.mean(np.stack(probabilities, axis=0), axis=0)


def degree_router(
    probabilities: list[np.ndarray], degrees: np.ndarray
) -> RouterResult:
    """Fuse predictions with transparent degree-dependent weights.

    Low-degree nodes rely more on NetMF. High-degree nodes rely more on the
    neighborhood-based GNNs. The middle group uses nearly equal weights.
    """
    rows, _ = _validate_probabilities(probabilities)
    degrees = np.asarray(degrees, dtype=np.float64).reshape(-1)
    if len(degrees) != rows:
        raise ValueError("The number of degree values must match the number of nodes")

    low_threshold, high_threshold = np.quantile(degrees, [1.0 / 3.0, 2.0 / 3.0])
    weights = np.empty((rows, 3), dtype=np.float64)
    low = degrees <= low_threshold
    high = degrees > high_threshold
    middle = ~(low | high)

    weights[low] = np.array([0.50, 0.25, 0.25])
    weights[middle] = np.array([0.34, 0.33, 0.33])
    weights[high] = np.array([0.20, 0.40, 0.40])

    stacked = np.stack(probabilities, axis=1)
    fused = np.sum(stacked * weights[:, :, None], axis=1)
    return RouterResult(
        probabilities=fused,
        weights=weights,
        low_threshold=float(low_threshold),
        high_threshold=float(high_threshold),
    )


def build_router_features(
    probabilities: list[np.ndarray], degrees: np.ndarray, fit_mask: np.ndarray,
    feature_set: str = "all",
) -> np.ndarray:
    """Create degree, confidence, entropy, and disagreement features."""
    rows, classes = _validate_probabilities(probabilities)
    degrees = np.asarray(degrees, dtype=np.float64).reshape(-1)
    fit_mask = np.asarray(fit_mask, dtype=bool).reshape(-1)
    if len(degrees) != rows or len(fit_mask) != rows:
        raise ValueError("Degrees and fit_mask must contain one value per node")
    if not np.any(fit_mask):
        raise ValueError("fit_mask must select at least one node")

    stacked = np.stack(probabilities, axis=1).astype(np.float64)
    clipped = np.clip(stacked, 1e-12, 1.0)
    confidence = clipped.max(axis=2)
    entropy = -(clipped * np.log(clipped)).sum(axis=2) / np.log(classes)
    predictions = stacked.argmax(axis=2)
    disagreements = np.column_stack([
        predictions[:, 0] != predictions[:, 1],
        predictions[:, 0] != predictions[:, 2],
        predictions[:, 1] != predictions[:, 2],
    ]).astype(np.float64)

    log_degree = np.log1p(degrees)
    mean = log_degree[fit_mask].mean()
    std = log_degree[fit_mask].std()
    normalized_degree = (log_degree - mean) / max(std, 1e-8)
    groups = {
        "degree": normalized_degree[:, None],
        "confidence": np.column_stack([confidence, entropy]),
        "all": np.column_stack([
            normalized_degree, confidence, entropy, disagreements
        ]),
    }
    if feature_set not in groups:
        raise ValueError(f"Unknown router feature set: {feature_set}")
    return groups[feature_set].astype(np.float32)


class _RouterMLP(torch.nn.Module):
    def __init__(self, in_features: int, hidden: int):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).softmax(dim=-1)


def learned_router(
    probabilities: list[np.ndarray],
    degrees: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    *,
    seed: int,
    device: torch.device,
    epochs: int = 300,
    patience: int = 40,
    hidden: int = 16,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    feature_set: str = "all",
) -> LearnedRouterResult:
    """Train a small validation-only gate over NetMF, GCN, and GraphSAGE."""
    rows, _ = _validate_probabilities(probabilities)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    train_mask = np.asarray(train_mask, dtype=bool).reshape(-1)
    val_mask = np.asarray(val_mask, dtype=bool).reshape(-1)
    if any(len(array) != rows for array in (labels, train_mask, val_mask)):
        raise ValueError("Labels and router masks must contain one value per node")
    if not np.any(train_mask) or not np.any(val_mask):
        raise ValueError("Router training and validation masks cannot be empty")
    if np.any(train_mask & val_mask):
        raise ValueError("Router training and validation masks must be disjoint")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    features = torch.from_numpy(
        build_router_features(probabilities, degrees, train_mask, feature_set)
    ).to(device)
    expert_probabilities = torch.from_numpy(
        np.stack(probabilities, axis=1).astype(np.float32)
    ).to(device)
    targets = torch.from_numpy(labels).to(device)
    train = torch.from_numpy(train_mask).to(device)
    validation = torch.from_numpy(val_mask).to(device)

    model = _RouterMLP(features.shape[1], hidden).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        weights = model(features)
        fused = (expert_probabilities * weights.unsqueeze(-1)).sum(dim=1)
        chosen = fused[train, targets[train]].clamp_min(1e-12)
        loss = -chosen.log().mean()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_weights = model(features)
            val_fused = (expert_probabilities * val_weights.unsqueeze(-1)).sum(dim=1)
            val_chosen = val_fused[validation, targets[validation]].clamp_min(1e-12)
            val_loss = float(-val_chosen.log().mean().item())
        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    train_seconds = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError("Learned router training did not produce a model state")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        weights = model(features)
        fused = (expert_probabilities * weights.unsqueeze(-1)).sum(dim=1)
    low_threshold, high_threshold = np.quantile(degrees, [1.0 / 3.0, 2.0 / 3.0])
    return LearnedRouterResult(
        probabilities=fused.cpu().numpy(),
        weights=weights.cpu().numpy(),
        low_threshold=float(low_threshold),
        high_threshold=float(high_threshold),
        best_epoch=best_epoch,
        train_seconds=train_seconds,
    )


def oracle_best_expert(
    probabilities: list[np.ndarray], labels: np.ndarray
) -> OracleResult:
    """Select a correct expert when one exists, giving an unattainable upper bound."""
    rows, _ = _validate_probabilities(probabilities)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(labels) != rows:
        raise ValueError("Labels must contain one value per node")
    stacked = np.stack(probabilities, axis=1)
    predictions = stacked.argmax(axis=2)
    correct = predictions == labels[:, None]
    confidence = stacked.max(axis=2)
    eligible_score = np.where(correct, confidence, -1.0)
    selected = eligible_score.argmax(axis=1)
    none_correct = ~correct.any(axis=1)
    selected[none_correct] = confidence[none_correct].argmax(axis=1)
    chosen = stacked[np.arange(rows), selected]
    return OracleResult(
        probabilities=chosen,
        any_correct=correct.any(axis=1),
        selected_expert=selected,
    )


def router_weight_summary(result: RouterResult, degrees: np.ndarray) -> list[dict]:
    """Return mean model weights for low-, medium-, and high-degree nodes."""
    degrees = np.asarray(degrees).reshape(-1)
    groups = {
        "low": degrees <= result.low_threshold,
        "medium": (degrees > result.low_threshold) & (degrees <= result.high_threshold),
        "high": degrees > result.high_threshold,
    }
    rows = []
    for group, mask in groups.items():
        if not np.any(mask):
            continue
        means = result.weights[mask].mean(axis=0)
        for model, value in zip(("NetMF", "GCN", "GraphSAGE"), means, strict=True):
            rows.append({"degree_group": group, "base_model": model, "weight": float(value)})
    return rows
