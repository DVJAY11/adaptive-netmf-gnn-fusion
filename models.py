"""PyTorch Geometric models and a shared early-stopping training loop."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MLP(torch.nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int, dropout: float):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden)
        self.lin2 = torch.nn.Linear(hidden, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        del edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)


class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int, dropout: float):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden, cached=False)
        self.conv2 = GCNConv(hidden, out_channels, cached=False)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int, dropout: float):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


@dataclass(frozen=True)
class TrainResult:
    probabilities: np.ndarray
    best_epoch: int
    train_seconds: float
    peak_memory_mb: float


@torch.no_grad()
def _accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    prediction = logits[mask].argmax(dim=-1)
    return float((prediction == labels[mask]).float().mean().item())


def train_model(
    model: torch.nn.Module,
    data,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    device: torch.device,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
) -> TrainResult:
    model = model.to(device)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    labels = data.y.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    process = psutil.Process()
    start_rss = process.memory_info().rss
    start = time.perf_counter()
    best_score = -1.0
    best_epoch = 0
    best_state = None
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, edge_index)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x, edge_index)
            val_score = _accuracy(val_logits, labels, val_mask)
        if val_score > best_score + 1e-8:
            best_score = val_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    train_seconds = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError("Training did not produce a model state")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probabilities = model(x, edge_index).softmax(dim=-1).cpu().numpy()

    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        peak_memory_mb = max(0, process.memory_info().rss - start_rss) / (1024**2)
    return TrainResult(probabilities, best_epoch, train_seconds, peak_memory_mb)
