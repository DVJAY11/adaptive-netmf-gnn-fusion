"""Small-graph NetMF embedding used for the Planetoid experiments."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd


def edge_index_to_adjacency(edge_index, num_nodes: int) -> sp.csr_matrix:
    edges = edge_index.detach().cpu().numpy()
    values = np.ones(edges.shape[1], dtype=np.float64)
    adjacency = sp.coo_matrix(
        (values, (edges[0], edges[1])), shape=(num_nodes, num_nodes)
    ).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.data[:] = 1.0
    return adjacency


def compute_netmf_embedding(
    edge_index,
    num_nodes: int,
    dimension: int = 128,
    window: int = 5,
    negative_samples: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Compute the explicit NetMF positive-PMI matrix and factorize it.

    This implementation is intended for Cora, CiteSeer, and PubMed. Cora and
    CiteSeer are inexpensive. PubMed can require several GB of memory because
    the explicit matrix is dense, so use a smaller dimension/window if needed.
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    if dimension < 2 or dimension >= num_nodes:
        raise ValueError("dimension must be between 2 and num_nodes - 1")

    adjacency = edge_index_to_adjacency(edge_index, num_nodes)
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    degree = np.maximum(degree, 1.0)
    transition = sp.diags(1.0 / degree) @ adjacency

    power = transition.copy()
    walk_sum = power.copy()
    for _ in range(2, window + 1):
        power = power @ transition
        walk_sum = walk_sum + power

    volume = float(degree.sum())
    scale = volume / (negative_samples * window)
    matrix = scale * (walk_sum @ sp.diags(1.0 / degree))
    dense = matrix.toarray()
    dense = np.log(np.maximum(dense, 1.0))

    left, singular_values, _ = randomized_svd(
        dense,
        n_components=dimension,
        n_iter=5,
        random_state=seed,
    )
    embedding = left * np.sqrt(np.maximum(singular_values, 1e-12))
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    return (embedding / np.maximum(norms, 1e-12)).astype(np.float32)


def load_or_compute_netmf(
    edge_index,
    num_nodes: int,
    cache_path: Path,
    dimension: int,
    window: int,
    seed: int,
) -> tuple[np.ndarray, float, bool]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return np.load(cache_path), 0.0, True

    start = time.perf_counter()
    embedding = compute_netmf_embedding(
        edge_index=edge_index,
        num_nodes=num_nodes,
        dimension=dimension,
        window=window,
        seed=seed,
    )
    seconds = time.perf_counter() - start
    np.save(cache_path, embedding)
    return embedding, seconds, False
