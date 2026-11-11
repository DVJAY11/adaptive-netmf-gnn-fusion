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

    Intended for Cora and CiteSeer. The sparse walk powers fill in quickly,
    so for PubMed use ``compute_netmf_embedding_eigen`` instead.
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


def compute_netmf_embedding_eigen(
    edge_index,
    num_nodes: int,
    dimension: int = 128,
    window: int = 5,
    negative_samples: float = 1.0,
    rank: int = 256,
    seed: int = 0,
    block_rows: int = 2048,
) -> np.ndarray:
    """NetMF for larger graphs (Qiu et al., 2018, "NetMF for a large window").

    Instead of summing sparse powers of the transition matrix, which fill in
    and need tens of GB on PubMed, approximate the normalized adjacency
    D^-1/2 A D^-1/2 by its top ``rank`` eigenpairs U, lambda. Then

        M ~= vol / b * D^-1/2 U diag(f(lambda)) U^T D^-1/2,
        f(lambda) = (1/T) * sum_{r=1..T} lambda^r.

    The dense log(max(M, 1)) matrix is built in float32 row blocks
    (about 1.6 GB for PubMed) and factorized as in the exact version.
    """
    from scipy.sparse.linalg import eigsh

    if window < 1:
        raise ValueError("window must be at least 1")
    if dimension < 2 or dimension >= num_nodes:
        raise ValueError("dimension must be between 2 and num_nodes - 1")
    rank = min(rank, num_nodes - 2)
    if rank < dimension:
        raise ValueError("rank must be at least the embedding dimension")

    adjacency = edge_index_to_adjacency(edge_index, num_nodes)
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1.0))
    inv_sqrt[degree == 0] = 0.0
    normalized = sp.diags(inv_sqrt) @ adjacency @ sp.diags(inv_sqrt)

    rng = np.random.default_rng(seed)
    eigenvalues, eigenvectors = eigsh(
        normalized, k=rank, which="LA", v0=rng.standard_normal(num_nodes)
    )
    powers = np.stack([eigenvalues ** r for r in range(1, window + 1)])
    filtered = powers.sum(axis=0) / window

    volume = float(degree.sum())
    left = (inv_sqrt[:, None] * eigenvectors) * (filtered * volume / negative_samples)
    right = inv_sqrt[:, None] * eigenvectors
    left, right = left.astype(np.float32), right.astype(np.float32)

    dense = np.empty((num_nodes, num_nodes), dtype=np.float32)
    for begin in range(0, num_nodes, block_rows):
        end = min(begin + block_rows, num_nodes)
        block = left[begin:end] @ right.T
        np.maximum(block, 1.0, out=block)
        np.log(block, out=dense[begin:end])

    left_vectors, singular_values, _ = randomized_svd(
        dense, n_components=dimension, n_iter=5, random_state=seed,
    )
    del dense
    embedding = left_vectors * np.sqrt(np.maximum(singular_values, 1e-12))
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    return (embedding / np.maximum(norms, 1e-12)).astype(np.float32)


def load_or_compute_netmf(
    edge_index,
    num_nodes: int,
    cache_path: Path,
    dimension: int,
    window: int,
    seed: int,
    method: str = "exact",
    rank: int = 256,
) -> tuple[np.ndarray, float, bool]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return np.load(cache_path), 0.0, True

    start = time.perf_counter()
    if method == "exact":
        embedding = compute_netmf_embedding(
            edge_index=edge_index,
            num_nodes=num_nodes,
            dimension=dimension,
            window=window,
            seed=seed,
        )
    elif method == "eigen":
        embedding = compute_netmf_embedding_eigen(
            edge_index=edge_index,
            num_nodes=num_nodes,
            dimension=dimension,
            window=window,
            rank=rank,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown NetMF method: {method}")
    seconds = time.perf_counter() - start
    np.save(cache_path, embedding)
    return embedding, seconds, False
