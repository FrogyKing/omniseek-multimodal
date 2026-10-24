"""Cosine search, MRL compression and explicit model compatibility checks."""

import faiss
import numpy as np
from numpy.typing import NDArray

from .config import DIMENSIONS


def normalize(vectors: NDArray, dimension: int = 768) -> NDArray[np.float32]:
    if dimension not in DIMENSIONS:
        raise ValueError("Supported dimensions: 768, 512, 256, 128")
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim not in (1, 2) or array.shape[-1] < dimension:
        raise ValueError("Embedding shape is incompatible with requested dimension")
    array = array[..., :dimension].copy()
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    if not np.all(np.isfinite(array)) or np.any(norm <= 1e-12):
        raise ValueError("Embedding contains nonfinite values or has zero norm")
    return np.asarray(array / norm, dtype=np.float32)


def search(
    vectors: NDArray, query: NDArray, dimension: int, k: int, approximate: bool = False
) -> tuple[NDArray, NDArray]:
    corpus = normalize(vectors, dimension)
    q = normalize(query, dimension).reshape(1, -1)
    index: faiss.Index
    if approximate:
        index = faiss.IndexHNSWFlat(dimension, 16, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = 64
    else:
        index = faiss.IndexFlatIP(dimension)
    index.add(corpus)
    scores, ids = index.search(q, min(k, len(corpus)))
    return scores[0], ids[0]
