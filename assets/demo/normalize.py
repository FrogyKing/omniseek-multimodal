"""Normalize vectors for cosine similarity."""
import numpy as np

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("zero vector")
    return vector / norm
