from __future__ import annotations

from typing import List, Tuple

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0

    return float(np.dot(a, b) / (a_norm * b_norm))


def find_nearest_neighbors(
    query_word: str,
    word_to_id: dict[str, int],
    id_to_word: dict[int, str],
    embeddings: np.ndarray,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Find the nearest neighbors of a word using cosine similarity.

    Returns:
        List of (word, similarity) tuples.
    """
    if query_word not in word_to_id:
        raise ValueError(f"Word '{query_word}' is not in the vocabulary.")

    query_id = word_to_id[query_word]
    query_vector = embeddings[query_id]

    similarities = []
    for idx in range(len(embeddings)):
        if idx == query_id:
            continue

        sim = cosine_similarity(query_vector, embeddings[idx])
        similarities.append((id_to_word[idx], sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]