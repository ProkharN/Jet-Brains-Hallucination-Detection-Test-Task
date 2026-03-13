from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np


def read_text(path: str) -> str:
    """Read raw text from a file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def tokenize(text: str) -> List[str]:
    """
    Simple tokenizer for English text.

    Steps:
    - lowercase
    - keep only letters and apostrophes
    - collapse whitespace
    - split into tokens
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def build_vocab(
    tokens: List[str],
    min_count: int = 1,
    max_vocab_size: int | None = None,
) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, int]]:
    """
    Build vocabulary mappings and filtered word counts.

    Returns:
    - word_to_id
    - id_to_word
    - filtered_counts
    """
    counts = Counter(tokens)

    filtered_items = [(word, count) for word, count in counts.items() if count >= min_count]
    filtered_items.sort(key=lambda x: (-x[1], x[0]))

    if max_vocab_size is not None:
        filtered_items = filtered_items[:max_vocab_size]

    filtered_counts = {word: count for word, count in filtered_items}

    word_to_id = {word: idx for idx, (word, _) in enumerate(filtered_items)}
    id_to_word = {idx: word for word, idx in word_to_id.items()}

    return word_to_id, id_to_word, filtered_counts


def encode_tokens(tokens: List[str], word_to_id: Dict[str, int]) -> List[int]:
    """
    Convert tokens into token ids.
    Tokens outside the vocabulary are skipped.
    """
    return [word_to_id[token] for token in tokens if token in word_to_id]


def generate_skipgram_pairs(token_ids: List[int], window_size: int) -> List[Tuple[int, int]]:
    """
    Generate (target_id, context_id) pairs for skip-gram.

    Example:
    sequence = [w1, w2, w3, w4], window_size = 1
    pairs:
    (w1, w2), (w2, w1), (w2, w3), (w3, w2), ...
    """
    pairs: List[Tuple[int, int]] = []
    n_tokens = len(token_ids)

    for center_pos, target_id in enumerate(token_ids):
        left = max(0, center_pos - window_size)
        right = min(n_tokens, center_pos + window_size + 1)

        for context_pos in range(left, right):
            if context_pos == center_pos:
                continue
            context_id = token_ids[context_pos]
            pairs.append((target_id, context_id))

    return pairs


def build_negative_sampling_distribution(
    word_counts: Dict[str, int],
    word_to_id: Dict[str, int],
    power: float = 0.75,
) -> np.ndarray:
    """
    Build the unigram distribution^power used for negative sampling.

    Returns:
    - probs: numpy array of shape (vocab_size,)
    """
    vocab_size = len(word_to_id)
    probs = np.zeros(vocab_size, dtype=np.float64)

    for word, idx in word_to_id.items():
        probs[idx] = float(word_counts[word]) ** power

    probs_sum = probs.sum()
    if probs_sum == 0:
        raise ValueError("Negative sampling distribution sum is zero.")

    probs /= probs_sum
    return probs