from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from src.model import SkipGramNegativeSampling


def sample_negative_ids(
    distribution: np.ndarray,
    num_negative: int,
    forbidden_ids: set[int] | None = None,
) -> np.ndarray:
    """
    Sample negative word ids from the negative sampling distribution.

    Repeated ids are allowed, which is standard for negative sampling.
    Words in forbidden_ids are excluded.
    """
    vocab_size = len(distribution)
    forbidden_ids = forbidden_ids or set()

    negative_ids = []
    while len(negative_ids) < num_negative:
        sampled_id = np.random.choice(vocab_size, p=distribution)
        if sampled_id in forbidden_ids:
            continue
        negative_ids.append(sampled_id)

    return np.array(negative_ids, dtype=np.int64)


def train(
    model: SkipGramNegativeSampling,
    pairs: Sequence[Tuple[int, int]],
    distribution: np.ndarray,
    epochs: int,
    learning_rate: float,
    num_negative: int,
    log_every: int = 10000,
) -> List[float]:
    """
    Train the SGNS model over skip-gram pairs.

    Returns:
        List of average losses per epoch.
    """
    epoch_losses = []

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for step, (target_id, context_id) in enumerate(pairs, start=1):
            negative_ids = sample_negative_ids(
                distribution=distribution,
                num_negative=num_negative,
                forbidden_ids={target_id, context_id},
            )

            loss = model.train_example(
                target_id=target_id,
                context_id=context_id,
                negative_ids=negative_ids,
                learning_rate=learning_rate,
            )

            total_loss += loss

            if step % log_every == 0:
                avg_loss_so_far = total_loss / step
                print(
                    f"Epoch {epoch}/{epochs} | "
                    f"Step {step}/{len(pairs)} | "
                    f"Average loss: {avg_loss_so_far:.4f}"
                )

        avg_epoch_loss = total_loss / len(pairs)
        epoch_losses.append(avg_epoch_loss)

        print(f"\nEpoch {epoch} finished.")
        print(f"Average epoch loss: {avg_epoch_loss:.4f}\n")

    return epoch_losses