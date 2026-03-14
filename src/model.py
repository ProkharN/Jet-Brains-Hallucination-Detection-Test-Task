from __future__ import annotations

import numpy as np


class SkipGramNegativeSampling:
    def __init__(self, vocab_size: int, embedding_dim: int, seed: int = 42) -> None:
        """
        Initialize input and output embedding matrices.

        W_in  - embeddings for target words
        W_out - embeddings for context words
        """
        rng = np.random.default_rng(seed)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.W_in = rng.uniform(
            low=-0.5 / embedding_dim,
            high=0.5 / embedding_dim,
            size=(vocab_size, embedding_dim),
        )

        self.W_out = np.zeros((vocab_size, embedding_dim), dtype=np.float64)

    @staticmethod
    def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
        """Numerically stable sigmoid."""
        x = np.clip(x, -10, 10)
        return 1.0 / (1.0 + np.exp(-x))

    def train_example(
        self,
        target_id: int,
        context_id: int,
        negative_ids: np.ndarray,
        learning_rate: float,
    ) -> float:
        """
        Perform one SGD update for a single skip-gram example with negative sampling.

        Returns:
            loss for this training example
        """

        v = self.W_in[target_id]
        u_pos = self.W_out[context_id]
        u_negs = self.W_out[negative_ids]

        v_old = v.copy()
        u_pos_old = u_pos.copy()
        u_negs_old = u_negs.copy()

        pos_score = np.dot(u_pos_old, v_old)
        pos_sigmoid = self.sigmoid(pos_score)
        pos_loss = -np.log(pos_sigmoid + 1e-10)

        neg_scores = np.dot(u_negs_old, v_old)
        neg_sigmoids = self.sigmoid(neg_scores)
        neg_loss = -np.sum(np.log(self.sigmoid(-neg_scores) + 1e-10))

        loss = pos_loss + neg_loss

        grad_pos_score = pos_sigmoid - 1.0
        grad_neg_scores = neg_sigmoids

        grad_u_pos = grad_pos_score * v_old

        grad_u_negs = grad_neg_scores[:, None] * v_old[None, :]

        grad_v = grad_pos_score * u_pos_old + np.sum(
            grad_neg_scores[:, None] * u_negs_old,
            axis=0,
        )

        self.W_in[target_id] -= learning_rate * grad_v
        self.W_out[context_id] -= learning_rate * grad_u_pos

        for i, neg_id in enumerate(negative_ids):
            self.W_out[neg_id] -= learning_rate * grad_u_negs[i]

        return float(loss)

    def get_input_embeddings(self) -> np.ndarray:
        """Return target/input embeddings."""
        return self.W_in

    def get_output_embeddings(self) -> np.ndarray:
        """Return context/output embeddings."""
        return self.W_out

    def get_word_embeddings(self) -> np.ndarray:
        """
        Return final word embeddings.
        A common choice is to use W_in or average W_in and W_out.
        """
        return (self.W_in + self.W_out) / 2.0