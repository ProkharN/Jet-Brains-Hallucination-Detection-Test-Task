import numpy as np

from src.data import (
    build_negative_sampling_distribution,
    build_vocab,
    encode_tokens,
    generate_skipgram_pairs,
    read_text,
    tokenize,
)
from src.model import SkipGramNegativeSampling
from src.train import train
from src.utils import find_nearest_neighbors


def main() -> None:
    # Hyperparameters and configuration
    data_path = "data/sample.txt"
    min_count = 1
    max_vocab_size = 3000
    window_size = 2
    embedding_dim = 50
    num_negative = 5
    learning_rate = 0.025
    epochs = 2
    log_every = 10000
    query_words = ["alice", "rabbit", "queen", "king", "cat"]

    # Data preparation
    text = read_text(data_path)
    tokens = tokenize(text)

    word_to_id, id_to_word, word_counts = build_vocab(
        tokens,
        min_count=min_count,
        max_vocab_size=max_vocab_size,
    )

    token_ids = encode_tokens(tokens, word_to_id)
    pairs = generate_skipgram_pairs(token_ids, window_size=window_size)
    neg_dist = build_negative_sampling_distribution(word_counts, word_to_id)

    print(f"Number of tokens: {len(tokens)}")
    print(f"Vocabulary size: {len(word_to_id)}")
    print(f"Encoded token count: {len(token_ids)}")
    print(f"Number of skip-gram pairs: {len(pairs)}")

    print("\nFirst 20 tokens:")
    print(tokens[:20])

    print("\nFirst 10 vocabulary items:")
    for i, (word, idx) in enumerate(word_to_id.items()):
        if i >= 10:
            break
        print(f"{word} -> {idx}")

    print("\nFirst 10 skip-gram pairs:")
    for target_id, context_id in pairs[:10]:
        print(f"({id_to_word[target_id]}, {id_to_word[context_id]})")

    print("\nNegative sampling distribution sanity check:")
    print(f"Sum = {neg_dist.sum():.6f}")
    print(f"Shape = {neg_dist.shape}")

    # Single-example smoke test
    model = SkipGramNegativeSampling(
        vocab_size=len(word_to_id),
        embedding_dim=embedding_dim,
        seed=42,
    )

    target_id, context_id = pairs[0]

    candidate_negatives = [
        idx for idx in range(len(word_to_id))
        if idx != target_id and idx != context_id
    ]
    negative_ids = np.array(candidate_negatives[:num_negative], dtype=np.int64)

    loss = model.train_example(
        target_id=target_id,
        context_id=context_id,
        negative_ids=negative_ids,
        learning_rate=learning_rate,
    )

    print("\nSingle training example test:")
    print(f"Target word: {id_to_word[target_id]}")
    print(f"Context word: {id_to_word[context_id]}")
    print(f"Negative words: {[id_to_word[i] for i in negative_ids]}")
    print(f"Loss: {loss:.6f}")

    # Full training
    print("\nStarting training...")
    epoch_losses = train(
        model=model,
        pairs=pairs,
        distribution=neg_dist,
        epochs=epochs,
        learning_rate=learning_rate,
        num_negative=num_negative,
        log_every=log_every,
    )

    print("Training finished.")
    print("Epoch losses:", [round(loss, 4) for loss in epoch_losses])

    # Qualitative evaluation
    embeddings = model.get_word_embeddings()

    print("\nNearest neighbors:")
    for word in query_words:
        if word in word_to_id:
            neighbors = find_nearest_neighbors(
                query_word=word,
                word_to_id=word_to_id,
                id_to_word=id_to_word,
                embeddings=embeddings,
                top_k=5,
            )
            print(f"\n{word}:")
            for neighbor, score in neighbors:
                print(f"  {neighbor:<15} {score:.4f}")
        else:
            print(f"\n{word}: not in vocabulary")


if __name__ == "__main__":
    main()