from src.data import (
    build_negative_sampling_distribution,
    build_vocab,
    encode_tokens,
    generate_skipgram_pairs,
    read_text,
    tokenize,
)


def main() -> None:
    text = read_text("data/sample.txt")
    tokens = tokenize(text)

    word_to_id, id_to_word, word_counts = build_vocab(
        tokens,
        min_count=1,
        max_vocab_size=None,
    )

    token_ids = encode_tokens(tokens, word_to_id)
    pairs = generate_skipgram_pairs(token_ids, window_size=2)
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


if __name__ == "__main__":
    main()