"""
build_vocab.py

Run ONCE before training to build a reduced vocabulary from the corpus.
Saves vocab to the path defined in config.yaml (data.vocab_path).

Usage:
    python build_vocab.py
"""

import json
import yaml
from pathlib import Path
from collections import Counter
from transformers import BertTokenizer


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_vocab(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    token_counter = Counter()

    # Count tokens across all splits
    for split in ["train", "val", "test"]:
        index_path = Path(config["data"]["processed_dir"]) / split / "index.json"
        if not index_path.exists():
            print(f"  Skipping {split} (not found)")
            continue

        with open(index_path) as f:
            index = json.load(f)

        for item in index:
            tokens = tokenizer.tokenize(item["sentence"])
            token_counter.update(tokens)

        print(f"  {split}: {len(index)} sentences")

    # Always include special tokens
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    # Keep tokens that appear at least min_freq times
    min_freq = config["data"].get("vocab_min_freq", 1)
    corpus_tokens = [tok for tok, count in token_counter.items() if count >= min_freq]

    vocab_tokens = special_tokens + sorted(corpus_tokens)
    vocab = {token: idx for idx, token in enumerate(vocab_tokens)}

    vocab_path = Path(config["data"]["vocab_path"])
    vocab_path.parent.mkdir(parents=True, exist_ok=True)

    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)

    print(f"\nVocab size: {len(vocab)} tokens (down from 30,522)")
    print(f"Saved to: {vocab_path}")


if __name__ == "__main__":
    build_vocab()