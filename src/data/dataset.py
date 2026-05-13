import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from transformers import BertTokenizer


class CorpusTokenizer:
    """
    Lightweight tokenizer built on top of BertTokenizer but limited to
    the vocabulary found in the corpus (built by build_vocab.py).

    Special token IDs are fixed:
        PAD  = 0
        UNK  = 1
        CLS  = 2
        SEP  = 3
        MASK = 4
    """

    PAD_TOKEN  = "[PAD]"
    UNK_TOKEN  = "[UNK]"
    CLS_TOKEN  = "[CLS]"
    SEP_TOKEN  = "[SEP]"
    MASK_TOKEN = "[MASK]"

    def __init__(self, vocab_path: str):
        with open(vocab_path) as f:
            self.token2id: dict = json.load(f)

        self.id2token: dict = {v: k for k, v in self.token2id.items()}

        # Special token IDs (always present — build_vocab guarantees this)
        self.pad_token_id  = self.token2id[self.PAD_TOKEN]
        self.unk_token_id  = self.token2id[self.UNK_TOKEN]
        self.cls_token_id  = self.token2id[self.CLS_TOKEN]
        self.sep_token_id  = self.token2id[self.SEP_TOKEN]
        self.mask_token_id = self.token2id[self.MASK_TOKEN]

        # Use BertTokenizer only for subword splitting
        self._bert = BertTokenizer.from_pretrained("bert-base-uncased")

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    def tokenize(self, text: str) -> list[str]:
        """Subword-tokenize using BERT rules, map unknowns to [UNK]."""
        bert_tokens = self._bert.tokenize(text)
        return [t if t in self.token2id else self.UNK_TOKEN for t in bert_tokens]

    def encode(
        self,
        text: str,
        max_length: int = 30,
        padding: bool = True,
        truncation: bool = True,
    ) -> dict:
        tokens = [self.CLS_TOKEN] + self.tokenize(text) + [self.SEP_TOKEN]

        if truncation and len(tokens) > max_length:
            # Keep [CLS] ... [SEP]
            tokens = tokens[: max_length - 1] + [self.SEP_TOKEN]

        input_ids = [self.token2id.get(t, self.unk_token_id) for t in tokens]
        attention_mask = [1] * len(input_ids)

        if padding and len(input_ids) < max_length:
            pad_len = max_length - len(input_ids)
            input_ids     += [self.pad_token_id] * pad_len
            attention_mask += [0] * pad_len

        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        special_ids = {
            self.pad_token_id,
            self.cls_token_id,
            self.sep_token_id,
            self.mask_token_id,
        }
        tokens = []
        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            tokens.append(self.id2token.get(tid, self.UNK_TOKEN))

        # BertTokenizer handles ## subword joining
        return self._bert.convert_tokens_to_string(tokens).strip()


class How2SignDataset(Dataset):
    """Loads preprocessed keypoint sequences and tokenized translations for a given split."""

    def __init__(self, processed_dir: str, split: str, vocab_path: str, max_output_length: int = 30):
        self.processed_dir = Path(processed_dir) / split
        self.max_output_length = max_output_length
        self.tokenizer = CorpusTokenizer(vocab_path)

        with open(self.processed_dir / "index.json") as f:
            self.index = json.load(f)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        item = self.index[idx]

        sequence = np.load(self.processed_dir / f"{item['clip_id']}.npy")
        sequence = torch.tensor(sequence, dtype=torch.float32)

        encoded = self.tokenizer.encode(
            item["sentence"],
            max_length=self.max_output_length,
            padding=True,
            truncation=True,
        )

        labels        = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        valid_frames  = (sequence.abs().sum(dim=1) > 0)

        return sequence, labels, attention_mask, valid_frames