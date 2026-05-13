import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from transformers import BertTokenizer


class How2SignDataset(Dataset):
    """Loads preprocessed keypoint sequences and tokenized translations for a given split."""

    def __init__(self, processed_dir: str, split: str, max_output_length: int = 30):
        self.processed_dir = Path(processed_dir) / split
        self.max_output_length = max_output_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        with open(self.processed_dir / 'index.json') as f:
            self.index = json.load(f)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        item = self.index[idx]

        sequence = np.load(self.processed_dir / f"{item['clip_id']}.npy")
        sequence = torch.tensor(sequence, dtype=torch.float32)

        tokenized = self.tokenizer(
            item['sentence'],
            max_length=self.max_output_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        labels = tokenized['input_ids'].squeeze(0)
        attention_mask = tokenized['attention_mask'].squeeze(0)
        valid_frames = (sequence.abs().sum(dim=1) > 0)

        return sequence, labels, attention_mask, valid_frames