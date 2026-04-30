import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


def build_label_map(index_path: str) -> dict:
    """
    Builds a mapping from gloss strings to integer class indices,
    sorted alphabetically for consistency across runs.
    """
    with open(index_path) as f:
        index = json.load(f)
    glosses = sorted(set(item["gloss"] for item in index))
    return {gloss: i for i, gloss in enumerate(glosses)}


class WLASLDataset(Dataset):
    """
    Loads pre-extracted keypoint sequences (.npy files) and their
    corresponding integer class labels for a given data split.

    Each sample is a tuple of:
      - sequence: FloatTensor of shape (T, 258) with keypoint features
      - label:    LongTensor scalar with the gloss class index

    Args:
        index_path:    Path to video_index.json generated during extraction.
        keypoints_dir: Directory containing the .npy keypoint files.
        split:         One of "train", "val", or "test".
        label_map:     Dict mapping gloss strings to integer indices,
                       as returned by build_label_map().
    """

    def __init__(self, index_path: str, keypoints_dir: str, split: str, label_map: dict):
        with open(index_path) as f:
            full_index = json.load(f)

        self.keypoints_dir = Path(keypoints_dir)
        self.label_map     = label_map

        self.samples = [
            item for item in full_index
            if item["split"] == split
            and (self.keypoints_dir / f"{item['video_id']}.npy").exists()
        ]

    def __len__(self):
        """Returns the number of samples in this split."""
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Loads and returns the keypoint sequence and label for sample at idx.

        Returns:
            sequence: FloatTensor of shape (T, 258)
            label: LongTensor scalar
        """
        item = self.samples[idx]
        sequence = np.load(self.keypoints_dir / f"{item['video_id']}.npy")
        sequence = torch.tensor(sequence, dtype=torch.float32)
        label = torch.tensor(self.label_map[item["gloss"]], dtype=torch.long)
        return sequence, label