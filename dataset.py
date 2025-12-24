"""
π/2 Dataset

Loads tokenized data for training with latent space rotation.
Each sample is presented 4 times with different rotation angles (0, π/2, π, 3π/2).
Rotation is applied during model forward pass, not here.
"""
import json
import random
from pathlib import Path
from typing import List, Union, Iterator

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader

from config import ROTATION_ANGLES


class Pi2Dataset(Dataset):
    """
    PyTorch Dataset for π/2 training.

    Loads pre-tokenized JSONL files. Each sample is duplicated 4 times
    with different rotation phases (applied in model forward pass).

    One tokenized file → 4 training samples (one per rotation angle)
    """

    def __init__(self, data_path: Union[str, Path], max_seq_len: int = 512):
        self.data_path = Path(data_path)
        self.max_seq_len = max_seq_len
        self.base_samples = []  # Original tokenized samples
        self.samples = []  # Expanded with rotation phases

        self._load_data()

    def _load_data(self):
        """Load samples and expand with 4 rotation phases each."""
        # Load base samples
        if self.data_path.suffix == '.jsonl':
            with open(self.data_path, 'r') as f:
                for line in f:
                    record = json.loads(line)
                    self.base_samples.append(record)
        else:
            raise ValueError(f"Expected .jsonl file, got: {self.data_path}")

        # Expand each sample into 4 rotation phases
        for sample in self.base_samples:
            for phase, angle in enumerate(ROTATION_ANGLES):
                self.samples.append({
                    "tokens": sample["tokens"],
                    "modality": sample["modality"],
                    "rotation_phase": phase,
                    "rotation_angle": angle,
                    "length": sample["length"]
                })

        print(f"Loaded {len(self.base_samples)} base samples from {self.data_path}")
        print(f"  Expanded to {len(self.samples)} samples (4 rotations each)")
        print(f"  Rotation angles: {ROTATION_ANGLES}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        tokens = sample["tokens"]

        # Ensure proper length
        if len(tokens) < self.max_seq_len:
            tokens = tokens + [0] * (self.max_seq_len - len(tokens))
        elif len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "labels": torch.tensor(tokens, dtype=torch.long),
            "rotation_angle": sample["rotation_angle"],
            "rotation_phase": sample["rotation_phase"]
        }


class Pi2StreamingDataset(IterableDataset):
    """
    Streaming dataset for large files.

    Doesn't load everything into memory. Streams samples with rotation expansion.
    """

    def __init__(self, data_path: Union[str, Path], max_seq_len: int = 512):
        self.data_path = Path(data_path)
        self.max_seq_len = max_seq_len

    def __iter__(self) -> Iterator[dict]:
        with open(self.data_path, 'r') as f:
            for line in f:
                sample = json.loads(line)
                tokens = sample["tokens"]

                if len(tokens) < self.max_seq_len:
                    tokens = tokens + [0] * (self.max_seq_len - len(tokens))
                elif len(tokens) > self.max_seq_len:
                    tokens = tokens[:self.max_seq_len]

                # Yield 4 rotation phases for each sample
                for phase, angle in enumerate(ROTATION_ANGLES):
                    yield {
                        "input_ids": torch.tensor(tokens, dtype=torch.long),
                        "labels": torch.tensor(tokens, dtype=torch.long),
                        "rotation_angle": angle,
                        "rotation_phase": phase
                    }


class Pi2RotationBatchSampler:
    """
    Custom batch sampler that ensures each batch has balanced rotation phases.

    Each batch contains samples from all 4 rotation phases.
    """

    def __init__(self, dataset: Pi2Dataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Group sample indices by base sample
        self.num_base_samples = len(dataset.base_samples)

    def __iter__(self):
        # Get indices grouped by base sample (4 per group)
        base_indices = list(range(self.num_base_samples))

        if self.shuffle:
            random.shuffle(base_indices)

        # Yield batches
        batch = []
        for base_idx in base_indices:
            # Add all 4 rotations of this sample
            for phase in range(4):
                sample_idx = base_idx * 4 + phase
                batch.append(sample_idx)

                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []

        if batch:
            yield batch

    def __len__(self):
        return (self.num_base_samples * 4 + self.batch_size - 1) // self.batch_size


def create_dataloader(
    data_path: str,
    batch_size: int = 8,
    max_seq_len: int = 512,
    shuffle: bool = True,
    num_workers: int = 4,
    streaming: bool = False
) -> DataLoader:
    """
    Create a DataLoader for π/2 training.

    Each base sample produces 4 training samples (one per rotation angle).
    Rotation is applied during model forward pass.
    """

    if streaming:
        dataset = Pi2StreamingDataset(data_path, max_seq_len)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        dataset = Pi2Dataset(data_path, max_seq_len)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )


def collate_fn(batch: List[dict]) -> dict:
    """Custom collate that groups by rotation angle."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "rotation_angles": torch.tensor([b["rotation_angle"] for b in batch]),
        "rotation_phases": torch.tensor([b["rotation_phase"] for b in batch])
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        dataset = Pi2Dataset(data_path)
        print(f"\nDataset size: {len(dataset)}")
        print(f"Base samples: {len(dataset.base_samples)}")
        print(f"Expanded samples: {len(dataset.samples)} (4x rotation)")

        if len(dataset) > 0:
            # Show samples from first base sample (all 4 rotations)
            print("\nFirst base sample (4 rotations):")
            for i in range(4):
                sample = dataset[i]
                print(f"  Phase {i} (angle={sample['rotation_angle']:.6f}):")
                print(f"    Input shape: {sample['input_ids'].shape}")
                print(f"    First 5 tokens: {sample['input_ids'][:5].tolist()}")
    else:
        print("Usage: python dataset.py <data_path.jsonl>")
