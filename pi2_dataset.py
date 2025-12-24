"""
π/2 Dataset Loader

HuggingFace-style dataset loader for π/2 tokenized training data.
Supports text, image, and audio modalities from JSONL files.

Usage:
    from datasets import load_dataset

    # Load specific file
    dataset = load_dataset("pi2_dataset.py", data_files="phase2_all_combined.jsonl")

    # Load with train/val split
    dataset = load_dataset("pi2_dataset.py", data_files="phase2_all_combined.jsonl", split_ratio=0.1)

    # Load multiple files
    dataset = load_dataset("pi2_dataset.py", data_files={
        "train": "phase2_combined.jsonl",
        "images": "phase2_images_tokenized.jsonl",
        "audio": "phase2_audio_tokenized.jsonl",
    })
"""

import json
import os
from typing import Dict, List, Optional, Any

import datasets


logger = datasets.logging.get_logger(__name__)

_CITATION = """\
@misc{pi2_training,
  title={π/2 Phase-Encoded Training},
  author={The Union},
  year={2024},
  note={Rotation-consistent multimodal learning}
}
"""

_DESCRIPTION = """\
π/2 tokenized training data for phase-encoded language models.

This dataset contains tokenized samples from multiple modalities:
- Text: Educational content, code, documentation
- Image: Patch-based tokenization of images
- Audio: Spectrogram-based tokenization of audio

Rotation is applied in model latent space during training, not in the tokenization.
The four rotation angles (0, π/2, π, 3π/2) produce exact integer values for cos/sin,
enabling lossless phase quantization.
"""

_HOMEPAGE = "https://github.com/anthropics/claude-code"

_LICENSE = "MIT"

_FEATURES = datasets.Features({
    "tokens": datasets.Sequence(datasets.Value("int32")),
    "modality": datasets.Value("string"),
    "source": datasets.Value("string"),
    "length": datasets.Value("int32"),
})


class Pi2Config(datasets.BuilderConfig):
    """BuilderConfig for π/2 datasets."""

    def __init__(
        self,
        name: str = "default",
        data_files: Optional[Any] = None,
        split_ratio: float = 0.0,
        max_samples: Optional[int] = None,
        modality_filter: Optional[List[str]] = None,
        **kwargs
    ):
        super(Pi2Config, self).__init__(name=name, version=datasets.Version("1.0.0"), **kwargs)
        self.data_files = data_files
        self.split_ratio = split_ratio  # Fraction for validation split
        self.max_samples = max_samples
        self.modality_filter = modality_filter  # Filter by modality: ["text", "image", "audio"]


class Pi2Dataset(datasets.GeneratorBasedBuilder):
    """π/2 tokenized dataset for phase-encoded training."""

    BUILDER_CONFIG_CLASS = Pi2Config
    BUILDER_CONFIGS = [
        Pi2Config(name="default", description="Default π/2 dataset configuration"),
        Pi2Config(name="text_only", modality_filter=["text"], description="Text modality only"),
        Pi2Config(name="image_only", modality_filter=["image"], description="Image modality only"),
        Pi2Config(name="audio_only", modality_filter=["audio"], description="Audio modality only"),
        Pi2Config(name="multimodal", description="All modalities combined"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Generate splits from data files."""
        data_files = self.config.data_files

        if data_files is None:
            # Default to looking for common files in current directory
            default_files = [
                "phase2_all_combined.jsonl",
                "phase2_combined.jsonl",
                "phase1_combined.jsonl",
            ]
            for f in default_files:
                if os.path.exists(f):
                    data_files = f
                    break

            if data_files is None:
                raise ValueError(
                    "No data_files specified and no default files found. "
                    "Please specify data_files parameter."
                )

        # Handle different data_files formats
        if isinstance(data_files, str):
            # Single file - optionally split into train/val
            if self.config.split_ratio > 0:
                return [
                    datasets.SplitGenerator(
                        name=datasets.Split.TRAIN,
                        gen_kwargs={
                            "filepath": data_files,
                            "split": "train",
                            "split_ratio": self.config.split_ratio,
                        },
                    ),
                    datasets.SplitGenerator(
                        name=datasets.Split.VALIDATION,
                        gen_kwargs={
                            "filepath": data_files,
                            "split": "validation",
                            "split_ratio": self.config.split_ratio,
                        },
                    ),
                ]
            else:
                return [
                    datasets.SplitGenerator(
                        name=datasets.Split.TRAIN,
                        gen_kwargs={"filepath": data_files, "split": None, "split_ratio": 0},
                    ),
                ]

        elif isinstance(data_files, dict):
            # Multiple named splits
            splits = []
            for split_name, filepath in data_files.items():
                split_enum = getattr(datasets.Split, split_name.upper(), split_name)
                splits.append(
                    datasets.SplitGenerator(
                        name=split_enum,
                        gen_kwargs={"filepath": filepath, "split": None, "split_ratio": 0},
                    )
                )
            return splits

        elif isinstance(data_files, list):
            # List of files - concatenate all into train
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={"filepath": data_files, "split": None, "split_ratio": 0},
                ),
            ]

        else:
            raise ValueError(f"Unsupported data_files type: {type(data_files)}")

    def _generate_examples(
        self,
        filepath: Any,
        split: Optional[str],
        split_ratio: float,
    ):
        """Generate examples from JSONL files."""
        # Handle single file or list of files
        if isinstance(filepath, str):
            filepaths = [filepath]
        else:
            filepaths = filepath

        idx = 0
        for fpath in filepaths:
            with open(fpath, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Handle train/val splitting
            if split is not None and split_ratio > 0:
                split_idx = int(len(lines) * (1 - split_ratio))
                if split == "train":
                    lines = lines[:split_idx]
                else:  # validation
                    lines = lines[split_idx:]

            for line in lines:
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line in {fpath}")
                    continue

                # Apply modality filter
                if self.config.modality_filter is not None:
                    if record.get("modality") not in self.config.modality_filter:
                        continue

                # Apply max samples limit
                if self.config.max_samples is not None and idx >= self.config.max_samples:
                    return

                yield idx, {
                    "tokens": record.get("tokens", []),
                    "modality": record.get("modality", "unknown"),
                    "source": record.get("source", ""),
                    "length": record.get("length", len(record.get("tokens", []))),
                }
                idx += 1


# Convenience functions for direct loading

def load_pi2_dataset(
    data_files: Any,
    split_ratio: float = 0.0,
    modality_filter: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
) -> datasets.Dataset:
    """
    Load π/2 tokenized dataset directly from JSONL.

    Args:
        data_files: Path to JSONL file(s)
        split_ratio: Fraction for validation split (0 = no split)
        modality_filter: List of modalities to include ["text", "image", "audio"]
        max_samples: Maximum number of samples to load

    Returns:
        HuggingFace Dataset

    Example:
        >>> dataset = load_pi2_dataset("phase2_all_combined.jsonl")
        >>> print(len(dataset))
        63956
    """
    # Direct loading without builder for simplicity
    samples = []

    if isinstance(data_files, str):
        filepaths = [data_files]
    else:
        filepaths = data_files

    for fpath in filepaths:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Apply modality filter
                if modality_filter is not None:
                    if record.get("modality") not in modality_filter:
                        continue

                samples.append({
                    "tokens": record.get("tokens", []),
                    "modality": record.get("modality", "unknown"),
                    "source": record.get("source", ""),
                    "length": record.get("length", len(record.get("tokens", []))),
                })

                # Apply max samples limit
                if max_samples is not None and len(samples) >= max_samples:
                    break

        if max_samples is not None and len(samples) >= max_samples:
            break

    dataset = datasets.Dataset.from_list(samples)

    if split_ratio > 0:
        split_dataset = dataset.train_test_split(test_size=split_ratio)
        return split_dataset
    else:
        return dataset


def get_modality_stats(dataset: datasets.Dataset) -> Dict[str, int]:
    """Get count of samples per modality."""
    from collections import Counter
    return dict(Counter(dataset["modality"]))


def get_length_stats(dataset: datasets.Dataset) -> Dict[str, float]:
    """Get length statistics."""
    lengths = dataset["length"]
    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": sum(lengths) / len(lengths),
        "total_tokens": sum(lengths),
    }


if __name__ == "__main__":
    import sys

    print("π/2 Dataset Loader")
    print("=" * 50)

    # Test with default or specified file
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        # Try to find a default file
        for f in ["phase2_all_combined.jsonl", "phase2_combined.jsonl"]:
            if os.path.exists(f):
                data_file = f
                break
        else:
            print("No data file found. Usage: python pi2_dataset.py <file.jsonl>")
            sys.exit(1)

    print(f"Loading: {data_file}")

    try:
        dataset = load_pi2_dataset(data_file)
        print(f"\nDataset loaded successfully!")
        print(f"  Samples: {len(dataset)}")
        print(f"  Features: {list(dataset.features.keys())}")

        # Modality breakdown
        print(f"\nModality breakdown:")
        for modality, count in get_modality_stats(dataset).items():
            print(f"  {modality}: {count}")

        # Length stats
        print(f"\nLength statistics:")
        for stat, value in get_length_stats(dataset).items():
            if isinstance(value, float):
                print(f"  {stat}: {value:.2f}")
            else:
                print(f"  {stat}: {value}")

        # Sample
        print(f"\nSample record:")
        sample = dataset[0]
        print(f"  modality: {sample['modality']}")
        print(f"  source: {sample['source'][:50]}...")
        print(f"  length: {sample['length']}")
        print(f"  tokens[:10]: {sample['tokens'][:10]}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
