#!/usr/bin/env python3
"""
Data Tokenization Script

Tokenizes files (text, audio, images) to token sequences.
Rotation happens in model latent space, NOT during tokenization.

Usage:
    # Single file
    python tokenize_data.py --input data.txt --output tokenized.jsonl

    # Directory (recursive)
    python tokenize_data.py --input data/ --output tokenized.jsonl

    # Custom settings
    python tokenize_data.py --input data/ --output tokenized.jsonl --max-len 1024
"""
import argparse
import json
from pathlib import Path

from tokenizer import Pi2Tokenizer
from config import ROTATION_ANGLES


def main():
    parser = argparse.ArgumentParser(description="π/2 Data Tokenizer")
    parser.add_argument("--input", required=True, help="Input file or directory")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--max-len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--vocab-size", type=int, default=65536, help="Vocabulary size")
    parser.add_argument("--no-recursive", action="store_true", help="Don't recurse into subdirectories")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print("π/2 Data Tokenizer")
    print("=" * 50)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Max seq len: {args.max_len}")
    print(f"Vocab size: {args.vocab_size}")
    print(f"Rotation: Applied in model latent space (not here)")
    print(f"  Angles: {ROTATION_ANGLES}")
    print("=" * 50)

    tokenizer = Pi2Tokenizer(
        vocab_size=args.vocab_size,
        max_seq_len=args.max_len
    )

    total_samples = 0

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as out_file:
        if input_path.is_file():
            # Single file
            print(f"\nTokenizing: {input_path}")
            sample = tokenizer.tokenize_file(input_path)
            record = {
                "tokens": sample.tokens,
                "modality": sample.modality,
                "source": sample.source,
                "length": sample.length
            }
            out_file.write(json.dumps(record) + '\n')
            total_samples += 1
            print(f"  → 1 sample, {sample.length} tokens")

        elif input_path.is_dir():
            # Directory
            recursive = not args.no_recursive
            print(f"\nScanning directory (recursive={recursive})...")

            for sample in tokenizer.tokenize_directory(input_path, recursive=recursive):
                record = {
                    "tokens": sample.tokens,
                    "modality": sample.modality,
                    "source": sample.source,
                    "length": sample.length
                }
                out_file.write(json.dumps(record) + '\n')
                total_samples += 1

                if total_samples % 100 == 0:
                    print(f"  Processed {total_samples} files...")

        else:
            print(f"Error: {input_path} does not exist")
            return

    print("\n" + "=" * 50)
    print("Tokenization Complete!")
    print(f"  Files processed: {total_samples}")
    print(f"  Training samples: {total_samples * 4} (4 rotations each)")
    print(f"  Output: {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
