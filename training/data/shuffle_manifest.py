"""
Module: shuffle_manifest
Project: AI Image Detector

Shuffles a manifest CSV's rows in place (deterministically, given a seed).

training/data/import_hf.py writes rows in Hugging Face streaming order, which
is not guaranteed to interleave classes -- if the source dataset's shards are
block-ordered by label, a `--limit`-truncated export can end up
class-skewed. That matters here specifically because
training/data/datamodule.py's AIGCDataModule splits its single manifest into
train/val by a *contiguous* index range (not a shuffled one), so an
unshuffled, class-skewed manifest can produce an almost single-class
validation split. Run this once on any manifest before pointing
configs/base_config.yaml's data.manifest_path at it.

Usage:
    python -m training.data.shuffle_manifest --input data/sid_subset/manifest.csv \
        --output data/sid_subset/manifest_shuffled.csv
"""

import argparse
import csv
import random
from pathlib import Path
from typing import List, Optional


def shuffle_manifest(input_path: Path, output_path: Path, seed: int = 42) -> int:
    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows: List[List[str]] = list(reader)

    random.Random(seed).shuffle(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return len(rows)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shuffle a manifest CSV's rows deterministically")
    parser.add_argument("--input", type=str, required=True, help="Path to the unshuffled manifest.csv")
    parser.add_argument("--output", type=str, required=True, help="Path to write the shuffled manifest to")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    count = shuffle_manifest(Path(args.input), Path(args.output), seed=args.seed)
    print(f"Shuffled {count} rows -> {args.output}")


if __name__ == "__main__":
    main()
