"""Build the self-describing final detector checkpoint bundle.

Example::

    python -m training.build_detector_bundle \
        --semantic-checkpoint checkpoints/semantic_stream.pt \
        --forensic-checkpoint checkpoints/bayar_srm_stream.pt \
        --fusion-checkpoint checkpoints/detector_fusion.pt \
        --output checkpoints/detector_bundle.pt \
        --parity-image test_sample.jpg

The source files are recorded as hashed provenance.  They are not required by
the bundle loader after this command succeeds.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.models.checkpoint_bundle import build_checkpoint_bundle


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a strict final detector checkpoint bundle")
    parser.add_argument("--semantic-checkpoint", default="checkpoints/semantic_stream.pt")
    parser.add_argument("--forensic-checkpoint", default="checkpoints/bayar_srm_stream.pt")
    parser.add_argument("--fusion-checkpoint", default="checkpoints/detector_fusion.pt")
    parser.add_argument("--output", default="checkpoints/detector_bundle.pt")
    parser.add_argument(
        "--parity-image",
        default=None,
        help="Optional image used to compare the bundle with an independently loaded three-file scorer",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    output = build_checkpoint_bundle(
        semantic_checkpoint=args.semantic_checkpoint,
        forensic_checkpoint=args.forensic_checkpoint,
        fusion_checkpoint=args.fusion_checkpoint,
        output_path=args.output,
        parity_image=args.parity_image,
    )
    print(f"Wrote canonical detector bundle to {output}")
    return output


if __name__ == "__main__":
    main()
