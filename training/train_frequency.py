"""
Module: train_frequency
Project: AI Image Detector

Entry point to train/research the frequency stream in isolation. Run from the
repo root as:
    python -m training.train_frequency --config configs/base_config.yaml

Trains FrequencyStream + a small standalone classifier head (NOT the fused
DetectorPipeline -- fusion.py/semantic_stream.py aren't touched here), using
the shared augmentation/data pipeline, and saves a checkpoint. This is a real,
runnable MOCK: the loop is real (forward/backward/optimizer + checkpoint
save), but FrequencyStream is still the FFT-magnitude+linear stub, so the
saved weights aren't meaningful yet -- see TODO.md SS3.

Deliberately self-contained (no shared module with train_semantic.py) so two
teammates can iterate on a stream each without touching the same file.
"""

import argparse
import sys
import time
from itertools import cycle
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import yaml

# Make `src` importable when this script is run directly (python training/train_frequency.py)
# rather than as a module (python -m training.train_frequency).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.frequency_stream import FrequencyStream, OUTPUT_DIM  # noqa: E402
from training.data.datamodule import AIGCDataModule  # noqa: E402
from training.logging_utils import setup_logger  # noqa: E402


class FrequencyProbe(nn.Module):
    """FrequencyStream + a linear classification head, trained standalone."""

    def __init__(self, feature_dim: int = OUTPUT_DIM) -> None:
        super().__init__()
        self.stream = FrequencyStream(output_dim=feature_dim)
        self.head = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.stream(x)
        return self.head(features)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI image detector -- frequency stream training (mock loop)")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to base_config.yaml")
    parser.add_argument("--augmentations-config", type=str, default="configs/augmentations.yaml")
    parser.add_argument("--batch_size", type=int, default=None, help="Override config's batch_size")
    parser.add_argument("--lr", type=float, default=None, help="Override config's lr")
    parser.add_argument("--steps", type=int, default=2, help="Number of mock optimizer steps to run")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-level", type=str, default=None, help="Override config's log_level")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.augmentations_config) as f:
        augmentations_config = yaml.safe_load(f)

    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    lr = args.lr if args.lr is not None else config["lr"]
    log_level = args.log_level or config["log_level"]

    logger = setup_logger("training", log_level)
    logger.info("MOCK FREQUENCY STREAM TRAINING -- FrequencyStream is still an FFT-magnitude+linear stub, see TODO.md SS3")

    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device=%s", device)

    datamodule = AIGCDataModule(config, augmentations_config)
    train_loader = datamodule.train_dataloader()

    model = FrequencyProbe().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    logger.info("Model built: %d parameters, lr=%g", sum(p.numel() for p in model.parameters()), lr)

    model.train()
    batch_iter = cycle(train_loader)
    for step in range(1, args.steps + 1):
        start = time.perf_counter()
        images, labels = next(batch_iter)
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logit = model(images)
        loss = loss_fn(logit, labels)
        loss.backward()
        optimizer.step()

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "step=%d/%d batch=%s loss=%.4f elapsed_ms=%.1f",
            step,
            args.steps,
            tuple(images.shape),
            loss.item(),
            elapsed_ms,
        )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "frequency_stream.pt"
    torch.save(model.stream.state_dict(), checkpoint_path)
    logger.info("Saved frequency stream checkpoint to %s (mock weights, not yet meaningful)", checkpoint_path)


if __name__ == "__main__":
    main()
