"""
Module: train_semantic
Project: AI Image Detector

Entry point to train/research the semantic stream in isolation. Run from the
repo root as:
    python -m training.train_semantic --config configs/base_config.yaml

Trains SemanticStream + a small standalone classifier head (NOT the fused
DetectorPipeline -- fusion.py/frequency_stream.py aren't touched here), using
the shared augmentation/data pipeline, and saves a checkpoint. This is a real,
runnable MOCK: the loop is real (forward/backward/optimizer + checkpoint
save), but SemanticStream is still the pool+linear stub, so the saved weights
aren't meaningful yet -- see TODO.md SS3.

Deliberately self-contained (no shared module with train_frequency.py) so two
teammates can iterate on a stream each without touching the same file.
"""

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import yaml

# Make `src` importable when this script is run directly (python training/train_semantic.py)
# rather than as a module (python -m training.train_semantic).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.semantic_stream import OUTPUT_DIM, SemanticStream  # noqa: E402
from training.data.datamodule import AIGCDataModule  # noqa: E402
from training.logging_utils import setup_logger  # noqa: E402


class SemanticProbe(nn.Module):
    """SemanticStream + a linear classification head, trained standalone."""

    def __init__(self, feature_dim: int = OUTPUT_DIM, pretrained: bool = False,
                 freeze_backbone: bool = True, unfreeze_last_n_blocks: int = 0) -> None:
        super().__init__()
        self.stream = SemanticStream(
            output_dim=feature_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        )
        self.head = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.stream(x)
        return self.head(features)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI image detector -- semantic stream training (mock loop)")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to base_config.yaml")
    parser.add_argument("--augmentations-config", type=str, default="configs/augmentations.yaml")
    parser.add_argument("--batch_size", type=int, default=None, help="Override config's batch_size")
    parser.add_argument("--lr", type=float, default=None, help="Override config's lr")
    parser.add_argument("--steps", type=int, default=2, help="Number of mock optimizer steps to run")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
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
    logger.info("SEMANTIC STREAM TRAINING -- ViT-B/16 semantic backbone")

    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device=%s", device)

    datamodule = AIGCDataModule(config, augmentations_config)
    train_loader = datamodule.train_dataloader()

    semantic_config = config.get("semantic", {})
    model = SemanticProbe(
        feature_dim=semantic_config.get("output_dim", OUTPUT_DIM),
        pretrained=semantic_config.get("pretrained", False),
        freeze_backbone=semantic_config.get("freeze_backbone", True),
        unfreeze_last_n_blocks=semantic_config.get("unfreeze_last_n_blocks", 0),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    counts = model.stream.parameter_counts()
    logger.info("Model built: total=%d trainable=%d frozen=%d parameters, lr=%g",
                counts["total"], counts["trainable"], counts["frozen"], lr)

    model.train()
    global_step = 0
    best_accuracy = -1.0
    best_stream_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, labels in train_loader:
            start = time.perf_counter()
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logit = model(images)
            loss = loss_fn(logit, labels)
            loss.backward()
            optimizer.step()
            global_step += 1
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.info("epoch=%d/%d step=%d loss=%.4f elapsed_ms=%.1f",
                        epoch, args.epochs, global_step, loss.item(), elapsed_ms)
            if global_step >= args.steps:
                break
        if global_step >= args.steps:
            break

        model.eval()
        correct = total = 0
        validation_loss = 0.0
        with torch.no_grad():
            for images, labels in datamodule.val_dataloader():
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                validation_loss += loss_fn(logits, labels).item() * labels.size(0)
                correct += ((torch.sigmoid(logits) >= 0.5) == (labels >= 0.5)).sum().item()
                total += labels.numel()
        accuracy = correct / max(total, 1)
        logger.info("epoch=%d validation_loss=%.4f validation_accuracy=%.3f",
                    epoch, validation_loss / max(total, 1), accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_stream_state = copy.deepcopy(model.stream.state_dict())

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "semantic_stream.pt"
    torch.save(best_stream_state or model.stream.state_dict(), checkpoint_path)
    logger.info("Saved semantic stream checkpoint to %s (mock weights, not yet meaningful)", checkpoint_path)


if __name__ == "__main__":
    main()
