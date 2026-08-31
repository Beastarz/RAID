"""
Module: test_npr
Project: AI Image Detector

Evaluates a trained NPR stream + head checkpoint on its held-out val split.
Run from the repo root as:
    python -m training.test_npr --config configs/base_config.yaml

Rebuilds the exact same NPRCropDataset + seeded train/val split that
train_npr.py used (same manifest, crop size, and seed => the same
random_split partition), then only ever touches the val subset -- this is a
genuine held-out check, not a re-score of data the model trained on. Loads
`checkpoints/npr_stream.pt` (backbone) and `checkpoints/npr_head.pt`
(classifier head) together, since train_npr.py saves them separately -- the
stream-only file matches train_semantic.py's convention for a future
DetectorPipeline fine-tune, but reconstructing predictions needs both.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.logging_utils import setup_logger  # noqa: E402
from training.train_npr import (  # noqa: E402
    NPRCropDataset,
    NPRProbe,
    _evaluate,
    _split_dataset,
)
from torch.utils.data import DataLoader  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI image detector -- NPR stream evaluation")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to base_config.yaml")
    parser.add_argument("--batch_size", type=int, default=None, help="Override config's batch_size")
    parser.add_argument("--crop-size", type=int, default=256, help="Must match the crop size used for training")
    parser.add_argument(
        "--backbone", type=str, default="resnet_shallow", choices=["resnet_shallow", "convnext_tiny"]
    )
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-level", type=str, default=None, help="Override config's log_level")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    batch_size = args.batch_size if args.batch_size is not None else config["batch_size"]
    log_level = args.log_level or config["log_level"]
    run_logger = setup_logger("training", log_level)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_logger.info("Using device=%s", device)

    checkpoint_dir = Path(args.checkpoint_dir)
    stream_path = checkpoint_dir / "npr_stream.pt"
    head_path = checkpoint_dir / "npr_head.pt"
    if not stream_path.exists() or not head_path.exists():
        raise FileNotFoundError(
            f"Expected both {stream_path} and {head_path} -- run training.train_npr first "
            "(it saves both together whenever val AUC improves)."
        )

    data_cfg = config["data"]
    full_dataset = NPRCropDataset(
        manifest_path=data_cfg["manifest_path"],
        crop_size=args.crop_size,
        num_synthetic_samples=data_cfg["num_synthetic_samples"],
        seed=config["seed"],
    )
    # Same seed + same val_split => the identical partition train_npr.py
    # validated against; only the val half is ever used here.
    _, val_subset = _split_dataset(full_dataset, data_cfg["val_split"], config["seed"])
    if len(val_subset) == 0:
        raise ValueError("Val split is empty -- increase data.val_split or num_synthetic_samples in the config.")
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=config["num_workers"])
    run_logger.info("Evaluating on %d held-out val samples", len(val_subset))

    model = NPRProbe(backbone=args.backbone).to(device)
    model.stream.load_state_dict(torch.load(stream_path, map_location=device))
    model.head.load_state_dict(torch.load(head_path, map_location=device))
    run_logger.info("Loaded checkpoint: %s + %s", stream_path, head_path)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    metrics = _evaluate(model, val_loader, device, loss_fn)
    run_logger.info(
        "val_loss=%.4f val_acc=%.4f val_auc=%.4f (n=%d)",
        metrics["loss"],
        metrics["accuracy"],
        metrics["auc"],
        len(val_subset),
    )


if __name__ == "__main__":
    main()
