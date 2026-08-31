"""
Module: train_bayar_srm
Project: AI Image Detector

Entry point to train/research the Bayar+SRM forensic frontend -- the M3
fallback for NPR (src/models/frontend_bayar.py). Run from the repo root as:
    python -m training.train_bayar_srm --config configs/base_config.yaml

training/evaluate_npr.py's M3 resize stress test showed NPR's fixed
nearest-neighbour operator collapses under a resize round trip (clean AUC
0.9194 -> 0.48-0.62). This script trains the same NPRStream backbone with
that operator swapped for BayarSRMFrontend instead, via NPRStream's existing
`frontend=` injection point -- NPR itself (src/models/npr_stream.py,
checkpoints/npr_stream.pt) is untouched; this is an additional model, not a
replacement.

Deliberately duplicates train_npr.py's probe class and training loop rather
than adding a `--frontend` flag to that script -- one script per
architecture keeps the CLI surface for each simple (no flag whose choices
need to be kept in sync with which hyperparameters make sense for which
frontend) at the cost of ~100 duplicated lines, which is the tradeoff this
was explicitly asked for. The DATA pipeline is NOT duplicated: NPRCropDataset,
_split_dataset, _compute_pos_weight, and _evaluate are imported directly from
training.train_npr, so both scripts load, split, and evaluate data through
the exact same code path -- only the model construction differs.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.frontend_bayar import BayarSRMFrontend  # noqa: E402
from src.models.npr_stream import BackboneName, NPRStream  # noqa: E402
from training.logging_utils import setup_logger  # noqa: E402
from training.train_npr import NPRCropDataset, _compute_pos_weight, _evaluate, _split_dataset  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

# See train_npr.py's identical comment: __name__ is "__main__" when this
# file is run as the entry point via `python -m training.train_bayar_srm`,
# which would silently detach these log calls from the "training" logger
# hierarchy setup_logger() configures below if we used __name__ instead.
logger = logging.getLogger("training.train_bayar_srm")


class BayarSRMProbe(nn.Module):
    """NPRStream (with the Bayar+SRM frontend) + a linear head, trained standalone.

    Identical to train_npr.py's NPRProbe except the frontend is hardcoded to
    BayarSRMFrontend() instead of NPRStream's default NPR() operator.
    """

    def __init__(self, backbone: BackboneName = "resnet_shallow") -> None:
        super().__init__()
        self.stream = NPRStream(backbone=backbone, frontend=BayarSRMFrontend())
        self.head = nn.Linear(self.stream.output_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.stream(x)
        return self.head(features)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI image detector -- Bayar+SRM frontend training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config_bayar.yaml",
        help="Path to a base_config.yaml-shaped config (defaults to the ~10K-sample Bayar-only variant)",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Override config's batch_size")
    parser.add_argument("--lr", type=float, default=None, help="Override config's lr")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Stop if val AUC doesn't improve for this many consecutive epochs (0 disables)",
    )
    parser.add_argument("--crop-size", type=int, default=256, help="Native-resolution crop size fed to NPRStream")
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
    lr = args.lr if args.lr is not None else config["lr"]
    log_level = args.log_level or config["log_level"]

    run_logger = setup_logger("training", log_level)
    run_logger.info(
        "BAYAR+SRM FRONTEND TRAINING -- %d epoch(s), backbone=%s. M3 fallback for NPR (npr_stream_guide.md "
        "SS7); NPR itself is untouched -- see this file's module docstring.",
        args.epochs,
        args.backbone,
    )

    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_logger.info("Using device=%s", device)

    # Identical data path to train_npr.py -- same class, same manifest,
    # same split logic, imported rather than duplicated.
    data_cfg = config["data"]
    full_dataset = NPRCropDataset(
        manifest_path=data_cfg["manifest_path"],
        crop_size=args.crop_size,
        num_synthetic_samples=data_cfg["num_synthetic_samples"],
        seed=config["seed"],
    )
    train_subset, val_subset = _split_dataset(full_dataset, data_cfg["val_split"], config["seed"])
    run_logger.info("Dataset split: %d train / %d val", len(train_subset), len(val_subset))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=config["num_workers"])
    val_loader = (
        DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=config["num_workers"])
        if len(val_subset)
        else None
    )

    model = BayarSRMProbe(backbone=args.backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    pos_weight = _compute_pos_weight(train_subset).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    run_logger.info(
        "Model built: %d parameters, backbone=%s, lr=%g, weight_decay=%g, pos_weight=%.3f",
        sum(p.numel() for p in model.parameters()),
        args.backbone,
        lr,
        args.weight_decay,
        pos_weight.item(),
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Distinct filenames from npr_stream.pt/npr_head.pt so both models can
    # coexist and be compared -- NPR is not deleted or overwritten.
    checkpoint_path = checkpoint_dir / "bayar_srm_stream.pt"
    head_checkpoint_path = checkpoint_dir / "bayar_srm_head.pt"
    best_val_auc = float("-inf")
    epochs_without_improvement = 0

    model.train()
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        running_loss, n_seen = 0.0, 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logit = model(images)
            loss = loss_fn(logit, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            n_seen += images.size(0)
            if batch_idx % 50 == 0:
                logger.debug("epoch=%d batch=%d loss=%.4f", epoch, batch_idx, loss.item())
        scheduler.step()
        train_loss = running_loss / max(n_seen, 1)

        val_metrics = (
            _evaluate(model, val_loader, device, loss_fn)
            if val_loader is not None
            else {"loss": float("nan"), "accuracy": float("nan"), "auc": float("nan")}
        )

        elapsed_s = time.perf_counter() - start
        run_logger.info(
            "epoch=%d/%d train_loss=%.4f val_loss=%.4f val_acc=%.4f val_auc=%.4f lr=%.2e elapsed_s=%.1f",
            epoch,
            args.epochs,
            train_loss,
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["auc"],
            optimizer.param_groups[0]["lr"],
            elapsed_s,
        )

        current_auc = val_metrics["auc"]
        is_best = (not np.isnan(current_auc) and current_auc > best_val_auc) or (
            np.isnan(current_auc) and epoch == args.epochs
        )
        if is_best:
            if not np.isnan(current_auc):
                best_val_auc = current_auc
            epochs_without_improvement = 0
            torch.save(model.stream.state_dict(), checkpoint_path)
            torch.save(model.head.state_dict(), head_checkpoint_path)
            run_logger.info("New best (val_auc=%.4f) -- saved checkpoint to %s", current_auc, checkpoint_path)
        elif not np.isnan(current_auc):
            epochs_without_improvement += 1
            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                run_logger.info(
                    "Early stopping: val_auc hasn't improved for %d epoch(s) (best=%.4f)",
                    epochs_without_improvement,
                    best_val_auc,
                )
                break

    run_logger.info("Training complete. Best val_auc=%.4f, checkpoint at %s", best_val_auc, checkpoint_path)


if __name__ == "__main__":
    main()
