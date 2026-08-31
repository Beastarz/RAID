"""
Module: train_fusion
Project: AI Image Detector

Entry point to train the fusion layer + classification head on top of the
two already-trained, frozen feature streams. Run from the repo root as:
    python -m training.train_fusion --config configs/base_config.yaml

This is the "compromise" pipeline wired into src/models/detector.py: one
shared resized + ImageNet-normalized tensor is fed to both streams
(denormalized back to ~[0,1] for the NPR stream inside DetectorPipeline
itself). That's a real limitation, not just a code-style choice -- resizing
destroys the exact artifact NPR reads (npr_stream_guide.md SS2.1), so this
run is meant to verify the fused pipeline trains end-to-end and produce a
first baseline number, not to reproduce train_npr.py's ~0.90 standalone AUC.

Loads `checkpoints/semantic_stream.pt` into SemanticStream and
`checkpoints/npr_stream.pt` into NPRStream if they exist (raises if either is
missing -- training a fusion head on two randomly-initialized streams isn't a
meaningful baseline). Both streams are frozen; only `fusion` and `classifier`
are optimized, matching the "train the fusion head" framing this script is
for -- joint fine-tuning of the streams themselves is separate future work
(TODO.md SS3).
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.detector import DetectorPipeline  # noqa: E402
from src.models.fusion import FeatureFusion  # noqa: E402
from src.models.npr_stream import BackboneName, NPRStream  # noqa: E402
from src.models.semantic_stream import OUTPUT_DIM as SEMANTIC_OUTPUT_DIM  # noqa: E402
from src.models.semantic_stream import SemanticStream  # noqa: E402
from training.data.datamodule import AIGCDataModule  # noqa: E402
from training.logging_utils import setup_logger  # noqa: E402

logger = logging.getLogger("training.train_fusion")


def _freeze(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False
    module.eval()


@torch.no_grad()
def _evaluate(model: DetectorPipeline, loader, device: torch.device, loss_fn: nn.Module) -> Dict[str, float]:
    model.eval()
    total_loss, n_seen = 0.0, 0
    all_probs, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = loss_fn(output["logit"], labels)
        total_loss += loss.item() * images.size(0)
        n_seen += images.size(0)
        all_probs.append(output["prob"].cpu())
        all_labels.append(labels.cpu())

    probs = torch.cat(all_probs).squeeze(1).numpy()
    labels_np = torch.cat(all_labels).squeeze(1).numpy()
    accuracy = float(((probs > 0.5).astype(np.float32) == labels_np).mean()) if n_seen else float("nan")
    try:
        auc = float(roc_auc_score(labels_np, probs)) if len(set(labels_np.tolist())) > 1 else float("nan")
    except ValueError:
        auc = float("nan")
    return {"loss": total_loss / max(n_seen, 1), "accuracy": accuracy, "auc": auc}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI image detector -- fusion head training")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to base_config.yaml")
    parser.add_argument("--augmentations-config", type=str, default="configs/augmentations.yaml")
    parser.add_argument("--batch_size", type=int, default=None, help="Override config's batch_size")
    parser.add_argument("--lr", type=float, default=1e-3, help="LR for the fusion+classifier head only")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--npr-backbone", type=str, default="resnet_shallow", choices=["resnet_shallow", "convnext_tiny"]
    )
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
    log_level = args.log_level or config["log_level"]

    run_logger = setup_logger("training", log_level)
    run_logger.info(
        "FUSION HEAD TRAINING -- %d epoch(s). COMPROMISE wiring (shared resized tensor for both "
        "streams, not NPR's native crop) -- see this file's module docstring.",
        args.epochs,
    )

    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_logger.info("Using device=%s", device)

    checkpoint_dir = Path(args.checkpoint_dir)
    semantic_ckpt = checkpoint_dir / "semantic_stream.pt"
    npr_ckpt = checkpoint_dir / "npr_stream.pt"
    if not semantic_ckpt.exists() or not npr_ckpt.exists():
        raise FileNotFoundError(
            f"Expected both {semantic_ckpt} and {npr_ckpt} -- train each stream first "
            "(python -m training.train_semantic ..., python -m training.train_npr ...)."
        )

    semantic_config = config.get("semantic", {})
    semantic_stream = SemanticStream(
        output_dim=semantic_config.get("output_dim", SEMANTIC_OUTPUT_DIM),
        pretrained=False,  # weights come from the checkpoint below, not torchvision
        freeze_backbone=semantic_config.get("freeze_backbone", True),
        unfreeze_last_n_blocks=semantic_config.get("unfreeze_last_n_blocks", 0),
    )
    semantic_stream.load_state_dict(torch.load(semantic_ckpt, map_location=device))
    npr_stream = NPRStream(backbone=args.npr_backbone)
    npr_stream.load_state_dict(torch.load(npr_ckpt, map_location=device))
    run_logger.info("Loaded checkpoints: %s + %s", semantic_ckpt, npr_ckpt)

    _freeze(semantic_stream)
    _freeze(npr_stream)

    model = DetectorPipeline(
        semantic_stream=semantic_stream,
        npr_stream=npr_stream,
        fusion=FeatureFusion(freq_dim=npr_stream.output_dim),
    ).to(device)

    trainable_params = list(model.fusion.parameters()) + list(model.classifier.parameters())
    n_trainable = sum(p.numel() for p in trainable_params)
    n_frozen = sum(p.numel() for p in model.parameters()) - n_trainable
    run_logger.info("Fusion+classifier trainable params=%d, frozen (both streams)=%d", n_trainable, n_frozen)

    datamodule = AIGCDataModule(config, augmentations_config)
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    loss_fn = nn.BCEWithLogitsLoss()

    fusion_checkpoint_path = checkpoint_dir / "fusion_head.pt"
    best_val_auc = float("-inf")

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        model.fusion.train()
        model.classifier.train()
        running_loss, n_seen = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output["logit"], labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            n_seen += images.size(0)
        scheduler.step()
        train_loss = running_loss / max(n_seen, 1)

        val_metrics = _evaluate(model, val_loader, device, loss_fn)
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
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"fusion": model.fusion.state_dict(), "classifier": model.classifier.state_dict()},
                fusion_checkpoint_path,
            )
            run_logger.info("New best (val_auc=%.4f) -- saved checkpoint to %s", current_auc, fusion_checkpoint_path)

    run_logger.info("Training complete. Best val_auc=%.4f, checkpoint at %s", best_val_auc, fusion_checkpoint_path)


if __name__ == "__main__":
    main()
