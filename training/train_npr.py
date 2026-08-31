"""
Module: train_npr
Project: AI Image Detector

Entry point to train/research the NPR (Neighboring Pixel Relationships)
forensic stream in isolation. Run from the repo root as:
    python -m training.train_npr --config configs/base_config.yaml

Trains NPRStream + a small standalone classifier head (NOT the fused
DetectorPipeline -- fusion.py/semantic_stream.py/frequency_stream.py aren't
touched here). Unlike the fixed-step wiring smoke test this started as, this
is a real short training loop (5 epochs by default, per npr_stream_guide.md
SS5): a train/val split, AdamW + cosine LR schedule, a pos_weight-balanced
BCE loss, and per-epoch val loss/accuracy/AUC -- the checkpoint is only
overwritten when val AUC improves, so `checkpoints/npr_stream.pt` always
holds the best epoch seen, not just the last one.

Deliberately self-contained (no shared module with train_semantic.py /
train_frequency.py) so a teammate can iterate on this stream without
touching a shared file -- and, unlike those two, NPR also cannot share their
dataloader transform even in principle: per npr_stream_guide.md SS2.1, NPR's
residual signal is destroyed by resizing, so its input must be raw [0, 1]
pixels at a native-resolution crop, never the resized + ImageNet-normalized
tensor training/data/augmentations.py produces for the other streams. This
file therefore owns a minimal crop-only dataset instead of importing
training/data/dataset.py.
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset, random_split

# Make `src` importable when this script is run directly (python training/train_npr.py)
# rather than as a module (python -m training.train_npr).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.npr_stream import BackboneName, NPRStream  # noqa: E402
from training.logging_utils import setup_logger  # noqa: E402


# A fixed name, not __name__: when this file is run as the entry point via
# `python -m training.train_npr`, Python sets its __name__ to "__main__",
# which would silently detach these log calls from the "training" logger
# hierarchy that setup_logger() configures below.
logger = logging.getLogger("training.train_npr")


class NPRCropDataset(Dataset):
    """Synthetic or manifest-CSV dataset of native-resolution NPR crops.

    Reuses the same manifest CSV format as training/data/dataset.py
    (`image_path,label`, 0=Real/1=AI-Generated) so the same manifest can
    back both pipelines, but the loading itself is different on purpose:
    images are randomly cropped to a fixed `crop_size` (reflect-padded first
    if smaller), never resized, and cast to [0, 1] float without ImageNet
    normalization -- exactly the input NPRStream expects. Falls back to
    synthetic in-memory samples (random "native resolution" noise images,
    then cropped) when no manifest is given, mirroring AIGCDataset's
    synthetic fallback. Note that on the synthetic fallback, image content
    and label are independent by construction, so real learning is only
    possible once a real manifest is supplied -- see training/data/import_hf.py.
    """

    def __init__(
        self,
        manifest_path: Optional[str],
        crop_size: int = 256,
        num_synthetic_samples: int = 32,
        seed: int = 42,
        resize_augmentation: bool = False,
    ) -> None:
        self.crop_size = crop_size
        self._rng = np.random.default_rng(seed)
        self.resize_augmentation = resize_augmentation
        self.train_indices = set()
        self._samples: List[Tuple[Optional[Path], int]] = []

        if manifest_path and Path(manifest_path).exists():
            with Path(manifest_path).open(newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self._samples.append((Path(row["image_path"]), int(row["label"])))
            logger.info("NPRCropDataset: loaded %d samples from manifest %s", len(self._samples), manifest_path)
        else:
            self._samples = [(None, int(self._rng.integers(0, 2))) for _ in range(num_synthetic_samples)]
            logger.info(
                "NPRCropDataset: SYNTHETIC MODE (no manifest given) -- generated %d in-memory samples",
                len(self._samples),
            )

    def __len__(self) -> int:
        return len(self._samples)

    def label_at(self, index: int) -> int:
        """Label lookup without decoding the image -- used for pos_weight."""
        return self._samples[index][1]

    def _load_native_image(self, index: int, path: Optional[Path]) -> np.ndarray:
        if path is not None:
            return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        # Synthetic sample: deterministic-per-index random "native resolution"
        # (>= crop_size, so the random crop below is realistically exercised)
        # noise image -- never pre-cropped to crop_size directly, so the crop
        # step is not a no-op.
        rng = np.random.default_rng(index)
        h = int(rng.integers(self.crop_size, self.crop_size * 2 + 1))
        w = int(rng.integers(self.crop_size, self.crop_size * 2 + 1))
        return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)

    def _random_crop(self, image: np.ndarray) -> np.ndarray:
        h, w, _ = image.shape
        pad_h, pad_w = max(0, self.crop_size - h), max(0, self.crop_size - w)
        if pad_h or pad_w:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            h, w, _ = image.shape
        top = int(self._rng.integers(0, h - self.crop_size + 1))
        left = int(self._rng.integers(0, w - self.crop_size + 1))
        return image[top : top + self.crop_size, left : left + self.crop_size, :]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self._samples[index]
        image = self._load_native_image(index, path)
        if self.resize_augmentation and index in self.train_indices:
            scale = float(self._rng.uniform(0.25, 0.75))
            h, w = image.shape[:2]
            small = Image.fromarray(image).resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
            image = np.asarray(small.resize((w, h), Image.BILINEAR), dtype=np.uint8)
        crop = self._random_crop(image)
        tensor = torch.from_numpy(crop.astype(np.float32) / 255.0).permute(2, 0, 1)  # [3, crop, crop] in [0, 1]
        label_tensor = torch.tensor([float(label)], dtype=torch.float32)
        logger.debug(
            "NPRCropDataset[%d]: source=%s label=%.1f shape=%s", index, path or "synthetic", label, tuple(tensor.shape)
        )
        return tensor, label_tensor


class NPRProbe(nn.Module):
    """NPRStream + a linear classification head, trained standalone."""

    def __init__(self, backbone: BackboneName = "resnet_shallow") -> None:
        super().__init__()
        self.stream = NPRStream(backbone=backbone)
        self.head = nn.Linear(self.stream.output_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.stream(x)
        return self.head(features)


def _split_dataset(dataset: NPRCropDataset, val_split: float, seed: int) -> Tuple[Subset, Subset]:
    """Randomly (not contiguously) splits into train/val subsets.

    Unlike training/data/datamodule.py's AIGCDataModule -- which slices a
    single manifest by a *contiguous* index range -- this uses a random
    split so an unshuffled or class-blocked manifest still yields a
    representative val set.
    """
    n_val = max(1, int(len(dataset) * val_split)) if len(dataset) > 1 else 0
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val], generator=generator)


def _compute_pos_weight(train_subset: Subset) -> torch.Tensor:
    """BCEWithLogitsLoss pos_weight = n_negative / n_positive, per
    npr_stream_guide.md SS5 -- SID_Set's own validation split is
    4998 real / 8843 fake, and any real manifest is unlikely to be
    perfectly balanced either."""
    dataset: NPRCropDataset = train_subset.dataset  # type: ignore[assignment]
    labels = [dataset.label_at(i) for i in train_subset.indices]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return torch.tensor(1.0)
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn: nn.Module) -> Dict[str, float]:
    model.eval()
    total_loss, n_seen = 0.0, 0
    all_probs, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logit = model(images)
        loss = loss_fn(logit, labels)
        total_loss += loss.item() * images.size(0)
        n_seen += images.size(0)
        all_probs.append(torch.sigmoid(logit).cpu())
        all_labels.append(labels.cpu())
    model.train()

    probs = torch.cat(all_probs).squeeze(1).numpy()
    labels_np = torch.cat(all_labels).squeeze(1).numpy()
    accuracy = float(((probs > 0.5).astype(np.float32) == labels_np).mean()) if n_seen else float("nan")
    try:
        auc = float(roc_auc_score(labels_np, probs)) if len(set(labels_np.tolist())) > 1 else float("nan")
    except ValueError:
        auc = float("nan")
    return {"loss": total_loss / max(n_seen, 1), "accuracy": accuracy, "auc": auc}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI image detector -- NPR stream training")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to base_config.yaml")
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
        "NPR STREAM TRAINING -- %d epoch(s), backbone=%s. Standalone (fusion.py untouched); see "
        "npr_stream_guide.md for the milestones this short run is meant to sanity-check.",
        args.epochs,
        args.backbone,
    )

    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_logger.info("Using device=%s", device)

    data_cfg = config["data"]
    full_dataset = NPRCropDataset(
        manifest_path=data_cfg["manifest_path"],
        crop_size=args.crop_size,
        num_synthetic_samples=data_cfg["num_synthetic_samples"],
        seed=config["seed"],
    )
    train_subset, val_subset = _split_dataset(full_dataset, data_cfg["val_split"], config["seed"])
    full_dataset.train_indices = set(train_subset.indices)
    run_logger.info("NPR dataset split: %d train / %d val", len(train_subset), len(val_subset))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=config["num_workers"])
    val_loader = (
        DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=config["num_workers"])
        if len(val_subset)
        else None
    )

    model = NPRProbe(backbone=args.backbone).to(device)
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
    checkpoint_path = checkpoint_dir / "npr_stream.pt"
    head_checkpoint_path = checkpoint_dir / "npr_head.pt"
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
        # A NaN AUC (degenerate/single-class val split, common on tiny
        # synthetic runs) can't be compared -- fall back to always saving the
        # final epoch so a checkpoint is guaranteed to exist.
        is_best = (not np.isnan(current_auc) and current_auc > best_val_auc) or (
            np.isnan(current_auc) and epoch == args.epochs
        )
        if is_best:
            if not np.isnan(current_auc):
                best_val_auc = current_auc
            epochs_without_improvement = 0
            torch.save(model.stream.state_dict(), checkpoint_path)
            # The stream-only checkpoint above matches train_semantic.py /
            # train_frequency.py's convention (a future DetectorPipeline
            # fine-tune only wants the backbone). The head is saved
            # separately so this run's classifier can still be evaluated
            # standalone -- see test_npr.py -- without overloading that
            # shared checkpoint format.
            torch.save(model.head.state_dict(), head_checkpoint_path)
            run_logger.info("New best (val_auc=%.4f) -- saved checkpoint to %s", current_auc, checkpoint_path)
        elif not np.isnan(current_auc):
            # Only a real (non-NaN) AUC that failed to improve counts against
            # patience -- a degenerate val split shouldn't trigger an early
            # stop on its own.
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
