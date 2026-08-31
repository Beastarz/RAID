"""
Module: evaluate_npr
Project: AI Image Detector

M3 resize/downscale stress test (npr_stream_guide.md SS7): evaluates a
trained NPR checkpoint on its held-out val split under a real
downscale-then-upscale round trip, applied to the full native image before
cropping (never after -- degrading an already-cropped patch simulates the
wrong thing, per SS2.3). No retraining -- this reuses whatever checkpoint
`python -m training.train_npr` already produced.

Run from the repo root as:
    python -m training.evaluate_npr --config configs/base_config.yaml

Severities come from configs/augmentations.yaml's `eval.downscale_levels`,
the same ones the semantic/frequency streams are eventually measured at, so
NPR's robustness story stays on the same scale as the rest of the project.

Go/no-go rule (the guide's own, SS7): if a resize condition's AUC stays
within ~0.05 of the clean baseline on the same held-out samples, NPR holds
up -- proceed to fusion work as-is. If it collapses toward 0.5, the
prescribed fix is swapping NPRStream's `frontend=` to a Bayar+SRM module
(see src/models/npr_stream.py) -- that injection point exists specifically
for this outcome, so the swap is a config change, not a rewrite.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.logging_utils import setup_logger  # noqa: E402
from training.train_npr import NPRCropDataset, NPRProbe, _evaluate, _split_dataset  # noqa: E402

GO_NO_GO_TOLERANCE = 0.05


def _resize_round_trip(scale: float):
    """Builds a real PIL downscale-then-upscale-back round trip at `scale`.

    A genuine image resampler, not a tensor-space approximation (SS2.3) --
    the whole point of this stress test is to impose a second resampling
    kernel on top of whatever the generator's own upsampling left behind.
    """

    def _degrade(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        small_h = max(1, round(h * scale))
        small_w = max(1, round(w * scale))
        pil_image = Image.fromarray(image)
        downsized = pil_image.resize((small_w, small_h), Image.BILINEAR)
        restored = downsized.resize((w, h), Image.BILINEAR)
        return np.asarray(restored, dtype=np.uint8)

    return _degrade


class DegradedNPRCropDataset(NPRCropDataset):
    """NPRCropDataset with an optional degradation applied to the full
    native image before the random crop -- everything else (manifest
    parsing, labels, crop/pad logic) is inherited unchanged."""

    def __init__(self, *args, degrade_fn=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._degrade_fn = degrade_fn

    def _load_native_image(self, index: int, path):
        image = super()._load_native_image(index, path)
        if self._degrade_fn is not None:
            image = self._degrade_fn(image)
        return image


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI image detector -- NPR M3 resize stress test")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to base_config.yaml")
    parser.add_argument("--augmentations-config", type=str, default="configs/augmentations.yaml")
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
    with open(args.augmentations_config) as f:
        aug_config = yaml.safe_load(f)

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

    model = NPRProbe(backbone=args.backbone).to(device)
    model.stream.load_state_dict(torch.load(stream_path, map_location=device))
    model.head.load_state_dict(torch.load(head_path, map_location=device))
    run_logger.info("Loaded checkpoint: %s + %s", stream_path, head_path)

    data_cfg = config["data"]
    downscale_levels = aug_config["eval"]["downscale_levels"]
    conditions = [("clean", None)] + [(f"resize_{scale}", scale) for scale in downscale_levels]

    loss_fn = torch.nn.BCEWithLogitsLoss()
    results = []
    for name, scale in conditions:
        degrade_fn = _resize_round_trip(scale) if scale is not None else None
        dataset = DegradedNPRCropDataset(
            manifest_path=data_cfg["manifest_path"],
            crop_size=args.crop_size,
            num_synthetic_samples=data_cfg["num_synthetic_samples"],
            seed=config["seed"],
            degrade_fn=degrade_fn,
        )
        # Same seed + same val_split as train_npr.py -> identical held-out
        # indices across every condition, so the comparison is apples-to-apples.
        _, val_subset = _split_dataset(dataset, data_cfg["val_split"], config["seed"])
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=config["num_workers"])

        metrics = _evaluate(model, val_loader, device, loss_fn)
        results.append((name, metrics, len(val_subset)))
        run_logger.info(
            "condition=%-12s auc=%.4f accuracy=%.4f loss=%.4f n=%d",
            name,
            metrics["auc"],
            metrics["accuracy"],
            metrics["loss"],
            len(val_subset),
        )

    clean_auc = results[0][1]["auc"]
    run_logger.info("--- M3 go/no-go (tolerance=%.2f AUC vs. clean=%.4f) ---", GO_NO_GO_TOLERANCE, clean_auc)
    any_fail = False
    for name, metrics, _n in results[1:]:
        auc = metrics["auc"]
        if np.isnan(auc) or np.isnan(clean_auc):
            verdict = "SKIP (AUC undefined -- degenerate val split)"
        elif abs(auc - clean_auc) <= GO_NO_GO_TOLERANCE:
            verdict = "PASS"
        else:
            verdict = "FAIL"
            any_fail = True
        run_logger.info("%-12s auc=%.4f delta=%+.4f -> %s", name, auc, auc - clean_auc, verdict)

    if any_fail:
        run_logger.warning(
            "M3 FAILED for at least one resize severity -- consider swapping NPRStream's frontend= "
            "to a Bayar+SRM module (see src/models/npr_stream.py) before building further on top of NPR."
        )
    else:
        run_logger.info("M3 PASSED -- NPR's signal holds up under the resize stress test. Safe to proceed to fusion.")


if __name__ == "__main__":
    main()
