"""
Module: evaluate_bayar_srm
Project: AI Image Detector

M3 resize/downscale stress test for the Bayar+SRM checkpoint -- identical
test to training/evaluate_npr.py, just pointed at
checkpoints/bayar_srm_stream.pt + bayar_srm_head.pt (training/train_bayar_srm.py's
output) instead of NPR's. Run from the repo root as:
    python -m training.evaluate_bayar_srm --config configs/base_config.yaml

Duplicated rather than adding a `--model {npr,bayar_srm}` flag to
evaluate_npr.py, for the same reason train_bayar_srm.py duplicates
train_npr.py's probe/loop instead of taking a `--frontend` flag: one script
per architecture keeps each one's CLI surface simple. The degradation logic
itself (DegradedNPRCropDataset, the resize round trip, the go/no-go
tolerance) is imported from training.evaluate_npr, not copied, so both
scripts stress-test under the exact same degradation code path -- only the
model/checkpoint differs.

CAVEAT: unlike the checkpoint evaluate_npr.py tests (which had 3+ epochs to
converge before evaluation), a 2-epoch sanity-check checkpoint was still
improving each epoch when this was first run against it. The go/no-go delta
here is still valid (it's self-relative -- resize AUC vs. this same
checkpoint's own clean AUC), but a PASS is only an early signal, not proof of
convergence-invariant robustness; a FAIL is ambiguous between a genuine
architectural problem and simply needing more training.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.evaluate_npr import GO_NO_GO_TOLERANCE, DegradedNPRCropDataset, _resize_round_trip  # noqa: E402
from training.logging_utils import setup_logger  # noqa: E402
from training.train_bayar_srm import BayarSRMProbe  # noqa: E402
from training.train_npr import _evaluate, _split_dataset  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI image detector -- Bayar+SRM M3 resize stress test")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config_bayar.yaml",
        help="Path to a base_config.yaml-shaped config (defaults to the same ~10K-sample config train_bayar_srm.py trains against)",
    )
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
    stream_path = checkpoint_dir / "bayar_srm_stream.pt"
    head_path = checkpoint_dir / "bayar_srm_head.pt"
    if not stream_path.exists() or not head_path.exists():
        raise FileNotFoundError(
            f"Expected both {stream_path} and {head_path} -- run training.train_bayar_srm first."
        )

    model = BayarSRMProbe(backbone=args.backbone).to(device)
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
        run_logger.warning("M3 FAILED for at least one resize severity for the Bayar+SRM frontend too.")
    else:
        run_logger.info("M3 PASSED -- Bayar+SRM's signal holds up under the resize stress test.")


if __name__ == "__main__":
    main()
