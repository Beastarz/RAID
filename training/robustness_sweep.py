"""Score a labeled manifest through the published detector under every
condition in the organizers' degradation table, writing one JSONL file of
PredictionRecords that tables 1, 3, and 4 (and part of 2) all derive from.

Run once:
    python -m training.robustness_sweep \
        --manifest data/sid_eval/manifest.csv \
        --checkpoint checkpoints/detector_bundle.pt \
        --output outputs/robustness/predictions.jsonl

Every degraded condition is applied to the *full native image before* the
canonical 512x512 resize (never after -- degrading a resized image
approximates the wrong physical process; a real photo is degraded in the
world, then a fixed pipeline resizes whatever comes out of that). Each
image is then scored through the exact same
``src.models.fused_detector.prepare_fused_inputs`` path ``predict.py`` uses,
so these numbers reflect the actual deployed pipeline, not an approximation
of it.

Severities match the organizers' table as given (not
``configs/augmentations.yaml``'s eval block, which uses slightly different
numbers for a different purpose): JPEG quality 90/70/50/30, Gaussian blur
sigma 0.5/1.0/2.0, downscale-then-upscale-back 0.5x/0.25x, Gaussian noise
std 0.02/0.05/0.10, a fixed +20% brightness/contrast/saturation jitter, an
80%-of-frame center crop, plus one "chained" condition (JPEG q70 -> 0.5x
resize round trip -> JPEG q50) approximating a real re-upload/redistribution
path.
"""

import argparse
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import albumentations as A
import numpy as np
import torch
from PIL import Image

from src.explainability.contracts import PredictionRecord
from src.explainability.serialization import write_prediction_jsonl
from src.models.checkpoint_bundle import load_checkpoint_bundle
from src.models.fused_detector import PreparedFusedInputs, prepare_fused_inputs

DegradeFn = Callable[[Image.Image], Image.Image]


def _to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _to_pil(array: np.ndarray) -> Image.Image:
    return Image.fromarray(array)


def _albumentations_transform(transform: A.BasicTransform) -> DegradeFn:
    pipeline = A.Compose([transform])

    def _apply(image: Image.Image) -> Image.Image:
        return _to_pil(pipeline(image=_to_array(image))["image"])

    return _apply


def jpeg(quality: int) -> DegradeFn:
    return _albumentations_transform(A.ImageCompression(quality_range=(quality, quality), p=1.0))


def blur(sigma: float) -> DegradeFn:
    return _albumentations_transform(A.GaussianBlur(sigma_limit=(sigma, sigma), p=1.0))


def resize_round_trip(scale: float) -> DegradeFn:
    """Downscale by ``scale`` then upscale back to the original size."""
    return _albumentations_transform(A.Downscale(scale_range=(scale, scale), p=1.0))


def gauss_noise(std: float) -> DegradeFn:
    return _albumentations_transform(A.GaussNoise(std_range=(std, std), p=1.0))


def color_jitter(factor: float = 0.2) -> DegradeFn:
    bump = 1.0 + factor
    return _albumentations_transform(
        A.ColorJitter(brightness=(bump, bump), contrast=(bump, bump), saturation=(bump, bump), hue=(0.0, 0.0), p=1.0)
    )


def center_crop(percent: float) -> DegradeFn:
    def _apply(image: Image.Image) -> Image.Image:
        width, height = image.size
        crop_width = max(1, round(width * percent))
        crop_height = max(1, round(height * percent))
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        return image.crop((left, top, left + crop_width, top + crop_height))

    return _apply


def chained_redistribution() -> DegradeFn:
    first_pass, resize, second_pass = jpeg(70), resize_round_trip(0.5), jpeg(50)

    def _apply(image: Image.Image) -> Image.Image:
        return second_pass(resize(first_pass(image)))

    return _apply


# (condition, severity, degrade_fn) -- severity is None only for "clean";
# every degraded condition needs a real, finite severity value for
# training.evaluation.robustness_suite.aggregate_robustness to group by.
CONDITIONS: List[Tuple[str, Optional[float], Optional[DegradeFn]]] = [
    ("clean", None, None),
    ("jpeg", 90.0, jpeg(90)),
    ("jpeg", 70.0, jpeg(70)),
    ("jpeg", 50.0, jpeg(50)),
    ("jpeg", 30.0, jpeg(30)),
    ("blur", 0.5, blur(0.5)),
    ("blur", 1.0, blur(1.0)),
    ("blur", 2.0, blur(2.0)),
    ("resize", 0.5, resize_round_trip(0.5)),
    ("resize", 0.25, resize_round_trip(0.25)),
    ("noise", 0.02, gauss_noise(0.02)),
    ("noise", 0.05, gauss_noise(0.05)),
    ("noise", 0.10, gauss_noise(0.10)),
    ("jitter", 0.20, color_jitter(0.20)),
    ("crop", 0.80, center_crop(0.80)),
    ("chained", 1.0, chained_redistribution()),  # nominal severity marker, not a magnitude
]


def _read_manifest(path: Path) -> List[dict]:
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _prepare_batch(image_paths: Sequence[Path], degrade_fn: Optional[DegradeFn]) -> List[PreparedFusedInputs]:
    prepared = []
    for path in image_paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
        if degrade_fn is not None:
            image = degrade_fn(image)
        prepared.append(prepare_fused_inputs(image))
    return prepared


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robustness sweep: degrade -> canonical detector -> tagged predictions")
    parser.add_argument("--manifest", required=True, help="CSV with image_path,label[,sid_label,native_width,native_height]")
    parser.add_argument("--checkpoint", default="checkpoints/detector_bundle.pt")
    parser.add_argument("--output", default="outputs/robustness/predictions.jsonl")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of manifest rows used, for a fast dry run")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, manifest = load_checkpoint_bundle(args.checkpoint, map_location=device)
    model.eval().to(device)
    model_id = str(manifest.get("model_id", "raid-detector-fusion"))
    threshold = float(manifest.get("decision", {}).get("threshold", 0.5))

    rows = _read_manifest(Path(args.manifest))
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"Scoring {len(rows)} images x {len(CONDITIONS)} conditions = {len(rows) * len(CONDITIONS)} forward passes")

    records: List[PredictionRecord] = []
    for condition_name, severity, degrade_fn in CONDITIONS:
        start = time.perf_counter()
        for batch_start in range(0, len(rows), args.batch_size):
            batch_rows = rows[batch_start : batch_start + args.batch_size]
            paths = [Path(row["image_path"]) for row in batch_rows]
            prepared = _prepare_batch(paths, degrade_fn)
            semantic = torch.cat([p.semantic for p in prepared], dim=0).to(device)
            forensic = torch.cat([p.forensic for p in prepared], dim=0).to(device)
            with torch.no_grad():
                logits = model(semantic, forensic)
                probabilities = torch.sigmoid(logits).squeeze(1).cpu().tolist()
                logit_values = logits.squeeze(1).cpu().tolist()

            for row, logit_value, probability in zip(batch_rows, logit_values, probabilities):
                predicted_label = int(probability >= threshold)
                metadata = {"condition": condition_name}
                if severity is not None:
                    metadata["severity"] = severity
                if "sid_label" in row and row["sid_label"] != "":
                    metadata["sid_label"] = int(row["sid_label"])
                if "native_width" in row and row["native_width"] != "":
                    metadata["native_width"] = int(row["native_width"])
                if "native_height" in row and row["native_height"] != "":
                    metadata["native_height"] = int(row["native_height"])
                records.append(
                    PredictionRecord(
                        sample_id=f"{Path(row['image_path']).stem}__{condition_name}_{severity}",
                        model_id=model_id,
                        predicted_logit=float(logit_value),
                        predicted_probability=float(probability),
                        predicted_label=predicted_label,
                        decision_threshold=threshold,
                        source_reference=row["image_path"],
                        ground_truth_label=int(row["label"]),
                        metadata=metadata,
                    )
                )
        elapsed_s = time.perf_counter() - start
        print(f"  condition={condition_name:<10} severity={severity}  {elapsed_s:.1f}s")

    output_path = write_prediction_jsonl(records, args.output)
    print(f"Wrote {len(records)} predictions to {output_path}")


if __name__ == "__main__":
    main()
