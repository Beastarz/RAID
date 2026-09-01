"""Stream a held-out SID_Set split into a labeled evaluation manifest.

Deliberately separate from training/data/import_hf.py rather than extending
it: that script builds *training* manifests (train split, binary label
only, no provenance beyond the image itself). This one builds *evaluation*
manifests and is not used by any training script, so it's free to capture
what evaluation actually needs without touching the training data contract
other scripts rely on:

- Pulls from SID_Set's ``validation`` split by default, not ``train`` --
  the published checkpoints were fit against a train-split pull, so scoring
  them against train-split images again would overstate performance.
- Preserves SID_Set's original 3-way label (0=real, 1=fully synthetic,
  2=tampered) alongside the binary label the project's contract uses, so
  error-breakdown tooling can ask "do folded-in tampered images behave
  differently from fully-synthetic ones" -- a question the binary-only
  manifest training scripts produce cannot answer.
- Records each image's native (pre-download-resize) width/height, since
  SID_Set ships images from 338px to 6020px wide and any resolution-based
  shortcut needs that logged at source, not recovered later.

Output manifest columns: image_path,label,sid_label,native_width,native_height
"""

import argparse
from itertools import chain
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from PIL import Image


def _column(sample: dict, requested: Optional[str], names: tuple[str, ...], kind: str) -> str:
    if requested:
        if requested not in sample:
            raise KeyError(f"Unknown {kind} column {requested!r}; available: {list(sample)}")
        return requested
    for name in names:
        if name in sample:
            return name
    raise KeyError(f"Could not infer {kind} column; available: {list(sample)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a labeled SID_Set evaluation manifest")
    parser.add_argument("--dataset", default="RAID-techjam/SID_Set")
    parser.add_argument("--split", default="validation", help="Use a split disjoint from training's --split train")
    parser.add_argument("--output", default="data/sid_eval")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--image-column", default=None)
    parser.add_argument("--label-column", default=None)
    args = parser.parse_args()

    output = Path(args.output)
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    samples = iter(load_dataset(args.dataset, split=args.split, streaming=True))
    first = next(samples)
    image_col = _column(first, args.image_column, ("image", "img", "pixel_values"), "image")
    label_col = _column(first, args.label_column, ("label", "labels", "class"), "label")

    manifest = output / "manifest.csv"
    count = 0
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        handle.write("image_path,label,sid_label,native_width,native_height\n")
        for sample in chain((first,), samples):
            image = sample[image_col]
            if isinstance(image, dict) and image.get("path"):
                image = Image.open(image["path"])
            if not isinstance(image, Image.Image):
                raise TypeError(f"Unsupported image value: {type(image)!r}")
            rgb_image = image.convert("RGB")
            native_width, native_height = rgb_image.size
            path = image_dir / f"sample_{count:06d}.jpg"
            # Same fixed re-encode quality as training/data/import_hf.py, so
            # this manifest's images carry the same compression-history
            # control (both classes re-encoded identically) rather than a
            # second, inconsistent one.
            rgb_image.save(path, quality=95)
            # SID_Set: 0=real, 1=fully synthetic, 2=tampered.
            sid_label = int(sample[label_col])
            if sid_label not in (0, 1, 2):
                raise ValueError(f"Unexpected SID_Set label: {sid_label}")
            binary_label = 0 if sid_label == 0 else 1
            handle.write(f"{path.as_posix()},{binary_label},{sid_label},{native_width},{native_height}\n")
            count += 1
            if count >= args.limit:
                break
    print(f"Exported {count} samples to {manifest}")


if __name__ == "__main__":
    main()
