"""Stream a Hugging Face image dataset into a small local manifest."""
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
    parser = argparse.ArgumentParser(description="Export a streamed HF dataset subset")
    parser.add_argument("--dataset", default="RAID-techjam/SID_Set")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="data/sid_subset")
    parser.add_argument("--limit", type=int, default=100)
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
        handle.write("image_path,label\n")
        for sample in chain((first,), samples):
            image = sample[image_col]
            if isinstance(image, dict) and image.get("path"):
                image = Image.open(image["path"])
            if not isinstance(image, Image.Image):
                raise TypeError(f"Unsupported image value: {type(image)!r}")
            path = image_dir / f"sample_{count:06d}.jpg"
            image.convert("RGB").save(path, quality=95)
            handle.write(f"{path.as_posix()},{int(sample[label_col])}\n")
            count += 1
            if count >= args.limit:
                break
    print(f"Exported {count} samples to {manifest}")


if __name__ == "__main__":
    main()
