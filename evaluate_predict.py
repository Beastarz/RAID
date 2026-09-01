"""Evaluate the canonical bundled detector against a labeled manifest."""
import argparse
import csv
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from predict import build_adapter, get_device, predict_single


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predict.py on a labeled manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/detector_bundle.pt")
    args = parser.parse_args()

    rows = list(csv.DictReader(Path(args.manifest).open(newline="", encoding="utf-8")))
    device = get_device()
    model = build_adapter(args.checkpoint, device)
    truth, predicted, scores = [], [], []
    for row in rows:
        result = predict_single(model, Path(row["image_path"]), device)
        label = int(row["label"])
        score = float(result["ai_probability"])
        truth.append(label)
        predicted.append(int(score >= 0.5))
        scores.append(score)
        print(f"{result['filename']}: actual={label} predicted={predicted[-1]} probability={score:.4f}")

    print(f"accuracy={accuracy_score(truth, predicted):.4f} n={len(rows)}")
    print(f"confusion_matrix=\n{confusion_matrix(truth, predicted, labels=[0, 1])}")
    if len(set(truth)) == 2:
        print(f"auc={roc_auc_score(truth, scores):.4f}")
    else:
        print("auc=undefined (manifest contains only one class)")


if __name__ == "__main__":
    main()
