"""Table 4: error breakdown by native resolution bucket x class, plus a
second slice by original SID_Set label (0=real/1=fully-synthetic/2=tampered)
-- the diagnostic for a resolution-based shortcut and for whether folded-in
tampered images behave differently from fully-synthetic ones.

Uses metadata training.robustness_sweep captured at source (native_width/
native_height/sid_label survive from the manifest -- see
training/import_eval_manifest.py) rather than re-deriving anything, and
defaults to the clean condition (an intrinsic-bias question, not a
degradation one) -- pass --condition to look at a degraded condition instead.

Usage:
    python -m training.tables.table4_error_breakdown \
        --predictions outputs/robustness/predictions.jsonl \
        --output outputs/robustness/table4_error_breakdown.md
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

from src.explainability.serialization import read_prediction_jsonl

RESOLUTION_BUCKETS = [
    (0, 512, "<512"),
    (512, 1024, "512-1024"),
    (1024, 2048, "1024-2048"),
    (2048, float("inf"), ">2048"),
]


def _resolution_bucket(width: int, height: int) -> str:
    # Larger dimension: a 6000x400 banner and a 400x6000 poster are both
    # "large" in the sense that matters for a resample-ratio shortcut.
    longest_side = max(width, height)
    for low, high, label in RESOLUTION_BUCKETS:
        if low <= longest_side < high:
            return label
    return RESOLUTION_BUCKETS[-1][2]


def _rate_row(bucket_label: str, records: list) -> str:
    real = [r for r in records if r.ground_truth_label == 0]
    ai = [r for r in records if r.ground_truth_label == 1]
    fp_rate = (sum(1 for r in real if r.predicted_label == 1) / len(real)) if real else None
    fn_rate = (sum(1 for r in ai if r.predicted_label == 0) / len(ai)) if ai else None
    fp_str = "n/a" if fp_rate is None else f"{fp_rate:.4f}"
    fn_str = "n/a" if fn_rate is None else f"{fn_rate:.4f}"
    return f"| {bucket_label} | real | {len(real)} | {fp_str} | - |\n| {bucket_label} | AI | {len(ai)} | - | {fn_str} |"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Table 4: error breakdown by image property")
    parser.add_argument("--predictions", default="outputs/robustness/predictions.jsonl")
    parser.add_argument("--output", default="outputs/robustness/table4_error_breakdown.md")
    parser.add_argument("--condition", default="clean")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    all_records = read_prediction_jsonl(args.predictions)
    records = [r for r in all_records if r.metadata.get("condition") == args.condition]
    if not records:
        raise ValueError(f"no records found for condition={args.condition!r}")
    missing_metadata = [r for r in records if "native_width" not in r.metadata or "sid_label" not in r.metadata]
    if missing_metadata:
        raise ValueError(
            f"{len(missing_metadata)} record(s) are missing native_width/sid_label metadata -- "
            "this manifest wasn't built with training.import_eval_manifest, or metadata was stripped"
        )

    lines = [
        "# Table 4 -- Error breakdown by image property",
        "",
        f"Condition: `{args.condition}`, n={len(records)}. Resolution bucket uses the longer image side.",
        "",
        "## By native resolution",
        "",
        "| Resolution | Class | n | FP rate | FN rate |",
        "|---|---|---:|---:|---:|",
    ]
    by_resolution = defaultdict(list)
    for record in records:
        bucket = _resolution_bucket(record.metadata["native_width"], record.metadata["native_height"])
        by_resolution[bucket].append(record)
    for _low, _high, label in RESOLUTION_BUCKETS:
        if label in by_resolution:
            lines.append(_rate_row(label, by_resolution[label]))

    lines += [
        "",
        "## By original SID_Set label (0=real, 1=fully synthetic, 2=tampered->folded into AI)",
        "",
        "| SID_Set label | n | Error rate | What it measures |",
        "|---|---:|---:|---|",
    ]
    by_sid_label = defaultdict(list)
    for record in records:
        by_sid_label[record.metadata["sid_label"]].append(record)
    label_names = {0: "0 (real)", 1: "1 (fully synthetic)", 2: "2 (tampered)"}
    for sid_label in sorted(by_sid_label):
        group = by_sid_label[sid_label]
        if sid_label == 0:
            rate = sum(1 for r in group if r.predicted_label == 1) / len(group)
            note = "FP rate"
        else:
            rate = sum(1 for r in group if r.predicted_label == 0) / len(group)
            note = "FN rate"
        lines.append(f"| {label_names.get(sid_label, sid_label)} | {len(group)} | {rate:.4f} | {note} |")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
