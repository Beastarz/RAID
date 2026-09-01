"""Table 6: parameter counts (proves the 2B budget, ~337M target) and
single-image inference latency, per branch and total.

Standalone -- doesn't need a robustness sweep, just the bundle. Latency is
measured through the exact predict.py path (prepare_fused_inputs -> model),
single image at a time (batch=1), matching real deployment rather than the
batched sweep numbers.

Usage:
    python -m training.tables.table6_model_and_compute \
        --checkpoint checkpoints/detector_bundle.pt --image test_sample.jpg
"""

import argparse
import statistics
import time
from pathlib import Path
from typing import Optional, Sequence

import torch

from src.models.checkpoint_bundle import load_checkpoint_bundle
from src.models.fused_detector import prepare_fused_inputs


def _param_counts(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Table 6: model size and compute")
    parser.add_argument("--checkpoint", default="checkpoints/detector_bundle.pt")
    parser.add_argument("--image", default="test_sample.jpg")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output", default="outputs/robustness/table6_model_and_compute.md")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _manifest = load_checkpoint_bundle(args.checkpoint, map_location=device)
    model.eval().to(device)

    branches = {
        "Semantic (ViT-B/16)": model.semantic_stream,
        "Forensic (Bayar+SRM + shallow ResNet)": model.forensic_stream,
        "Fusion + classifier head": torch.nn.ModuleList([model.fusion, model.classifier]),
    }
    rows = []
    total_all, trainable_all = 0, 0
    for name, module in branches.items():
        total, trainable = _param_counts(module)
        total_all += total
        trainable_all += trainable
        rows.append((name, total, trainable))

    prepared = prepare_fused_inputs(Path(args.image))
    semantic = prepared.semantic.to(device)
    forensic = prepared.forensic.to(device)
    with torch.no_grad():
        for _ in range(args.warmup):
            model(semantic, forensic)
        latencies_ms = []
        for _ in range(args.runs):
            start = time.perf_counter()
            model(semantic, forensic)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

    lines = [
        "# Table 6 -- Model size and compute",
        "",
        f"Device: `{device}`. Latency: {args.runs} single-image forward passes (batch=1) "
        f"after {args.warmup} warmup runs, through the exact `predict.py` scoring path.",
        "",
        "| Component | Params | Trainable params |",
        "|---|---:|---:|",
    ]
    for name, total, trainable in rows:
        lines.append(f"| {name} | {total:,} | {trainable:,} |")
    lines.append(f"| **Total** | **{total_all:,}** | **{trainable_all:,}** |")
    lines += [
        "",
        "Note: `load_checkpoint_bundle` explicitly sets `requires_grad=False` on every "
        "parameter after strict loading (correct for inference/Grad-CAM, which "
        "differentiates w.r.t. inputs, not weights) -- so \"Trainable params\" above is "
        "always 0 for a loaded bundle and does not reflect which parameters were "
        "actually trained. During training: the semantic backbone is frozen except its "
        "last N transformer blocks (config-controlled), the forensic branch and "
        "fusion+classifier head train fully.",
        "",
        f"Budget check: {total_all:,} / 2,000,000,000 params "
        f"({100 * total_all / 2_000_000_000:.2f}% of the 2B ceiling, "
        f"{'under' if total_all < 337_000_000 else 'over'} the ~337M target).",
        "",
        f"Single-image latency ({device}): mean={statistics.mean(latencies_ms):.1f}ms, "
        f"median={statistics.median(latencies_ms):.1f}ms, "
        f"min={min(latencies_ms):.1f}ms, max={max(latencies_ms):.1f}ms, n={args.runs}.",
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
