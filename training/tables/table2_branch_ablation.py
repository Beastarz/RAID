"""Table 2: per-branch ablation under degradation -- semantic-only,
forensic-only, and fused, each scored across the same conditions as Table 1.

Why this needs its own probes rather than reusing anything that exists:
- `src/models/detector.py`'s standalone `DetectorPipeline` uses a *different*
  (untrained, stub-weight) forensic stream -- scoring through it would not
  tell you anything about the published model.
- The explainability module's branch-coalition/Shapley path
  (`src/explainability/branch_contributions.py`) is explicitly UNSUPPORTED
  for the published bundle: it requires a complete coalition power set
  including an empty-coalition baseline, and the bundle doesn't embed a
  calibration-set-mean feature baseline. The code deliberately refuses to
  fake that with zeros (see `ARCHITECTURE.md` SS4.E) -- so this script does
  not either.
- The standalone `train_bayar_srm.py` checkpoint was pretrained on
  unaugmented native crops, a different distribution than the fused model's
  resize-based input -- not a fair "forensic-only" stand-in for THIS model.

So: this script trains two small linear probes (`nn.Linear(1024, 1)` for
semantic, `nn.Linear(256, 1)` for forensic) on top of the *same frozen,
already-fine-tuned* `semantic_stream`/`forensic_stream` feature extractors
embedded in the published bundle -- the only way to ask "how much does each
already-trained branch know on its own" without changing what was actually
published. Probes are fit on a small held-out-from-eval training pool (a
fresh pull from SID_Set's `train` split, never touched by the robustness
sweep's `validation`-split eval manifest) using the SAME canonical
resize-based preprocessing, then scored across Table 1's conditions.

Usage:
    python -m training.tables.table2_branch_ablation \
        --eval-manifest data/sid_eval/manifest.csv \
        --checkpoint checkpoints/detector_bundle.pt \
        --probe-train-limit 400 \
        --output outputs/robustness/table2_branch_ablation.md
"""

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score

from src.models.checkpoint_bundle import load_checkpoint_bundle
from src.models.fused_detector import prepare_fused_inputs
from training.robustness_sweep import CONDITIONS, _prepare_batch


def _read_manifest(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _pull_probe_training_pool(output_dir: Path, limit: int) -> Path:
    """Fetches a small `train`-split pool for fitting the linear probes,
    disjoint from the `validation`-split eval manifest the sweep scores.

    Shells out to training.data.import_hf as a subprocess rather than
    importing its ``main()`` directly -- that function reads ``sys.argv``
    itself with no ``argv`` parameter, so it can't be called in-process with
    a custom argument list without fragile ``sys.argv`` patching.
    """
    import subprocess
    import sys

    manifest_path = output_dir / "manifest.csv"
    if manifest_path.exists():
        return manifest_path
    subprocess.run(
        [
            sys.executable,
            "-m",
            "training.data.import_hf",
            "--split",
            "train",
            "--output",
            str(output_dir),
            "--limit",
            str(limit),
        ],
        check=True,
    )
    return manifest_path


@torch.no_grad()
def _extract_features(model, rows: Sequence[dict], device: torch.device, batch_size: int = 16):
    semantic_features, forensic_features, labels = [], [], []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        prepared = [prepare_fused_inputs(Path(row["image_path"])) for row in batch_rows]
        semantic = torch.cat([p.semantic for p in prepared], dim=0).to(device)
        forensic = torch.cat([p.forensic for p in prepared], dim=0).to(device)
        semantic_features.append(model.semantic_stream(semantic).cpu())
        forensic_features.append(model.forensic_stream(forensic).cpu())
        labels.extend(int(row["label"]) for row in batch_rows)
    return torch.cat(semantic_features), torch.cat(forensic_features), torch.tensor(labels, dtype=torch.float32)


def _fit_probe(features: torch.Tensor, labels: torch.Tensor, epochs: int = 200, lr: float = 1e-2) -> nn.Linear:
    probe = nn.Linear(features.shape[1], 1)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = probe(features).squeeze(1)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
    probe.eval()
    return probe


@torch.no_grad()
def _score_branch(
    extractor, probe: nn.Linear, rows: Sequence[dict], device: torch.device, branch: str, degrade_fn, batch_size: int = 16
):
    labels, probabilities = [], []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        paths = [Path(row["image_path"]) for row in batch_rows]
        prepared = _prepare_batch(paths, degrade_fn)
        view = torch.cat([getattr(p, branch) for p in prepared], dim=0).to(device)
        features = extractor(view)
        logits = probe(features).squeeze(1)
        probabilities.extend(torch.sigmoid(logits).cpu().tolist())
        labels.extend(int(row["label"]) for row in batch_rows)
    return labels, probabilities


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Table 2: per-branch ablation under degradation")
    parser.add_argument("--eval-manifest", default="data/sid_eval/manifest.csv")
    parser.add_argument("--checkpoint", default="checkpoints/detector_bundle.pt")
    parser.add_argument("--probe-train-dir", default="data/sid_probe_train")
    parser.add_argument("--probe-train-limit", type=int, default=400)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--output", default="outputs/robustness/table2_branch_ablation.md")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _manifest = load_checkpoint_bundle(args.checkpoint, map_location=device)
    model.eval().to(device)

    probe_manifest_path = _pull_probe_training_pool(Path(args.probe_train_dir), args.probe_train_limit)
    probe_rows = _read_manifest(probe_manifest_path)
    print(f"Fitting probes on {len(probe_rows)} train-split images (disjoint from the eval manifest)")
    semantic_features, forensic_features, probe_labels = _extract_features(model, probe_rows, device)
    semantic_probe = _fit_probe(semantic_features, probe_labels, epochs=args.probe_epochs).to(device)
    forensic_probe = _fit_probe(forensic_features, probe_labels, epochs=args.probe_epochs).to(device)

    eval_rows = _read_manifest(Path(args.eval_manifest))
    per_branch_results: dict[str, dict] = {"semantic": {}, "forensic": {}, "fused": {}}
    for condition_name, severity, degrade_fn in CONDITIONS:
        semantic_labels, semantic_probs = _score_branch(
            model.semantic_stream, semantic_probe, eval_rows, device, "semantic", degrade_fn
        )
        forensic_labels, forensic_probs = _score_branch(
            model.forensic_stream, forensic_probe, eval_rows, device, "forensic", degrade_fn
        )
        fused_probs = []
        for start in range(0, len(eval_rows), 16):
            batch_rows = eval_rows[start : start + 16]
            paths = [Path(row["image_path"]) for row in batch_rows]
            prepared = _prepare_batch(paths, degrade_fn)
            semantic_view = torch.cat([p.semantic for p in prepared], dim=0).to(device)
            forensic_view = torch.cat([p.forensic for p in prepared], dim=0).to(device)
            with torch.no_grad():
                logits = model(semantic_view, forensic_view)
            fused_probs.extend(torch.sigmoid(logits).squeeze(1).cpu().tolist())

        key = (condition_name, severity)
        for branch, labels, probs in (
            ("semantic", semantic_labels, semantic_probs),
            ("forensic", forensic_labels, forensic_probs),
            ("fused", semantic_labels, fused_probs),
        ):
            auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else None
            per_branch_results[branch][key] = auc
        print(f"  condition={condition_name} severity={severity} done")

    lines = ["# Table 2 -- Per-branch ablation under degradation", ""]
    lines += ["| Branch | Clean AUC | Mean robust AUC | Worst-condition AUC (named) |", "|---|---:|---:|---|"]
    clean_key = ("clean", None)
    degraded_keys = [(name, severity) for name, severity, _fn in CONDITIONS if name != "clean"]
    for branch, label in (("semantic", "Semantic-only"), ("forensic", "Forensic-only"), ("fused", "Fused (published)")):
        results = per_branch_results[branch]
        clean_auc = results.get(clean_key)
        robust_values = [results[k] for k in degraded_keys if results.get(k) is not None]
        mean_robust = sum(robust_values) / len(robust_values) if robust_values else None
        worst_key, worst_auc = min(
            ((k, results[k]) for k in degraded_keys if results.get(k) is not None),
            key=lambda item: item[1],
            default=(None, None),
        )
        worst_name = f"{worst_key[0]}={worst_key[1]}" if worst_key is not None else "n/a"
        lines.append(
            f"| {label} | {clean_auc:.4f} | {mean_robust:.4f} | {worst_auc:.4f} ({worst_name}) |"
            if clean_auc is not None and mean_robust is not None and worst_auc is not None
            else f"| {label} | n/a | n/a | n/a |"
        )
    lines += [
        "",
        "Probes: `nn.Linear` fit on frozen, already-fused-trained branch features "
        f"({len(probe_rows)} train-split images, {args.probe_epochs} epochs, disjoint from the eval set above). "
        "\"Fused\" uses the published joint classifier, not a probe.",
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
