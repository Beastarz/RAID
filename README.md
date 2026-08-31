# RAID - Robust AI-generated Image Detector

Modular dual-stream framework for robust AI-generated image detection, designed to
stay accurate under real-world post-processing (JPEG compression, blur, rescaling,
noise, color jitter, and cropping).

The architecture combines two feature-extraction streams — a high-level **semantic**
stream (ViT / DINOv2) and a low-level **forensic** stream (Neighboring Pixel
Relationships (NPR) residual + a shallow ResNet/ConvNeXt-Tiny backbone, both
swappable) — fused via a cross-attention layer feeding a classification head. The
forensic stream started as an FFT-based "frequency" stream; `src/models/npr_stream.py`
now supersedes it (see Status below). Full architecture and module ownership are
documented in [`BLUEPRINT.md`](BLUEPRINT.md) and [`.claude/CLAUDE.md`](.claude/CLAUDE.md).

## Status

The project is transitioning out of the **scaffolding / stub phase**: the semantic
stream now wraps a real backbone, while the frequency stream and fusion layer are
still stubs. This lets the data, evaluation, and explainability workstreams develop
in parallel against a stable interface while the remaining heavy models land.

**Implemented so far:**

- `src/models/base_stream.py` — abstract `BaseFeatureStream` interface shared by
  both extraction streams.
- `src/models/semantic_stream.py` — `SemanticStream` wraps torchvision's ViT-B/16,
  producing `[B, 1024]` features via a linear projection off the 768-dim backbone
  output. Controlled by `configs/model_config.yaml`'s `semantic` block:
  `pretrained` (download ImageNet weights vs. random init), `freeze_backbone`,
  and `unfreeze_last_n_blocks` (partial fine-tuning of the last N encoder blocks
  + final LayerNorm while the rest stays frozen). Exposes
  `parameter_counts()` (`total`/`trainable`/`frozen`) for logging, and keeps the
  frozen backbone in `eval()` mode even when the wrapping module is in `train()`.
- `src/models/npr_stream.py` — `NPRStream`, the low-level forensic stream (replaces
  `frequency_stream.py`, which remains in the tree but is superseded and no longer
  part of the active architecture). Computes the fixed, parameter-free NPR residual
  (a nearest-neighbour downsample-upsample round trip subtracted from the input,
  isolating the periodic artifact every GAN/diffusion decoder's upsampling stack
  leaves behind), rescales it with `BatchNorm2d`, and reads it with a backbone —
  producing `[B, 256]` features by default (truncated ResNet-50 stem+layer1) or
  `[B, 768]` with the `convnext_tiny` ablation. Both the backbone and the residual
  frontend itself are swappable via constructor args (`backbone=`, `frontend=`), the
  latter specifically so a future Bayar+SRM frontend can drop in with no other code
  changes if the resize-robustness stress test calls for it. Unlike the other two
  streams, its input contract is raw `[0, 1]` pixels at a native-resolution crop,
  never resized — see `training/train_npr.py`.
- `src/models/fusion.py` — `FeatureFusion`: concatenates both streams and projects
  to a fused `[B, 512]` vector. Marked for a future cross-attention upgrade. Its
  `freq_dim` constructor arg defaults to 768 (the old FrequencyStream's output) —
  pass `freq_dim=256` (or `NPRStream.output_dim`) when pairing it with NPR's default
  backbone.
- `src/models/detector.py` — `DetectorPipeline`: wraps both streams, fusion, and a
  2-layer MLP head; `forward()` returns `{"logit", "prob", "features"}` per the
  project's data contract.
- `predict.py` — CLI for single-image or directory inference. Standardizes input to
  `[1, 3, 512, 512]` (ImageNet-normalized), runs the pipeline, and prints one JSON
  result per image (`filename`, `ai_probability`, `label`, `execution_time_ms`).
- `app.py` — Gradio frontend: image upload, Run button, prediction label, a
  P(AI-Generated) percentage slider, and a placeholder saliency heatmap overlay
  (to be replaced by the real Grad-CAM / attention visualizer).
- `training/data/augmentations.py` — `RobustnessTransforms` is a real,
  fully-implemented Albumentations pipeline (JPEG compression, Gaussian blur,
  downscale rescaling, Gaussian noise, color jitter, 80% center crop), with a
  train mode (stochastic stack) and an eval mode (one isolated transform at a
  fixed severity, for degradation curves).
- `training/data/dataset.py` / `datamodule.py` — `AIGCDataset` reads a real
  manifest CSV when `configs/base_config.yaml`'s `data.manifest_path` is set,
  else falls back to an in-memory synthetic dataset; `AIGCDataModule` wraps it
  into train/val `DataLoader`s.
- `training/data/import_hf.py` — CLI that streams an image classification
  dataset from the Hugging Face Hub (default `RAID-techjam/SID_Set`) without
  downloading it in full, exports up to `--limit` samples as local JPEGs, and
  writes a `manifest.csv` compatible with `AIGCDataset`. Infers the image/label
  columns (overridable via `--image-column`/`--label-column`) and remaps
  `SID_Set`'s three-way label (`0=real, 1=fully synthetic, 2=tampered`) onto
  this project's binary contract (`0=Real`, `1=AI-Generated`).
- `training/data/shuffle_manifest.py` — shuffles a manifest's rows (seeded) before
  training reads it, since `AIGCDataModule` splits train/val by a contiguous index
  range rather than a shuffled one.
- `training/train_semantic.py` — real, runnable training loop for the semantic
  stream: config → datamodule → `SemanticProbe` (`SemanticStream` + a linear
  head) → multi-epoch (`--epochs`) forward/BCE-loss/backward/optimizer loop,
  capped by `--steps` total optimizer steps, with a validation pass
  (loss + accuracy) logged after each epoch → saves
  `checkpoints/semantic_stream.pt`. Stream config (`pretrained`,
  `freeze_backbone`, `unfreeze_last_n_blocks`, `output_dim`) comes from
  `configs/base_config.yaml`'s `semantic` block.
- `training/train_npr.py` — unlike the semantic script above, this is a **real**
  (not mock) short training loop for the NPR stream: a train/val split, AdamW +
  cosine LR schedule, a class-imbalance-aware BCE loss, and per-epoch val
  loss/accuracy/AUC, checkpointing only when val AUC improves. Owns its own
  crop-only dataset (native-resolution random crops, raw `[0, 1]` pixels) rather
  than reusing `AIGCDataset`, since NPR's input contract can't share a resize step
  with the semantic stream. Saves `checkpoints/npr_stream.pt` (the backbone, same
  convention as `semantic_stream.pt` for a future `DetectorPipeline` fine-tune)
  and `checkpoints/npr_head.pt` (the classifier head) separately.
- `training/test_npr.py` — evaluates a trained NPR checkpoint (both files above)
  on the exact same held-out val split `train_npr.py` used, reporting loss,
  accuracy, and AUC.
- `training/evaluate.py` — a light mock sweep through the eval-mode
  `RobustnessTransforms`, logging shapes/probabilities per severity level; real
  metric computation (`training/evaluation/`) isn't implemented yet.
- `training/logging_utils.py` — shared logger setup; the data/training path
  logs dataset/datamodule sizes at INFO and per-sample augmentation params at
  DEBUG, for tracing data flow and debugging.

**Not yet implemented:** joint fine-tuning of the fused `DetectorPipeline` (and loading the per-stream
checkpoints into it for evaluation), a real (non-`import_hf.py`-subset) image
dataset for full training runs, the robustness metrics/benchmark suite
(`training/evaluation/metrics.py`, `robustness_suite.py`), and the real
explainability visualizer (`src/explainability/`) — these currently exist only
as empty module stubs or mocks.

The semantic stream can now be trained meaningfully (real ViT-B/16 backbone +
real data via `import_hf.py`), but end-to-end predictions through
`DetectorPipeline` are **still not meaningful** until the frequency stream and
fusion layer get real backbones and the streams are jointly fine-tuned.


## Project Structure

The repo separates two independent pipelines that both build on `src/`:

- **Training pipeline** (`training/`) — offline: `training/train_semantic.py`
  reads `training/data/` (`AIGCDataset`/`RobustnessTransforms`); `training/train_npr.py`
  owns its own crop-only dataset instead (see Status above) since NPR's input
  contract can't share that resize step. Each fits its own stream and writes its
  own checkpoint, independently of the other; `training/test_npr.py` evaluates
  NPR's checkpoint standalone, and `training/evaluate.py` benchmarks the fused
  `DetectorPipeline` with `training/evaluation/`. Run as modules from the repo
  root, e.g. `python -m training.train_semantic --config configs/base_config.yaml`.
  `training/data/` and `training/evaluation/` are training-only.
- **Inference pipeline** (`predict.py`, `app.py`) — online: loads a checkpoint into
  `DetectorPipeline` and serves predictions via CLI or the Gradio app, using
  `src/explainability/` for saliency heatmaps. Never imports `training/`.

See [`BLUEPRINT.md`](BLUEPRINT.md) for the full directory map, the data-flow
diagrams for each pipeline, and the tensor contracts.

## Setup & Test Guide

### 1. Prerequisites

- Python 3.10+
- (Optional) an NVIDIA GPU with CUDA for faster inference — the pipeline runs fine
  on CPU too.

### 2. Clone and enter the repo

```bash
git clone <repo-url>
cd ai-image-detector
```

### 3. Create and activate a virtual environment

```bash
python -m venv .venv
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

Includes `datasets` (Hugging Face) for `training/data/import_hf.py`. Set
`configs/base_config.yaml`'s `semantic.pretrained: true` to download
torchvision's ImageNet-pretrained ViT-B/16 weights on first run; the default
(`false`) keeps everything offline for smoke tests.

### 5. Verify the tensor contracts

Confirms Stream 1, Stream 2, Fusion, and the Detector output all match the shapes
defined in `CLAUDE.md`:

```bash
python -c "
import torch
from src.models.detector import DetectorPipeline

x = torch.randn(2, 3, 512, 512)
out = DetectorPipeline()(x)
assert out['logit'].shape == (2, 1)
assert out['prob'].shape == (2, 1)
assert out['features'].shape == (2, 512)
print('Tensor contracts OK:', {k: tuple(v.shape) for k, v in out.items()})
"
```

### 6. Run inference on a single image

A synthetic test image (`test_sample.jpg`) is included at the repo root for a quick
smoke test:

```bash
python predict.py --image test_sample.jpg
```

Expected output — one JSON line, e.g.:

```json
{
  "filename": "test_sample.jpg",
  "ai_probability": 0.49,
  "label": "Authentic",
  "execution_time_ms": 8.7
}
```

You can also point `--image` at a directory to run batch inference (one JSON line
per image):

```bash
python predict.py --image test_data
```

Pass `--checkpoint path/to/model.ckpt` once a trained checkpoint exists; without it,
the model runs with randomly initialized stub weights.

### 7. Launch the frontend

```bash
python app.py
```

This starts a local Gradio server (URL printed in the console). Upload a `.jpg`,
`.png`, or `.webp` image and click **Run** to see the predicted label, the
P(AI-Generated) percentage, and a placeholder saliency overlay.

### 8. Run the test suite

```bash
pytest tests/
```

`tests/test_data.py` covers the augmentation shape contract, the dataset
label/shape contract, and a smoke test per stream-training script (including
that it writes its checkpoint file). `test_models.py` covers `SemanticStream`'s
output contract, non-RGB input rejection, freeze/unfreeze-last-N-blocks
behavior, `parameter_counts()` consistency, and `DetectorPipeline`'s output
contract. `test_evaluation.py` is still an empty stub.

### 9. (Optional) Import a real training subset

```bash
python -m training.data.import_hf --dataset RAID-techjam/SID_Set --split train --limit 100 --output data/sid_subset
```

Streams samples from a Hugging Face dataset (no full download), exports them
as JPEGs, and writes `data/sid_subset/manifest.csv`. Point
`configs/base_config.yaml`'s `data.manifest_path` at that CSV to train against
real images instead of the synthetic fallback.


### 10. Run the training pipeline

```bash
# Semantic stream: still a mock loop (fixed step count, stub backbone)
python -m training.train_semantic --config configs/base_config.yaml --epochs 2 --steps 200 --log-level DEBUG
python -m training.train_frequency --config configs/base_config.yaml --steps 2 --log-level DEBUG

# NPR stream: a real training loop (5 epochs by default)
python -m training.train_npr --config configs/base_config.yaml --log-level INFO
python -m training.test_npr --config configs/base_config.yaml

# Evaluation:
python -m training.evaluate --config configs/augmentations.yaml
```

All of these run against the synthetic in-memory dataset by default (no real
dataset needed) and log the data flow at each stage — DEBUG level shows which
augmentations fired per sample. Point `configs/base_config.yaml`'s
`data.manifest_path` at a manifest produced by `training/data/import_hf.py` +
`shuffle_manifest.py` to train on real data instead — every script here falls
back to synthetic automatically if that path doesn't exist yet. `train_npr.py`/`test_npr.py` 
do, once pointed at real data.
`train_semantic.py` runs a real multi-epoch
loop (validation loss/accuracy logged after each epoch) capped by
`--steps` total optimizer steps; Both write a checkpoint to `checkpoints/`; 
see the "Not yet implemented" list above for what's still missing before predictions are meaningful.

## Contributing

Follow the module ownership, tensor contracts, and coding conventions defined in
[`.claude/CLAUDE.md`](.claude/CLAUDE.md) before opening a PR. In particular:

- Keep total model parameters under 2B (target ~337M).
- Never modify `src/models/base_stream.py` without team consensus.
- Apply augmentations before `ToTensorV2()`.
- Always use `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
- Run `pytest tests/` before committing.
