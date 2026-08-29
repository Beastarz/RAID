# AI Image Detector

Modular dual-stream framework for robust AI-generated image detection, designed to
stay accurate under real-world post-processing (JPEG compression, blur, rescaling,
noise, color jitter, and cropping).

The architecture combines two feature-extraction streams — a high-level **semantic**
stream (ViT / DINOv2) and a mid-level **frequency** stream (2D FFT + ConvNeXt-Tiny) —
fused via a cross-attention layer feeding a classification head. Full architecture
and module ownership are documented in [`BLUEPRINT.md`](BLUEPRINT.md) and
[`.claude/CLAUDE.md`](.claude/CLAUDE.md).

## Status

The project is currently in the **scaffolding / stub phase**: all tensor contracts
are implemented and verified end-to-end, but the real pretrained backbones have not
been dropped in yet. This lets the data, evaluation, and explainability workstreams
develop in parallel against a stable interface before the heavy models land.

**Implemented so far:**

- `src/models/base_stream.py` — abstract `BaseFeatureStream` interface shared by
  both extraction streams.
- `src/models/semantic_stream.py` — `SemanticStream` stub (pool + linear projection)
  producing `[B, 1024]` features. Marked for a DINOv2 / ViT backbone drop-in.
- `src/models/frequency_stream.py` — `FrequencyStream` stub using a real 2D FFT
  magnitude spectrum + linear projection, producing `[B, 768]` features. Marked for
  an FFT high-pass mask + ConvNeXt-Tiny backbone drop-in.
- `src/models/fusion.py` — `FeatureFusion`: concatenates both streams and projects
  to a fused `[B, 512]` vector. Marked for a future cross-attention upgrade.
- `src/models/detector.py` — `DetectorPipeline`: wraps both streams, fusion, and a
  2-layer MLP head; `forward()` returns `{"logit", "prob", "features"}` per the
  project's data contract.
- `predict.py` — CLI for single-image or directory inference. Standardizes input to
  `[1, 3, 512, 512]` (ImageNet-normalized), runs the pipeline, and prints one JSON
  result per image (`filename`, `ai_probability`, `label`, `execution_time_ms`).
- `app.py` — Gradio frontend: image upload, Run button, prediction label, a
  P(AI-Generated) percentage slider, and a placeholder saliency heatmap overlay
  (to be replaced by the real Grad-CAM / attention visualizer).

**Not yet implemented:** real backbone weights, training loop (`train.py`), dataset
loading (`src/data/`), the robustness evaluation suite (`src/evaluation/`), and the
real explainability visualizer (`src/explainability/`) — these currently exist only
as empty module stubs.

Because no real backbone is trained yet, current predictions are **not meaningful**
— they reflect randomly initialized weights and exist only to prove the pipeline is
wired correctly end-to-end.

## Project Structure

See [`BLUEPRINT.md`](BLUEPRINT.md) for the full directory map and data contracts.

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
{"filename": "test_sample.jpg", "ai_probability": 0.49, "label": "Authentic", "execution_time_ms": 8.7}
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

Note: `tests/test_*.py` are currently empty stubs, so this will report "no tests
ran" until test cases are added — this is expected at this stage, not a failure.

## Contributing

Follow the module ownership, tensor contracts, and coding conventions defined in
[`.claude/CLAUDE.md`](.claude/CLAUDE.md) before opening a PR. In particular:

- Keep total model parameters under 2B (target ~337M).
- Never modify `src/models/base_stream.py` without team consensus.
- Apply augmentations before `ToTensorV2()`.
- Always use `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
- Run `pytest tests/` before committing.
