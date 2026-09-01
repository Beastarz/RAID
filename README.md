# RAID — Robust AI-generated Image Detector

Catch AI-generated images before they are mistaken for authentic ones. RAID is
a dual-stream detector designed to remain useful after real-world image
processing such as recompression and resizing. The final published model combines:

- a ViT-B/16 semantic branch for high-level structure and content;
- a Bayar+SRM shallow-ResNet forensic branch for residual, texture, and
  pixel-correlation evidence; and
- a learned fusion module and binary classifier.

Both branches are derived from one deterministic 512x512 Pillow bilinear resize.
The semantic view is ImageNet-normalized, while the forensic view retains the
same pixels in float32 `[0, 1]` form. Runtime inference requires a validated,
self-describing checkpoint bundle and never falls back to random weights.

See [MODEL_README.md](MODEL_README.md) for training details and reported model
results, [ARCHITECTURE.md](ARCHITECTURE.md) for the broader architecture, and
[docs/final_model_contract.md](docs/final_model_contract.md) for the exact
runtime contract.

## Features

- Single-image and directory inference through `predict.py`.
- Gradio application with an adjustable classification threshold.
- Semantic and forensic Grad-CAM.
- Bayar+SRM forensic intermediate visualization.
- Semantic Integrated Gradients as an explicit, expensive analysis.
- Raw-image deletion/insertion faithfulness as an explicit, expensive analysis.
- Strict JSON explanation envelopes plus lossless NPY and rendered PNG artifacts.
- Dataset-level metrics, calibration, robustness reports, and plots kept
  separate from per-image explanations.

Attention rollout is explicitly unsupported because the torchvision ViT forward
path does not expose attention matrices. Branch contributions are also
unsupported for the published bundle because it does not include a trained
feature-ablation baseline. The application reports these limitations instead of
synthesizing misleading results.

## Setup

Requirements:

- Python 3.10 or newer
- An optional CUDA-capable NVIDIA GPU; CPU inference is supported

Create an environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate` instead.

## Download and build the canonical bundle

Download the three published source checkpoints:

```powershell
pip install huggingface_hub
hf download RAID-techjam/raid-detector-fusion `
  --repo-type model `
  --local-dir checkpoints
```

Build the self-describing detector bundle and run the independent parity check:

```powershell
python -m training.build_detector_bundle `
  --semantic-checkpoint checkpoints/semantic_stream.pt `
  --forensic-checkpoint checkpoints/bayar_srm_stream.pt `
  --fusion-checkpoint checkpoints/detector_fusion.pt `
  --output checkpoints/detector_bundle.pt `
  --parity-image test_sample.jpg
```

The builder validates topology, strict state loading, source hashes, embedded
state integrity, explainability targets, and numerical parity with an
independently loaded three-file scorer. The application and CLI use
`checkpoints/detector_bundle.pt` by default.

## Run the integrated application

From the repository root:

```powershell
.\.venv\Scripts\python.exe app.py
```

Open the local URL printed in the terminal, normally
`http://127.0.0.1:7860`. Upload an image, choose a decision threshold and an
explanation view, then select **Run**.

Available application views:

- **Semantic Grad-CAM**: class-conditioned activity from the final ViT block.
  Its native 14x14 patch grid is enlarged to 512x512 with nearest-neighbor
  display scaling so it fills the panel without inventing extra detail.
- **Forensic Grad-CAM**: class-conditioned activity from the forensic backbone.
  Its native 128x128 grid is enlarged to 512x512 with bilinear display scaling.
- **Bayar+SRM fused intermediate**: a 512x512 view of low-level residual activity
  before forensic classification.
- **Attention rollout (unsupported)**: displays the structured reason that this
  model cannot provide attention matrices.

These are standalone feature-grid visualizations, not overlays on the uploaded
image. The raw JSON panel records the native grid size, display size,
interpolation, coordinate space, raw scale, model identity, and preprocessing
context.

### How the three explanation modes help

The three supported views describe different stages of the detector. They are
most useful when interpreted together rather than as independent proof that a
specific region was generated.

#### Semantic Grad-CAM

Semantic Grad-CAM examines the final ViT semantic layer and shows which image
patches most positively influenced the AI-generated logit. It answers: **which
high-level parts of the image supported the classification?**

The semantic branch can respond to features such as repeated or implausible
structures, inconsistent object geometry, malformed text or anatomy, and
unusual global composition or lighting. A bright patch means that activation in
that semantic region supported the AI-generated score; it does not necessarily
mean that a visible defect exists in every bright patch.

The model produces this explanation as a native 14x14 ViT patch grid. The app
uses nearest-neighbor enlargement so the patch boundaries remain visible and no
additional spatial precision is implied.

#### Forensic Grad-CAM

Forensic Grad-CAM examines a late convolutional layer after the Bayar+SRM
frontend and shows which forensic feature regions positively influenced the
AI-generated logit. It answers: **where did low-level forensic evidence affect
the final decision?**

This branch is designed to react to abnormal pixel correlations, inconsistent
noise, unusual edge statistics, repeated or overly smooth textures, resampling
traces, and generator-related high-frequency patterns. Bright regions indicate
where those learned forensic features supported the AI-generated class. Dark
regions are not proof of authenticity; they simply contributed less positive
evidence at the selected layer.

Its native 128x128 feature grid is enlarged with bilinear interpolation for
display. The raw NPY artifact retains the original grid values.

#### Bayar+SRM fused intermediate

This mode visualizes activity directly after the forensic frontend combines
Bayar prediction-error filters and fixed SRM high-pass residual filters. It
answers: **what low-level residual patterns were available to the forensic
backbone before classification?**

Bayar filters emphasize differences between a pixel and the value predicted by
its neighbors. SRM filters emphasize high-frequency residuals commonly used in
image-forensics analysis. Their fused response can reveal edges, texture
transitions, noise discontinuities, and processing artifacts that may be weak or
invisible in the RGB image.

Unlike Grad-CAM, this intermediate is **not class-conditioned**. A bright value
means strong residual activity, not automatically evidence of AI generation.
Use Forensic Grad-CAM to determine whether the classifier actually used those
residual features in support of its decision.

In summary:

- Semantic Grad-CAM indicates **what high-level content mattered**.
- Forensic Grad-CAM indicates **where low-level evidence affected the AI score**.
- Bayar+SRM intermediate indicates **what residual signals the forensic branch
  detected before deciding**.

These explanations describe model behavior and should not be treated as causal
proof, pixel-level segmentation, or a guarantee that highlighted regions were
generated or manipulated.

## Run prediction from the CLI

Score one image:

```powershell
.\.venv\Scripts\python.exe predict.py --image test_sample.jpg
```

The command prints one JSON object containing the filename, AI probability,
label, and execution time. Point `--image` at a directory for deterministic
batch inference, or use `--checkpoint` to select another validated bundle.

Generate lightweight explanations:

```powershell
# Semantic Grad-CAM
.\.venv\Scripts\python.exe predict.py `
  --image test_sample.jpg `
  --explanation-method semantic-gradcam `
  --output-directory outputs/explanations

# Forensic Grad-CAM
.\.venv\Scripts\python.exe predict.py `
  --image test_sample.jpg `
  --explanation-method forensic-gradcam

# All declared semantic and forensic intermediates
.\.venv\Scripts\python.exe predict.py `
  --image test_sample.jpg `
  --explanation-method intermediates
```

`--save_heatmap` remains a compatibility alias for forensic Grad-CAM.

Expensive analyses are disabled during normal prediction and must be selected
explicitly:

```powershell
# Semantic Integrated Gradients
.\.venv\Scripts\python.exe predict.py `
  --image test_sample.jpg `
  --explanation-method semantic-integrated-gradients `
  --ig-steps 32

# Forensic Grad-CAM followed by raw-image deletion/insertion faithfulness
.\.venv\Scripts\python.exe predict.py `
  --image test_sample.jpg `
  --explanation-method forensic-gradcam-faithfulness `
  --faithfulness-steps 8 `
  --faithfulness-patch-size 64
```

Explanation output defaults to `outputs/explanations/<sample-id>/` and includes
an `explanation.json` envelope, rendered PNG files, and lossless NPY arrays.
Feature-grid artifacts retain explicit coordinate-space metadata and are never
silently treated as source-image coordinates.

To record the structured attention limitation from the CLI:

```powershell
.\.venv\Scripts\python.exe predict.py `
  --image test_sample.jpg `
  --explanation-method attention
```

## Evaluate a labeled manifest

`evaluate_predict.py` accepts a CSV with `image_path,label` columns:

```powershell
.\.venv\Scripts\python.exe evaluate_predict.py `
  --manifest data/example_manifest.csv `
  --checkpoint checkpoints/detector_bundle.pt
```

It reports accuracy, a confusion matrix, and ROC AUC when both classes are
present. Offline report generation and robustness plotting are available under
`training/evaluation/`; they do not load the per-image explanation pipeline.

## Tests

Run the focused final integration tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_detector_adapter.py `
  tests/test_faithfulness.py `
  tests/test_predict_integration.py `
  tests/test_app_integration.py `
  -p no:cacheprovider
```

Run the complete suite with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests -p no:cacheprovider
```

Some Windows installations can encounter pytest temporary-directory ACL errors.
See [docs/verification-environment.md](docs/verification-environment.md) for the
known environment limitation. It is distinct from detector or explainability
test failures.

## Project layout

```text
app.py                                  Gradio application
predict.py                              Inference and explanation CLI
evaluate_predict.py                     Labeled-manifest evaluation
src/models/fused_detector.py            Canonical fused detector and preparation
src/models/checkpoint_bundle.py         Strict bundle builder/loader contract
src/explainability/adapters/            Final-model adapter
src/explainability/faithfulness.py       Deletion/insertion analysis
src/explainability/                      Generic attribution/rendering/contracts
training/build_detector_bundle.py        Published-checkpoint bundle builder
training/evaluation/                     Dataset-level evaluation and reports
tests/                                   Unit and integration tests
```

Checkpoint files, datasets, generated explanations, and evaluation outputs are
gitignored local artifacts.

## Team contributions

- **morpheuschoo** — semantic model and training pipeline, fusion training,
  configuration, and Hugging Face data import.
- **Goh Jin Yu ([@Beastarz](https://github.com/Beastarz))** — project
  scaffolding, architecture documentation, data pipeline, and Gradio frontend.
- **Shawn Wee ([@McFishhh](https://github.com/McFishhh))** — NPR and Bayar+SRM
  forensic streams, training, and robustness evaluation.
- **Shen Yu Chen, Keith ([@blurfrost](https://github.com/blurfrost))** —
  evaluation, explainability, output contracts, and integration tests.

See [ABOUT.md](ABOUT.md) for the longer project retrospective.

## Training and legacy components

The repository retains standalone semantic, NPR, Bayar+SRM, fusion-training, and
legacy detector components for research and checkpoint reproduction. They are
not runtime fallbacks for the final application. Refer to
[MODEL_README.md](MODEL_README.md) for those workflows rather than using legacy
training modules as inference entry points.
