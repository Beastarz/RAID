# TODO

Tracks remaining work for the AI Image Detector project. See [`README.md`](README.md)
for what's already done and [`.claude/CLAUDE.md`](.claude/CLAUDE.md) /
[`BLUEPRINT.md`](BLUEPRINT.md) for the architecture contracts everything below must
follow.

## Done

- [x] `src/models/base_stream.py` — `BaseFeatureStream` abstract interface
- [x] `src/models/semantic_stream.py` — stub, `[B, 3, H, W] -> [B, 1024]`
- [x] `src/models/frequency_stream.py` — stub (real FFT magnitude), `[B, 3, H, W] -> [B, 768]`
- [x] `src/models/fusion.py` — `FeatureFusion` stub, `-> [B, 512]`
- [x] `src/models/detector.py` — `DetectorPipeline`, returns `{logit, prob, features}`
- [x] `predict.py` — single-image / directory CLI inference, JSON output
- [x] `app.py` — Gradio frontend with placeholder saliency overlay
- [x] `.gitignore`, `README.md` setup guide

## 1. Data & Augmentations (`src/data/`)

- [ ] `dataset.py` — implement `AIGCDataset(Dataset)` reading a manifest CSV
      (`image_path,label`; `0=Real`, `1=AI-Generated`), returning `[3, 512, 512]`
      ImageNet-normalized tensors.
- [ ] `augmentations.py` — implement `RobustnessTransforms` (Albumentations) with
      JPEG compression (Q 30-90), Gaussian Blur (sigma 0.5-2.0), Rescaling
      (0.25x-0.5x), Gaussian Noise, Color Jitter, and 80% Center Crop. Apply
      augmentations **before** `ToTensorV2()`.
- [ ] `datamodule.py` — PyTorch/Lightning `DataLoader` wrapper (train/val/test
      splits, batch size + workers from `configs/base_config.yaml`).
- [ ] Populate `configs/base_config.yaml` and `configs/augmentations.yaml` (both
      currently empty placeholders).
- [ ] Source or assemble the real/AI-generated image dataset and build the
      manifest CSV.
- [ ] `tests/test_data.py` — unit tests for dataset shape/label contracts and
      each augmentation.

## 2. Model Backbones (`src/models/`)

- [ ] `semantic_stream.py` — replace the pool+linear stub with a DINOv2 / ViT
      backbone (frozen or fine-tuned), keeping the `[B, 3, H, W] -> [B, 1024]`
      contract.
- [ ] `frequency_stream.py` — replace the FFT-magnitude stub with real FFT
      high-pass masking feeding a lightweight ConvNeXt-Tiny backbone, keeping the
      `[B, 3, H, W] -> [B, 768]` contract.
- [ ] `fusion.py` — upgrade concat+linear to cross-attention fusion, keeping the
      `-> [B, 512]` contract.
- [ ] Populate `configs/model_config.yaml` (currently empty) with backbone and
      fusion hyperparameters.
- [ ] Verify total parameter count stays under the 2B budget (target ~337M) once
      real backbones are in — add an automated check.
- [ ] `tests/test_models.py` — shape/contract tests for each stream, fusion, and
      `DetectorPipeline`, including a frozen-weights determinism test.

## 3. Training (`train.py`)

- [ ] Implement the PyTorch Lightning training loop: optimizer, LR schedule,
      loss (BCE on `logit`), checkpointing, logging.
- [ ] Wire up `--config`, `--batch_size`, `--lr` CLI args per `CLAUDE.md`.
- [ ] Decide freeze/fine-tune strategy for the semantic and frequency backbones.
- [ ] Produce a first trained checkpoint under `checkpoints/`.

## 4. Robustness Evaluation (`src/evaluation/`, `evaluate.py`)

- [ ] `metrics.py` — Accuracy, ROC-AUC, F1, FPR@95%TPR, degradation-curve helpers.
- [ ] `robustness_suite.py` — `RobustnessBenchmark` running the model across the
      full transform severity spectrum from `configs/augmentations.yaml`.
- [ ] `evaluate.py` — CLI wiring `--checkpoint` + `--config` to the benchmark,
      producing CSV/JSON degradation-curve outputs.
- [ ] `tests/test_evaluation.py` — metric correctness tests using a dummy
      random-logit model.

## 5. Explainability (`src/explainability/`)

- [ ] `visualizer.py` — real `AttributionVisualizer` (Grad-CAM for the frequency
      stream / ConvNeXt, attention-rollout for the semantic ViT), producing an
      `[H, W]` saliency map.
- [ ] Swap `app.py`'s `_mock_saliency_overlay` placeholder for the real
      visualizer output.
- [ ] Add `--save_heatmap` support to `predict.py` per `CLAUDE.md`.

## 6. Integration & Polish

- [ ] End-to-end run: real dataset -> `train.py` -> checkpoint -> `evaluate.py`
      robustness report -> `predict.py --checkpoint ...` -> `app.py`.
- [ ] Load a real checkpoint into `app.py` (currently always runs stub/random
      weights).
- [ ] Restrict `app.py` image upload to `.jpg`/`.png`/`.webp` at the component
      level (currently relies on PIL's default decode support).
- [ ] Add CI (GitHub Actions or similar) running `pytest tests/` on every PR.
- [ ] Remove `test_sample.jpg` / stub-only smoke-test artifacts once real fixtures
      exist, or move them into `test_data/` for consistency.
