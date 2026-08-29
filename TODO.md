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
- [x] `training/data/augmentations.py` — `RobustnessTransforms` (Albumentations):
      JPEG compression (Q 30-90), Gaussian Blur (sigma 0.5-2.0), Downscale
      rescaling (0.25x-0.5x), Gaussian Noise, Color Jitter, 80% Center Crop, all
      before `ToTensorV2()`. Train mode (stochastic stack) and eval mode
      (isolated transform + severity, for degradation curves).
- [x] `training/data/dataset.py` — `AIGCDataset(Dataset)`, manifest-CSV mode or
      synthetic in-memory mode when no manifest is given.
- [x] `training/data/datamodule.py` — plain-PyTorch `AIGCDataModule`
      (train/val split, `DataLoader`s from `configs/base_config.yaml`) — not
      Lightning yet, see §3.
- [x] `configs/base_config.yaml`, `configs/augmentations.yaml` populated.
- [x] `training/logging_utils.py` + logging throughout the data path for
      debugging (per-sample augmentation params at DEBUG, dataset/datamodule
      sizes at INFO).
- [x] `tests/test_data.py` — minimal critical-path tests: augmentation output
      shape contract, dataset label/shape contract, and a smoke test per
      stream-training script (`train_semantic.py`/`train_frequency.py`,
      including checkpoint-file creation).

## 1. Data & Augmentations (`training/data/`) — remaining

- [ ] Source or assemble the real/AI-generated image dataset and build the
      manifest CSV (currently only exercised via the synthetic fallback).
- [ ] Broaden `tests/test_data.py` beyond the minimal critical-path set (e.g.
      manifest-CSV loading, eval-mode isolation, severity bounds) once useful.

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

## 3. Training (`training/train_semantic.py`, `training/train_frequency.py`)

Each stream trains independently (own script, own checkpoint, no shared file)
so two teammates can research a stream each in parallel; neither touches
`fusion.py` yet.

- [x] Wire up `--config`, `--batch_size`, `--lr` CLI args per `CLAUDE.md`, plus
      a mock loop (real forward/BCE-loss/backward/optimizer.step over a few
      `--steps`) that proves the data flow end-to-end, for each stream.
- [x] Save a checkpoint per stream (`checkpoints/semantic_stream.pt`,
      `checkpoints/frequency_stream.pt`) — currently mock weights, not yet
      meaningful.
- [ ] Implement the full (PyTorch Lightning or plain) per-stream training
      loop: LR schedule, multi-epoch training, real loss curves.
- [ ] Once both streams are validated individually: joint fine-tuning of the
      fused `DetectorPipeline` (`fusion.py` + both streams together), and
      wiring `training/evaluate.py` to load the two per-stream checkpoints
      into it instead of running a fresh random-init model.

## 4. Robustness Evaluation (`training/evaluation/`, `training/evaluate.py`)

- [x] `evaluate.py` — CLI wiring `--checkpoint` + `--config`, running a mock
      sweep through the real eval-mode `RobustnessTransforms` and logging
      shapes/probabilities per severity (no real metrics yet).
- [ ] `metrics.py` — Accuracy, ROC-AUC, F1, FPR@95%TPR, degradation-curve helpers.
- [ ] `robustness_suite.py` — `RobustnessBenchmark` running the model across the
      full transform severity spectrum from `configs/augmentations.yaml`,
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

- [ ] End-to-end run: real dataset -> `training/train_semantic.py` +
      `training/train_frequency.py` -> per-stream checkpoints -> joint
      fine-tuning -> `training/evaluate.py` robustness report ->
      `predict.py --checkpoint ...` -> `app.py`.
- [ ] Load a real checkpoint into `app.py` (currently always runs stub/random
      weights).
- [ ] Restrict `app.py` image upload to `.jpg`/`.png`/`.webp` at the component
      level (currently relies on PIL's default decode support).
- [ ] Add CI (GitHub Actions or similar) running `pytest tests/` on every PR.
- [ ] Remove `test_sample.jpg` / stub-only smoke-test artifacts once real fixtures
      exist, or move them into `test_data/` for consistency.
