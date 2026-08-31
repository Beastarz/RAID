# TODO

Tracks remaining work for the AI Image Detector project. See [`README.md`](README.md)
for what's already done and [`.claude/CLAUDE.md`](.claude/CLAUDE.md) /
[`BLUEPRINT.md`](BLUEPRINT.md) for the architecture contracts everything below must
follow.

## Done

- [x] `src/models/base_stream.py` — `BaseFeatureStream` abstract interface
- [x] `src/models/semantic_stream.py` — stub, `[B, 3, H, W] -> [B, 1024]`
- [x] `src/models/npr_stream.py` — `NPRStream`, replaces the `frequency_stream.py`
      stub (kept on disk, superseded). Real NPR residual operator + swappable
      backbone (`resnet_shallow`/`convnext_tiny`) + swappable frontend (for a
      future Bayar+SRM fallback), `[B, 3, H, W] (raw [0,1], native crop) ->
      [B, 256 or 768]`. Has an actual trained checkpoint (see below), not just a
      stub.
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
      including checkpoint-file creation). Predates `train_npr.py` and doesn't
      cover it yet -- see §1 below. Still covers the now-superseded
      `train_frequency.py`; that coverage can move to `train_npr.py` once
      `frequency_stream.py`/`train_frequency.py` are removed for good.
- [x] `training/data/import_hf.py` — streams a real Hugging Face dataset
      (`RAID-techjam/SID_Set`) into a local `image_path,label` manifest CSV,
      collapsing its 3-way label to the project's binary contract.
- [x] `training/data/shuffle_manifest.py` — shuffles a manifest's rows (seeded)
      so `AIGCDataModule`'s contiguous train/val split isn't class-skewed.
- [x] `training/train_npr.py` — real (not mock) short training loop for the NPR
      stream: train/val split, AdamW + cosine LR, pos_weight-balanced BCE,
      per-epoch val loss/accuracy/AUC, checkpoints only on best val AUC. Trained
      once already on a partial real SID_Set pull (~3.4K samples): val AUC
      reached ~0.91 after 3 epochs.
- [x] `training/test_npr.py` — evaluates a trained NPR checkpoint on its
      held-out val split.

## 1. Data & Augmentations (`training/data/`) — remaining

- [ ] `import_hf.py` only exports a `--limit`-sized sample count, not a byte
      budget -- pull the full ~32GB/~137K-sample target (or more) once ready,
      and re-run `train_npr.py`/`test_npr.py` against it.
- [ ] `tests/test_data.py` doesn't cover `train_npr.py` or the new
      `import_hf.py`/`shuffle_manifest.py` scripts -- add coverage.
- [ ] Broaden `tests/test_data.py` beyond the minimal critical-path set (e.g.
      manifest-CSV loading, eval-mode isolation, severity bounds) once useful.

## 2. Model Backbones (`src/models/`)

- [ ] `semantic_stream.py` — replace the pool+linear stub with a DINOv2 / ViT
      backbone (frozen or fine-tuned), keeping the `[B, 3, H, W] -> [B, 1024]`
      contract.
- [ ] `npr_stream.py` — run the M3 resize/downscale stress test (go/no-go, see
      npr_stream_guide.md §7): does val AUC hold up (within ~0.05) when images
      go through a downscale-then-upscale round trip before the crop? If it
      collapses toward 0.5, swap the frontend to Bayar constrained-conv + SRM
      filters via the already-built `frontend=` injection point -- no other
      code changes needed.
- [ ] `fusion.py` — upgrade concat+linear to cross-attention fusion, keeping the
      `-> [B, 512]` contract; note its `freq_dim` default (768) needs
      `freq_dim=256` (or `NPRStream.output_dim`) to match NPR's default backbone.
- [ ] Populate `configs/model_config.yaml` (currently empty) with backbone and
      fusion hyperparameters.
- [ ] Verify total parameter count stays under the 2B budget (target ~337M) once
      real backbones are in — add an automated check.
- [ ] `tests/test_models.py` — shape/contract tests for each stream, fusion, and
      `DetectorPipeline`, including a frozen-weights determinism test.

## 3. Training (`training/train_semantic.py`, `training/train_npr.py`)

Each stream trains independently (own script, own checkpoint, no shared file)
so two teammates can research a stream each in parallel; neither touches
`fusion.py` yet. The two scripts are no longer at the same maturity level --
NPR has a real multi-epoch loop already; semantic is still the mock wiring
test.

- [x] Wire up `--config`, `--batch_size`, `--lr` CLI args per `CLAUDE.md`, plus
      a mock loop (real forward/BCE-loss/backward/optimizer.step over a few
      `--steps`) that proves the data flow end-to-end, for `train_semantic.py`.
- [x] Save a checkpoint per stream (`checkpoints/semantic_stream.pt` — mock
      weights, not yet meaningful).
- [x] `train_npr.py` — full real per-stream training loop: AdamW + cosine LR
      schedule, multi-epoch (5 by default), train/val split, real val
      loss/accuracy/AUC curves, checkpointing on best val AUC
      (`checkpoints/npr_stream.pt` + `checkpoints/npr_head.pt`).
- [ ] Give `train_semantic.py` the same real-training upgrade `train_npr.py`
      already got (epochs, val split, metrics, best-checkpoint saving) once a
      real semantic backbone lands.
- [ ] Once both streams are validated individually: joint fine-tuning of the
      fused `DetectorPipeline` (`fusion.py` + both streams together), and
      wiring `training/evaluate.py` to load the two per-stream checkpoints
      into it instead of running a fresh random-init model. Note this needs a
      key-remapping step for NPR's checkpoint (`NPRStream`'s flat state dict
      vs. `DetectorPipeline`'s `frequency_stream.*`-prefixed keys) and a
      resolution for the input-contract mismatch (NPR wants a raw native crop,
      semantic wants a resized/normalized tensor) -- `DetectorPipeline.forward`
      currently feeds one shared tensor to both streams.

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

- [ ] `visualizer.py` — real `AttributionVisualizer` (Grad-CAM for the NPR
      stream's backbone, attention-rollout for the semantic ViT), producing an
      `[H, W]` saliency map.
- [ ] Swap `app.py`'s `_mock_saliency_overlay` placeholder for the real
      visualizer output.
- [ ] Add `--save_heatmap` support to `predict.py` per `CLAUDE.md`.

## 6. Integration & Polish

- [ ] End-to-end run: real dataset -> `training/train_semantic.py` +
      `training/train_npr.py` -> per-stream checkpoints -> joint
      fine-tuning -> `training/evaluate.py` robustness report ->
      `predict.py --checkpoint ...` -> `app.py`.
- [ ] Decide the fate of `src/models/frequency_stream.py` and
      `training/train_frequency.py` now that `npr_stream.py`/`train_npr.py`
      supersede them -- either delete both (and their `tests/test_data.py`
      smoke-test coverage) or explicitly keep the FFT stream as a third,
      optional stream rather than a replacement.
- [ ] Load a real checkpoint into `app.py` (currently always runs stub/random
      weights).
- [ ] Restrict `app.py` image upload to `.jpg`/`.png`/`.webp` at the component
      level (currently relies on PIL's default decode support).
- [ ] Add CI (GitHub Actions or similar) running `pytest tests/` on every PR.
- [ ] Remove `test_sample.jpg` / stub-only smoke-test artifacts once real fixtures
      exist, or move them into `test_data/` for consistency.
