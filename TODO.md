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
      including checkpoint-file creation).
- [x] `training/data/import_hf.py` — streams a real image dataset from the
      Hugging Face Hub (default `RAID-techjam/SID_Set`), exports a `--limit`-sized
      JPEG subset, and writes an `AIGCDataset`-compatible `manifest.csv`
      (remaps SID_Set's 3-way label onto the project's binary contract).
- [x] `training/data/shuffle_manifest.py` — shuffles a manifest's rows (seeded)
      so `AIGCDataModule`'s contiguous train/val split isn't class-skewed.
- [x] `src/models/semantic_stream.py` — replaced the pool+linear stub with a
      real torchvision ViT-B/16 backbone (`pretrained`/`freeze_backbone`/
      `unfreeze_last_n_blocks` config, `parameter_counts()` for logging),
      keeping the `[B, 3, H, W] -> [B, 1024]` contract.
- [x] Populate `configs/model_config.yaml` (was empty) with semantic/frequency/
      fusion hyperparameters.
- [x] `training/train_semantic.py` — real multi-epoch (`--epochs`) training
      loop with a post-epoch validation pass (loss + accuracy), reading the new
      `semantic` config block; capped by `--steps` total optimizer steps.
- [x] `training/train_npr.py` — real (not mock) short training loop for the NPR
      stream: train/val split, AdamW + cosine LR, pos_weight-balanced BCE,
      per-epoch val loss/accuracy/AUC, checkpoints only on best val AUC. Trained
      once already on a partial real SID_Set pull (~3.4K samples): val AUC
      reached ~0.91 after 3 epochs.
- [x] `training/test_npr.py` — evaluates a trained NPR checkpoint on its
      held-out val split.
- [x] `tests/test_models.py` — output-contract, non-RGB-rejection,
      freeze/unfreeze-last-N-blocks, and parameter-count tests for
      `SemanticStream`, plus an output-contract test for `DetectorPipeline`.

## 1. Data & Augmentations (`training/data/`) — remaining

- [ ] Run `training/data/import_hf.py` against the full `SID_Set` (or another
      source) at training scale, not just a `--limit`-sized smoke subset.
- [ ] `tests/test_data.py` doesn't cover `train_npr.py` or the new
      `import_hf.py`/`shuffle_manifest.py` scripts -- add coverage.
- [ ] Broaden `tests/test_data.py` beyond the minimal critical-path set (e.g.
      manifest-CSV loading, eval-mode isolation, severity bounds) once useful.

## 2. Model Backbones (`src/models/`)


- [ ] `npr_stream.py` — run the M3 resize/downscale stress test (go/no-go, see
      npr_stream_guide.md §7): does val AUC hold up (within ~0.05) when images
      go through a downscale-then-upscale round trip before the crop? If it
      collapses toward 0.5, swap the frontend to Bayar constrained-conv + SRM
      filters via the already-built `frontend=` injection point -- no other
      code changes needed.
- [ ] Populate `configs/model_config.yaml` (currently empty) with backbone and
      fusion hyperparameters.
- [ ] `fusion.py` — upgrade concat+linear to cross-attention fusion, keeping the
      `-> [B, 512]` contract.
- [ ] Verify total parameter count stays under the 2B budget (target ~337M) once
      the frequency backbone is in — add an automated check (semantic ViT-B/16
      alone is ~86M; `parameter_counts()` on `SemanticStream` already reports
      its share).
- [ ] `tests/test_models.py` — add shape/contract tests for `frequency_stream.py`
      and `fusion.py`, plus a frozen-weights determinism test, once those
      backbones are real.

## 3. Training (`training/train_semantic.py`, `training/train_npr.py`)

Each stream trains independently (own script, own checkpoint, no shared file)
so two teammates can research a stream each in parallel; neither touches
`fusion.py` yet. The two scripts are no longer at the same maturity level --
NPR has a real multi-epoch loop already; semantic is still the mock wiring
test.

- [x] Wire up `--config`, `--batch_size`, `--lr` CLI args per `CLAUDE.md`, plus
      a mock loop (real forward/BCE-loss/backward/optimizer.step over a few
      `--steps`) that proves the data flow end-to-end, for each stream.
- [x] Save a checkpoint per stream (`checkpoints/semantic_stream.pt`,
      `checkpoints/frequency_stream.pt`) — currently mock weights, not yet
      meaningful.
- [x] `train_semantic.py` — real multi-epoch loop (`--epochs`) with
      post-epoch validation loss/accuracy logging.
- [x] `train_npr.py` — full real per-stream training loop: AdamW + cosine LR
      schedule, multi-epoch (5 by default), train/val split, real val
      loss/accuracy/AUC curves, checkpointing on best val AUC
      (`checkpoints/npr_stream.pt` + `checkpoints/npr_head.pt`).
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

Implement model-independent primitives first, validate them with deterministic
toy models and prediction records, then add one architecture adapter after the
model branches merge. Do not modify `src/models/`, `app.py`, `predict.py`, or
training entry points until the integration phases.

### Phase 0: Freeze contracts

- [x] Add versioned contracts for prediction records, explanation results,
      branch-coalition logits, artifact references, and JSON output schemas.
- [x] Define an `ExplainabilityAdapter` protocol that can expose prediction,
      semantic and NPR attribution targets, attention tensors, NPR residuals,
      and branch-subset logits when supported.
- [x] Keep semantic/NPR preprocessing, target-layer selection, ViT token-grid
      reshaping, NPR crop coordinates, crop aggregation, and model-output
      extraction inside the adapter rather than generic algorithms.
- [x] Represent unavailable model capabilities explicitly so an implementation
      without NPR, attention capture, or branch ablation degrades cleanly.
- [x] Document example prediction and per-image explanation JSON records.

### Phase 1: Metrics and report schemas

- [x] Implement model-independent ROC-AUC, average precision, explicitly named
      trapezoidal PR-AUC, confusion counts, accuracy, precision, recall,
      specificity, F1, Brier score, binary log loss, ECE, and reliability bins.
- [x] Add bootstrap confidence intervals for dataset-level metrics.
- [x] Accept arrays or prediction records rather than a model, and use a
      caller-supplied fixed decision threshold.
- [x] Return `None` for undefined metrics instead of non-standard JSON `NaN`.
- [x] Include class counts, sample counts, threshold, and schema version in
      serialized reports.
- [x] Keep metric calculation separate from plotting.
- [x] Add correctness tests using fixed labels and scores with independently
      verified expected values.
- [x] Acceptance: validated prediction records can produce a complete
      JSON-serializable report without importing or loading a PyTorch model;
      JSONL parsing/file orchestration remains Phase 7 work.

Suggested files: `training/evaluation/metrics.py`,
`training/evaluation/calibration.py`, `training/evaluation/schemas.py`,
`training/evaluation/report.py`, and `tests/test_evaluation.py`.

### Phase 2: Robustness reporting

- [ ] Aggregate existing prediction records by transform condition and severity
      without applying transforms or invoking a detector.
- [ ] Report clean and per-condition ROC-AUC, average precision, confusion,
      calibration, absolute degradation, and relative degradation.
- [ ] Preserve one fixed threshold across clean and degraded conditions.
- [ ] Produce machine-readable JSON/CSV plus ROC, PR, calibration, and
      degradation plots.
- [ ] Add tests for severity ordering, missing conditions, and clean-relative
      deltas.
- [ ] Acceptance: synthetic condition records produce correctly ordered
      robustness curves and clean-relative summaries.

Suggested files: `training/evaluation/robustness_suite.py` and
`training/evaluation/robustness_report.py`.

### Phase 3: Rendering and artifact management

- [x] Implement heatmap resizing, percentile normalization, color-map overlays,
      signed diverging residual views, absolute residual magnitude views, and
      side-by-side panels from arrays/tensors rather than model objects.
- [x] Save visual outputs as PNG and optionally preserve lossless map data as
      NumPy files.
- [x] Record artifact paths and raw statistics before display normalization so
      weak NPR residuals are not made to look artificially strong.
- [x] Make rendering deterministic and keep visualization separate from
      attribution generation.
- [x] Acceptance: fixed arrays produce deterministic dimensions, finite values,
      valid image ranges, and serializable artifact metadata.

Suggested files: `src/explainability/rendering.py`,
`src/explainability/artifacts.py`, `src/explainability/serialization.py`,
`src/explainability/visualizer.py`, and `tests/test_visualizer.py`.

### Phase 4: Generic attribution engines

- [ ] Implement generic Grad-CAM around a scoring callable, target `nn.Module`,
      logit selector, and optional activation-to-spatial transformation.
- [ ] Ensure hooks are always removed, model mode is restored, and attribution
      targets logits rather than saturated sigmoid probabilities.
- [ ] Implement Integrated Gradients as the primary architecture-independent
      semantic attribution method, with vanilla gradients as a diagnostic.
- [ ] Implement attention rollout from supplied attention matrices: head
      averaging, residual identity, row normalization, layer multiplication,
      CLS-to-patch extraction, and patch-grid reshaping.
- [ ] Label plain attention rollout as class-agnostic and defer
      gradient-weighted rollout until the final ViT exposes attention tensors
      from its real forward graph.
- [ ] Keep ViT CLS removal and token-grid reshaping adapter-supplied so Grad-CAM
      also works with the final semantic architecture.
- [ ] Test Grad-CAM with a tiny CNN and attention rollout with a miniature
      transformer containing known salient regions.
- [ ] Acceptance: toy models produce correctly localized finite maps and leave
      no hooks registered after execution.

Suggested files: `src/explainability/gradcam.py`,
`src/explainability/attention_rollout.py`,
`src/explainability/attribution.py`, `tests/test_gradcam.py`, and
`tests/test_attention_rollout.py`.

### Phase 5: Branch contributions

- [ ] Implement exact Shapley contributions for a generic set of named branches
      when the complete coalition power set is supplied.
- [ ] Validate unique branches, complete coalitions, finite logits, and a
      practical branch-count limit.
- [ ] Preserve the optimized two-branch result while supporting a possible
      semantic + frequency + NPR topology.
- [ ] Return the baseline logit, signed branch contributions, optional absolute
      display shares, reconstruction error, and coalition metadata.
- [ ] Verify that baseline plus branch contributions reconstructs the combined
      logit within numerical tolerance.
- [ ] Leave branch replacement policy to the adapter; prefer calibration-set
      mean features over arbitrary zero vectors when the final model supports
      feature-level branch ablation.
- [ ] Acceptance: exact reconstruction holds for nonlinear two-branch and
      three-branch toy fusion models.

Suggested files: `src/explainability/branch_contributions.py` and
`tests/test_branch_contributions.py`.

### Phase 6: Deletion/insertion faithfulness

- [ ] Implement deletion and insertion against a raw-image scoring callback,
      configurable patch size, perturbation count, baseline, and logit selector.
- [ ] Perturb raw source images and regenerate every active branch input after
      each step rather than perturbing one model-specific tensor.
- [ ] Preserve adapter-owned deterministic context such as crop coordinates,
      resize policy, frequency transforms, and crop aggregation throughout every
      perturbation sequence.
- [ ] Prefer blur or dataset-mean baselines and patch-level perturbations to
      avoid introducing synthetic forensic edges.
- [ ] Report semantic, frequency/NPR, and combined-map faithfulness separately
      when those maps are available.
- [ ] Save score curves as well as normalized deletion/insertion AUC values.
- [ ] Acceptance: a patch-dependent toy classifier ranks a correct heatmap above
      a random heatmap.

Suggested files: `src/explainability/faithfulness.py` and
`tests/test_faithfulness.py`.

### Phase 7: Standalone CLI and outputs

- [ ] Add an offline report command that consumes prediction JSONL and writes
      `report.json`, metrics/robustness CSV, and plot artifacts without loading
      a model.
- [ ] Add a rendering command that consumes explanation metadata/maps and writes
      per-sample visual artifacts.
- [ ] Store per-image outputs under stable sample-ID directories and reference
      them by path from JSON rather than embedding large arrays by default.
- [ ] Acceptance: both commands run against fixtures before model integration.

Proposed commands:

```text
python -m training.evaluation.report --predictions predictions.jsonl --output reports/
python -m src.explainability.render --input explanation.json --output artifacts/
```

Proposed artifacts:

```text
report.json
predictions.jsonl
metrics.csv
robustness.csv
roc_curve.png
pr_curve.png
confusion_matrix.png
reliability_diagram.png
robustness_curves.png
explanations/<sample-id>/*.png
```

### Phase 8: Final-model adapter

- [ ] After model work merges, add one isolated
      `src/explainability/adapters/detector_adapter.py` implementation.
- [ ] Confirm the final branch manifest, checkpoint format, preprocessing
      contracts, and decision threshold before implementation.
- [ ] Strictly load and validate the final checkpoint and record its identity and
      preprocessing metadata in reports.
- [ ] Construct branch-specific prepared inputs from each raw source image.
- [ ] Expose the final AI logit plus named attribution targets and intermediate
      representations for every supported branch.
- [ ] Preserve deterministic preparation context, including crop coordinates,
      resize policy, frequency transforms, and crop aggregation where applicable.
- [ ] Expose complete branch coalitions when contribution analysis is supported.
- [ ] Omit unsupported explanation methods with a structured reason instead of
      inventing outputs.
- [ ] Keep this adapter as the only module containing knowledge of final branch
      names, layer paths, feature dimensions, and detector output structure.

### Phase 9: Application integration

- [ ] Replace `app.py`'s `_mock_saliency_overlay` with all adapter-supported
      explanation views, including semantic attribution, attention,
      frequency-plane visualization, NPR residuals, and branch Grad-CAM where
      available.
- [ ] Never overlay frequency-coordinate maps on the original image.
- [ ] Add branch contributions and a raw JSON explanation component to Gradio.
- [ ] Extend `predict.py` with explanation-method and output-directory options,
      including `--save_heatmap` compatibility.
- [ ] Preserve the existing threshold slider, generator-based loading state,
      label-confidence behavior, and CSS during Gradio integration.
- [ ] Keep expensive Integrated Gradients and deletion/insertion disabled by
      default and expose them as explicit opt-in analyses.
- [ ] Keep dataset-level ROC/PR, confusion, calibration, and robustness reports
      separate from per-image explanations.

### Execution workflow

`TODO.md` is the source of truth for scope, dependencies, and acceptance
criteria. GPT-5.6 Sol medium is the parent planner/reviewer and GPT-5.6 Luna
xhigh is the sole active implementation worker. Sol must generate each Luna
prompt immediately before work starts; do not prewrite fixed prompts for later
waves because the repository and model contracts may change between reviews.

#### Wave 0: Contracts (sequential)

- [x] Assign one worker to Phase 0 only.
- [x] Parent review gate: verify that contracts do not import the current
      detector, hard-code branch dimensions/layer paths, or conflate raw images,
      model inputs, logits, and probabilities.
- [x] Do not start later waves until contract tests and example serialization
      pass.

Wave 0 accepted handoff:

- Schema version `1.0` uses strict JSON-compatible records and rejects
  non-finite values.
- `source_reference` is optional and opaque so paths, URIs, uploads, dataset
  references, and in-memory images are all representable.
- Predictions preserve model ID, logit, probability, predicted/ground-truth
  labels, and the applied decision threshold as distinct fields.
- Runtime model inputs and attribution targets remain opaque adapter-owned
  values; unsupported capabilities require a structured reason.
- Branch coalitions use canonical model-defined branch names and do not assume
  semantic/NPR dimensions or fusion implementation details.
- Verification: 24 focused contract tests pass, strict JSON examples serialize,
  Python compilation and `git diff --check` pass. The full pre-existing suite
  was attempted but cannot collect in the available pytest environment because
  `torch` and `albumentations` are not installed there.

#### Wave 1: Independent foundations (parallel where isolated)

- [x] Assign one worker to Phase 1 metrics/calibration.
- [x] Assign one worker to Phase 3 rendering/artifact management.
- [x] Give workers disjoint implementation and test files, preferably in
      isolated branches/worktrees.
- [x] Parent review gate: run each focused test set, inspect both public APIs,
      then run the full suite after integration.

Wave 1 accepted handoff:

- Evaluation accepts validated arrays or `PredictionRecord` sequences and
  produces strict-JSON discrimination, threshold, calibration, reliability-bin,
  and deterministic percentile-bootstrap results without importing a model.
- Undefined metrics are represented as `None`; public schemas reject invalid
  ranges, inconsistent confusion counts, malformed bins, and invalid confidence
  intervals.
- Rendering accepts NumPy and tensor-like values without importing PyTorch,
  preserves raw scale statistics, requires explicit coordinate-space labels,
  and prevents frequency-plane maps from being silently overlaid on images.
- Artifact storage writes PNG and optional lossless NPY outputs, rejects unsafe
  paths/IDs, and keeps generated provenance authoritative over caller metadata.
- Verification: 110 combined Wave 0/1 tests pass, Python compilation and
  `git diff --check` pass. The full pre-existing suite was attempted but cannot
  collect in the available pytest environment because `torch` and
  `albumentations` are not installed there.

#### Wave 2.0: Verification environment

- [ ] Create a repository `.venv` and install `requirements.txt`.
- [ ] Record the Python and dependency versions used for verification.
- [ ] Run the existing full test suite and record the baseline before adding
      attribution algorithms.
- [ ] Use the same environment for every remaining focused and full-suite run.

#### Wave 2: Algorithms (sequential reviewed increments)

- [ ] Sol prepares one bounded task card at a time.
- [ ] Luna first amends the adapter capabilities to expose generic named branch
      targets and intermediate representations. The active detector now uses an
      FFT high-pass ConvNeXt branch, while NPR remains a possible replacement or
      additional branch, so Wave 0's NPR-specific methods are too narrow.
- [ ] Replace branch-specific protocol methods with generic
      `attribution_targets()` and `intermediate_representations()` capability
      results keyed by canonical branch names; keep `branch_subset_logits()`
      generic over those same names.
- [ ] Luna then implements, in order: robustness aggregation, branch
      contributions, attention rollout, Integrated Gradients, and Grad-CAM.
- [ ] Sol reviews and accepts each increment before Luna starts the next one.
- [ ] Shared facades are updated only after their underlying implementations
      pass review.
- [ ] Parent review gate: verify consistent validation, dtype/device behavior,
      logit targeting, hook cleanup, output contracts, deterministic behavior,
      and architecture independence across all implementations.

#### Wave 3: Sequential composition

- [ ] Add plot generation and output assembly on top of the accepted evaluation
      report schemas.
- [ ] Integrate robustness report generation second.
- [ ] Implement deletion/insertion faithfulness after raw-image scoring and
      deterministic-context contracts are stable.
- [ ] Add JSONL parsing and standalone CLIs only after report and serialization
      APIs are accepted.
- [ ] Parent review gate: run the complete model-free workflow from prediction
      fixtures to metrics/plots and from a toy model/image to explanation maps,
      faithfulness curves, rendering, and JSON.

#### Wave 4: Final-model adapter (sequential, blocked on final model decision)

- [ ] Confirm whether the final detector is semantic + frequency, semantic +
      NPR, or semantic + frequency + NPR, along with its preprocessing and
      checkpoint contracts, before assigning Phase 8.
- [ ] Assign one Luna increment to the architecture adapter; keep branch-specific
      preprocessing, target selection, crop handling, branch ablation, and
      checkpoint validation together.
- [ ] Sol review gate: validate explanations against a real checkpoint and
      deterministic samples, including unsupported-capability behavior.

#### Wave 5: Product integration (sequential)

- [ ] Integrate `predict.py` and complete its Sol review first.
- [ ] Integrate `app.py` only after the CLI path is accepted.
- [ ] Add integration documentation and end-to-end tests last.
- [ ] Route required shared-adapter changes through a separate reviewed Luna
      correction rather than changing the adapter incidentally in product work.
- [ ] Parent review gate: run CLI, Gradio prediction function, focused tests, and
      the complete test suite before marking explainability integration complete.

### Parent orchestration protocol

For every worker iteration, the parent agent must:

- [ ] Inspect the current branch, worktree diff, relevant implementation files,
      accepted contracts, and previous handoff before preparing the task.
- [ ] Create a task card using the template below and derive the worker prompt
      from current code rather than copying stale instructions.
- [ ] Assign exact owned and prohibited files for one bounded Luna increment.
- [ ] State required tests and objective acceptance criteria in the prompt.
- [ ] Review the complete worker diff rather than relying only on its summary.
- [ ] Run focused tests, `git diff --check`, and the full suite at appropriate
      review gates.
- [ ] Return concrete findings to the same worker for correction when possible.
- [ ] Update phase checkboxes and record newly discovered constraints before
      constructing the next task.
- [ ] Do not commit worker changes unless the user explicitly authorizes commits.

### Task-card template

Create or update this card in the active parent task context before launching a
worker. It may be copied into a temporary handoff note when work spans sessions;
only stable requirements belong permanently in this file.

```text
Wave / phase:
Status: pending | in_progress | review | completed | blocked
Base commit:
Implementation model: GPT-5.6 Luna xhigh
Review model: GPT-5.6 Sol medium
Previous accepted handoff:
Objective:
Prerequisites satisfied:
Owned files:
Prohibited files:
Required behavior:
Acceptance criteria:
Verification commands:
Known integration constraints:
Expected handoff:
```

### Worker-prompt template

```text
As the sole active GPT-5.6 Luna xhigh implementation worker, implement
<wave/phase and bounded task> on the current explainability branch.

Read first:
- TODO.md section 5, especially <relevant phase>
- <accepted contract and implementation files>
- <relevant tests>

Owned files:
- <exact paths>

Do not modify:
- <exact paths or globs>
- src/models/** unless this is the accepted Phase 8 adapter task
- app.py and predict.py before Phase 9
- unrelated user or agent changes

Required behavior:
- <requirements derived from current code and TODO.md>

Acceptance criteria:
- <observable outcomes>

Verification:
- <focused commands>

Before finishing, inspect your diff and return one handoff containing changed
files, public API decisions, tests and results, assumptions, known limitations,
integration requirements, and the recommended next task. Do not commit unless
explicitly instructed.
```

### Worker handoff template

```text
Implemented:
Files changed:
Public APIs introduced or changed:
Tests run and results:
Assumptions:
Known limitations:
Integration requirements:
Recommended next iteration:
```

### Sequential role-switch rules

- [ ] Keep exactly one Luna implementation task active.
- [ ] Sol owns planning, API decisions, review, acceptance, and handoff.
- [ ] Luna owns bounded implementation and focused tests.
- [ ] Return review findings to the same Luna session for correction.
- [ ] Use a fresh Luna context after an increment is accepted.
- [ ] Do not begin the next increment until the current Sol review gate passes.
- [ ] Do not combine unrelated algorithm or integration work in one diff.
- [ ] Integrate accepted work strictly in dependency order.

### Merge-safety rules

- [ ] Land sequential increments as small, independently tested commits.
- [ ] Do not modify model or training-loop files during Phases 0-7.
- [ ] Keep `src/explainability/` free of imports from `training/`.
- [ ] Keep metrics independent of PyTorch where practical.
- [ ] Use versioned schemas and capability checks rather than hard-coded branch
      names, feature dimensions, or target-layer paths.
- [ ] Rebase after the final model branches merge, then adapt to the resulting
      architecture rather than adding compatibility for the obsolete detector.
- [ ] Use existing `torch`, `numpy`, `scikit-learn`, and `matplotlib`
      dependencies by default; agree with the team before adding Captum.

## 6. Integration & Polish

- [ ] End-to-end run: real dataset -> train selected active streams -> final
      fusion/joint training -> strict checkpoint load -> robustness report ->
      explainability CLI -> Gradio app.
- [ ] Decide and document the final branch topology: frequency, NPR, or both.
      Remove obsolete training paths only after that decision and checkpoint
      compatibility are confirmed.
- [ ] Load a real checkpoint into `app.py` (currently always runs stub/random
      weights).
- [ ] Restrict `app.py` image upload to `.jpg`/`.png`/`.webp` at the component
      level (currently relies on PIL's default decode support).
- [ ] Add CI (GitHub Actions or similar) running `pytest tests/` on every PR.
- [ ] Remove `test_sample.jpg` / stub-only smoke-test artifacts once real fixtures
      exist, or move them into `test_data/` for consistency.
