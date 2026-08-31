# TODO

Tracks remaining work for the AI Image Detector project. See [`README.md`](README.md)
for what's already done and [`.claude/CLAUDE.md`](.claude/CLAUDE.md) /
[`BLUEPRINT.md`](BLUEPRINT.md) for the architecture contracts everything below must
follow.

## Done

- [x] `src/models/base_stream.py` — `BaseFeatureStream` abstract interface
- [x] `src/models/semantic_stream.py` — real ViT-B/16 stream,
      `[B, 3, H, W] -> [B, 1024]`
- [x] `src/models/npr_stream.py` — `NPRStream`, replaces the `frequency_stream.py`
      stub (kept on disk, superseded). Real NPR residual operator + swappable
      backbone (`resnet_shallow`/`convnext_tiny`) + swappable frontend (for a
      Bayar+SRM candidate), `[B, 3, H, W] (raw [0,1], native crop) ->
      [B, 256 or 768]`. Has an actual trained checkpoint (see below), not just a
      stub. The Bayar+SRM frontend is now implemented and trained as a
      separate forensic candidate; active-branch selection remains a
      robustness gate below.
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
      shape contract, dataset label/shape contract, and legacy stream-training
      smoke coverage (`train_semantic.py`/`train_frequency.py`, including
      checkpoint-file creation). Additional NPR/Bayar+SRM training coverage is
      tracked in the updated training section below.
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
- [x] Populate `configs/model_config.yaml` with baseline semantic and fusion
      hyperparameters; the selected forensic configuration remains pending.
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
- [ ] `tests/test_data.py` doesn't cover `train_npr.py`, `train_bayar_srm.py`, or the new
      `import_hf.py`/`shuffle_manifest.py` scripts -- add coverage.
- [ ] Broaden `tests/test_data.py` beyond the minimal critical-path set (e.g.
      manifest-CSV loading, eval-mode isolation, severity bounds) once useful.

## 2. Model Backbones (`src/models/`)

### Updated architecture status

The high-level stream is now a ViT-B/16 projected to 1024 features. The trained
low-level candidates are `NPRStream` (native raw-pixel crops, 256-d default
output) and `NPRStream(frontend=BayarSRMFrontend())` (Bayar+SRM forensic
frontend with the same downstream interface). `DetectorPipeline` still imports
the obsolete FFT `FrequencyStream` and accepts one shared tensor, so it is not
yet the final explainability target.

- [ ] `npr_stream.py` — run the M3 resize/downscale stress test (go/no-go, see
      npr_stream_guide.md §7) for NPR and Bayar+SRM. Select the active forensic
      frontend from the measured robustness result; this is a model-selection
      gate, not an automatic frontend swap.
- [ ] Replace the legacy `frequency` block in `configs/model_config.yaml` with
      explicit selected-forensic settings (frontend, backbone, output dimension,
      crop policy, and aggregation).
- [ ] `fusion.py` — upgrade concat+linear to cross-attention fusion, keeping the
      `-> [B, 512]` contract.
- [ ] Verify total parameter count stays under the 2B budget (target ~337M) for
      the selected semantic+forensic topology and add an automated check.
- [ ] `tests/test_models.py` — add shape/contract tests for `NPRStream`,
      `BayarSRMFrontend`, and fusion, including raw-crop versus
      normalized-resize input validation and frozen-weights determinism.

## 3. Training (`training/train_semantic.py`, `training/train_npr.py`, `training/train_bayar_srm.py`)

The semantic and forensic candidates train independently (own script, own
checkpoint) so each can be evaluated before fusion. NPR and Bayar+SRM use a
distinct native-crop/raw-[0,1] data path; semantic uses
resized/ImageNet-normalized tensors. These preprocessing contracts must be
retained when the streams are fused.

- [x] Wire up `--config`, `--batch_size`, `--lr` CLI args per `CLAUDE.md`, plus
      a mock loop (real forward/BCE-loss/backward/optimizer.step over a few
      `--steps`) that proves the data flow end-to-end, for each stream.
- [x] Save semantic and forensic stream checkpoints (`semantic_stream.pt`,
      `npr_stream.pt` + `npr_head.pt`, and `bayar_srm_stream.pt` +
      `bayar_srm_head.pt` where trained).
- [x] `train_semantic.py` — real multi-epoch loop (`--epochs`) with
      post-epoch validation loss/accuracy logging.
- [x] `train_npr.py` — full real per-stream training loop: AdamW + cosine LR
      schedule, multi-epoch (5 by default), train/val split, real val
      loss/accuracy/AUC curves, checkpointing on best val AUC
      (`checkpoints/npr_stream.pt` + `checkpoints/npr_head.pt`).
- [x] `train_bayar_srm.py` trains the Bayar+SRM forensic candidate through the
      same native-crop data path and writes distinct stream/head checkpoints for
      comparison with NPR.
- [ ] Add a self-describing final checkpoint bundle/manifest (complete fused
      detector state, topology, selected frontend and backbone, feature
      dimensions, crop size and count, resize/normalization policy, aggregation
      rule, threshold, and model identity). Standalone stream files and probe
      heads are initialization/provenance artifacts, not the final model state.
- [ ] Make the selected forensic output dimension explicit in fusion (NPR's
      default is 256; the ConvNeXt-Tiny option is 768) and decide whether a
      standalone semantic probe head must also be retained; never rely on the
      old `FeatureFusion(freq_dim=768)` default by accident.
- [ ] After individual validation and low-level selection: replace the current
      shared-input `DetectorPipeline` path with a canonical raw-image inference
      path that creates semantic resized/normalized input and deterministic
      native forensic crops, then performs the chosen crop aggregation and fused
      logit calculation.
- [ ] Jointly fine-tune/evaluate that final fused detector from the selected
      stream checkpoints. Remove the obsolete `FrequencyStream` loading path;
      do not rely on key remapping to make its incompatible checkpoint and input
      contract appear valid.

## 4. Robustness Evaluation (`training/evaluation/`, `training/evaluate.py`)

- [x] `evaluate.py` — CLI wiring `--checkpoint` + `--config`, running a mock
      sweep through the real eval-mode `RobustnessTransforms` and logging
      shapes/probabilities per severity. The CLI remains a legacy shared-input
      smoke path; record aggregation is provided by the model-free suite below.
- [x] `metrics.py` — model-independent binary metrics, calibration, and
      confidence-interval helpers used by robustness aggregation.
- [x] `robustness_suite.py` — `RobustnessBenchmark` and
      `aggregate_robustness()` consume existing prediction records by condition
      and severity; they do not apply transforms or invoke a detector.
- [x] `robustness_report.py` — strict JSON and flattened CSV serialization for
      ordered clean/degraded robustness points.
- [x] `tests/test_robustness_suite.py` — severity ordering, missing conditions,
      fixed identity/threshold, and clean-relative delta coverage.

## 5. Explainability (`src/explainability/`)

Implement model-independent primitives first, validate them with deterministic
toy models and prediction records, then add one architecture adapter after the
model branches merge. Do not modify `src/models/`, `app.py`, `predict.py`, or
training entry points until the integration phases.

### Phase 0: Freeze contracts

- [x] Add versioned contracts for prediction records, explanation results,
      branch-coalition logits, artifact references, and JSON output schemas.
- [x] Define an initial `ExplainabilityAdapter` protocol that can expose
      prediction, stream attribution targets, attention tensors, forensic
      intermediates, and branch-subset logits when supported. Wave 2 generalizes
      the initial NPR-specific method names to this capability model.
- [x] Keep branch-specific preprocessing, target-layer selection, ViT token-grid
      reshaping, forensic crop coordinates, crop aggregation, and model-output
      extraction inside the adapter rather than generic algorithms.
- [x] Represent unavailable model capabilities explicitly so an implementation
      without a forensic frontend, attention capture, or branch ablation
      degrades cleanly.
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

- [x] Aggregate existing prediction records by transform condition and severity
      without applying transforms or invoking a detector.
- [x] Report clean and per-condition ROC-AUC, average precision, confusion,
      calibration, absolute degradation, and relative degradation.
- [x] Preserve one fixed threshold across clean and degraded conditions.
- [x] Produce machine-readable JSON/CSV. Plot generation remains a later
      composition step over accepted report schemas.
- [x] Add tests for severity ordering, missing conditions, and clean-relative
      deltas.
- [x] Acceptance: synthetic condition records produce correctly ordered
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

- [x] Implement generic Grad-CAM around a scoring callable, target `nn.Module`,
      logit selector, and optional activation-to-spatial transformation.
- [x] Ensure hooks are always removed, model mode is restored, and attribution
      targets logits rather than saturated sigmoid probabilities.
- [x] Implement Integrated Gradients as the primary architecture-independent
      semantic attribution method, with vanilla gradients as a diagnostic.
- [x] Keep forensic attribution separate from semantic attribution: generic
      Grad-CAM consumes an adapter-selected target, while the accepted
      `intermediate_representations()` capability and Phase 3 signed/magnitude
      rendering provide frontend views with explicit coordinate space and raw
      scale. Final-model adapter wiring remains a later phase.
- [x] Implement attention rollout from supplied attention matrices: head
      averaging, residual identity, row normalization, layer multiplication,
      CLS-to-patch extraction, and patch-grid reshaping.
- [x] Label plain attention rollout as class-agnostic and defer
      gradient-weighted rollout until the final ViT exposes attention tensors
      from its real forward graph.
- [x] Keep ViT CLS removal and token-grid reshaping adapter-supplied so Grad-CAM
      also works with the final semantic architecture.
- [x] Test Grad-CAM with a tiny CNN and attention rollout with a miniature
      transformer containing known salient regions.
- [x] Acceptance: toy models produce correctly localized finite maps and leave
      no hooks registered after execution.

Suggested files: `src/explainability/gradcam.py`,
`src/explainability/attention_rollout.py`,
`src/explainability/attribution.py`, `tests/test_gradcam.py`, and
`tests/test_attention_rollout.py`.

### Phase 5: Branch contributions

- [x] Implement exact Shapley contributions for a generic set of named branches
      when the complete coalition power set is supplied.
- [x] Validate unique branches, complete coalitions, finite logits, and a
      practical branch-count limit.
- [x] Make the optimized two-branch result (`semantic` + selected `forensic`)
      the default. Support more branches only when the final detector exposes
      them as independently fused inputs; Bayar and SRM are internal forensic
      representations in the current model and are not separate coalitions.
- [x] Return the baseline logit, signed branch contributions, optional absolute
      display shares, reconstruction error, and coalition metadata.
- [x] Verify that baseline plus branch contributions reconstructs the combined
      logit within numerical tolerance.
- [x] Leave branch replacement policy to the adapter; prefer calibration-set
      mean features over arbitrary zero vectors when the final model supports
      feature-level branch ablation.
- [x] Acceptance: exact reconstruction holds for nonlinear two-branch and
      three-branch toy fusion models, with coalition names supplied by the
      adapter rather than hard-coded semantic/forensic names.

Suggested files: `src/explainability/branch_contributions.py` and
`tests/test_branch_contributions.py`.

### Phase 6: Deletion/insertion faithfulness

- [ ] Implement deletion and insertion against a raw-image scoring callback,
      configurable patch size, perturbation count, baseline, and logit selector.
- [ ] Perturb raw source images and regenerate every active branch input after
      each step rather than perturbing one model-specific tensor.
- [ ] Preserve adapter-owned deterministic context such as crop coordinates,
      resize policy, forensic frontend transforms, and crop aggregation
      throughout every perturbation sequence.
- [ ] Prefer blur or dataset-mean baselines and patch-level perturbations to
      avoid introducing synthetic forensic edges.
- [ ] Report semantic, forensic (including any Bayar/SRM intermediate maps),
      and combined-map faithfulness separately when those maps are available.
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

- [ ] After the final architecture gate and model work merge, add one isolated
      `src/explainability/adapters/detector_adapter.py` implementation.
- [ ] Consume and strictly validate the self-describing checkpoint manifest:
      final topology (`semantic` + selected `forensic`), frontend/backbone,
      feature dimensions, complete fused-detector state, crop policy,
      aggregation rule, preprocessing, and decision threshold. Treat standalone
      stream/probe heads as initialization or provenance only.
- [ ] Strictly load and validate the final checkpoint and record its identity and
      preprocessing metadata in reports.
- [ ] Construct branch-specific prepared inputs from each raw source image:
      semantic resized/normalized input plus deterministic native forensic crops.
- [ ] Expose the final AI logit plus named attribution targets and intermediate
      representations for each supported branch. For Bayar+SRM, expose Bayar,
      SRM, and fused frontend representations as forensic internals rather than
      pretending they are independent detector branches.
- [ ] Preserve deterministic preparation context, including crop coordinates,
      resize policy, forensic frontend transforms, and crop aggregation where
      applicable.
- [ ] Expose complete branch coalitions when contribution analysis is supported.
- [ ] Omit unsupported explanation methods with a structured reason instead of
      inventing outputs.
- [ ] Keep this adapter as the only module containing knowledge of final branch
      names, layer paths, feature dimensions, and detector output structure.

### Phase 9: Application integration

- [ ] Replace `app.py`'s `_mock_saliency_overlay` with all adapter-supported
      explanation views, including semantic attribution, attention,
      forensic frontend/intermediate visualization, and branch Grad-CAM where
      available. Keep the coordinate space and raw scale explicit for every
      forensic output.
- [ ] Never overlay non-image-coordinate maps (for example forensic/frequency
      planes) on the original image.
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
  semantic/forensic dimensions or fusion implementation details.
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
  and prevents non-image-coordinate maps from being silently overlaid on images.
- Artifact storage writes PNG and optional lossless NPY outputs, rejects unsafe
  paths/IDs, and keeps generated provenance authoritative over caller metadata.
- Verification: 110 combined Wave 0/1 tests pass, Python compilation and
  `git diff --check` pass. The full pre-existing suite was attempted but cannot
  collect in the available pytest environment because `torch` and
  `albumentations` are not installed there.

#### Wave 2.0: Verification environment

- [x] Create a repository `.venv` and install `requirements.txt`.
- [x] Record the Python and dependency versions used for verification.
- [x] Run the existing full test suite and record the baseline before adding
      attribution algorithms.
- [x] Use the same environment for every remaining focused and full-suite run.

Wave 2.0 accepted handoff:

- Repository-local environment: `.venv/`, currently verified on Windows with
  Python 3.14.7 at
  `C:\Users\keithPC\Documents\RAID\.venv\Scripts\python.exe`, and installed
  from the unpinned `requirements.txt`.
- Exact direct dependency versions are recorded in
  `docs/verification-environment.md`.
- Historical baseline verification command: `PYTHONPATH=. .venv/bin/python -m
  pytest -q`, with `120 passed`, recorded before attribution algorithms under
  the earlier macOS environment; that path is not current.
- Current focused and full-suite verification command:
  `.\.venv\Scripts\python.exe -m pytest -q tests`.
- All remaining verification uses the current repository-local Windows venv,
  not the system interpreter.

#### Wave 2: Algorithms (sequential reviewed increments)

- [x] Sol prepares one bounded task card at a time.
- [x] Luna first amends the adapter capabilities to expose generic named branch
      targets and intermediate representations. The updated trained models use
      a ViT semantic branch plus an NPR or Bayar+SRM forensic candidate, so
      Wave 0's NPR-specific methods are too narrow and the old FFT assumption
      must not be carried into the adapter.
- [x] Replace branch-specific protocol methods with generic
      `attribution_targets()` and `intermediate_representations()` capability
      results keyed by canonical branch names; keep `branch_subset_logits()`
      generic over those same names.
- [x] Luna implements robustness aggregation (this increment).
- [x] Luna then implements, in order: branch contributions, attention rollout,
      Integrated Gradients, and Grad-CAM.
- [x] Sol reviews and accepts each increment before Luna starts the next one.
- [x] Shared facades are updated only after their underlying implementations
      pass review.
- [x] Parent review gate: verify consistent validation, dtype/device behavior,
      logit targeting, hook cleanup, output contracts, deterministic behavior,
      and architecture independence across all implementations.

Wave 2 implementation handoff and parent review:

- Branch contributions, attention rollout, Integrated Gradients/vanilla
  gradients, and Grad-CAM were implemented as separate bounded Luna increments
  in that order; each increment received an implementation-focused review and
  its facade exports were added only after acceptance.
- Parent verification: focused Wave 2 suite `95 passed, 3 skipped in 2.62s`;
  full tracked suite via `.\.venv\Scripts\python.exe -m pytest -q tests`
  `190 passed, 4 skipped in 7.85s`; `compileall` passed; and `git diff --check`
  passed with only LF/CRLF warnings.
- Architecture-independence review found no implementation coupling; the only
  branch-topology mention was the explanatory docstring in
  `branch_contributions.py` stating that it does not know model topologies.
- Parent review gate accepted for Wave 2. Phase 6, Wave 3, final-model adapter,
  and product integration remain pending.

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

#### Wave 3.5: Final architecture and checkpoint gate

- [ ] Run the NPR and Bayar+SRM resize/downscale evaluations on comparable
      held-out data and select exactly one active low-level `forensic` branch
      for the first production detector.
- [ ] Implement and validate the final raw-image multi-input detector/trainer:
      semantic resized/normalized input, deterministic native forensic crops,
      explicit crop aggregation, and a fused logit. Do not adapt explainability
      to the obsolete shared-input FFT `DetectorPipeline`.
- [ ] Produce a self-describing checkpoint bundle and verify strict loading,
      deterministic preparation, output threshold, and real-checkpoint
      predictions before exposing any explanation target.
- [ ] Record the final branch names, internal representations, preprocessing,
      target layers, and unsupported capabilities as the input contract for
      Wave 4.

#### Wave 4: Final-model adapter (sequential, blocked on final model decision)

- [ ] Confirm the selected final detector is semantic + one selected forensic
      branch (NPR or Bayar+SRM), along with its preprocessing and checkpoint
      contracts, before assigning Phase 8. Add more top-level branches only if
      the fused model independently exposes them.
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
- [ ] Decide and document the final branch topology: semantic + selected
      forensic (NPR or Bayar+SRM). Remove obsolete FFT training/inference paths
      only after that decision and checkpoint compatibility are confirmed.
- [ ] Load a real checkpoint into `app.py` (currently always runs stub/random
      weights).
- [ ] Restrict `app.py` image upload to `.jpg`/`.png`/`.webp` at the component
      level (currently relies on PIL's default decode support).
- [ ] Add CI (GitHub Actions or similar) running `pytest tests/` on every PR.
- [ ] Remove `test_sample.jpg` / stub-only smoke-test artifacts once real fixtures
      exist, or move them into `test_data/` for consistency.
