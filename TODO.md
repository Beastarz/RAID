# TODO

Tracks remaining work for the AI Image Detector project. See [`README.md`](README.md)
for what's already done and [`.claude/CLAUDE.md`](.claude/CLAUDE.md) /
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the architecture contracts everything below must
follow.

## Done

- [x] `src/models/base_stream.py` — `BaseFeatureStream` abstract interface
- [x] `src/models/semantic_stream.py` — real ViT-B/16 stream,
      `[B, 3, H, W] -> [B, 1024]`
- [x] `src/models/npr_stream.py` — `NPRStream`, replaces the retired
      `frequency_stream.py` predecessor. Real NPR residual operator + swappable
      backbone (`resnet_shallow`/`convnext_tiny`) + swappable frontend (for a
      Bayar+SRM candidate), `[B, 3, H, W] (raw [0,1], native crop) ->
      [B, 256 or 768]`. Has an actual trained checkpoint (see below), not just a
      stub. The published fused detector selects the Bayar+SRM frontend with
      the shallow ResNet backbone and a 256-dimensional forensic output.
- [x] `src/models/fusion.py` — trained concat+linear `FeatureFusion`,
      `-> [B, 512]`
- [x] `src/models/detector.py` — `DetectorPipeline`, returns `{logit, prob, features}`
- [x] `predict.py` — single-image / directory CLI inference, JSON output
- [x] `app.py` — Gradio frontend with canonical-bundle inference and supported
      explanation views
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
- [x] `tests/test_data.py` — minimal critical-path tests for augmentation output,
      dataset label/shape contracts, and semantic stream-training smoke coverage.
      Additional NPR/Bayar+SRM training coverage is tracked below.
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
      hyperparameters; recording the now-selected Bayar+SRM forensic
      configuration remains pending.
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

The published fused detector fixes the topology at ViT-B/16 semantic features
projected to 1024 dimensions plus a Bayar+SRM forensic frontend with the shallow
ResNet backbone and 256-dimensional output. Its fusion checkpoint was trained
with both branch inputs derived from one 512x512 resize: ImageNet-normalized for
the semantic branch and raw `[0,1]` pixels for the forensic branch. The legacy
shared-input FFT `DetectorPipeline` remains obsolete and is not the final
explainability target.

- [x] Select Bayar+SRM with the shallow ResNet backbone as the active forensic
      branch for the published fused detector; retain NPR only as an experiment
      and provenance artifact.
- [x] Replace the legacy `frequency` block in `configs/model_config.yaml` with
      the selected Bayar+SRM frontend, shallow backbone, 256 output dimensions,
      and the canonical shared 512-resize preprocessing contract.
- [x] Retain the trained concat+linear `FeatureFusion -> [B, 512]` architecture
      for checkpoint compatibility. Cross-attention would require retraining and
      is not part of the final published model workflow.
- [ ] Verify total parameter count stays under the 2B budget (target ~337M) for
      the selected semantic+forensic topology and add an automated check.
- [ ] `tests/test_models.py` — add shape/contract tests for `NPRStream`,
      `BayarSRMFrontend`, the canonical fused detector, shared-resize branch
      preparation, raw-logit output, and frozen-weights determinism.

## 3. Training (`training/train_semantic.py`, `training/train_npr.py`, `training/train_bayar_srm.py`)

The semantic and forensic candidates were trained independently before fusion.
Standalone NPR and Bayar+SRM training use native raw-pixel crops, but the
published fusion checkpoint was trained through `AIGCDataModule`: one 512x512
resize is normalized for the semantic stream and denormalized back to `[0,1]`
for the frozen Bayar+SRM stream. That actual fused-training contract is
authoritative for the published final weights.

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
- [x] Add a self-describing final checkpoint bundle/manifest (complete fused
      detector state, topology, selected frontend and backbone, feature
      dimensions, 512 resize/interpolation and normalization policy, forensic
      pixel range, threshold, model identity, schema version, and source-file
      hashes). Bundle the semantic, Bayar+SRM, fusion, and classifier states;
      standalone files remain provenance artifacts.
- [x] Add a canonical fused detector under `src/models/` with explicit
      `semantic_dim=1024`, `forensic_dim=256`, and `fused_dim=512`; return raw
      logits from `forward()` without internal `no_grad` or sigmoid behavior.
- [x] Replace the provisional `predict.py`-local wrapper and obsolete
      `DetectorPipeline` path with one raw-image preparation path that performs
      a deterministic 512x512 resize once, derives normalized semantic and raw
      forensic tensors from the same pixels, and then calculates the fused
      logit. Do not retain the current separate 256x256 forensic resize.
- [x] Verify strict loading and numerical parity between the published
      three-file checkpoint and the self-describing bundle. Retraining or
      native-crop/top-k aggregation is out of scope for these final weights.

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
      JSONL parsing/file orchestration is implemented in Phase 7 below.

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

- [x] Start this phase only after the final-model adapter exposes the canonical
      deterministic raw-image scorer; it is no longer part of the model-free
      Wave 3 composition gate.
- [x] Implement deletion and insertion against a raw-image scoring callback,
      configurable patch size, perturbation count, baseline, and logit selector.
- [x] Perturb raw source images and regenerate every active branch input after
      each step rather than perturbing one model-specific tensor.
- [x] Preserve the adapter-owned 512 resize/interpolation, semantic
      normalization, forensic `[0,1]` conversion, and frontend transforms
      throughout every perturbation sequence.
- [x] Prefer blur or dataset-mean baselines and patch-level perturbations to
      avoid introducing synthetic forensic edges.
- [x] Report semantic, forensic (including any Bayar/SRM intermediate maps),
      and combined-map faithfulness separately when those maps are available.
- [x] Save score curves as well as normalized deletion/insertion AUC values.
- [x] Acceptance: a patch-dependent toy classifier ranks a correct heatmap above
      a random heatmap.

Suggested files: `src/explainability/faithfulness.py` and
`tests/test_faithfulness.py`.

### Phase 7: Standalone CLI and outputs

- [x] Add an offline report command that consumes prediction JSONL and writes
      `report.json`, metrics/robustness CSV, and plot artifacts without loading
      a model.
- [x] Add a rendering command that consumes explanation metadata/maps and writes
      per-sample visual artifacts.
- [x] Store per-image outputs under stable sample-ID directories and reference
      them by path from JSON rather than embedding large arrays by default.
- [x] Acceptance: both commands run against fixtures before model integration.

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

- [x] After the canonical fused detector and checkpoint bundle pass their gate,
      add one isolated
      `src/explainability/adapters/detector_adapter.py` implementation.
- [x] Consume and strictly validate the self-describing checkpoint manifest:
      `semantic` + `forensic` topology, Bayar+SRM/shallow-ResNet selection,
      feature dimensions, complete fused-detector state, shared 512 resize,
      normalization/pixel-range policies, decision threshold, identity, schema,
      and source hashes. Treat standalone files as provenance only.
- [x] Strictly load and validate the final checkpoint and record its identity and
      preprocessing metadata in reports.
- [x] Construct branch-specific prepared inputs from each raw source image:
      resize once to 512x512, then derive the normalized semantic tensor and raw
      `[0,1]` forensic tensor from the same resized pixels.
- [x] Expose the final AI logit plus named attribution targets and intermediate
      representations for each supported branch. For Bayar+SRM, expose Bayar,
      SRM, and fused frontend representations as forensic internals rather than
      pretending they are independent detector branches.
- [x] Preserve deterministic preparation context, including interpolation,
      normalization, pixel range, original/resized dimensions, and forensic
      frontend transforms.
- [x] Expose complete branch coalitions only when the bundle records an explicit
      feature-ablation baseline, preferably calibration-set means; otherwise
      return a structured unsupported reason rather than silently using zeros.
- [x] Mark plain attention rollout unsupported for the current torchvision ViT
      because its forward path does not expose attention matrices. Do not invent
      attention tensors; expose supported semantic attribution and token-grid
      Grad-CAM targets instead.
- [x] Omit unsupported explanation methods with a structured reason instead of
      inventing outputs.
- [x] Keep this adapter as the only module containing knowledge of final branch
      names, layer paths, feature dimensions, and detector output structure.

### Phase 9: Application integration

- [x] Replace the provisional `predict.py`-local `BayarFusionModel` with the
      canonical fused detector/adapter and remove the separate 256x256 forensic
      resize. Normal prediction must fail clearly when final weights are absent
      instead of silently returning random-weight predictions.
- [x] Replace `app.py`'s `_mock_saliency_overlay` with all adapter-supported
      explanation views, including semantic attribution, forensic
      frontend/intermediate visualization, and branch Grad-CAM where available.
      Display attention rollout as unsupported for this model. Keep coordinate
      space and raw scale explicit for every forensic output.
- [x] Never overlay non-image-coordinate maps (for example forensic/frequency
      planes) on the original image.
- [x] Add branch contributions and a raw JSON explanation component to Gradio.
- [x] Extend `predict.py` with explanation-method and output-directory options,
      including `--save_heatmap` compatibility.
- [x] Preserve the existing threshold slider, generator-based loading state,
      label-confidence behavior, and CSS during Gradio integration.
- [x] Keep expensive Integrated Gradients and deletion/insertion disabled by
      default and expose them as explicit opt-in analyses.
- [x] Keep dataset-level ROC/PR, confusion, calibration, and robustness reports
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
- Parent review gate accepted for Wave 2. Phase 6, the final-model adapter, and
  product integration remain pending; Wave 3 is accepted below.

#### Wave 3: Sequential composition

- [x] Add plot generation and output assembly on top of the accepted evaluation
      report schemas.
- [x] Integrate robustness report generation second.
- [x] Add JSONL parsing and standalone CLIs only after report and serialization
      APIs are accepted.
- [x] Parent review gate: run the complete model-free workflow from prediction
      fixtures to metrics, plots, robustness outputs, rendering, JSON, CSV, and
      standalone CLI artifacts. Deletion/insertion moves after the adapter gate.

Wave 3 implementation handoff and parent review:

- Added deterministic ROC, PR, confusion-matrix, reliability, and robustness
  curve plotting over validated model-free reports.
- Added strict prediction JSONL read/write, report output assembly, metrics and
  robustness CSV/JSON, standard PNG artifacts, and the documented report and
  explanation-rendering CLIs. Rendering keeps stable per-sample directories and
  rejects unsafe artifact paths.
- Hardened rendering against duplicate destinations, escaped valid contract IDs
  deterministically for filesystem paths, and preserved clean-baseline report
  semantics for condition-aware evaluation outputs.
- Focused Wave 3 suite: `20 passed`; full tracked suite: `210 passed, 4 skipped`;
  `compileall` and `git diff --check` pass.
- Wave 3 remains model-independent. Phase 6 faithfulness, the final-model
  adapter, and application integration remain pending behind the published
  checkpoint bundle gate.

#### Wave 3.5: Final architecture and checkpoint gate

- [x] Accept the published final topology as ViT-B/16 semantic plus one
      Bayar+SRM/shallow-ResNet forensic branch; NPR and the FFT frequency stream
      are not active final branches.
- [x] Implement and validate the canonical fused detector with raw-logit output
      and one deterministic 512x512 resize feeding both branch-specific tensor
      views. Remove the provisional separate 256 forensic resize; do not add
      native-crop/top-k behavior without retraining the fused checkpoint.
- [x] Produce a self-describing checkpoint bundle from the published semantic,
      Bayar+SRM, and fusion/classifier files. Verify strict loading, source
      hashes, deterministic preparation, threshold `0.5`, and numerical parity
      with the three-file scorer before exposing any explanation target.
- [x] Record the final branch names, internal representations, preprocessing,
      target layers, and unsupported capabilities as the input contract for
      Wave 4.

Wave 3.5 correction handoff:

- Canonical preparation now performs one Pillow bilinear 512x512 resize and
  derives both branch tensors from those shared pixels; the active CLI, app, and
  evaluation paths no longer use a separate 256x256 forensic resize or random
  fallback model.
- Bundle manifests bind source provenance to a derived `weights_id` and bind
  embedded values to a deterministic state digest. Loading validates both,
  resolves every declared explainability target, honors the requested device,
  and performs strict state loading.
- `training.build_detector_bundle --parity-image` compares the saved detector's
  score with an independently loaded three-file scorer before writing the bundle.
  The published checkpoint smoke test reproduces probability `0.4053786` on
  `test_sample.jpg`.
- Focused final-model suite: `40 passed, 1 skipped`; compileall and
  `git diff --check` pass. Full-suite attempts still encounter the existing
  Windows pytest temporary-directory ACL failure outside the code under test.

Wave 3.5 acceptance recheck (2026-09-01):

- Regenerated both `checkpoints/detector_bundle.pt` and the historical
  `checkpoints/detector_bundle_wave35.pt` from the three published source
  checkpoints with `--parity-image test_sample.jpg`.
- Both bundles now strict-load with source-hash validation and resolve the
  semantic and forensic attribution targets. Independent three-file parity is
  exact at logit `-0.38310351967811584` (probability `0.4053786098957062`),
  with threshold `0.5`; repeated canonical preparation is bitwise deterministic.
- The default `predict.py --image test_sample.jpg` path and app now load the
  validated bundle. Focused recheck: `22 passed, 1 skipped`; compileall and
  `git diff --check` pass. The full suite remains externally limited by the
  existing Windows pytest temporary-directory ACL failure.

Post-merge model audit:

- The downloaded stream checkpoints are bare state dictionaries and
  `detector_fusion.pt` contains only `fusion` and `classifier`; no file records
  topology, preprocessing, threshold, identity, or schema metadata.
- Fusion training used normalized 512x512 tensors and derived forensic
  `[0,1]` tensors by denormalization. The historical app/CLI path used a
  separate 256x256 forensic resize; the active paths now use the shared 512x512
  contract.
- On `test_sample.jpg`, the published weights produced `0.4153479` with the
  provisional 256 forensic resize and `0.4053786` with the 512 fused-training
  contract, confirming that the mismatch is behaviorally material.

#### Wave 4: Final-model adapter (sequential)

- [x] Confirm the canonical detector and self-describing bundle reproduce the
      published checkpoint before assigning Phase 8.
- [x] Assign one Luna increment to the architecture adapter; keep branch-specific
      preprocessing, target selection, branch ablation, and
      checkpoint validation together.
- [x] Sol review gate: validate explanations against a real checkpoint and
      deterministic samples, including unsupported-capability behavior.
- [x] Implement and review deletion/insertion faithfulness against the accepted
      adapter's raw-image scorer, then run toy-ranking and real-checkpoint smoke
      tests.

Wave 4 implementation handoff and parent review:

- Added the strict final-model adapter with canonical raw-image preparation,
  prediction metadata, semantic/forensic Grad-CAM targets, ViT token reshaping,
  declared intermediate capture, and structured unsupported capabilities.
- Added raw-image deletion/insertion curves with blur, dataset-mean, or explicit
  baselines, deterministic patch ordering, normalized AUCs, and independent
  named-map evaluation.
- Parent real-checkpoint review reproduced probability `0.4053786098957062`,
  produced finite nonzero semantic and forensic Grad-CAM maps with no leaked
  hooks, captured every declared intermediate, and ran a real raw-image
  faithfulness smoke. Focused gates: `46 passed, 2 skipped` for the adapter and
  `28 passed, 2 skipped` for faithfulness; compileall and `git diff --check`
  pass.

#### Wave 5: Product integration (sequential)

- [x] Migrate `predict.py` from its provisional model wrapper to the accepted
      detector/adapter and complete its Sol review first.
- [x] Migrate `app.py`, remove the placeholder saliency map and random-weight
      normal-operation fallback, and add supported explanation views only after
      the CLI path is accepted.
- [x] Add integration documentation and end-to-end tests last.
- [x] Route required shared-adapter changes through a separate reviewed Luna
      correction rather than changing the adapter incidentally in product work.
- [x] Parent review gate: run CLI, Gradio prediction function, focused tests, and
      the complete test suite before marking explainability integration complete.

Wave 5 implementation handoff and parent review:

- `predict.py` now uses the accepted adapter, has no provisional three-file
  wrapper, and emits deterministic strict explanation envelopes plus standalone
  PNG/NPY artifacts for semantic/forensic Grad-CAM and declared intermediates.
  Attention is structured unsupported; `--save_heatmap` remains compatible.
- Integrated Gradients and raw-image faithfulness are explicit CLI opt-ins and
  remain disabled during default inference. Faithfulness records both score
  curves and normalized AUCs.
- Gradio now shows real semantic/forensic Grad-CAM or the Bayar+SRM fused
  intermediate as standalone non-overlay views, preserves the threshold/loading
  flow and label CSS, and includes raw JSON with identity, preparation context,
  coordinate space, raw scale, and structured unsupported branch contributions.
- Parent real-checkpoint review reproduced `0.405379` through CLI and Gradio,
  exercised every UI view, built the Gradio `Blocks` interface, and validated
  lightweight and expensive CLI artifact envelopes. Integrated focused gate:
  `26 passed, 1 skipped`; compileall and `git diff --check` pass. The complete
  suite reached `198 passed, 4 skipped`; its remaining 39 setup errors are all
  the pre-existing Windows pytest temporary-directory ACL failure.

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
- [x] Rebase after the final model branches merge and audit the published
      semantic + Bayar/SRM fused architecture before adapter implementation.
- [ ] Use existing `torch`, `numpy`, `scikit-learn`, and `matplotlib`
      dependencies by default; agree with the team before adding Captum.

## 6. Integration & Polish

- [ ] End-to-end run: published checkpoint bundle -> strict canonical-detector
      load -> deterministic 512 preprocessing -> robustness report ->
      explainability CLI -> Gradio app.
- [x] Decide the final branch topology: ViT-B/16 semantic plus
      Bayar+SRM/shallow-ResNet forensic. Keep NPR as experimental provenance and
      remove obsolete FFT inference after bundle parity is confirmed.
- [ ] Replace `app.py`'s provisional three-file loader with the strict final
      bundle loader; missing weights must produce a clear unavailable state,
      not normal-operation random predictions.
- [ ] Restrict `app.py` image upload to `.jpg`/`.png`/`.webp` at the component
      level (currently relies on PIL's default decode support).
- [ ] Add CI (GitHub Actions or similar) running `pytest tests/` on every PR.
- [ ] Remove `test_sample.jpg` / stub-only smoke-test artifacts once real fixtures
      exist, or move them into `test_data/` for consistency.
