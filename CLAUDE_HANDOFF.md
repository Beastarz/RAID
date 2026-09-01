# Claude handoff: RAID (Robust AI-generated Image Detector)

Read this first for fast orientation, then follow the pointers below for
depth. Everything here is verified against the current code (not paraphrased
from other docs) as of this writing.

## Doc map — don't re-derive what's already written down

- **`README.md`** — project overview, setup, `git clone` → inference walkthrough
  with the published pretrained weights, reported numbers, limitations, team
  credits. Read this for "how do I run it."
- **`ARCHITECTURE.md`** — full directory map with module ownership, mermaid
  data-flow diagrams (training pipeline vs. inference pipeline), the tensor
  contract table, and a deep section on the explainability system (Grad-CAM,
  faithfulness, branch contributions, what's unsupported and why). Read this
  for "how do the pieces connect."
- **`ABOUT.md`** — the narrative: why dual-stream, why NPR got replaced by
  Bayar+SRM, challenges, what's next. Read this for "why does it look like
  this."
- **`docs/final_model_contract.md`** — the short, authoritative spec for the
  *published* model's topology, preprocessing, and explainability target
  paths. Read this before touching `src/models/fused_detector.py`,
  `checkpoint_bundle.py`, or the explainability adapter.
- **`TODO.md`** — very long, contains a multi-agent (Sol/Luna) orchestration
  history for the explainability build-out. Skip unless you need the
  phase-by-phase acceptance history for `src/explainability/`.

## The one thing to internalize first

There are **two different models in this codebase**, and mixing them up is
the most likely way to misunderstand the repo:

1. **`src/models/detector.py`'s `DetectorPipeline`** — the *original* stub
   pipeline. Still exists, still runs, but its forensic stream defaults to
   NPR (not Bayar+SRM), it has no trained fusion weights, and it's kept only
   for standalone-stream shape/contract testing. **Not an inference path.**
2. **`src/models/fused_detector.py`'s `CanonicalFusedDetector`** — the real,
   published, jointly-trained model. This is what `predict.py`, `app.py`,
   and the explainability adapter actually load (via
   `checkpoints/detector_bundle.pt`, built by
   `training.build_detector_bundle`). **This is "the model."**

## Preprocessing

**Two different preprocessing paths exist, for two different purposes —
know which one you're looking at.**

### Canonical (final model, inference) — `src/models/fused_detector.py::prepare_fused_inputs`

One deterministic step, no randomness, no augmentation:
1. Decode source image (path / PIL / numpy) → RGB.
2. Resize **once** to 512×512 with Pillow bilinear (`Image.Resampling.BILINEAR`).
3. Cast to float32 `[0, 1]` — this raw-pixel tensor **is** the forensic
   branch's input directly.
4. Semantic branch input = the *same* resized pixels, ImageNet-normalized
   (`mean=(0.485,0.456,0.406)`, `std=(0.229,0.224,0.225)`).

Both branches see the *same* 512×512 pixels, just normalized differently —
this is a single shared preprocessing contract, not two separate resizes.
`CanonicalFusedDetector.forward()` validates this strictly at runtime
(shape, dtype, and the forensic tensor's `[0,1]` range are all asserted, not
assumed). This is what the published bundle was actually trained and scored
against — see `docs/final_model_contract.md`.

**This is a deliberate trade-off, not an oversight — say so explicitly if
asked.** The Standalone section right below states that resizing destroys
the forensic signal, and the canonical path resizes anyway. That tension is
real, not a doc inconsistency: a single shared preprocessing contract is far
simpler to ship, validate (one parity-checked path, not two), and explain to
an explainability adapter than a two-view pipeline would be, and
`train_fusion.py` training the forensic branch on *augmented* resized images
(see Augmentation below) is the mitigation that was chosen instead of
avoiding the resize. What does **not** exist: an ablation comparing this
shared-resize forensic branch against a native-crop forensic branch inside
the fused model, so there's no number to cite for how much signal the resize
actually costs post-fusion-training — only the pre-fusion standalone numbers
below, which used a different (unaugmented, native-crop) branch entirely and
so aren't directly comparable. If asked for that ablation, the honest answer
is "not run — flagged as the highest-value next experiment," not a claim
that it was measured.

### Standalone per-stream training (research/pretraining only)

Before the two streams were fused, each was pretrained independently with
its **own** preprocessing, because the forensic stream's signal is destroyed
by resizing (a generator's upsampling artifact gets overwritten by whatever
resampling kernel resized the image) and can't share the semantic stream's
pipeline:

- **Semantic**: `training/data/dataset.py::AIGCDataset` → resize to 512×512
  + ImageNet-normalize (via `RobustnessTransforms`, see Augmentation below).
- **Forensic**: `training/train_npr.py::NPRCropDataset` → a random
  **native-resolution** crop (256×256 default), raw `[0,1]`, **never
  resized**.

These standalone pipelines are still what `train_npr.py`/`train_bayar_srm.py`
use today for independent stream research/ablation — they are not what
produced the published fusion weights (see Fusion head, below).

## Data augmentation

`training/data/augmentations.py::RobustnessTransforms` is the real,
Albumentations-based augmentation engine, driven by `configs/augmentations.yaml`.
Two modes:
- **`train`**: a stochastic stack — JPEG compression (Q30–90), Gaussian blur
  (σ 0.5–2.0), downscale rescale (0.25×–0.5×), Gaussian noise, color jitter,
  and an 80%-of-frame center crop (pad-if-needed first, so narrower source
  images don't break the crop) — each fires **independently** with
  probability `p_each=0.5`. Always ends in `Resize(512) → Normalize(ImageNet)
  → ToTensorV2`, so output shape is fixed regardless of what fired.
- **`eval`**: isolates exactly one named transform at one fixed severity (or
  `"clean"`) — used for degradation-curve sweeps, not training.

**Where augmentation actually applies, and where it doesn't — this is the
part worth being precise about:**

| training script | data path | augmented? |
|---|---|---|
| `train_semantic.py` (standalone) | `AIGCDataModule` → `RobustnessTransforms` (train mode) | yes |
| `train_fusion.py` (the one that produced the published weights) | also `AIGCDataModule` → `RobustnessTransforms`, forensic branch derived by denormalizing the same augmented+resized tensor back to `[0,1]` | **yes** |
| `train_npr.py` / `train_bayar_srm.py` (standalone forensic pretraining) | `NPRCropDataset`, imported by both scripts unmodified — random native crop only | **no augmentation at all**, not even JPEG/blur, and no resize |
| Canonical inference (`prepare_fused_inputs`) | one resize, no randomness | no |

So: the standalone forensic checkpoints (`npr_stream.pt`, early
`bayar_srm_stream.pt` snapshots) were pretrained on clean native crops only —
identically for both NPR and Bayar+SRM, since `train_bayar_srm.py` imports
`NPRCropDataset` straight from `train_npr.py` rather than building its own.
Their resize/downscale stress test (`evaluate_npr.py`/`evaluate_bayar_srm.py`)
is what first showed both collapsing under resize, with Bayar+SRM
consistently less badly than NPR at every severity despite the identical
unaugmented handicap — a controlled, fair comparison **between the two
frontends**, and the actual basis for selecting Bayar+SRM. What it does
**not** support is a claim about either frontend's *absolute* resize
robustness once trained normally: neither standalone run ever saw a resize
during training, so "NPR is fragile" and "NPR was never taught to survive a
resize" are both consistent with the same result, and only the second is
demonstrated. The *fused* model's forensic branch, separately, was later
exposed to the full augmentation stack during `train_fusion.py` — which is
part of why the published numbers (reported in `README.md`: ~0.87 clean AUC,
~0.85 AUC at 0.5× resize) hold up better under resize than the standalone
Bayar+SRM pretraining numbers, but see "Known gaps" below for how those two
published numbers were actually produced.

## Dual stream

Both conform to the shared `BaseFeatureStream` interface
(`src/models/base_stream.py`, `forward(x) -> [B, D]`) but with materially
different input contracts (see Preprocessing above).

### Semantic stream — `src/models/semantic_stream.py`

`SemanticStream(output_dim=1024, pretrained, freeze_backbone, unfreeze_last_n_blocks)`.
Wraps `torchvision.models.vit_b_16`; internally interpolates its input to
224×224 regardless of what it's given, so it tolerates the shared 512×512
input; backbone output (768-d) projected to 1024-d via a trainable
`nn.Linear`. Backbone frozen by default, optionally the last N transformer
blocks unfrozen for fine-tuning. Expects ImageNet-normalized input.

### Forensic stream — `src/models/npr_stream.py` + `src/models/frontend_bayar.py`

`NPRStream(backbone="resnet_shallow", frontend=BayarSRMFrontend())` — this
exact configuration (256-d output) is the **published, final** forensic
branch. `NPRStream` itself is architecture-agnostic over two axes, both
swappable via constructor args:
- **`frontend`**: turns raw `[0,1]` pixels into a residual,
  `[B,3,H,W] -> [B,3,H,W]`. Default is `NPR` (Tan et al., CVPR 2024 — a
  fixed, parameter-free nearest-neighbour downsample/upsample residual).
  `BayarSRMFrontend` (`frontend_bayar.py`) is the alternative actually
  selected for publication: a **learnable** Bayar constrained convolution
  (center tap fixed at −1, other 24 taps always renormalize to sum to +1)
  concatenated with 3 fixed classic SRM high-pass filters, fused back to 3
  channels via a learnable 1×1 conv. The swap happened because plain NPR's
  signal collapsed toward chance under a resize/downscale stress test — see
  `ABOUT.md` for the story, `training/evaluate_npr.py` /
  `training/evaluate_bayar_srm.py` for the actual harness.
- **`backbone`**: `"resnet_shallow"` (truncated ResNet-50 stem+layer1,
  ~0.2M params, 256-d — the published choice) or `"convnext_tiny"` (28M
  params, 768-d ablation).

Explainability hooks into this stream at `forensic_stream.backbone.4.2.conv3`
(Grad-CAM) and `frontend.bayar` / `frontend.srm` / `frontend.fuse` /
`backbone.4` / `pool` (raw intermediate dumps) — see
`docs/final_model_contract.md`.

## Fusion head

`src/models/fusion.py::FeatureFusion` — concatenate `[B,1024]` (semantic) +
`[B,256]` (forensic) → `[B,1280]` → `nn.Linear` → `LayerNorm` → `GELU` →
`[B,512]`. Simple concat+linear, not cross-attention (a documented future
upgrade, not yet done — retraining would be required).

The classification head is a plain 2-layer MLP (`512 → 128 → 1`, GELU
between), living directly on `CanonicalFusedDetector`
(`src/models/fused_detector.py`) alongside `fusion`, `semantic_stream`, and
`forensic_stream` — `forward()` runs both streams, fuses, classifies, and
returns a **raw logit** (no internal sigmoid/`no_grad`, deliberately, so
attribution algorithms can differentiate straight through the whole model).
Decision threshold is `0.5`.

**How it was actually trained**: `training/train_fusion.py` — freezes both
already-pretrained streams, trains only `fusion` + `classifier`, using
`AIGCDataModule` (see Augmentation table above) as the single shared data
source for both branches. Then `training/build_detector_bundle.py` packages
the three trained pieces (semantic checkpoint, forensic checkpoint, fusion
checkpoint) into one self-describing `checkpoints/detector_bundle.pt` —
source file hashes, a deterministic state digest, the full manifest
(topology/preprocessing/threshold/identity), and an independent parity check
against a fresh three-file scorer, all verified at bundle-build time and
re-verified at load time by `src/models/checkpoint_bundle.py`. This bundle
is what's published to `RAID-techjam/raid-detector-fusion` on the Hugging
Face Hub and what `predict.py`/`app.py` actually load.

## Known gaps, verified against the code — read before citing the reported numbers

Everything below was checked directly against the current repo (grep +
reading the actual scripts), not inferred from the docs' prose. In priority
order:

1. **The published robustness numbers (~0.87 clean, ~0.85 at 0.5× resize)
   have no backing artifact and can't be regenerated from what's in the
   repo.** No JSON/CSV report exists anywhere with these numbers — they only
   appear as prose in `README.md`. More importantly,
   `training/evaluation/robustness_suite.py` explicitly documents that it
   *never applies transforms or invokes a detector* — it's a pure aggregator
   over prediction records someone else must already have generated and
   tagged with condition/severity. There is no script anywhere that wires
   `RobustnessTransforms`'s eval mode (which covers JPEG/blur/downscale/
   noise/crop, not just resize) to `CanonicalFusedDetector`. So the two
   published numbers are a spot check from some ad-hoc process, not the
   output of a full per-transform, per-severity sweep — and that sweep
   infrastructure, despite existing for the *legacy* stub pipeline, doesn't
   reach the real model. This is the single highest-priority gap.
2. **No `scores.csv`-shaped batch output.** `predict.py --image <dir>`
   already works (`collect_image_paths` handles directories) and prints one
   JSON line per image (`ai_probability`, `execution_time_ms`) to stdout.
   `evaluate_predict.py` computes accuracy/confusion-matrix/AUC against a
   labeled manifest, also stdout-only. Neither writes a `filename,probability`
   CSV a reviewer can diff without parsing JSON lines first.
3. **Compression-history control exists but is partial.** `import_hf.py`
   re-encodes *every* image, both classes, at a fixed JPEG quality=95 — a
   real control on re-encode quality. It does **not** normalize native
   resolution (SID_Set spans 338–6020px, untouched), and a single high-quality
   re-encode doesn't erase prior generational compression history on the
   real-photo side. Don't claim this rules out a provenance shortcut; claim
   it partially mitigates one specific axis of it.
4. **Label mapping**: `import_hf.py`'s `binary_label = 0 if raw_label == 0
   else 1` — SID_Set's label 2 (tampered) is folded into positive/AI, same
   as label 1 (fully synthetic). This was previously undocumented anywhere.
5. **Split is unstratified, no generator holdout.** `import_hf.py --limit N`
   takes the first N samples in HF streaming order; `shuffle_manifest.py`
   shuffles rows afterward for train/val, but that's a random row shuffle,
   not stratification, and SID_Set exposes no generator-family column via
   this importer. Cross-generator generalization is genuinely untested —
   state this as a limitation, not an open question.
6. **Decision threshold is a hardcoded default, not calibrated.**
   `DECISION_THRESHOLD = 0.5` in `src/models/checkpoint_bundle.py`, and every
   bundle manifest is required to match it exactly. No calibration script or
   ROC-based operating-point search exists anywhere in the repo. At ~0.87
   AUC the operating point matters for false-positive rate — flag this
   plainly rather than imply 0.5 was chosen for a reason.
7. **No seed control or variance reporting for the fusion training run
   that produced the published weights.** `train_npr.py`/`train_bayar_srm.py`
   read a seed from config; `train_fusion.py` has zero seed references. No
   multi-seed run exists anywhere. Treat any single-run AUC delta under
   ~0.02 as possibly noise until proven otherwise.
8. **Not a gap, contrary to one worry**: input-format robustness in
   `predict.py` is actually fine — `VALID_EXTENSIONS` already covers
   `.jpg/.jpeg/.png/.webp`, and `prepare_fused_inputs`'s resize doesn't
   assume a square or minimum-size source image. Per-image latency is
   already reported (`execution_time_ms`); what's missing is only an
   aggregate throughput/hardware disclosure, not per-image timing.

## Quick commands (condensed from `README.md` — see it for full context)

```bash
# Run the published, real model
pip install huggingface_hub
hf download RAID-techjam/raid-detector-fusion --repo-type model --local-dir checkpoints
python -m training.build_detector_bundle \
  --semantic-checkpoint checkpoints/semantic_stream.pt \
  --forensic-checkpoint checkpoints/bayar_srm_stream.pt \
  --fusion-checkpoint checkpoints/detector_fusion.pt \
  --output checkpoints/detector_bundle.pt --parity-image test_sample.jpg
python predict.py --image test_sample.jpg --checkpoint checkpoints/detector_bundle.pt

# Train from scratch (each stream independently, then fuse)
python -m training.train_semantic --config configs/base_config.yaml
python -m training.train_bayar_srm --config configs/base_config_bayar.yaml
python -m training.train_fusion --config configs/base_config.yaml \
  --semantic-checkpoint checkpoints/semantic_stream.pt \
  --forensic-checkpoint checkpoints/bayar_srm_stream.pt
```

## Current status (verified, not aspirational)

- Semantic + Bayar+SRM forensic + fusion: **published, working, real
  weights** (`RAID-techjam/raid-detector-fusion`). `predict.py`/`app.py`
  load this by default when `checkpoints/detector_bundle.pt` is present.
- `DetectorPipeline` (`detector.py`): stub-weight, standalone-research-only,
  not an inference path — don't confuse it with the above.
- Explainability (`src/explainability/`): real, wired to the canonical
  bundle — semantic Grad-CAM, forensic Grad-CAM, Bayar/SRM intermediate
  dumps, Integrated Gradients, deletion/insertion faithfulness. Attention
  rollout and branch-coalition logits are explicitly reported as
  *unsupported* (not faked) — see `ARCHITECTURE.md` §4.E for why.
- Training data is still hackathon-scale (10K-image `SID_Set` subset, not
  the full dataset) — see `ABOUT.md`'s "What's next" for the honest
  priority-ordered remaining-work list (full dataset scale-up, full
  robustness benchmark, cross-attention fusion, CI).
- Before repeating the published robustness/accuracy numbers to anyone
  external, read "Known gaps" above — several of the claims that sound most
  load-bearing (the resize numbers, the label split, the threshold) are less
  rigorously established than the README's prose implies, and a reviewer
  checking the code will find that quickly.
