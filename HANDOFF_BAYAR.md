# Handoff: Bayar+SRM Forensic Stream

Written for a fresh Claude instance picking up this work cold. Read this
before touching `src/models/`, `training/train_*.py`, or `training/evaluate_*.py`.

## TL;DR

- The project is a dual-stream AI-image detector: **semantic** (ViT-B/16) +
  **forensic**. The forensic stream is now **Bayar+SRM**
  (`src/models/frontend_bayar.py`), not NPR and not the original FFT
  `frequency_stream.py` -- both of those are dead ends, kept on disk but
  superseded. The two streams train independently right now and are meant
  to be combined via `fusion.py` + a classifier head later.
- Semantic stream: trained, real checkpoint (`checkpoints/semantic_stream.pt`,
  ViT-B/16, 86.6M params, 787K trainable / rest frozen).
- Bayar+SRM stream: architecture is done and verified correct, but its
  checkpoint (`checkpoints/bayar_srm_stream.pt`) is **stale** -- trained for
  only 2 sanity-check epochs on the small 3,420-sample set, predates the
  10,000-sample real pull that's already downloaded and ready. **Retraining
  it on the 10K set is the immediate next step**, command below.
- `src/models/detector.py` and `training/train_fusion.py` have **not** been
  updated to use Bayar+SRM yet -- they still reference the old
  `FrequencyStream` / plain `NPRStream` respectively. Wiring the real
  semantic+Bayar+SRM fusion is unbuilt. See "Known gaps" below.

## Why NPR was abandoned for Bayar+SRM

The original forensic stream was NPR (Neighboring Pixel Relationships,
Tan et al. CVPR 2024): a fixed, parameter-free residual operator
(`x - upsample(downsample(x))`) that isolates the periodic artifact a
generator's upsampling stack leaves behind. It trained well on real data
(val AUC 0.9194 on a held-out split) but **failed the M3 resize stress
test** badly -- when images go through a downscale-then-upscale round trip
before the crop (simulating real-world resizing/recompression), AUC
collapsed to near-chance:

| condition | NPR AUC | Δ vs clean |
|---|---|---|
| clean | 0.9194 | -- |
| resize 0.25x | 0.4819 | -0.4375 |
| resize 0.35x | 0.5434 | -0.3760 |
| resize 0.5x | 0.6230 | -0.2965 |

Go/no-go tolerance is ±0.05 AUC vs. clean -- NPR missed it by a huge margin
at every severity. Per `npr_stream_guide.md` §7/§10's own prescribed
fallback, the fix is swapping the fixed residual operator for a **learnable**
one: Bayar constrained convolution + fixed SRM filters. `NPRStream` was
already built with a `frontend=` injection point for exactly this swap (see
`src/models/npr_stream.py`), so this didn't require touching that class.

Bayar+SRM, evaluated the same way (only 2 epochs of training so far, so this
comparison isn't fully apples-to-apples -- see caveat below), is
consistently ~0.06-0.09 AUC better than NPR at every severity, but **still
fails M3**:

| condition | Bayar+SRM AUC (2 epochs) | Δ vs clean | vs. NPR |
|---|---|---|---|
| clean | 0.8973 | -- | -- |
| resize 0.25x | 0.5702 | -0.3271 | +0.088 |
| resize 0.35x | 0.6124 | -0.2849 | +0.069 |
| resize 0.5x | 0.6794 | -0.2179 | +0.056 |

**Caveat**: the Bayar+SRM checkpoint above was still improving each epoch
(0.868 -> 0.908 val AUC from epoch 1 to 2) when M3 was run against it --
unlike NPR's checkpoint, which had 3+ epochs to converge first. Whether
Bayar+SRM's robustness gap closes further with real training on the 10K set
is genuinely unknown -- that's the open question the next training run
should answer.

## Architecture

### Semantic stream (`src/models/semantic_stream.py`)

`SemanticStream(output_dim=1024, pretrained=False, freeze_backbone=True,
unfreeze_last_n_blocks=0)`. Wraps `torchvision.models.vit_b_16`, backbone
output (768-dim) projected to 1024 via a trainable `nn.Linear`. Backbone
frozen by default (only `proj` + optionally the last N transformer blocks
train). Internally `F.interpolate`s any input to 224x224, so it tolerates
whatever spatial size it's given -- but still expects **ImageNet-normalized**
input (normalization happens upstream, not inside this class).

Input: `[B, 3, H, W]`, ImageNet-normalized. Output: `[B, 1024]`.

### Forensic stream: Bayar+SRM (`src/models/frontend_bayar.py` + `src/models/npr_stream.py`)

`NPRStream(backbone="resnet_shallow", frontend=BayarSRMFrontend())` -- the
backbone (truncated ResNet-50 stem+layer1, ~0.2M params, output_dim=256; or
`"convnext_tiny"` ablation, output_dim=768) and the pooling/flatten wrapper
are unchanged from NPR; only the frontend differs.

`BayarSRMFrontend` (`[B,3,H,W]` in `[0,1]` -> `[B,3,H,W]`, same interface as
the old fixed `NPR` operator):
- `BayarConv2d`: depthwise (`groups=3`) 5x5 **learnable** constrained conv.
  Center tap hard-fixed at -1, the other 24 taps always renormalize to sum
  to +1 -- reparameterized fresh every forward call from raw learnable
  weights, so the constraint is exact and differentiable (no
  post-optimizer-step hook needed). **Bug fixed here during development**:
  the original normalization used `.clamp_min(1e-8)` on the raw-weight sum,
  which silently flips a legitimately negative sum positive and explodes the
  kernel (~10^6 magnitude instead of the constraint holding). Fixed by
  preserving sign and only flooring the *magnitude* near zero -- see the
  comment in `_constrained_kernel()` if touching this again.
- `SRMFilterBank`: 3 fixed (non-learnable, `register_buffer`) classic SRM
  high-pass kernels, applied per channel -> 9 output channels.
- Concatenate (3+9=12 channels) -> learnable `1x1 Conv2d(12, 3)` fuse ->
  `[B, 3, H, W]`. Forced fp32 (`torch.autocast(..., enabled=False)`) like
  NPR -- residual magnitudes are small enough that fp16 can underflow.

Total Bayar+SRM stream params: 225,461 (resnet_shallow backbone). Well
inside the 2B budget.

Input contract (same as NPR, and this is the part that matters most if
you're tempted to simplify anything): **raw `[0, 1]` pixels at a
native-resolution random crop, never resized**. Resizing is exactly what
M3 showed destroys the signal. This is why the forensic stream cannot share
a dataloader with the semantic stream (see Data section).

### Fusion (`src/models/fusion.py`) -- unmodified stub, not yet wired to Bayar+SRM

`FeatureFusion(semantic_dim=1024, freq_dim=768, fused_dim=512)`: concat +
`nn.Linear` + `LayerNorm` + `GELU`. `freq_dim` defaults to 768 (matches the
old `FrequencyStream`/`convnext_tiny` ablation) -- **must be passed
explicitly as 256** when pairing with Bayar+SRM's default `resnet_shallow`
backbone, e.g. `FeatureFusion(freq_dim=256)` or
`FeatureFusion(freq_dim=stream.output_dim)`.

## Data

Two separate real-data pulls from `RAID-techjam/SID_Set` on Hugging Face
(via `training/data/import_hf.py`, which streams and re-encodes at JPEG
q=95, collapsing SID_Set's 3-way label to this project's binary
`0=real/1=AI`), each shuffled via `training/data/shuffle_manifest.py`
(required -- `AIGCDataModule`/the crop dataset's train/val split is by
contiguous/seeded-random index, not re-shuffled internally, so an
unshuffled class-blocked manifest can produce a skewed val split):

- `data/sid_subset/manifest_shuffled.csv` -- 3,420 samples (~1.6% of
  SID_Set's 210K train split). Used by `configs/base_config.yaml`
  (shared: `train_semantic.py`, `train_npr.py`, `train_fusion.py`).
- `data/sid_subset_10k/manifest_shuffled.csv` -- 10,000 samples (3,338
  real / 6,662 AI), 2.4GB. Used by `configs/base_config_bayar.yaml`
  (Bayar+SRM-only, so scaling this up doesn't affect the other scripts'
  data budget). **Verified loadable** (10,000 samples, 8,000/2,000
  train/val split, `pos_weight≈0.50`) but **not yet actually trained on**.

Both live under `data/`, which is gitignored (`/data/` in `.gitignore`,
currently uncommitted -- see Known gaps).

The forensic stream's dataset class, `NPRCropDataset`, lives inside
`training/train_npr.py` (not a separate module) and is imported by
`train_bayar_srm.py`/`evaluate_bayar_srm.py` rather than duplicated --
same manifest format (`image_path,label`) as `AIGCDataset`, but loads raw
`[0,1]` pixels and takes a random native-resolution crop (reflect-padded if
the source image is smaller than the crop) instead of resizing.

## Checkpoints (as of this handoff)

| file | stream | status |
|---|---|---|
| `checkpoints/semantic_stream.pt` | ViT-B/16 semantic | trained (real, ~346MB) |
| `checkpoints/bayar_srm_stream.pt` + `bayar_srm_head.pt` | Bayar+SRM | **stale** -- only 2 epochs on the 3,420-sample set, predates the 10K pull |
| `checkpoints/npr_stream.pt` + `npr_head.pt` | NPR (abandoned) | trained, failed M3 -- kept for comparison only |
| `checkpoints/frequency_stream.pt` | old FFT stub (abandoned) | irrelevant, ignore |

Note the stream/head split: `*_stream.pt` holds only the backbone (matches
`train_semantic.py`'s convention, meant for a future `DetectorPipeline`
fine-tune), `*_head.pt` holds the standalone probe's linear classifier head
separately -- both are needed to reconstruct a working standalone predictor
(see the test/eval commands below).

## Commands

### Train

```bash
# Semantic stream (already trained -- rerun only if you need a fresh checkpoint)
# NOTE: --steps caps TOTAL optimizer steps across ALL epochs (default 2!),
# not per-epoch -- set it explicitly or --epochs silently does nothing past
# the first couple of steps. 2,736 train samples / batch 8 = 342 steps/epoch.
python -m training.train_semantic --config configs/base_config.yaml --epochs 5 --steps 1710 --log-level INFO

# Bayar+SRM forensic stream -- THE NEXT THING TO RUN, on the 10K set
python -m training.train_bayar_srm --epochs 5 --early-stopping-patience 2 --log-level INFO
# (defaults to configs/base_config_bayar.yaml, i.e. the 10K-sample set --
# no --config flag needed unless you want to point elsewhere)
```

`train_bayar_srm.py` is a deliberate near-duplicate of `train_npr.py`
(same training loop: AdamW + cosine LR, pos_weight-balanced BCE,
early stopping on val AUC, checkpoint only on best val AUC) rather than a
`--frontend` flag on one shared script -- keeps each script's CLI surface
simple. The dataset/split/eval *code* is still shared via import (not
copied), only the model construction differs.

### Evaluate / M3 resize stress test

```bash
# Standard held-out val evaluation (loss/accuracy/AUC)
python -m training.test_npr --config configs/base_config.yaml   # NPR only, no bayar_srm equivalent exists yet

# M3 resize/downscale stress test (the real robustness check)
python -m training.evaluate_npr --config configs/base_config.yaml           # against NPR's checkpoint
python -m training.evaluate_bayar_srm                                       # against Bayar+SRM's checkpoint, defaults to base_config_bayar.yaml
```

`evaluate_bayar_srm.py` re-runs the exact same held-out val samples (same
seed) under `clean` + each `configs/augmentations.yaml`
`eval.downscale_levels` severity (0.25/0.35/0.5 -- a real PIL
downscale-then-upscale round trip applied to the *full native image before
cropping*, never after), and prints a PASS/FAIL verdict per severity against
a ±0.05 AUC tolerance vs. clean. No retraining needed -- pure inference
against whatever checkpoint already exists.

## Known gaps / next steps, in priority order

1. **Retrain Bayar+SRM on the 10K set** (`python -m training.train_bayar_srm
   --epochs 5 --early-stopping-patience 2`), then re-run
   `python -m training.evaluate_bayar_srm` to see if M3 actually passes with
   a properly-converged, more-data checkpoint. This is the open question the
   whole Bayar+SRM pivot exists to answer.
2. **`src/models/detector.py` still imports `FrequencyStream`**, not
   `NPRStream`/Bayar+SRM at all. `DetectorPipeline.__init__`'s second-stream
   parameter is named `frequency_stream` and defaults to the old FFT stub.
   Needs updating to default to `NPRStream(frontend=BayarSRMFrontend())` (or
   accept injection the way it already does) if you want the fused pipeline
   to reflect the actual current architecture.
3. **`training/train_fusion.py` wires plain `NPRStream`** (not
   `BayarSRMFrontend`), loading `checkpoints/npr_stream.pt`. It freezes both
   streams and trains only `fusion` + `classifier`, using a "compromise"
   input (the semantic stream's resized+ImageNet-normalized tensor,
   denormalized back to `~[0,1]` for the forensic stream inside
   `DetectorPipeline.forward` -- NOT the forensic stream's real
   native-resolution crop path, since one shared tensor can't satisfy both
   input contracts simultaneously). This script needs its model-construction
   lines swapped to Bayar+SRM and probably a checkpoint-path update once (1)
   is done. Expect this compromise to underperform `evaluate_bayar_srm.py`'s
   real native-crop numbers, same caveat as NPR's fusion attempt.
4. **`.gitignore`'s `/data/` fix is uncommitted** (`git status` shows ` M
   .gitignore`). Worth committing so the 2.4GB `data/sid_subset_10k/` never
   risks being staged.
5. No `test_bayar_srm.py` equivalent to `test_npr.py` exists yet (plain
   held-out loss/accuracy/AUC without the M3 degradation sweep) -- low
   priority, `evaluate_bayar_srm.py`'s `clean` condition already covers this.

## Gotchas worth knowing before changing anything here

- **Never resize the forensic stream's input.** Every script in this stream
  (`NPRCropDataset`, `BayarSRMFrontend`, `NPRStream`) is built around raw
  `[0,1]` native-resolution crops. Feeding it a resized tensor is exactly
  the failure mode M3 measures -- it's not a style preference.
- **fp32 forcing**: both `NPR` (abandoned) and `BayarSRMFrontend` wrap their
  forward pass in `torch.autocast(..., enabled=False)`. Residual magnitudes
  are small enough that fp16/autocast can underflow them to zero.
- **`__name__` vs `logging.getLogger`**: `train_npr.py`, `train_bayar_srm.py`,
  etc. use a hardcoded logger name (`"training.train_npr"`, not `__name__`)
  because `python -m training.X` sets that module's `__name__` to
  `"__main__"`, which would silently detach its logs from the `"training"`
  logger hierarchy `setup_logger()` configures. Keep this pattern in any new
  entry-point script.
- **`AIGCDataModule`'s train/val split is contiguous**, not shuffled --
  that's why manifests need `shuffle_manifest.py` run on them first. The
  forensic stream's own split (`_split_dataset` in `train_npr.py`) uses a
  seeded `random_split` instead and doesn't have this problem, but the
  manifest itself should still be pre-shuffled for consistency/safety.
