# Final detector contract

This is the Wave 4 handoff for the published fused detector. The authoritative
runtime copy is the `manifest` embedded in the self-describing bundle produced
by `python -m training.build_detector_bundle`.

## Topology

- `semantic`: `SemanticStream` backed by torchvision ViT-B/16, projected to
  1024 features. Its trained internal input resize is 224×224 bilinear.
- `forensic`: one `NPRStream` instance configured with the `BayarSRMFrontend`
  and shallow ResNet backbone, producing 256 features. Bayar and SRM are
  forensic internals, not independent detector branches.
- Fusion is concatenation → linear projection → LayerNorm → GELU to 512
  features, followed by a 512 → 128 → 1 classifier.
- Canonical branch names are exactly `semantic` and `forensic`.

## Preparation and scoring

Decode the source as RGB and resize it once to 512×512 with Pillow bilinear
interpolation (`Image.Resampling.BILINEAR`). Convert those pixels to float32 `[0,1]` values. Pass the shared
pixels to the forensic branch and derive the semantic branch by ImageNet
normalization (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`). The model
returns a raw AI logit; the decision threshold is `0.5`, with equality treated
as AI-generated.

The bundle manifest records source filenames, sizes, SHA-256 provenance,
`weights_id`, and a deterministic digest of every embedded state tensor. The
loader verifies those values before strict loading. Bundle creation can (and
for the published release should) receive `--parity-image` to compare against
an independently loaded three-file scorer before the bundle is exposed.

## Explainability handoff

Supported target paths are:

- Semantic attribution/Grad-CAM: `semantic_stream.backbone.encoder.layers.encoder_layer_11.ln_1`.
- Forensic attribution/Grad-CAM: `forensic_stream.backbone.4.2.conv3`.
- Forensic intermediates: `frontend.bayar`, `frontend.srm`, `frontend.fuse`,
  `backbone.4`, and `pool`.
- Semantic token-grid handling must remove the CLS token and reshape the 196
  patch tokens to 14×14.

Plain attention rollout is unsupported because the current torchvision ViT
forward path does not expose attention matrices. Branch-subset logits are also
unsupported until a bundle explicitly records a feature-ablation baseline;
zeros must not be used as an implicit baseline.

The strict bundle loader resolves every declared target and intermediate path
against the canonical module tree before returning the detector.
