# About RAID

**RAID — Robust AI-generated Image Detector.** Catch AI fakes before they slip past you.

## Inspiration

AI-generated imagery got good — fast. A year ago you could spot the extra
fingers; today diffusion models render skin pores and correct shadows better
than most people notice on a phone screen. That gap between "how convincing
AI images are" and "how equipped people are to question them" is widening
every month, and it isn't landing evenly. Our parents and grandparents grew
up trusting that a photo was proof something happened. They didn't grow up
scrolling past a hundred synthetic images a day, and they're exactly the
audience targeted by AI-generated scam ads, fake product photos, and
fabricated "news" images shared in family group chats.

We wanted to build something that could sit between an image and a person
and just say, plainly: *this was probably made by AI.* Not a research paper,
not a benchmark leaderboard entry — a tool simple enough that someone who's
never heard of a GAN could upload a photo and get a straight answer.

## What we learned

- **AI-generated images leave two different kinds of fingerprints.** High-level
  semantic irrelevancies (impossible anatomy, warped text, physically
  inconsistent lighting) are one signal; low-level generation artifacts
  hiding in the frequency domain (upsampling checkerboard patterns, unnatural
  spectral energy distributions from GAN/diffusion decoders) are a completely
  different one. Neither signal alone is robust — a model trained purely on
  semantic cues gets fooled by well-composed AI art, and a model trained
  purely on frequency artifacts collapses the moment an image is
  re-compressed. That's what pushed us toward a **dual-stream architecture**
  instead of a single classifier.
- **Robustness has to be a first-class design constraint, not an
  afterthought.** Real images people actually encounter have been through
  JPEG re-compression, screenshotted, cropped, resized, and color-corrected
  half a dozen times before they ever reach a detector. A model that only
  ever sees pristine training images looks great on a benchmark and falls
  apart on a WhatsApp forward. We had to bake degradation (JPEG Q30-90,
  Gaussian blur, 0.25x-0.5x rescaling, Gaussian noise, color jitter, 80%
  cropping) into the training and evaluation loop from day one, not bolt it
  on at the end.
- **A hard parameter budget forces better decisions.** Capping the whole
  architecture near ~337M parameters (well under a 2B ceiling) meant we
  couldn't just bolt together the biggest foundation models available. We
  had to be deliberate about freezing backbones, projecting down to compact
  shared feature dimensions, and asking whether a bigger model actually
  bought us robustness or just latency.
- **Clean interfaces are what let a small team move fast.** None of us could
  wait for someone else's backbone to be "done" before starting our own
  piece.

## How we built it

We split the problem the same way the architecture is split, and it mapped
cleanly onto the team:

- **Data & Augmentations** — a dataset loader (`AIGCDataset`) plus an
  Albumentations-based `RobustnessTransforms` pipeline that reproduces the
  real-world degradations above, with a stochastic training mode and an
  isolated-severity eval mode for plotting degradation curves. We also built
  a streaming Hugging Face dataset importer so we could pull a real labeled
  AI/real image subset without downloading an entire dataset up front.
- **Feature Extraction** — two independent streams behind a shared
  `BaseFeatureStream` interface: a **semantic stream** (a ViT backbone
  projected to a 1024-d vector, with configurable partial fine-tuning of the
  last few transformer blocks) that reasons about high-level content and
  composition, and a **frequency stream** (2D FFT magnitude spectrum feeding
  a lightweight ConvNeXt-style backbone, projected to 768-d) that picks up
  the low-level spectral artifacts generative decoders leave behind.
- **Fusion & Classification** — a fusion layer that merges both feature
  vectors into a shared representation feeding a compact classification
  head, returning a single logit/probability plus the fused feature vector
  for downstream explainability.
- **Explainability & Interface** — a `predict.py` CLI for single-image and
  batch inference, and a Gradio app so a non-technical user can drag in a
  photo and get a probability and a saliency overlay showing *why* the model
  thinks what it thinks — because "trust me" isn't good enough when the
  audience is exactly the people being targeted by scams.

The whole thing was built stub-first: every module honored its tensor
contract (`[B, 3, H, W] -> [B, D]` for each stream, a fixed output dict for
the detector) from hour one, using lightweight mock networks in place of the
real backbones. That let data, model, evaluation, and explainability work
happen in parallel on the same interfaces instead of one person blocking the
other three.

## Challenges we faced

- **Fitting real backbones into the parameter budget** without sacrificing
  the robustness we were designing for — every architecture decision was a
  three-way trade-off between accuracy, robustness, and parameter count.
- **Sourcing labeled AI-vs-real image data** fast enough for a hackathon
  timeline. Streaming a Hugging Face dataset and remapping its label scheme
  onto our binary real/AI contract (some datasets label real, fully
  synthetic, *and* tampered separately) was its own small integration
  project.
- **Keeping two independently-developed feature streams honest to the same
  contract** so fusion wouldn't require a rewrite the moment either backbone
  changed — this is where the abstract base interface and disciplined tensor
  shape testing earned its keep.
- **Balancing robustness against everything at once.** Tuning augmentation
  severity so the model gets meaningfully harder to fool without also making
  it too hard to train in the time we had was a constant back-and-forth.
- **Designing for our actual target user.** It's easy to build a tool for
  people who already know what a "confidence score" means. Building
  something legible to someone who just wants a straight "real or fake"
  answer — clear labels, a simple probability, a visual explanation instead
  of a wall of numbers — took as much thought as the model itself.

## What's next

- **Finish joint fusion training.** The semantic and NPR streams have each
  been validated independently (NPR alone is already hitting ~0.91 val AUC),
  but they haven't been fine-tuned together through `fusion.py` yet — that's
  the step that turns "two decent models" into one detector that's actually
  better than either alone.
- **Settle the frequency branch.** We need to run the resize/downscale
  stress test on the NPR stream and decide, for real, whether NPR alone
  holds up or whether we fall back to the Bayar constrained-conv + SRM
  frontend we already built a swap-in point for.
- **Load a real checkpoint into the app.** `app.py` still runs on stub
  weights — wiring the trained, fused checkpoint into the Gradio demo (and
  the CLI) is what turns this from "a set of validated components" into
  something a non-technical user can actually try.
- **Ship real explainability, not a placeholder.** The saliency overlay in
  the demo today is a mock. We've already built the harder, model-independent
  half (metrics, calibration, rendering, contracts) — what's left is Grad-CAM,
  attention rollout, and branch-contribution analysis wired to the real
  fused model, so "this was probably made by AI" comes with a "here's why."
- **Run the full robustness benchmark end-to-end.** We've built the
  degradation transforms and the eval scaffolding; we haven't yet generated
  the actual degradation curves (accuracy vs. JPEG quality, blur sigma, crop
  ratio) that prove the "robust" in RAID holds up on the full spectrum,
  not just spot checks.
- **Scale past the smoke-test dataset.** Training so far has used a partial,
  hackathon-sized pull from SID_Set. Pulling the full dataset and retraining
  at scale is the biggest lever left for real accuracy gains.
- **Add CI.** Once the architecture stabilizes, wire `pytest tests/` into
  GitHub Actions so regressions get caught before they hit a checkpoint.

## Built With

#python #pytorch #torchvision #vit #convnext #albumentations #huggingface #datasets #gradio #scikitlearn #numpy #pillow #pyyaml #pytest
