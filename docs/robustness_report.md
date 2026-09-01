# Robustness & Error Analysis Report

Generated against the published bundle (`checkpoints/detector_bundle.pt`,
built from `RAID-techjam/raid-detector-fusion`) on **200 held-out images**
from SID_Set's `validation` split — disjoint from the `train` split the
model was fit on. See [Reproducing this report](#reproducing-this-report)
for exact commands; every number below traces back to
`outputs/robustness/predictions.jsonl`, not hand-computed figures.

**Sample size caveat, stated once up front rather than per table**: n=200
is enough to see real, consistent signal (the confusion-matrix and
error-breakdown findings below hold up), but it's too small for some slices
to be meaningful — most visibly the resolution breakdown, where 199 of 200
images landed in a single bucket. Treat this as a first real report from a
working harness, not a final benchmark.

## Key findings

1. **The deployed threshold (0.5) is materially uncalibrated.** At the
   published threshold, 25% of real photos are flagged as AI-generated. An
   ROC-chosen threshold gives the *same* recall at less than half that false
   positive rate. This is the single most actionable finding here.
2. **Tampered images are the model's weak point, not fully-synthetic
   ones.** 0% of fully-synthetic images were missed; 14.1% of tampered
   images were. SID_Set's tampered class gets folded into the same "AI"
   label as fully-synthetic during training, and that merge is hiding a real
   accuracy gap between the two.
3. **The resolution-shortcut question wasn't actually tested.** Not because
   the model passed it, but because this particular 200-image pull doesn't
   span SID_Set's stated resolution range. This needs a resolution-aware
   sample, not a bigger random one, to answer honestly.
4. Every design choice audited in [Table 5](#table-5--design-trade-offs) has
   real evidence for its *direction* but not its *counterfactual* — none of
   them were tested against the alternative that was passed over.

## Confusion matrices at the published threshold

### Clean (`clean`, n=200)

|  | Predicted: AI | Predicted: Real |
|---|---:|---:|
| **Actual: AI** | TP = 123 | FN = 9 |
| **Actual: Real** | FP = 17 | TN = 51 |

Precision = 0.8786 · Recall (TPR) = 0.9318 · **FPR = 0.2500** · Accuracy = 0.8700 · threshold = 0.5

### Worst condition by AUC (`crop` at 80%, n=200)

|  | Predicted: AI | Predicted: Real |
|---|---:|---:|
| **Actual: AI** | TP = 121 | FN = 11 |
| **Actual: Real** | FP = 15 | TN = 53 |

Precision = 0.8897 · Recall (TPR) = 0.9167 · FPR = 0.2206 · Accuracy = 0.8700 · threshold = 0.5

### Threshold calibration note

The published threshold is a hardcoded `0.5` default (`DECISION_THRESHOLD`
in `src/models/checkpoint_bundle.py`), not calibrated against any ROC curve.
An ROC-chosen operating point on the clean condition (Youden's J) would
instead use **threshold≈0.864**, giving **TPR≈0.9167 / FPR≈0.1029** on clean
data — the same recall as the published threshold's clean-condition
performance, at under half the false-positive rate. That gap (25.0% → 10.3%
FPR) is the concrete cost of shipping an uncalibrated default.

## Error breakdown by image property

Condition: `clean`, n=200. Resolution bucket uses the image's longer side.

### By native resolution

| Resolution | Class | n | FP rate | FN rate |
|---|---|---:|---:|---:|
| 512–1024 | real | 1 | 0.0000 | – |
| 512–1024 | AI | 0 | – | n/a |
| 1024–2048 | real | 67 | 0.2537 | – |
| 1024–2048 | AI | 132 | – | 0.0682 |

**Not usable as evidence**: 199 of 200 images fall in the 1024–2048px
bucket; the `<512` and `>2048` buckets are empty and `512–1024` has a single
sample. SID_Set's card claims a 338–6020px range, so either this validation
slice is more resolution-uniform than the full dataset, or a larger/
resolution-stratified pull is needed before this diagnostic says anything.
Reporting the near-empty buckets as a finding would overstate what 200
random samples actually tested.

### By original SID_Set label (0=real, 1=fully synthetic, 2=tampered → folded into AI)

| SID_Set label | n | Error rate | What it measures |
|---|---:|---:|---|
| 0 (real) | 68 | 0.2500 | FP rate |
| 1 (fully synthetic) | 68 | **0.0000** | FN rate |
| 2 (tampered) | 64 | **0.1406** | FN rate |

This slice *is* well-populated (all three sub-classes have 64+ samples) and
shows a real, non-trivial gap: the model essentially never misses a
fully-synthetic image, but misses roughly 1 in 7 tampered ones. Binary
accuracy/AUC numbers that only report the folded label hide this.

## Design trade-offs

| Choice | What it buys | What it costs | Evidence |
|---|---|---|---|
| Shared 512×512 resize feeding both branches (`prepare_fused_inputs`) | One deterministic, parity-checked preprocessing contract instead of two; simpler to validate, ship, and explain to the explainability adapter | The forensic branch's signal is, by the project's own stated mechanism, partially destroyed by the resize — a generator's upsampling artifact gets overwritten by whatever resampling kernel produced the 512×512 image | **Not measured.** No ablation exists comparing this shared-resize forensic branch against a native-crop forensic branch inside the fused model. The mitigation actually taken was training the forensic branch on augmented resized images (`train_fusion.py`), not avoiding the resize. |
| Concat + linear fusion, not cross-attention | Simple, fast, already trained and shipped; retraining risk avoided | Caps how much the two branches can interact — a linear projection can't learn conditional "trust semantic more when forensic is ambiguous" behavior the way attention could | Not measured against a cross-attention alternative; documented in `ARCHITECTURE.md`/`TODO.md` as a deliberate scope cut, not a tested comparison |
| Shallow ResNet-50 (stem+layer1) forensic backbone over ConvNeXt-Tiny | ~0.2M params vs. ~28M — 140× smaller, comfortably inside the parameter budget, faster inference | Less representational capacity if the forensic signal turns out to need deeper features than a 2-layer stem provides | `docs/../outputs/robustness/table6_model_and_compute.md` confirms the 0.2M param count; no head-to-head clean/robust AUC comparison against the ConvNeXt-Tiny ablation was run in this pass |
| Bayar+SRM frontend over plain NPR | At every resize severity tested standalone (both frontends pretrained identically, with zero augmentation), Bayar+SRM's learnable filter degraded less badly than NPR's fixed one | Bayar+SRM adds ~150 learnable params over NPR's zero, and — per the standalone sweep — still failed the organizers' resize conditions outright, just less badly than NPR | `training/evaluate_npr.py` / `training/evaluate_bayar_srm.py`, both under the same unaugmented pretraining handicap (see `CLAUDE_HANDOFF.md`). Only supports "Bayar+SRM generalized better to an unseen resize than NPR did" — not "Bayar+SRM would still win if both were resize-augmented," which is untested. Note the *fused* model (Table 1) is far more resize-robust than either standalone checkpoint, which is a separate effect (see the augmentation row below). |
| Decision threshold fixed at 0.5 | One documented, manifest-enforced constant; nothing to drift between environments | Not calibrated to any validation ROC curve — Table 3 shows this costs roughly 2.4x the false-positive rate a calibrated threshold would give at the same recall | **Measured in this report** (Table 3) — the one row here where the cost is no longer hypothetical. |
| Fusion-head training on JPEG/blur/downscale/noise/jitter/crop-augmented data (`train_fusion.py` via `AIGCDataModule`) | The published forensic branch, unlike its standalone pretraining checkpoint, was actually exposed to degradation during training — and Table 1's flat AUC across every condition suggests this worked | Augmentation strength (`p_each=0.5` per transform, independently) wasn't itself tuned against a robustness target; no ablation isolates how much of the fused model's robustness this training choice is responsible for versus the architecture itself | `training/data/augmentations.py`'s `RobustnessTransforms`; Table 1 (`outputs/robustness/table1_robustness_sweep.md`) is the evidence for the combined effect, not this choice's isolated contribution |

**Reading this table**: most rows have real evidence behind the *direction*
of the claim but not the *counterfactual* (what the alternative would have
scored) — writing "not measured" here rather than omitting untested
ablations is the honest version of this table. See `CLAUDE_HANDOFF.md`'s
"Known gaps" section for the full list, including ones outside this report's
scope (compression-history control, split stratification, seed variance).

## Reproducing this report

```bash
python -m training.import_eval_manifest --split validation --output data/sid_eval --limit 200
python -m training.build_detector_bundle \
  --semantic-checkpoint checkpoints/semantic_stream.pt \
  --forensic-checkpoint checkpoints/bayar_srm_stream.pt \
  --fusion-checkpoint checkpoints/detector_fusion.pt \
  --output checkpoints/detector_bundle.pt --parity-image test_sample.jpg
python -m training.robustness_sweep \
  --manifest data/sid_eval/manifest.csv --checkpoint checkpoints/detector_bundle.pt \
  --output outputs/robustness/predictions.jsonl
python -m training.tables.table1_robustness_sweep --predictions outputs/robustness/predictions.jsonl
python -m training.tables.table3_confusion_matrices --predictions outputs/robustness/predictions.jsonl
python -m training.tables.table4_error_breakdown --predictions outputs/robustness/predictions.jsonl
```

Increase `--limit` in the first command for a larger (and, for the
resolution slice specifically, ideally stratified) sample — the sweep and
table scripts need no changes to scale up.
