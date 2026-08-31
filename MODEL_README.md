# RAID Image Detector Models

Experimental AI-image detection models trained on the SID_Set dataset.

## Model files

- `semantic_stream.pt`: pretrained ViT-B/16 semantic stream checkpoint.
- `bayar_srm_stream.pt`: Bayar+SRM low-level forensic stream checkpoint.
- `bayar_srm_head.pt`: standalone Bayar+SRM classifier head for low-level evaluation.
- `detector_fusion.pt`: fusion/classifier checkpoint trained with frozen semantic and Bayar+SRM streams.
- `base_config.yaml`: training configuration.

The stream checkpoints and fusion checkpoint must be used with the matching source
code from the RAID repository. These are experimental weights, not a production
detector.

## Download the models

```powershell
python -m pip install huggingface_hub
hf auth login
hf download RAID-techjam/raid-detector-fusion --repo-type model --local-dir checkpoints
```

This downloads the files into the local `checkpoints` directory.

## Install the project

```powershell
git clone https://github.com/Beastarz/RAID.git
cd RAID
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For an NVIDIA GPU, install the CUDA-enabled PyTorch wheel appropriate for the
machine before installing the remaining requirements. CPU execution also works,
but is considerably slower.

## Run standalone Bayar+SRM evaluation

The Bayar+SRM stream expects native-resolution crops with raw RGB values in
`[0, 1]`. It is not interchangeable with the semantic stream's normalized input.
Evaluation requires a local CSV manifest with this format:

```csv
image_path,label
data/example.jpg,0
data/generated.jpg,1
```

Labels are binary: `0` is authentic and `1` is AI-generated or manipulated.

Set the manifest path in `configs/base_config_bayar.yaml`, then run:

```powershell
python -m training.evaluate_bayar_srm `
  --config configs/base_config_bayar.yaml `
  --crop-size 256 `
  --backbone resnet_shallow
```

## Reproduce fusion training

The fusion stage loads the semantic and Bayar+SRM stream checkpoints, freezes
both streams, and trains only the fusion and classifier layers:

```powershell
python -m training.train_fusion `
  --config configs/base_config.yaml `
  --semantic-checkpoint checkpoints/semantic_stream.pt `
  --frequency-checkpoint checkpoints/bayar_srm_stream.pt `
  --epochs 5 `
  --steps 0
```

The resulting checkpoint is written to `checkpoints/detector_fusion.pt`.

## Important limitation

The current `predict.py` entry point still targets the original semantic plus
frequency-stream `DetectorPipeline`. The Bayar-aware fusion checkpoint is not
yet wired into that CLI. Use the training/evaluation scripts above until a
Bayar-aware inference wrapper is added.

## Reported experiment

On a 10,000-image SID_Set subset, the Bayar+SRM model reached approximately
`0.8738` clean validation AUC. Resize robustness improved after training with
resize augmentation, reaching approximately `0.8459` AUC at 0.5 scale, while
more aggressive 0.25 and 0.35 scale conditions remained challenging.
