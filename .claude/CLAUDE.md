### `CLAUDE.md` for the `ai-image-detector` Codebase

Below is the raw Markdown content for the `CLAUDE.md` file tailored specifically to the project blueprint. Save this as `CLAUDE.md` in the root of your repository.

# CLAUDE.md - AI Image Detector Project Guidelines

## Project Overview

This repository contains a modular prototype for distinguishing AI-generated images from authentic photographs with high robustness against real-world post-processing (JPEG Q=30-90, Gaussian Blur, 0.25x Rescaling, Gaussian Noise, Color Jitter, and 80% Cropping).

## Core Project Constraints

- **Parameter Budget**: Total model architecture MUST stay under **2 Billion parameters** (target architecture is ~337M parameters).
- **Python Version**: Python 3.10+
- **Primary Stack**: PyTorch, PyTorch Lightning / Albumentations, OpenCV, timm, pytest.
- **Hardware Assumption**: Hackathon-scale single GPU execution. Optimize for parameter efficiency and fast iteration.

---

## Directory Map & Module Ownership

- `configs/`: YAML configurations for base hyperparameters, model architecture, and robustness transformations.
- `src/data/`: Dataset loaders (`AIGCDataset`) and Albumentations robustness transform pipelines (`RobustnessTransforms`).
- `src/models/`:
- `base_stream.py`: Abstract class `BaseFeatureStream`. DO NOT modify without team consensus.
- `semantic_stream.py`: Stream 1 (High-Level ViT / DINOv2 wrapper).
- `frequency_stream.py`: Stream 2 (Mid-Level 2D FFT + ConvNeXt-Tiny wrapper).
- `fusion.py`: Cross-attention / Feature fusion layer.
- `detector.py`: End-to-end `DetectorPipeline` PyTorch Lightning module.

- `src/evaluation/`: Automated benchmark runner testing performance across degradation spectrums.
- `src/explainability/`: Grad-CAM and ViT attention heatmap visualizers.

---

## Key Development & Execution Commands

### Environment Setupbash

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

````

### Running Unit & Integration Tests
```bash
# Run all tests (ALWAYS run before committing changes)
pytest tests/

# Test specific modules
pytest tests/test_data.py
pytest tests/test_models.py
pytest tests/test_evaluation.py

````

### Training

```bash
# Train detector using default config
python train.py --config configs/base_config.yaml

# Train with specific batch size and learning rate
python train.py --config configs/base_config.yaml --batch_size 32 --lr 1e-4

```

### Robustness Evaluation

```bash
# Run benchmark across all transform suites (JPEG, Blur, Rescale, Noise, Crop)
python evaluate.py --checkpoint checkpoints/best_model.ckpt --config configs/augmentations.yaml

```

### Single Image Inference & Explainability

```bash
# Predict probability for a single image and output saliency heatmap
python predict.py --image path/to/sample.jpg --checkpoint checkpoints/best_model.ckpt --save_heatmap

```

---

## Data Contracts & Interface Specifications

All team members must maintain strict compatibility with these tensor formats:

1. **Dataset Tensor Format**:

- Input shape: `[B, 3, 512, 512]` (RGB, normalized using ImageNet mean/std).
- Label shape: `[B, 1]` where `0.0 = Real` and `1.0 = AI-Generated`.

2. **Feature Stream Interface (`BaseFeatureStream`)**:

- `forward(x: torch.Tensor) -> torch.Tensor`
- Input: Image tensor `[B, 3, H, W]`.
- Output: Feature vector `[B, D]` (e.g., `D1=1024` for Semantic, `D2=768` for Frequency).

3. **Detector Output Dictionary**:
   `DetectorPipeline.forward()` MUST return a Python dictionary with the following keys:

```python
{
    "logit": torch.Tensor,       # [B, 1] Raw output logit
    "prob": torch.Tensor,        # [B, 1] Sigmoid probability [0.0, 1.0]
    "features": torch.Tensor     # [B, D_fused] Fused vector representation
}

```

---

## Coding Guidelines & Conventions

- **Type Hints**: Explicitly type-hint all function arguments and returns (e.g., `def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:`).
- **No Direct Hardware Hardcoding**: Always use `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
- **Mocking During Development**: When testing fusion or pipeline logic without GPU access, use lightweight stub networks (e.g., `nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())`) to mock feature streams.
- **Transform Normalization**: Apply image augmentations BEFORE tensor normalization (`ToTensorV2()`).

---

## Verification Criteria Before Marking Tasks Complete

When modifying code in this repository:

1. Verify tensor shape consistency across Stream 1, Stream 2, and Fusion.
2. Run `pytest tests/` to confirm no core data contracts are broken.
3. Ensure no API keys, credentials, or absolute file paths are hardcoded.
