# Base Project Blueprint: Modular AI Image Detector

This blueprint provides a clean, decoupled skeleton codebase for a hackathon team. It abstracts specific neural network models behind pluggable interfaces, allowing teammates to work in parallel on data pipelines, feature extraction backbones, fusion layers, evaluation benchmarks, and explainability modules without breaking each other's code.

---

## 1. Repository Directory Structure

ai-image-detector/
├── configs/
│ ├── base_config.yaml # Global parameters (batch size, image size, learning rate)
│ ├── augmentations.yaml # Robustness test transform parameters
│ └── model_config.yaml # Hyperparameters for streams and fusion
├── src/
│ ├── **init**.py
│ ├── data/ # Team Member A: Data & Augmentations
│ │ ├── **init**.py
│ │ ├── dataset.py # Custom Dataset class loading Real/AI images
│ │ ├── augmentations.py # Robustness transform pipeline (Albumentations)
│ │ └── datamodule.py # PyTorch DataLoader wrapper
│ ├── models/ # Team Member B: Network Architecture
│ │ ├── **init**.py
│ │ ├── base_stream.py # Abstract class for extraction streams
│ │ ├── semantic_stream.py # Stream 1: High-level semantic backbone wrapper
│ │ ├── frequency_stream.py # Stream 2: Mid-level/frequency domain backbone
│ │ ├── fusion.py # Cross-stream feature fusion layer
│ │ └── detector.py # End-to-end PyTorch Lightning / PyTorch Module
│ ├── evaluation/ # Team Member C: Benchmark & Metrics
│ │ ├── **init**.py
│ │ ├── robustness_suite.py # Automated robustness test runner across transforms
│ │ └── metrics.py # Accuracy, ROC-AUC, F1, degradation curve tools
│ └── explainability/ # Team Member D: Diagnostic Tools
│ ├── **init**.py
│ └── visualizer.py # Heatmap & attention map generation
├── tests/ # Integration and unit tests
│ ├── test_data.py
│ ├── test_models.py
│ └── test_evaluation.py
├── train.py # Main training script
├── evaluate.py # Main evaluation CLI against benchmark transforms
├── predict.py # Single-image / Batch inference script
├── requirements.txt # Dependencies
└── README.md # Setup and usage guide

---

## 2. Core Data Contracts & Tensor Flow

To ensure all components seamlessly connect, teammates must strictly follow these standardization contracts:

| Stage                   | Input Artifact               | Output Artifact                      | Interface / Function            |
| ----------------------- | ---------------------------- | ------------------------------------ | ------------------------------- |
| **Data Ingestion**      | Raw image path (`str`)       | Tensor `[B, 3, H, W]`                | `ImageDataset.__getitem__()`    |
| **Augmentation**        | Clean Tensor `[B, 3, H, W]`  | Transformed Tensor `[B, 3, H, W]`    | `AugmentationEngine.apply()`    |
| **Stream 1 Extraction** | Tensor `[B, 3, H, W]`        | Feature Tensor `[B, D1]`             | `SemanticStream.forward()`      |
| **Stream 2 Extraction** | Tensor `[B, 3, H, W]`        | Feature Tensor `[B, D2]`             | `FrequencyStream.forward()`     |
| **Fusion Layer**        | Tensors `([B, D1], [B, D2])` | Unified Vector `[B, D_fused]`        | `FusionModule.forward()`        |
| **Classification**      | Vector `[B, D_fused]`        | Logit `[B, 1]` / Prob `[0.0 to 1.0]` | `ClassificationHead.forward()`  |
| **Explainability**      | Image Tensor + Model         | Spatial Saliency Matrix `[H, W]`     | `Visualizer.generate_heatmap()` |

---

## 3. Modular Component Blueprint

### A. Data & Augmentation Module (`src/data/`)

**Responsibility:** Load dataset pairs, apply standardized spatial cropping, and simulate real-world degradations (JPEG compression, blur, noise, downscaling) during both training and robustness testing.

Key classes to implement:

- `RobustnessTransforms`: Configurable wrappers using `Albumentations` or `TorchVision` to dynamically inject JPEG degradation (quality in [30, 90]), Gaussian Blur (sigma in [0.5, 2.0]), Rescaling (0.25x to 0.5x), Gaussian Noise, Color Jitter, and Center Cropping (80%).
- `AIGCDataset`: PyTorch `Dataset` that reads a manifest CSV (`image_path, label`) where real images are labeled `0` and AI-generated images are labeled `1`.

### B. Feature Extraction Streams (`src/models/`)

**Responsibility:** Implement two isolated extraction backbones that convert input image tensors into fixed-size feature vectors.

1. **Abstract Base Interface (`base_stream.py`)**:
   Defines a standardized PyTorch interface `BaseFeatureStream` with an abstract method `extract_features(x: torch.Tensor) -> torch.Tensor`.
2. **Stream 1: High-Level Semantic Stream (`semantic_stream.py`)**:

- _Purpose_: Extracts deep semantic and contextual representations resilient to surface noise and downsampling.
- _Wrapper Contract_: Accepts `[B, 3, H, W]`, outputs a high-level representation vector `[B, D1]`. Any suitable vision foundation backbone can be plugged into this class without changing downstream code.

3. **Stream 2: Mid-Level / Frequency Stream (`frequency_stream.py`)**:

- _Purpose_: Processes mid-frequency spectral features or texture residuals that survive spatial compression.
- _Wrapper Contract_: Converts spatial tensors `[B, 3, H, W]` to frequency representations (e.g., using 2D Fast Fourier Transforms or High-Pass Filters) before passing them to a lightweight backbone to output `[B, D2]`.

### C. Fusion & Classification Head (`src/models/fusion.py` & `detector.py`)

**Responsibility:** Combine feature representations from Stream 1 and Stream 2, process them through a fusion layer (e.g., Concatenation, Cross-Attention, or Bilinear Pooling), and predict the final probability score.

Key classes to implement:

- `FeatureFusion`: Accepts `feat_stream1` and `feat_stream2`, merges them, and applies normalization.
- `DetectorPipeline`: Main wrapper holding Stream 1, Stream 2, Fusion, and a final Multi-Layer Perceptron (MLP) classification head. Returns a dictionary containing prediction logits, probabilities, and intermediate feature vectors:
  `Output Dict = {'logit': Tensor[B, 1], 'prob': Tensor[B, 1], 'features': Tensor[B, D_fused]}`

### D. Evaluation Suite (`src/evaluation/`)

**Responsibility:** Automate evaluation across specific transformation suites to compute degradation curves (e.g., Accuracy vs. JPEG Quality Factor).

Key tasks to implement:

- `RobustnessBenchmark`: Runs the model against transformed test sets and outputs performance metrics per transform severity:
- Accuracy
- ROC-AUC
- False Positive Rate (FPR) at 95% True Positive Rate (TPR)

- Generates structured output tables (CSV/JSON) summarizing metric degradation across transformation parameters.

### E. Explainability Module (`src/explainability/`)

**Responsibility:** Provide diagnostic maps indicating which image patches or features triggered the classification score.

Key tasks to implement:

- `AttributionVisualizer`: Uses spatial gradient attribution (e.g., Grad-CAM) or cross-attention token weights to project importance scores back onto the original image space.
- Saves diagnostic visual outputs overlaid on input images for error analysis.

---

## 4. Parallel Workstreams & Task Allocation

To maximize team efficiency during a hackathon, team members can work simultaneously on separate modules using mock interfaces:

| Team Member  | Module Focus                    | Primary Deliverables                                                    | Independence Strategy (Mocking)                                                                                                               |
| ------------ | ------------------------------- | ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Member A** | **Data & Augmentations**        | `dataset.py`, `augmentations.py`, transform parameters configuration    | Can build data loaders and test transformations on dummy images independently of model development.                                           |
| **Member B** | **Model & Fusion Architecture** | `semantic_stream.py`, `frequency_stream.py`, `fusion.py`, `detector.py` | Can build backbone wrappers using lightweight stub networks (e.g., simple CNNs) to verify tensor shapes before dropping in pretrained models. |
| **Member C** | **Evaluation & Robustness**     | `robustness_suite.py`, `metrics.py`, `evaluate.py`                      | Can write the evaluation benchmark runner using a dummy model that outputs random logits to ensure metric calculation works.                  |
| **Member D** | **Explainability & Scripts**    | `visualizer.py`, `predict.py`, CLI interface, documentation             | Can develop visualization scripts using random gradient tensors or mock spatial attention maps.                                               |

---

## 5. Execution Strategy

1. **Step 1: Set Up Interfaces**: Clone repository, build virtual environment, and verify that `train.py` runs using dummy input tensors and basic placeholder backbones.
2. **Step 2: Component Implementation**: Team members complete their respective modules using the agreed-upon data contracts.
3. **Step 3: Integration**: Replace stub models with actual pretrained extraction backbones and connect real datasets to the robust augmentation pipeline.
4. **Step 4: Benchmarking**: Run `evaluate.py` to test performance under JPEG compression, blur, downscaling, noise, jitter, and cropping, generating final metrics and visual heatmaps.
