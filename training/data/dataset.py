"""
Module: dataset
Project: AI Image Detector

AIGCDataset: PyTorch Dataset over the AI-vs-Real image classification task.

Real data source when configs/base_config.yaml's `data.manifest_path` points
at a CSV with an `image_path,label` header (0=Real, 1=AI-Generated), per the
contract in BLUEPRINT.md SS4.A. Mocked (synthetic, in-memory random images)
when no manifest is given -- this lets the rest of the pipeline (augmentation,
datamodule, training loop) be exercised and tested without a real dataset.
"""

import csv
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from training.data.augmentations import RobustnessTransforms

logger = logging.getLogger(__name__)


class AIGCDataset(Dataset):
    def __init__(
        self,
        manifest_path: Optional[str],
        transforms: RobustnessTransforms,
        num_synthetic_samples: int = 32,
        image_size: int = 512,
        seed: int = 42,
    ) -> None:
        self.transforms = transforms
        self.image_size = image_size
        self._samples: List[Tuple[Optional[Path], int]] = []
        self._rng = np.random.default_rng(seed)

        if manifest_path and Path(manifest_path).exists():
            self._samples = self._load_manifest(Path(manifest_path))
            logger.info("AIGCDataset: loaded %d samples from manifest %s", len(self._samples), manifest_path)
        else:
            self._samples = [
                (None, int(self._rng.integers(0, 2))) for _ in range(num_synthetic_samples)
            ]
            logger.info(
                "AIGCDataset: SYNTHETIC MODE (no manifest given) -- generated %d in-memory samples",
                len(self._samples),
            )

    @staticmethod
    def _load_manifest(manifest_path: Path) -> List[Tuple[Optional[Path], int]]:
        samples: List[Tuple[Optional[Path], int]] = []
        with manifest_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append((Path(row["image_path"]), int(row["label"])))
        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def _load_image(self, index: int, path: Optional[Path]) -> np.ndarray:
        if path is not None:
            return np.asarray(Image.open(path).convert("RGB"))
        # Synthetic sample: deterministic-per-index random RGB image.
        rng = np.random.default_rng(index)
        return rng.integers(0, 256, size=(self.image_size, self.image_size, 3), dtype=np.uint8)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self._samples[index]
        image = self._load_image(index, path)
        tensor = self.transforms(image)
        label_tensor = torch.tensor([float(label)], dtype=torch.float32)
        logger.debug("AIGCDataset[%d]: source=%s label=%.1f shape=%s", index, path or "synthetic", label, tuple(tensor.shape))
        return tensor, label_tensor
