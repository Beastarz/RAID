"""
Module: datamodule
Project: AI Image Detector

AIGCDataModule: thin wrapper building train/val DataLoaders around
AIGCDataset. Not a PyTorch Lightning DataModule -- plain PyTorch, per the
project's current mock-the-training-loop stage (no lightning dependency yet).
"""

import logging
from typing import Any, Dict, Tuple

from torch.utils.data import DataLoader, Subset

from training.data.augmentations import RobustnessTransforms
from training.data.dataset import AIGCDataset

logger = logging.getLogger(__name__)


class AIGCDataModule:
    def __init__(self, config: Dict[str, Any], augmentations_config: Dict[str, Any]) -> None:
        self.config = config
        data_cfg = config["data"]
        image_size = config["image_size"]

        train_transforms = RobustnessTransforms(augmentations_config, image_size=image_size, mode="train")
        full_dataset = AIGCDataset(
            manifest_path=data_cfg["manifest_path"],
            transforms=train_transforms,
            num_synthetic_samples=data_cfg["num_synthetic_samples"],
            image_size=image_size,
            seed=config["seed"],
        )

        val_split = data_cfg["val_split"]
        n_val = max(1, int(len(full_dataset) * val_split)) if len(full_dataset) > 1 else 0
        n_train = len(full_dataset) - n_val
        self.train_dataset = Subset(full_dataset, range(0, n_train))
        self.val_dataset = Subset(full_dataset, range(n_train, len(full_dataset)))
        logger.info("AIGCDataModule: %d train / %d val samples", len(self.train_dataset), len(self.val_dataset))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
        )
