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

        # Validation must not receive random training degradations.
        val_transforms = RobustnessTransforms(
            augmentations_config, image_size=image_size, mode="eval",
            eval_transform="clean", eval_severity=None,
        )
        val_full_dataset = AIGCDataset(
            manifest_path=data_cfg["manifest_path"], transforms=val_transforms,
            num_synthetic_samples=data_cfg["num_synthetic_samples"],
            image_size=image_size, seed=config["seed"],
        )
        val_split = data_cfg["val_split"]
        n_val = max(1, int(len(full_dataset) * val_split)) if len(full_dataset) > 1 else 0
        n_train = len(full_dataset) - n_val
        rng = __import__("numpy").random.default_rng(config["seed"])
        indices_by_label = {}
        for index, (_path, label) in enumerate(full_dataset._samples):
            indices_by_label.setdefault(label, []).append(index)
        train_indices, val_indices = [], []
        for label, label_indices in indices_by_label.items():
            shuffled = list(label_indices)
            rng.shuffle(shuffled)
            label_val = max(1, round(len(shuffled) * val_split)) if len(shuffled) > 1 else 0
            val_indices.extend(shuffled[:label_val])
            train_indices.extend(shuffled[label_val:])
        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(val_full_dataset, val_indices)
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
