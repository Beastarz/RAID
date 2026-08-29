"""
Module: test_data
Project: AI Image Detector

Minimal, critical-path tests for the training pipeline's data path: the
augmentation shape contract, the dataset label/shape contract, and one
end-to-end smoke test per stream-training entry point.
"""

import importlib

import numpy as np
import pytest
import yaml

from training.data.augmentations import RobustnessTransforms
from training.data.dataset import AIGCDataset

AUGMENTATIONS_CONFIG = {
    "train": {
        "p_each": 0.5,
        "jpeg_quality_range": [30, 90],
        "blur_sigma_range": [0.5, 2.0],
        "downscale_range": [0.25, 0.5],
        "noise_std_range": [0.03, 0.12],
        "color_jitter": {
            "brightness": [0.8, 1.2],
            "contrast": [0.8, 1.2],
            "saturation": [0.8, 1.2],
            "hue": [-0.1, 0.1],
        },
        "crop_percent": 0.8,
    },
    "eval": {
        "jpeg_quality_levels": [30, 90],
        "blur_sigma_levels": [0.5, 2.0],
        "downscale_levels": [0.25, 0.5],
        "noise_std_levels": [0.03, 0.12],
        "crop_percent_levels": [0.8],
    },
}

IMAGE_SIZE = 64


def test_robustness_transforms_output_shape():
    transforms = RobustnessTransforms(AUGMENTATIONS_CONFIG, image_size=IMAGE_SIZE, mode="train")
    image = np.random.default_rng(0).integers(0, 256, size=(100, 80, 3), dtype=np.uint8)

    tensor = transforms(image)

    assert tuple(tensor.shape) == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert tensor.dtype.is_floating_point


def test_aigc_dataset_synthetic_mode_contract():
    transforms = RobustnessTransforms(AUGMENTATIONS_CONFIG, image_size=IMAGE_SIZE, mode="train")
    dataset = AIGCDataset(
        manifest_path=None,
        transforms=transforms,
        num_synthetic_samples=4,
        image_size=IMAGE_SIZE,
    )

    assert len(dataset) == 4
    tensor, label = dataset[0]
    assert tuple(tensor.shape) == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert tuple(label.shape) == (1,)
    assert label.item() in (0.0, 1.0)


@pytest.mark.parametrize(
    "module_name,checkpoint_name",
    [
        ("training.train_semantic", "semantic_stream.pt"),
        ("training.train_frequency", "frequency_stream.pt"),
    ],
)
def test_train_stream_main_smoke(tmp_path, module_name, checkpoint_name):
    train_module = importlib.import_module(module_name)

    base_config = {
        "seed": 0,
        "image_size": IMAGE_SIZE,
        "batch_size": 2,
        "num_workers": 0,
        "lr": 1e-3,
        "data": {"manifest_path": None, "num_synthetic_samples": 4, "val_split": 0.25},
        "log_level": "WARNING",
    }
    base_config_path = tmp_path / "base_config.yaml"
    augmentations_config_path = tmp_path / "augmentations.yaml"
    base_config_path.write_text(yaml.dump(base_config))
    augmentations_config_path.write_text(yaml.dump(AUGMENTATIONS_CONFIG))
    checkpoint_dir = tmp_path / "checkpoints"

    train_module.main(
        [
            "--config",
            str(base_config_path),
            "--augmentations-config",
            str(augmentations_config_path),
            "--steps",
            "1",
            "--checkpoint-dir",
            str(checkpoint_dir),
        ]
    )

    assert (checkpoint_dir / checkpoint_name).exists()
