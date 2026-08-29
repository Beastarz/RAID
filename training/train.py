"""
Module: train
Project: AI Image Detector

Entry point for the training pipeline. Run from the repo root as:
    python -m training.train --config configs/base_config.yaml
"""

import sys
from pathlib import Path

# Make `src` importable when this script is run directly (python training/train.py)
# rather than as a module (python -m training.train).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
