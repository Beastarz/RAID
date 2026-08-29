"""
Module: evaluate
Project: AI Image Detector

Entry point for the robustness evaluation pipeline. Run from the repo root as:
    python -m training.evaluate --checkpoint checkpoints/best_model.ckpt --config configs/augmentations.yaml
"""

import sys
from pathlib import Path

# Make `src` importable when this script is run directly (python training/evaluate.py)
# rather than as a module (python -m training.evaluate).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
