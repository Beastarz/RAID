"""Train fusion and classifier after independently training both streams."""
import argparse
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import yaml

from src.models.semantic_stream import SemanticStream
from src.models.npr_stream import NPRStream, OUTPUT_DIM_RESNET_SHALLOW
from src.models.frontend_bayar import BayarSRMFrontend
from src.models.fusion import FeatureFusion
from training.data.datamodule import AIGCDataModule
from training.logging_utils import setup_logger


def _load_stream(path: str, module: nn.Module, device: torch.device) -> None:
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Could not load {path}: missing={missing}, unexpected={unexpected}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frozen streams plus fusion/classifier")
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--augmentations-config", default="configs/augmentations.yaml")
    parser.add_argument("--semantic-checkpoint", default="checkpoints/semantic_stream.pt")
    parser.add_argument("--frequency-checkpoint", default="checkpoints/bayar_srm_stream.pt")
    parser.add_argument("--frequency-head", default="checkpoints/bayar_srm_head.pt")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=0, help="0 means all batches for each epoch")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--log-level", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    with open(args.augmentations_config) as handle:
        aug_config = yaml.safe_load(handle)
    logger = setup_logger("training", args.log_level or config["log_level"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    semantic_cfg = config.get("semantic", {})
    semantic_stream = SemanticStream(output_dim=semantic_cfg.get("output_dim", 1024), pretrained=False).to(device)
    npr_stream = NPRStream(backbone="resnet_shallow", frontend=BayarSRMFrontend()).to(device)
    _load_stream(args.semantic_checkpoint, semantic_stream, device)
    _load_stream(args.frequency_checkpoint, npr_stream, device)
    for stream in (semantic_stream, npr_stream):
        stream.eval()
        for parameter in stream.parameters():
            parameter.requires_grad = False
    fusion = FeatureFusion(semantic_dim=semantic_cfg.get("output_dim", 1024),
                           freq_dim=OUTPUT_DIM_RESNET_SHALLOW, fused_dim=512).to(device)
    classifier = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1)).to(device)
    trainable = list(fusion.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.Adam(trainable, lr=args.lr or config["lr"])
    loss_fn = nn.BCEWithLogitsLoss()
    data = AIGCDataModule(config, aug_config)
    logger.info("Fusion training: frozen streams, trainable fusion/classifier parameters=%d",
                sum(p.numel() for p in trainable))
    step = 0
    for epoch in range(1, args.epochs + 1):
        fusion.train(); classifier.train()
        for images, labels in data.train_dataloader():
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            raw_images = images * torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            raw_images = raw_images + torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            with torch.no_grad():
                semantic_features = semantic_stream(images)
                npr_features = npr_stream(raw_images)
            output_features = fusion(semantic_features, npr_features)
            logit = classifier(output_features)
            loss = loss_fn(logit, labels)
            loss.backward()
            optimizer.step()
            step += 1
            logger.info("epoch=%d/%d step=%d loss=%.4f", epoch, args.epochs, step, loss.item())
            if args.steps and step >= args.steps:
                break
        semantic_stream.eval(); npr_stream.eval(); fusion.eval(); classifier.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in data.val_dataloader():
                images = images.to(device)
                raw_images = images * torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
                raw_images = raw_images + torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                probability = torch.sigmoid(classifier(fusion(semantic_stream(images), npr_stream(raw_images))))
                correct += ((probability >= 0.5) == (labels.to(device) >= 0.5)).sum().item()
                total += labels.numel()
        logger.info("epoch=%d validation_accuracy=%.3f", epoch, correct / max(total, 1))
        if args.steps and step >= args.steps:
            break
    output_path = Path(args.checkpoint_dir) / "detector_fusion.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"fusion": fusion.state_dict(), "classifier": classifier.state_dict()}, output_path)
    logger.info("Saved fused detector checkpoint to %s", output_path)


if __name__ == "__main__":
    main()
