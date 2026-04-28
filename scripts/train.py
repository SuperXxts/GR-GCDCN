import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataloader.volume_dataset import NumpyVolumeDataset
from models.GCDCNet01 import GCDCNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train GR-GCDCN on paired NumPy 3D volumes.")
    parser.add_argument("--train-images", required=True, type=Path)
    parser.add_argument("--train-labels", required=True, type=Path)
    parser.add_argument("--val-images", required=True, type=Path)
    parser.add_argument("--val-labels", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("outputs/run"), type=Path)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--seed", default=11, type=int)
    parser.add_argument("--num-classes", default=2, type=int)
    parser.add_argument("--geometric-loss-weight", default=0.1, type=float)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_iou_from_logits(logits, labels, positive_class=1):
    preds = logits.argmax(dim=1)
    pred_pos = preds == positive_class
    label_pos = labels == positive_class
    intersection = torch.logical_and(pred_pos, label_pos).sum().float()
    pred_count = pred_pos.sum().float()
    label_count = label_pos.sum().float()
    union = torch.logical_or(pred_pos, label_pos).sum().float()
    dice = (2.0 * intersection) / (pred_count + label_count + 1e-6)
    iou = intersection / (union + 1e-6)
    return float(dice.item()), float(iou.item())


def run_epoch(model, loader, optimizer, device, geometric_weight, train):
    model.train(train)
    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "count": 0}

    iterator = tqdm(loader, leave=False, desc="train" if train else "val")
    for images, labels in iterator:
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(train):
            logits, geometric_loss = model(images, return_geometric_loss=True)
            segmentation_loss = F.cross_entropy(logits, labels)
            loss = segmentation_loss + geometric_weight * geometric_loss.mean()

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if hasattr(model, "step_warmup"):
                    model.step_warmup(1)

        dice, iou = dice_iou_from_logits(logits.detach(), labels)
        batch_size = images.size(0)
        totals["loss"] += float(loss.item()) * batch_size
        totals["dice"] += dice * batch_size
        totals["iou"] += iou * batch_size
        totals["count"] += batch_size

    count = max(totals["count"], 1)
    return {key: totals[key] / count for key in ("loss", "dice", "iou")}


def main():
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    train_set = NumpyVolumeDataset(args.train_images, args.train_labels)
    val_set = NumpyVolumeDataset(args.val_images, args.val_labels)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.workers)

    model = GCDCNet(
        num_channels=1,
        num_classes=args.num_classes,
        geometric_loss_weight=args.geometric_loss_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = []
    best_iou = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args.geometric_loss_weight, train=True)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, optimizer, device, args.geometric_loss_weight, train=False)

        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(json.dumps(record, indent=None))

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, args.output_dir / "best_model.pth")

    with (args.output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
