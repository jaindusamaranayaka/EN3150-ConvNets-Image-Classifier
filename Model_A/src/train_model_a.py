"""Train and evaluate Model A using the project's shared data splits."""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Common.evaluation import measure_inference_time


DATA_ROOT = ROOT / "data" / "gtsrb" / "GTSRB"
SPLIT_ROOT = ROOT / "splits"
RESULT_ROOT = ROOT / "results"
FIGURE_ROOT = RESULT_ROOT / "figures"
METRIC_ROOT = RESULT_ROOT / "metrics"
CHECKPOINT_ROOT = ROOT / "checkpoints"


class ModelA(nn.Module):
    def __init__(self, num_classes=43):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        if training:
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return loss_sum / total, correct / total


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)), transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(8), transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    # The local archive is the original GTSRB layout (class folders of PPMs).
    # Build the required disjoint 70/15/15 split from that exact archive.
    raw_train = ImageFolder(DATA_ROOT / "Training", transform=None)
    generator = torch.Generator().manual_seed(42)
    permutation = torch.randperm(len(raw_train), generator=generator).tolist()
    train_end = int(0.70 * len(permutation))
    val_end = train_end + int(0.15 * len(permutation))
    raw_train_transform = ImageFolder(DATA_ROOT / "Training", transform=train_transform)
    raw_eval_transform = ImageFolder(DATA_ROOT / "Training", transform=eval_transform)
    datasets = {
        "train": Subset(raw_train_transform, permutation[:train_end]),
        "val": Subset(raw_eval_transform, permutation[train_end:val_end]),
        "test": Subset(raw_eval_transform, permutation[val_end:]),
    }
    loaders = {name: DataLoader(ds, batch_size=32, shuffle=name == "train", num_workers=0) for name, ds in datasets.items()}
    model = ModelA().to(device)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = {key: [] for key in ("train_loss", "train_accuracy", "val_loss", "val_accuracy", "epoch_time")}
    for epoch in range(20):
        start = time.perf_counter()
        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion, device)
        elapsed = time.perf_counter() - start
        for key, value in (("train_loss", train_loss), ("train_accuracy", train_acc), ("val_loss", val_loss), ("val_accuracy", val_acc), ("epoch_time", elapsed)):
            history[key].append(value)
        print(f"Epoch {epoch + 1:02d}/20 | train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f} | {elapsed:.1f}s")
    model.eval()
    y_true, y_pred = [], []
    with torch.inference_mode():
        for images, labels in loaders["test"]:
            y_true.extend(labels.numpy())
            y_pred.extend(model(images.to(device)).argmax(1).cpu().numpy())
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {
        "Model": "Model A", "Parameters": parameter_count,
        "Estimated FP32 Size (KB)": parameter_count * 4 / 1024,
        "Average Epoch Time (s)": float(np.mean(history["epoch_time"])),
        "Test Accuracy": accuracy,
        "Macro Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
    RESULT_ROOT.mkdir(exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    METRIC_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(exist_ok=True)
    checkpoint_path = CHECKPOINT_ROOT / "model_a_final.pth"
    torch.save(model.state_dict(), checkpoint_path)
    metrics["Actual Model Size (KB)"] = checkpoint_path.stat().st_size / 1024
    metrics["Inference Time (ms/image)"] = measure_inference_time(
        model, loaders["test"], device
    )
    (METRIC_ROOT / "model_a_metrics.csv").write_text(pd.DataFrame([metrics]).to_csv(index=False), encoding="utf-8")
    (METRIC_ROOT / "model_a_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (METRIC_ROOT / "model_a_classification_report.json").write_text(json.dumps(classification_report(y_true, y_pred, output_dict=True, zero_division=0), indent=2), encoding="utf-8")
    np.savetxt(METRIC_ROOT / "model_a_confusion_matrix.csv", confusion_matrix(y_true, y_pred), fmt="%d", delimiter=",")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
