"""Detector training/evaluation utilities shared by the last pipeline stages.

The detector is deliberately vanilla - torchvision backbone, plain
cross-entropy, no class weighting or sampling tricks - because the experiment
measures the effect of *data* (real vs real+synthetic). Any loss-level
imbalance handling would confound exactly the thing we are trying to measure.

Metrics are numpy-only so they unit-test on the torch-less control machine.
"""

from __future__ import annotations

import csv
from pathlib import Path

EVAL_RESOLUTION = 224


# ------------------------------------------------------------------- indexes
def load_index(path: Path) -> list[dict]:
    """Read an index CSV (path,cls,source) written by the augment stage."""
    if not path.exists():
        raise FileNotFoundError(f"index not found: {path} (run augment first)")
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        if not {"path", "cls", "source"} <= set(row):
            raise ValueError(f"{path}: malformed row {row}")
    return rows


def label_map(rows: list[dict]) -> dict[str, int]:
    """Deterministic class -> id mapping (sorted by class name)."""
    return {cls: i for i, cls in enumerate(sorted({r["cls"] for r in rows}))}


# ------------------------------------------------------------------- metrics
def confusion_matrix(y_true: list[int], y_pred: list[int], n_classes: int):
    import numpy as np

    cm = np.zeros((n_classes, n_classes), dtype="int64")
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def metrics_from_confusion(cm, classes: list[str]) -> dict:
    """Per-class precision/recall/F1 + macro aggregates from a confusion matrix."""
    import numpy as np

    tp = np.diag(cm).astype("float64")
    support = cm.sum(axis=1).astype("float64")
    predicted = cm.sum(axis=0).astype("float64")

    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, tp / support, 0.0)
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        f1 = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )

    per_class = {
        cls: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i, cls in enumerate(classes)
    }
    present = support > 0  # macro over classes that actually appear
    return {
        "per_class": per_class,
        "macro_f1": round(float(f1[present].mean()) if present.any() else 0.0, 4),
        "macro_recall": round(float(recall[present].mean()) if present.any() else 0.0, 4),
        "accuracy": round(float(tp.sum() / max(1.0, cm.sum())), 4),
        "n": int(cm.sum()),
    }


# ----------------------------------------------------------------- ML pieces
def build_model(name: str, num_classes: int):
    import torch.nn as nn
    from torchvision import models

    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes
        )
    else:
        raise ValueError(f"unknown model {name!r} (resnet18, efficientnet_b0)")
    return model


def make_dataset(rows: list[dict], labels: dict[str, int], train: bool):
    import torch
    from PIL import Image
    from torch.utils.data import Dataset
    from torchvision import transforms

    if train:
        tf = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(EVAL_RESOLUTION),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(EVAL_RESOLUTION),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    class _DS(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, idx):
            row = rows[idx]
            with Image.open(row["path"]) as img:
                x = tf(img.convert("RGB"))
            return x, torch.tensor(labels[row["cls"]], dtype=torch.long)

    return _DS()


def predict(model, rows: list[dict], labels: dict[str, int], batch_size: int = 64):
    """Return (y_true, y_pred) over ``rows``."""
    import torch
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = DataLoader(
        make_dataset(rows, labels, train=False),
        batch_size=batch_size, num_workers=2,
    )
    model.eval().to(device)
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_true.extend(y.tolist())
    return y_true, y_pred
