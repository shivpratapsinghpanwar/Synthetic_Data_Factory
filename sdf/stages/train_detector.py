"""train_detector stage: train the condition classifier on a training index.

Usage:
  python -m sdf run-stage train_detector --opt tag=real_only
  python -m sdf run-stage train_detector --opt tag=augmented

The tag names the experiment arm; the same augment-produced indexes are used
(an index without synthetic rows *is* the real-only arm - run augment with no
manifest present, or pass --opt real_only=1 to filter synthetic rows here).
Model selection: best val macro-F1 across epochs. Output:
detector/<tag>/{model.pt, labels.json, report.json}.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .. import detect
from ..config import PipelineConfig, output_dir
from .base import StageResult


def run(cfg: PipelineConfig, opts: dict | None = None) -> StageResult:
    started = time.time()
    opts = opts or {}
    tag = str(opts.get("tag", "detector"))
    index_root = Path(str(opts.get("index_root", "")) or output_dir())

    try:
        train_rows = detect.load_index(index_root / "train_index.csv")
        val_rows = detect.load_index(index_root / "val_index.csv")
    except (FileNotFoundError, ValueError) as exc:
        return StageResult(
            stage="train_detector", success=False, error=str(exc),
            duration_s=round(time.time() - started, 2),
        )

    if int(opts.get("real_only", 0)):
        train_rows = [r for r in train_rows if r["source"] == "real"]

    labels = detect.label_map(train_rows + val_rows)
    source_counts = {"real": 0, "synthetic": 0}
    for row in train_rows:
        source_counts[row["source"]] = source_counts.get(row["source"], 0) + 1

    try:
        report = _train(cfg, train_rows, val_rows, labels, tag, opts)
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        return StageResult(
            stage="train_detector", success=False,
            error=f"{type(exc).__name__}: {exc}",
            metrics={"tag": tag, "train_sources": source_counts},
            duration_s=round(time.time() - started, 2),
        )

    report["train_sources"] = source_counts
    return StageResult(
        stage="train_detector", success=True, metrics=report,
        outputs=[f"detector/{tag}/"],
        duration_s=round(time.time() - started, 2),
    )


def _train(cfg, train_rows, val_rows, labels, tag, opts) -> dict:
    import torch
    from torch.utils.data import DataLoader

    model_name = str(opts.get("model", "efficientnet_b0"))
    epochs = int(opts.get("epochs", 12))
    batch_size = int(opts.get("batch_size", 32))
    lr = float(opts.get("lr", 3e-4))
    seed = int(opts.get("seed", cfg.splits.seed))

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classes = sorted(labels, key=labels.get)

    model = detect.build_model(model_name, len(labels)).to(device)
    loader = DataLoader(
        detect.make_dataset(train_rows, labels, train=True),
        batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    out = output_dir() / "detector" / tag
    out.mkdir(parents=True, exist_ok=True)

    best = {"epoch": -1, "macro_f1": -1.0}
    history = []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        total, batches = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
            batches += 1

        y_true, y_pred = detect.predict(model, val_rows, labels)
        cm = detect.confusion_matrix(y_true, y_pred, len(labels))
        val_metrics = detect.metrics_from_confusion(cm, classes)
        history.append({
            "epoch": epoch,
            "train_loss": round(total / max(1, batches), 4),
            "val_macro_f1": val_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
        })
        print(f"[train_detector] {tag} epoch {epoch + 1}/{epochs} "
              f"loss={history[-1]['train_loss']} val_f1={val_metrics['macro_f1']}",
              flush=True)

        if val_metrics["macro_f1"] > best["macro_f1"]:
            best = {"epoch": epoch, "macro_f1": val_metrics["macro_f1"]}
            torch.save(model.state_dict(), out / "model.pt")

    (out / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    report = {
        "tag": tag,
        "model": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "device": device,
        "classes": classes,
        "train_n": len(train_rows),
        "val_n": len(val_rows),
        "best": best,
        "history": history,
        "duration_s": round(time.time() - t0, 1),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
