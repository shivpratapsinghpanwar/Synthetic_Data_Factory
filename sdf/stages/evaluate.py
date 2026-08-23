"""evaluate stage: score trained detector arms on the real-only test split.

Usage: python -m sdf run-stage evaluate --opt tags=real_only,augmented

Loads each detector/<tag>/model.pt, predicts on test_index.csv (real images
only - enforced), and reports per-class + macro metrics per arm. With two or
more arms it also reports the deltas vs the first tag, which is treated as
the baseline. This stage produces THE number the project is judged by:
rare-class recall/F1 delta of augmented over real_only.
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
    tags = [t.strip() for t in str(opts.get("tags", "")).split(",") if t.strip()]
    if not tags:
        return StageResult(
            stage="evaluate", success=False,
            error="pass --opt tags=<tag>[,<tag>...]; first tag is the baseline",
            duration_s=round(time.time() - started, 2),
        )
    index_root = Path(str(opts.get("index_root", "")) or output_dir())

    try:
        test_rows = detect.load_index(index_root / "test_index.csv")
    except (FileNotFoundError, ValueError) as exc:
        return StageResult(
            stage="evaluate", success=False, error=str(exc),
            duration_s=round(time.time() - started, 2),
        )
    synthetic_in_test = [r for r in test_rows if r["source"] != "real"]
    if synthetic_in_test:
        return StageResult(
            stage="evaluate", success=False,
            error=f"test index contains {len(synthetic_in_test)} non-real rows - "
                  "evaluation on synthetic data is meaningless",
            duration_s=round(time.time() - started, 2),
        )

    try:
        arms = {tag: _score(tag, test_rows, opts) for tag in tags}
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        return StageResult(
            stage="evaluate", success=False,
            error=f"{type(exc).__name__}: {exc}",
            duration_s=round(time.time() - started, 2),
        )

    metrics: dict = {"test_n": len(test_rows), "arms": arms}
    if len(tags) > 1:
        metrics["deltas_vs_" + tags[0]] = _deltas(arms, tags)

    for tag in tags:
        print(f"[evaluate] {tag}: macro_f1={arms[tag]['macro_f1']} "
              f"acc={arms[tag]['accuracy']}", flush=True)
    if len(tags) > 1:
        print(f"[evaluate] deltas vs {tags[0]}: "
              f"{json.dumps(metrics['deltas_vs_' + tags[0]], indent=1)}", flush=True)

    return StageResult(
        stage="evaluate", success=True, metrics=metrics,
        duration_s=round(time.time() - started, 2),
    )


def _score(tag: str, test_rows: list[dict], opts: dict) -> dict:
    import torch

    model_dir = output_dir() / "detector" / tag
    labels = json.loads((model_dir / "labels.json").read_text(encoding="utf-8"))
    classes = sorted(labels, key=labels.get)

    report = json.loads((model_dir / "report.json").read_text(encoding="utf-8"))
    model = detect.build_model(report["model"], len(labels))
    model.load_state_dict(
        torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    )

    y_true, y_pred = detect.predict(model, test_rows, labels)
    cm = detect.confusion_matrix(y_true, y_pred, len(labels))
    scored = detect.metrics_from_confusion(cm, classes)
    scored["confusion"] = cm.tolist()
    scored["train_sources"] = report.get("train_sources", {})
    return scored


def _deltas(arms: dict, tags: list[str]) -> dict:
    base = arms[tags[0]]
    out: dict = {}
    for tag in tags[1:]:
        arm = arms[tag]
        per_class = {
            cls: {
                "recall": round(arm["per_class"][cls]["recall"]
                                - base["per_class"][cls]["recall"], 4),
                "f1": round(arm["per_class"][cls]["f1"]
                            - base["per_class"][cls]["f1"], 4),
            }
            for cls in arm["per_class"]
        }
        out[tag] = {
            "macro_f1": round(arm["macro_f1"] - base["macro_f1"], 4),
            "accuracy": round(arm["accuracy"] - base["accuracy"], 4),
            "per_class": per_class,
        }
    return out
