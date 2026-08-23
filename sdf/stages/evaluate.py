"""evaluate stage: score trained detector arms on the real-only test split.

Usage:
  python -m sdf run-stage evaluate --opt tags=real_only,augmented
  python -m sdf run-stage evaluate --opt pairs=real_only_s1:augmented_s1,real_only_s2:augmented_s2

`tags` mode: first tag is the baseline, deltas reported against it.
`pairs` mode (multi-seed): each baseline:treatment pair is one seed; deltas
are computed within each pair and aggregated as mean +/- sample std across
pairs - the honest form of the experiment's headline number.

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
    pairs_raw = str(opts.get("pairs", ""))
    pairs: list[tuple[str, str]] = []
    if pairs_raw:
        for item in pairs_raw.split(","):
            if ":" not in item:
                return StageResult(
                    stage="evaluate", success=False,
                    error=f"pairs entries must be baseline:treatment, got {item!r}",
                    duration_s=round(time.time() - started, 2),
                )
            base, _, treat = item.partition(":")
            pairs.append((base.strip(), treat.strip()))
        tags = sorted({t for pair in pairs for t in pair})
    else:
        tags = [t.strip() for t in str(opts.get("tags", "")).split(",") if t.strip()]
    if not tags:
        return StageResult(
            stage="evaluate", success=False,
            error="pass --opt tags=... or --opt pairs=base:treat[,base:treat...]",
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
    if pairs:
        metrics["seed_pairs"] = [f"{b}:{t}" for b, t in pairs]
        metrics["aggregate"] = aggregate_pairs(arms, pairs)
    elif len(tags) > 1:
        metrics["deltas_vs_" + tags[0]] = _deltas(arms, tags)

    for tag in tags:
        print(f"[evaluate] {tag}: macro_f1={arms[tag]['macro_f1']} "
              f"acc={arms[tag]['accuracy']}", flush=True)
    if pairs:
        print(f"[evaluate] aggregate over {len(pairs)} seed pair(s): "
              f"{json.dumps(metrics['aggregate']['summary'], indent=1)}", flush=True)
    elif len(tags) > 1:
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


def aggregate_pairs(arms: dict, pairs: list[tuple[str, str]]) -> dict:
    """Mean +/- sample std of within-pair deltas across seed pairs.

    Pure function over already-scored arms so it unit-tests without torch.
    """
    import statistics

    def spread(values: list[float]) -> dict:
        return {
            "mean": round(statistics.mean(values), 4),
            "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
            "values": [round(v, 4) for v in values],
        }

    macro = [arms[t]["macro_f1"] - arms[b]["macro_f1"] for b, t in pairs]
    acc = [arms[t]["accuracy"] - arms[b]["accuracy"] for b, t in pairs]
    classes = sorted(arms[pairs[0][0]]["per_class"])
    per_class = {
        cls: {
            "recall_delta": spread(
                [arms[t]["per_class"][cls]["recall"] - arms[b]["per_class"][cls]["recall"]
                 for b, t in pairs]
            ),
            "f1_delta": spread(
                [arms[t]["per_class"][cls]["f1"] - arms[b]["per_class"][cls]["f1"]
                 for b, t in pairs]
            ),
        }
        for cls in classes
    }
    return {
        "n_pairs": len(pairs),
        "summary": {"macro_f1_delta": spread(macro), "accuracy_delta": spread(acc)},
        "per_class": per_class,
    }


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
