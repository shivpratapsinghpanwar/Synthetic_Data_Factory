"""Audit stage: the pipeline's cheap, CPU-only ground-truth check.

Answers, with numbers: is the dataset actually present and structurally sound,
what does the class distribution look like, which classes are rare enough to
be generation targets, and do the leakage-safe splits come out sane?

This stage is the gate in front of every GPU stage - it must pass on Kaggle
before any training run is allowed to spend quota.
"""

from __future__ import annotations

import time
from collections import Counter

from ..config import PipelineConfig
from ..data.base import DataError, get_adapter
from ..splits import grouped_stratified_split
from .base import StageResult

MAX_MISSING_RATIO = 0.01
SIZE_PROBE_SAMPLES = 25


def _probe_sizes(records) -> dict[str, int]:
    """Open a small sample of images to learn the real pixel dimensions."""
    try:
        from PIL import Image
    except ImportError:
        return {"unavailable (no PIL)": 0}

    sizes: Counter[str] = Counter()
    probed = 0
    for rec in records:
        if probed >= SIZE_PROBE_SAMPLES:
            break
        if not rec.exists:
            continue
        try:
            with Image.open(rec.path) as img:
                sizes[f"{img.width}x{img.height}"] += 1
            probed += 1
        except OSError:
            sizes["unreadable"] += 1
            probed += 1
    return dict(sizes)


def run(cfg: PipelineConfig, opts: dict | None = None) -> StageResult:
    started = time.time()
    try:
        adapter = get_adapter(cfg)
        records, report = adapter.index()
    except DataError as exc:
        return StageResult(
            stage="audit",
            success=False,
            error=str(exc),
            duration_s=round(time.time() - started, 2),
        )

    class_counts = Counter(r.cls for r in records)
    splits, split_stats = grouped_stratified_split(
        records,
        seed=cfg.splits.seed,
        val_frac=cfg.splits.val_frac,
        test_frac=cfg.splits.test_frac,
    )

    train_counts = Counter(r.cls for r in splits["train"])
    rare = sorted(
        cls
        for cls, n in train_counts.items()
        if n < cfg.generator.rare_class_max_count
    )

    rows = report["metadata_rows"]
    missing_ratio = (report["missing_files"] / rows) if rows else 1.0
    problems = []
    if rows == 0:
        problems.append("metadata parsed to zero rows")
    if missing_ratio > MAX_MISSING_RATIO:
        problems.append(
            f"{report['missing_files']}/{rows} referenced images missing "
            f"(> {MAX_MISSING_RATIO:.0%})"
        )
    if not rare:
        problems.append(
            "no rare classes below the threshold - nothing for the generator to target"
        )
    problems.extend(split_stats["warnings"])

    metrics = {
        "discovery": report,
        "classes": dict(sorted(class_counts.items())),
        "splits": split_stats,
        "rare_classes": rare,
        "rare_class_max_count": cfg.generator.rare_class_max_count,
        "image_sizes_sampled": _probe_sizes(records),
    }

    for line in (
        f"[audit] root: {report['root']}",
        f"[audit] rows={rows} on_disk={report['images_on_disk']} missing={report['missing_files']}",
        f"[audit] classes: {dict(sorted(class_counts.items()))}",
        f"[audit] split totals: {split_stats['totals']}",
        f"[audit] rare classes (train < {cfg.generator.rare_class_max_count}): {rare}",
    ):
        print(line, flush=True)

    return StageResult(
        stage="audit",
        success=not problems,
        metrics=metrics,
        error="; ".join(problems),
        duration_s=round(time.time() - started, 2),
    )
