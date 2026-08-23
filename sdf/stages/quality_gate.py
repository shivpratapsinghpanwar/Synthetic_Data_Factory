"""quality_gate stage: decide whether synthetic images are usable for training.

Usage: python -m sdf run-stage quality_gate --opt cls=df
Reads synthetic images from $SDF_OUTPUT_DIR/synthetic/<cls> (or --opt
syn_dir=...), compares against the class's real train split (memorization)
and val split (FID). Gates on memorization and degeneracy; FID is reported
as a diagnostic with its sample count attached.
"""

from __future__ import annotations

import time
from pathlib import Path

from .. import quality
from ..config import PipelineConfig, output_dir
from ..data.base import DataError, get_adapter
from ..gen.prompts import CLASS_PROMPTS
from ..splits import grouped_stratified_split
from .base import StageResult


def run(cfg: PipelineConfig, opts: dict | None = None) -> StageResult:
    started = time.time()
    opts = opts or {}
    cls = str(opts.get("cls", ""))
    if cls not in CLASS_PROMPTS:
        return StageResult(
            stage="quality_gate", success=False,
            error=f"pass --opt cls=<class>; got {cls!r}, known: {sorted(CLASS_PROMPTS)}",
            duration_s=round(time.time() - started, 2),
        )

    syn_dir = Path(str(opts.get("syn_dir", "")) or output_dir() / "synthetic" / cls)
    syn_paths = sorted(syn_dir.glob("*.png")) + sorted(syn_dir.glob("*.jpg"))
    if not syn_paths:
        return StageResult(
            stage="quality_gate", success=False,
            error=f"no synthetic images found in {syn_dir}",
            duration_s=round(time.time() - started, 2),
        )

    # Degeneracy screen first: PIL-only, works even if torch is broken.
    screen = quality.pixel_screen(syn_paths)
    problems: list[str] = []
    if screen["flat"]:
        problems.append(f"{len(screen['flat'])} flat/blank images")
    if screen["exact_duplicates"]:
        problems.append(f"{len(screen['exact_duplicates'])} exact duplicate pairs")

    metrics: dict = {
        "cls": cls,
        "syn_dir": str(syn_dir),
        "syn_count": len(syn_paths),
        "pixel_screen": screen,
    }

    try:
        records, _ = get_adapter(cfg).index()
        splits, _ = grouped_stratified_split(
            records, seed=cfg.splits.seed,
            val_frac=cfg.splits.val_frac, test_frac=cfg.splits.test_frac,
        )
        train_real = [r.path for r in splits["train"] if r.cls == cls and r.exists]
        val_real = [r.path for r in splits["val"] if r.cls == cls and r.exists]
    except DataError as exc:
        return StageResult(
            stage="quality_gate", success=False,
            error=f"dataset unavailable for reference sets: {exc}",
            metrics=metrics, duration_s=round(time.time() - started, 2),
        )

    try:
        syn_feats = quality.embed_images(syn_paths)
        train_feats = quality.embed_images(train_real)
        mem = quality.memorization_check(
            syn_feats, train_feats, [p.name for p in syn_paths]
        )
        metrics["memorization"] = mem
        if mem["flagged"]:
            problems.append(
                f"{len(mem['flagged'])} images too similar to real training data "
                f"(max {mem['max_similarity']})"
            )

        if val_real:
            val_feats = quality.embed_images(val_real)
            metrics["fid"] = {
                "value": quality.frechet_distance(syn_feats, val_feats),
                "syn_n": len(syn_paths),
                "real_n": len(val_real),
                "stable": min(len(syn_paths), len(val_real)) >= quality.FID_UNSTABLE_BELOW,
            }
        else:
            metrics["fid"] = {"value": None, "note": "no val images for class"}
    except Exception as exc:  # noqa: BLE001 - embedding failures are stage failures
        import traceback

        traceback.print_exc()
        return StageResult(
            stage="quality_gate", success=False,
            error=f"embedding/metrics failed: {type(exc).__name__}: {exc}",
            metrics=metrics, duration_s=round(time.time() - started, 2),
        )

    for line in (
        f"[quality_gate] {cls}: {len(syn_paths)} synthetic, "
        f"train_ref={len(train_real)}, val_ref={len(val_real)}",
        f"[quality_gate] memorization max={metrics['memorization']['max_similarity']} "
        f"flagged={len(metrics['memorization']['flagged'])}",
        f"[quality_gate] fid={metrics['fid']}",
    ):
        print(line, flush=True)

    return StageResult(
        stage="quality_gate",
        success=not problems,
        metrics=metrics,
        error="; ".join(problems),
        duration_s=round(time.time() - started, 2),
    )
