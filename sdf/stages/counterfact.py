"""counterfact stage: CF-Edit real train images into labelled counterfactuals.

Takes real images of one class (train split ONLY - val/test never feed the
generator) and label-swaps them into the target class via partial diffusion
editing on the shared mdx checkpoint. Every output is a paired counterfactual
whose manifest row names its real source image and edit strength; the select
stage's detector scoring then filters edits too weak to express the target
class.

Usage: python -m sdf run-stage counterfact --config pipeline_cond_a.toml \
    --opt cls_from=<normal-class> --opt cls_to=<condition-class> \
    --opt strength=0.5 --opt count=100 [--opt adapter_dir=...]
"""

from __future__ import annotations

import random
import time
from pathlib import Path

from .. import manifest
from ..config import PipelineConfig, output_dir
from ..data.base import DataError, get_adapter
from ..splits import grouped_stratified_split
from .base import StageResult
from .sample import MANIFEST_NAME

BACKEND = "mdx"


def run(cfg: PipelineConfig, opts: dict | None = None) -> StageResult:
    started = time.time()
    opts = opts or {}
    cls_from = str(opts.get("cls_from", ""))
    cls_to = str(opts.get("cls_to", ""))
    if not cls_from or not cls_to or cls_from == cls_to:
        return StageResult(
            stage="counterfact", success=False,
            error="pass --opt cls_from=<source class> --opt cls_to=<target class> "
                  "(distinct)",
            duration_s=round(time.time() - started, 2),
        )
    count = int(opts.get("count", cfg.generator.sample_count))
    seed = int(opts.get("seed", cfg.splits.seed))

    from ..gen import mdx

    adapter_dir = Path(
        str(opts.get("adapter_dir", ""))
        or output_dir() / "lora" / BACKEND / "joint" / mdx.ADAPTER_DIR_NAME
    )
    if not adapter_dir.is_dir():
        return StageResult(
            stage="counterfact", success=False,
            error=f"mdx checkpoint not found: {adapter_dir} (run train_joint "
                  "first or pass --opt adapter_dir=...)",
            duration_s=round(time.time() - started, 2),
        )

    try:
        records, _ = get_adapter(cfg).index()
    except DataError as exc:
        return StageResult(
            stage="counterfact", success=False, error=str(exc),
            duration_s=round(time.time() - started, 2),
        )
    splits, _ = grouped_stratified_split(
        records, seed=cfg.splits.seed,
        val_frac=cfg.splits.val_frac, test_frac=cfg.splits.test_frac,
    )
    sources = sorted(
        (r for r in splits["train"] if r.cls == cls_from and r.exists),
        key=lambda r: r.image_id,
    )
    if not sources:
        known = sorted({r.cls for r in records})
        return StageResult(
            stage="counterfact", success=False,
            error=f"no train images for cls_from {cls_from!r}; dataset has: {known}",
            duration_s=round(time.time() - started, 2),
        )
    rng = random.Random(seed)
    rng.shuffle(sources)
    # Cycle sources when count exceeds them - each pass gets fresh noise seeds.
    paths = [sources[i % len(sources)].path for i in range(count)]

    out = output_dir() / "synthetic" / cls_to
    try:
        rows = mdx.edit(cfg, paths, cls_from, cls_to, adapter_dir, out, opts)
    except Exception as exc:  # noqa: BLE001 - convert to a structured failure
        import traceback

        traceback.print_exc()
        return StageResult(
            stage="counterfact", success=False,
            error=f"{type(exc).__name__}: {exc}",
            metrics={"cls_from": cls_from, "cls_to": cls_to,
                     "sources": len(sources)},
            duration_s=round(time.time() - started, 2),
        )

    manifest_path = output_dir() / MANIFEST_NAME
    for row in rows:
        record = manifest.new_record(**row)
        record["file"] = f"synthetic/{cls_to}/{row['file']}"
        manifest.append(manifest_path, record)

    return StageResult(
        stage="counterfact",
        success=len(rows) > 0,
        metrics={
            "cls_from": cls_from,
            "cls_to": cls_to,
            "count": len(rows),
            "unique_sources": len(sources),
            "strength": float(opts.get("strength", 0.5)),
            "adapter_dir": str(adapter_dir),
            "first_files": [r["file"] for r in rows[:5]],
        },
        outputs=[f"synthetic/{cls_to}/", MANIFEST_NAME],
        error="" if rows else "no counterfactuals generated",
        duration_s=round(time.time() - started, 2),
    )
