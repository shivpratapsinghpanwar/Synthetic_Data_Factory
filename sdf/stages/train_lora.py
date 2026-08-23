"""train_lora stage: fine-tune the class LoRA on the class's train split.

Usage (via runner): python -m sdf run-stage train_lora --opt cls=df
Common opts: cls (required), steps, resolution, batch_size, lr, seed.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..config import PipelineConfig, output_dir
from ..data.base import DataError, get_adapter
from ..gen import BackendError, get_backend
from ..gen.prompts import CLASS_PROMPTS
from ..splits import grouped_stratified_split
from .base import StageResult


def run(cfg: PipelineConfig, opts: dict | None = None) -> StageResult:
    started = time.time()
    opts = opts or {}
    cls = str(opts.get("cls", ""))
    if cls not in CLASS_PROMPTS:
        return StageResult(
            stage="train_lora",
            success=False,
            error=f"pass --opt cls=<class>; got {cls!r}, known: {sorted(CLASS_PROMPTS)}",
            duration_s=round(time.time() - started, 2),
        )

    try:
        records, _report = get_adapter(cfg).index()
    except DataError as exc:
        return StageResult(
            stage="train_lora", success=False, error=str(exc),
            duration_s=round(time.time() - started, 2),
        )

    # Train ONLY on the train split - never let val/test leak into the generator.
    splits, _stats = grouped_stratified_split(
        records, seed=cfg.splits.seed,
        val_frac=cfg.splits.val_frac, test_frac=cfg.splits.test_frac,
    )
    class_records = [r for r in splits["train"] if r.cls == cls]
    print(f"[train_lora] class {cls}: {len(class_records)} train images", flush=True)

    backend_name = str(opts.get("backend", cfg.generator.backend))
    try:
        backend = get_backend(backend_name)
    except BackendError as exc:
        return StageResult(
            stage="train_lora", success=False, error=str(exc),
            duration_s=round(time.time() - started, 2),
        )

    out = output_dir() / "lora" / backend_name / cls
    try:
        report = backend.train(cfg, class_records, cls, out, opts)
    except Exception as exc:  # noqa: BLE001 - convert to a structured failure
        import traceback

        traceback.print_exc()
        return StageResult(
            stage="train_lora", success=False,
            error=f"{type(exc).__name__}: {exc}",
            metrics={"cls": cls, "train_images": len(class_records)},
            duration_s=round(time.time() - started, 2),
        )

    return StageResult(
        stage="train_lora",
        success=True,
        metrics=report,
        outputs=[str(Path(report["adapter_dir"]).relative_to(output_dir()))],
        duration_s=round(time.time() - started, 2),
    )
