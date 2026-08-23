"""sample stage: generate synthetic images from a trained class LoRA.

Usage: python -m sdf run-stage sample --opt cls=df --opt count=16
The adapter is found under $SDF_OUTPUT_DIR/lora/<cls>/adapter by default
(i.e. produced by train_lora in the same kernel session), or at --opt
adapter_dir=<path>. Every image gets a provenance manifest row.
"""

from __future__ import annotations

import time
from pathlib import Path

from .. import manifest
from ..config import PipelineConfig, output_dir
from ..gen import sd15_lora
from ..gen.prompts import CLASS_PROMPTS
from .base import StageResult

MANIFEST_NAME = "synthetic_manifest.jsonl"


def run(cfg: PipelineConfig, opts: dict | None = None) -> StageResult:
    started = time.time()
    opts = opts or {}
    cls = str(opts.get("cls", ""))
    if cls not in CLASS_PROMPTS:
        return StageResult(
            stage="sample", success=False,
            error=f"pass --opt cls=<class>; got {cls!r}, known: {sorted(CLASS_PROMPTS)}",
            duration_s=round(time.time() - started, 2),
        )

    adapter_dir = Path(
        str(opts.get("adapter_dir", ""))
        or output_dir() / "lora" / cls / sd15_lora.ADAPTER_DIR_NAME
    )
    if not adapter_dir.is_dir():
        return StageResult(
            stage="sample", success=False,
            error=f"adapter not found: {adapter_dir} (run train_lora first "
                  f"or pass --opt adapter_dir=...)",
            duration_s=round(time.time() - started, 2),
        )

    out = output_dir() / "synthetic" / cls
    try:
        rows = sd15_lora.sample(cfg, cls, adapter_dir, out, opts)
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        return StageResult(
            stage="sample", success=False,
            error=f"{type(exc).__name__}: {exc}",
            metrics={"cls": cls, "adapter_dir": str(adapter_dir)},
            duration_s=round(time.time() - started, 2),
        )

    manifest_path = output_dir() / MANIFEST_NAME
    for row in rows:
        record = manifest.new_record(**row)
        record["file"] = f"synthetic/{cls}/{row['file']}"
        manifest.append(manifest_path, record)

    return StageResult(
        stage="sample",
        success=len(rows) > 0,
        metrics={
            "cls": cls,
            "count": len(rows),
            "adapter_dir": str(adapter_dir),
            "manifest": MANIFEST_NAME,
            "first_files": [r["file"] for r in rows[:5]],
        },
        outputs=[f"synthetic/{cls}/", MANIFEST_NAME],
        error="" if rows else "no images generated",
        duration_s=round(time.time() - started, 2),
    )
