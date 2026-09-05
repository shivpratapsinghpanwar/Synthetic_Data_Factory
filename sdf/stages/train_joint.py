"""train_joint stage: train ONE shared generator across several studies.

Unlike train_lora (one class of one dataset per run), this stage combines the
train splits of every listed pipeline config into a single labelled corpus
("<condition>/<class>" labels) and trains the mdx mixture-of-experts
diffusion transformer on all of it jointly.

Usage:
    python -m sdf run-stage train_joint --config pipeline_cond_a.toml \
        --opt configs=pipeline_cond_a.toml,pipeline_cond_b.toml \
        --opt data_roots=/kaggle/input/a,/kaggle/input/b \
        --opt steps=4000 --opt resolution=128 --opt tag=joint

configs= defaults to the primary --config alone. data_roots= (same order,
same length) overrides each config's data_root - needed on Kaggle where every
private dataset mounts at its own /kaggle/input/<slug> path and the
gitignored local overlays are absent. Onboarding opts (resume_from=,
freeze_trunk=1) pass straight through to the backend.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..config import PipelineConfig, load as load_config, output_dir
from ..data.base import DataError, get_adapter
from ..splits import grouped_stratified_split
from .base import StageResult

BACKEND = "mdx"


def _gather(cfg) -> tuple[list, dict]:
    """Train-split labelled items + per-study report for one config."""
    from ..gen import mdx

    records, _report = get_adapter(cfg).index()
    splits, _stats = grouped_stratified_split(
        records, seed=cfg.splits.seed,
        val_frac=cfg.splits.val_frac, test_frac=cfg.splits.test_frac,
    )
    labelled = mdx._labelled_from_records(cfg, splits["train"])
    condition = mdx._condition_tag(cfg)
    per_class: dict[str, int] = {}
    for _p, _r, label, _c in labelled:
        per_class[label] = per_class.get(label, 0) + 1
    info = {
        "condition": condition,
        "roi_only": bool(getattr(cfg.generator, "roi_only", False)),
        "train_images": per_class,
    }
    return labelled, info


def run(cfg: PipelineConfig, opts: dict | None = None) -> StageResult:
    started = time.time()
    opts = opts or {}

    config_paths = [p for p in str(opts.get("configs", "")).split(",") if p.strip()]
    data_roots = [p for p in str(opts.get("data_roots", "")).split(",") if p.strip()]
    if data_roots and config_paths and len(data_roots) != len(config_paths):
        return StageResult(
            stage="train_joint", success=False,
            error=f"data_roots has {len(data_roots)} entries for "
                  f"{len(config_paths)} configs - they must pair up",
            duration_s=round(time.time() - started, 2),
        )

    configs: list[PipelineConfig] = []
    try:
        if config_paths:
            for i, path in enumerate(config_paths):
                c = load_config(path.strip())
                if data_roots:
                    c.dataset.data_root = data_roots[i].strip()
                configs.append(c)
        else:
            configs = [cfg]
    except Exception as exc:  # noqa: BLE001 - config errors become stage failures
        return StageResult(
            stage="train_joint", success=False, error=f"{type(exc).__name__}: {exc}",
            duration_s=round(time.time() - started, 2),
        )

    labelled: list = []
    studies: list[dict] = []
    seen_conditions: set[str] = set()
    try:
        for c in configs:
            part, info = _gather(c)
            if info["condition"] in seen_conditions:
                raise DataError(
                    f"duplicate condition tag {info['condition']!r} - set a "
                    "distinct [dataset] condition in each config"
                )
            seen_conditions.add(info["condition"])
            labelled.extend(part)
            studies.append(info)
            print(f"[train_joint] {info['condition']}: {info['train_images']} "
                  f"(roi_only={info['roi_only']})", flush=True)
    except DataError as exc:
        return StageResult(
            stage="train_joint", success=False, error=str(exc),
            duration_s=round(time.time() - started, 2),
        )

    if not labelled:
        return StageResult(
            stage="train_joint", success=False,
            error="no usable training images across the listed configs",
            duration_s=round(time.time() - started, 2),
        )

    from ..gen import mdx

    out = output_dir() / "lora" / BACKEND / "joint"
    try:
        report = mdx.train_joint(labelled, out, opts, seed=cfg.splits.seed)
    except Exception as exc:  # noqa: BLE001 - convert to a structured failure
        import traceback

        traceback.print_exc()
        return StageResult(
            stage="train_joint", success=False,
            error=f"{type(exc).__name__}: {exc}",
            metrics={"studies": studies, "total_images": len(labelled)},
            duration_s=round(time.time() - started, 2),
        )

    report["studies"] = studies
    return StageResult(
        stage="train_joint",
        success=True,
        metrics=report,
        outputs=[str(Path(report["adapter_dir"]).relative_to(output_dir()))],
        duration_s=round(time.time() - started, 2),
    )
