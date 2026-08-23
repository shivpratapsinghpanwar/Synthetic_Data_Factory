"""augment stage: compose the real+synthetic training index.

Usage: python -m sdf run-stage augment [--opt syn_root=/kaggle/input/...-artifacts/<run>]

Inputs
  - the real dataset (adapter + deterministic splits)
  - a synthetic root containing synthetic_manifest.jsonl and the images it
    names (default: $SDF_OUTPUT_DIR, i.e. samples generated in this session;
    point it at an attached artifacts dataset for cross-session use)
  - if stage_quality_gate.json exists under the synthetic root, its flagged
    images are excluded automatically

Output: train_index.csv (path,cls,source) covering real-train + accepted
synthetic - and nothing from val/test, ever. Also val_index.csv and
test_index.csv (real-only) so downstream stages share one index format.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from .. import manifest as manifest_mod
from ..config import PipelineConfig, output_dir
from ..data.base import DataError, get_adapter
from ..splits import grouped_stratified_split
from .base import StageResult

MANIFEST_NAME = "synthetic_manifest.jsonl"


def _flagged_images(syn_root: Path) -> set[str]:
    """Images any quality gate flagged as memorized - excluded from training.

    Gates write stage_quality_gate_<cls>.json (one per class); the union of
    all flags applies. The unsuffixed legacy name is matched by the glob too.
    """
    flagged: set[str] = set()
    for gate in sorted(syn_root.glob("stage_quality_gate*.json")):
        try:
            data = json.loads(gate.read_text(encoding="utf-8"))
            entries = data.get("metrics", {}).get("memorization", {}).get("flagged", [])
            flagged.update(f["image"] for f in entries)
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return flagged


def run(cfg: PipelineConfig, opts: dict | None = None) -> StageResult:
    started = time.time()
    opts = opts or {}
    syn_root = Path(str(opts.get("syn_root", "")) or output_dir())

    try:
        records, _ = get_adapter(cfg).index()
    except DataError as exc:
        return StageResult(
            stage="augment", success=False, error=str(exc),
            duration_s=round(time.time() - started, 2),
        )
    splits, _ = grouped_stratified_split(
        records, seed=cfg.splits.seed,
        val_frac=cfg.splits.val_frac, test_frac=cfg.splits.test_frac,
    )

    # --- synthetic side ----------------------------------------------------
    manifest_path = syn_root / MANIFEST_NAME
    rows = manifest_mod.read(manifest_path)
    flagged = _flagged_images(syn_root)

    accepted: list[tuple[str, str]] = []  # (path, cls)
    skipped = {"flagged": 0, "missing_file": 0, "not_marked_synthetic": 0}
    for row in rows:
        name = Path(row.get("file", "")).name
        if not row.get("synthetic"):
            skipped["not_marked_synthetic"] += 1
            continue
        if name in flagged:
            skipped["flagged"] += 1
            continue
        path = syn_root / row["file"]
        if not path.exists():
            skipped["missing_file"] += 1
            continue
        accepted.append((str(path), row["cls"]))

    # --- write indexes -----------------------------------------------------
    out = output_dir()
    out.mkdir(parents=True, exist_ok=True)

    def write_index(name: str, entries: list[tuple[str, str, str]]) -> str:
        path = out / name
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["path", "cls", "source"])
            writer.writerows(entries)
        return name

    train_entries = [(str(r.path), r.cls, "real") for r in splits["train"] if r.exists]
    train_entries += [(p, c, "synthetic") for p, c in accepted]
    files = [
        write_index("train_index.csv", train_entries),
        write_index("val_index.csv",
                    [(str(r.path), r.cls, "real") for r in splits["val"] if r.exists]),
        write_index("test_index.csv",
                    [(str(r.path), r.cls, "real") for r in splits["test"] if r.exists]),
    ]

    per_class: dict[str, dict[str, int]] = {}
    for _, cls, source in train_entries:
        per_class.setdefault(cls, {"real": 0, "synthetic": 0})[source] += 1

    metrics = {
        "syn_root": str(syn_root),
        "manifest_rows": len(rows),
        "accepted_synthetic": len(accepted),
        "skipped": skipped,
        "train_total": len(train_entries),
        "per_class": per_class,
    }
    print(f"[augment] accepted {len(accepted)}/{len(rows)} synthetic; "
          f"train index = {len(train_entries)} rows", flush=True)
    print(f"[augment] per-class: {per_class}", flush=True)

    # Success requires a usable index; synthetic content is optional (the
    # real-only baseline uses the same stage with no manifest present).
    return StageResult(
        stage="augment",
        success=len(train_entries) > 0,
        metrics=metrics,
        outputs=files,
        error="" if train_entries else "empty training index",
        duration_s=round(time.time() - started, 2),
    )
