"""Publish run outputs as versions of a private Kaggle dataset.

Why: Kaggle kernel sessions are ephemeral. A LoRA adapter trained in one
session must be attachable to later sessions (sampling, resumed training)
without retraining. Kaggle datasets are the free persistence layer: publish a
run's output folder as a new version of one private dataset, then attach it
via ``dataset_sources`` like any other dataset.

Contract (verified live): each dataset VERSION is a full snapshot of one
run's output, served flattened at the dataset root - Kaggle extracts the
zip-mode upload and strips the staging folder. Consumers therefore see
/kaggle/input/<slug>/lora/..., synthetic/..., etc. directly. The version
message records which run a version came from.

CLI: ``python -m kaggle_runner publish [run_id]``
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import kaggle_cli
from .config import Config, REPO_ROOT

STAGING_DIR = REPO_ROOT / ".sdf_publish"


class PublishError(RuntimeError):
    pass


def dataset_slug(cfg: Config) -> str:
    """The artifacts dataset lives beside the runner kernel: <owner>/<slug>-artifacts."""
    return f"{cfg.kernel.owner}/{cfg.kernel.slug}-artifacts"


def dataset_exists(slug: str) -> bool:
    res = kaggle_cli.run("datasets", "files", slug, check=False, timeout=120)
    text = res.combined.lower()
    if res.returncode == 0 and "not found" not in text:
        return True
    return False


def stage_folder(cfg: Config, run_ids: list[str], sources: list[Path]) -> Path:
    """Merge one or more runs' outputs into a staging folder (later runs win
    on file collisions) and add dataset-metadata.json."""
    for source in sources:
        if not source.is_dir():
            raise PublishError(f"nothing to publish: {source} is not a directory")

    slug = dataset_slug(cfg)
    folder = STAGING_DIR / "+".join(run_ids)[:80]
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)

    # Content sits at the staging root: Kaggle flattens zip-mode uploads
    # server-side, so per-run nesting would not survive anyway.
    #
    # Collision policy:
    #   *.jsonl            -> concatenate (manifests must accumulate; dropping
    #                         a run's rows would silently exclude its images
    #                         from augmentation)
    #   stage_*.json       -> keep both under a deduped name (augment globs
    #                         stage_quality_gate*.json and unions the flags)
    #   everything else    -> later run wins
    for source in sources:
        for item in source.rglob("*"):
            if not item.is_file():
                continue
            rel = item.relative_to(source)
            target = folder / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and item.suffix == ".jsonl":
                with target.open("ab") as out_fh, item.open("rb") as in_fh:
                    out_fh.write(in_fh.read())
            elif target.exists() and item.name.startswith("stage_"):
                n = 2
                while (dedup := target.with_name(f"{target.stem}.{n}{target.suffix}")).exists():
                    n += 1
                shutil.copy2(item, dedup)
            else:
                shutil.copy2(item, target)

    (folder / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": f"SDF artifacts ({cfg.kernel.slug})",
                "id": slug,
                "licenses": [{"name": "CC0-1.0"}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return folder


def publish(cfg: Config, run_ids: list[str] | str) -> dict:
    """Create-or-version the artifacts dataset from one or more runs' output.

    Multiple run ids merge into one version (later wins on collisions) so a
    single attachable version can carry every class even when classes were
    generated in separate sessions.
    """
    if isinstance(run_ids, str):
        run_ids = [run_ids]
    sources = [cfg.runs_path / rid / "output" for rid in run_ids]
    folder = stage_folder(cfg, run_ids, sources)
    slug = dataset_slug(cfg)
    run_id = "+".join(run_ids)

    if dataset_exists(slug):
        res = kaggle_cli.run(
            "datasets", "version", "-p", str(folder), "-m", f"run {run_id}",
            "--dir-mode", "zip", timeout=1800,
        )
        action = "versioned"
    else:
        # No --private flag exists: datasets are private by default (-u opts
        # into public, which we never do here).
        res = kaggle_cli.run(
            "datasets", "create", "-p", str(folder), "--dir-mode", "zip",
            timeout=1800,
        )
        action = "created"

    shutil.rmtree(folder, ignore_errors=True)
    return {
        "action": action,
        "dataset": slug,
        "run_id": run_id,
        "url": f"https://www.kaggle.com/datasets/{slug}",
        "cli_output": res.combined[-500:],
    }
