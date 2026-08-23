"""Publish run outputs as versions of a private Kaggle dataset.

Why: Kaggle kernel sessions are ephemeral. A LoRA adapter trained in one
session must be attachable to later sessions (sampling, resumed training)
without retraining. Kaggle datasets are the free persistence layer: publish a
run's output folder as a new version of one private dataset, then attach it
via ``dataset_sources`` like any other dataset.

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


def stage_folder(cfg: Config, run_id: str, source: Path) -> Path:
    """Copy a run's output into a staging folder with dataset-metadata.json."""
    if not source.is_dir():
        raise PublishError(f"nothing to publish: {source} is not a directory")

    slug = dataset_slug(cfg)
    folder = STAGING_DIR / run_id
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)

    # Kaggle versions replace the whole file set; nest under the run id so one
    # dataset accumulates every published run side by side.
    shutil.copytree(source, folder / run_id)

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


def publish(cfg: Config, run_id: str) -> dict:
    """Create-or-version the artifacts dataset from runs/<run_id>/output."""
    source = cfg.runs_path / run_id / "output"
    folder = stage_folder(cfg, run_id, source)
    slug = dataset_slug(cfg)

    if dataset_exists(slug):
        res = kaggle_cli.run(
            "datasets", "version", "-p", str(folder), "-m", f"run {run_id}",
            "--dir-mode", "zip", timeout=1800,
        )
        action = "versioned"
    else:
        res = kaggle_cli.run(
            "datasets", "create", "-p", str(folder), "--private", "--dir-mode", "zip",
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
