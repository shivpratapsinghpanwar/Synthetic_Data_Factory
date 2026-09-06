"""Publish trained checkpoints as versions of a private weights-only dataset.

Why: Kaggle kernel sessions are ephemeral. A checkpoint trained in one
session must be attachable to later sessions (sampling, resumed training,
onboarding) without retraining. Kaggle datasets are the free persistence
layer - but they are content-moderated, and a predecessor dataset holding a
full run output (generated imagery included) was removed by moderation.
This publisher therefore ships ONLY model weights and small text metadata:
every staged file must pass an extension allowlist, and any other file -
images above all - fails the publish outright.

Contract (verified live): each dataset VERSION is a full snapshot, served
flattened at the dataset root. Only the ``lora/`` checkpoint subtree of a
run's output is staged, and because it is then the single top-level folder,
Kaggle strips it on extraction - consumers see the BACKEND directories at
the root: /kaggle/input/<slug>/mdx/..., /kaggle/input/<slug>/sd15_lora/...
Staging always ships exactly one root folder so this stays consistent
across versions.

CLI: ``python -m kaggle_runner publish [run_id ...]``
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import kaggle_cli
from .config import Config, REPO_ROOT

STAGING_DIR = REPO_ROOT / ".sdf_publish"

# Model weights and small text metadata only. No image, archive, or pickle
# side-channel may ever reach the dataset; publishing anything else is a hard
# error, not a skip - a skip would hide that a checkpoint tree is polluted.
ALLOWED_SUFFIXES = {".pt", ".safetensors", ".json", ".md", ".txt"}

# Subtree of a run's output that holds checkpoints (every backend writes
# under lora/<backend>/...).
CHECKPOINT_SUBDIR = "lora"


class PublishError(RuntimeError):
    pass


def dataset_slug(cfg: Config) -> str:
    """The weights dataset lives beside the runner kernel: <owner>/<slug>-weights."""
    return f"{cfg.kernel.owner}/{cfg.kernel.slug}-weights"


def dataset_exists(slug: str) -> bool:
    res = kaggle_cli.run("datasets", "files", slug, check=False, timeout=120)
    text = res.combined.lower()
    if res.returncode == 0 and "not found" not in text:
        return True
    return False


def stage_folder(cfg: Config, run_ids: list[str], sources: list[Path]) -> Path:
    """Merge one or more runs' checkpoint subtrees into a staging folder
    (later runs win on file collisions) and add dataset-metadata.json.

    Each source is a run's ``output/lora`` directory. Every file must pass
    ALLOWED_SUFFIXES or the whole publish fails.
    """
    for source in sources:
        if not source.is_dir():
            raise PublishError(
                f"nothing to publish: {source} is not a directory "
                f"(runs without a train stage have no checkpoint subtree)")

    slug = dataset_slug(cfg)
    folder = STAGING_DIR / "+".join(run_ids)[:80]
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)

    refused: list[str] = []
    for source in sources:
        for item in source.rglob("*"):
            if not item.is_file():
                continue
            if item.suffix.lower() not in ALLOWED_SUFFIXES:
                refused.append(str(item.relative_to(source)))
                continue
            rel = Path(CHECKPOINT_SUBDIR) / item.relative_to(source)
            target = folder / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
    if refused:
        shutil.rmtree(folder, ignore_errors=True)
        raise PublishError(
            "refusing to publish non-weight files (weights-only dataset): "
            + ", ".join(sorted(refused)[:10])
            + (f" (+{len(refused) - 10} more)" if len(refused) > 10 else ""))

    (folder / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": f"SDF weights ({cfg.kernel.slug})",
                "id": slug,
                "licenses": [{"name": "CC0-1.0"}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return folder


def publish(cfg: Config, run_ids: list[str] | str) -> dict:
    """Create-or-version the weights dataset from one or more runs' checkpoints.

    Multiple run ids merge into one version (later wins on collisions) so a
    single attachable version can carry every backend/class even when they
    were trained in separate sessions.
    """
    if isinstance(run_ids, str):
        run_ids = [run_ids]
    sources = [cfg.runs_path / rid / "output" / CHECKPOINT_SUBDIR for rid in run_ids]
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
