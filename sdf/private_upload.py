"""De-identify and upload private client imagery as a PRIVATE Kaggle dataset.

Usage: python -m sdf upload-private <source_dir> <owner/slug> [--title T] [--sanitize]

Default mode is VERBATIM: every file is copied byte-for-byte unmodified into
staging (owner's instruction: the data must not be changed), then uploaded
with `kaggle datasets create` - private by default; this tool never passes
the public flag - or `version` when the dataset already exists.

--sanitize opts into a transforming mode instead: images re-encoded via PIL
(drops EXIF/GPS by construction) and XML sidecars scrubbed of absolute paths.

Either way the SOURCE tree is never touched, the staging tree lives under
.sdf_publish/ (gitignored), and nothing here ever goes near git.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from kaggle_runner import kaggle_cli
from kaggle_runner.artifacts import dataset_exists
from kaggle_runner.config import REPO_ROOT

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
STAGING_ROOT = REPO_ROOT / ".sdf_publish"


class UploadError(RuntimeError):
    pass


def sanitize_tree(source: Path, staging: Path) -> dict:
    """Re-encode images (EXIF gone) and copy XML sidecars into ``staging``."""
    from PIL import Image

    if not source.is_dir():
        raise UploadError(f"source is not a directory: {source}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    counts = {"images": 0, "xml": 0, "skipped": 0, "unreadable": 0}
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(source)
        target = staging / rel
        suffix = path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            # Keep the original suffix: renaming .jpeg -> .jpg collides with a
            # sibling .jpg of the same stem and silently overwrites it. PIL
            # picks the output format from the suffix; re-encoding still
            # drops EXIF for every format.
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with Image.open(path) as img:
                    clean = Image.new("RGB", img.size)
                    clean.paste(img.convert("RGB"))
                if target.suffix == ".png":
                    clean.save(target, "PNG")
                else:
                    clean.save(target, "JPEG", quality=95)
                counts["images"] += 1
            except OSError:
                counts["unreadable"] += 1
        elif suffix == ".xml":
            text = path.read_text(encoding="utf-8", errors="replace")
            # keep only the box payload's integrity: refuse absolute paths
            if re.search(r"[A-Za-z]:\\|/home/|/Users/", text):
                text = re.sub(r"<path>.*?</path>", "<path />", text, flags=re.DOTALL)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            counts["xml"] += 1
        else:
            counts["skipped"] += 1
    if counts["images"] == 0:
        raise UploadError(f"no images found under {source}")
    return counts


def verbatim_tree(source: Path, staging: Path) -> dict:
    """Copy every file unmodified. The data owner's terms: change nothing."""
    if not source.is_dir():
        raise UploadError(f"source is not a directory: {source}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    counts = {"images": 0, "xml": 0, "skipped": 0, "unreadable": 0}
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        target = staging / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        suffix = path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            counts["images"] += 1
        elif suffix == ".xml":
            counts["xml"] += 1
        else:
            counts["skipped"] += 1
    if counts["images"] == 0:
        raise UploadError(f"no images found under {source}")
    return counts


def upload(source: Path, slug: str, title: str = "", sanitize: bool = False) -> dict:
    if slug.count("/") != 1:
        raise UploadError(f'slug must be "owner/dataset-name", got {slug!r}')
    staging = STAGING_ROOT / ("private_" + slug.split("/")[1])
    counts = (sanitize_tree if sanitize else verbatim_tree)(source, staging)

    (staging / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": title or slug.split("/")[1],
                "id": slug,
                "licenses": [{"name": "other"}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if counts.get("skipped"):
        print(f"[upload-private] note: {counts['skipped']} non-image/non-xml "
              f"file(s) included verbatim", flush=True)

    try:
        if dataset_exists(slug):
            res = kaggle_cli.run(
                "datasets", "version", "-p", str(staging), "-m", "refresh",
                "--dir-mode", "zip", timeout=3600,
            )
            action = "versioned"
        else:
            # datasets create is PRIVATE by default; -u (public) is never passed.
            res = kaggle_cli.run(
                "datasets", "create", "-p", str(staging), "--dir-mode", "zip",
                timeout=3600,
            )
            action = "created"
    finally:
        # The staging tree holds private imagery - never leave it behind,
        # least of all after a failure.
        shutil.rmtree(staging, ignore_errors=True)
    return {
        "action": action,
        "dataset": slug,
        "mode": "sanitize" if sanitize else "verbatim",
        "counts": counts,
        "cli_tail": res.combined[-300:],
    }


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    title = ""
    sanitize = "--sanitize" in args
    if sanitize:
        args = [a for a in args if a != "--sanitize"]
    if "--title" in args:
        i = args.index("--title")
        if i + 1 >= len(args):
            print("--title requires a value", file=sys.stderr)
            return 2
        title = args[i + 1]
        args = args[:i] + args[i + 2:]
    try:
        result = upload(Path(args[0]), args[1], title, sanitize=sanitize)
    except (UploadError, kaggle_cli.KaggleCliError) as exc:
        print(f"upload failed: {exc}", file=sys.stderr)
        return 1
    print(f"{result['action']} PRIVATE dataset {result['dataset']} "
          f"[{result['mode']}]: {result['counts']['images']} images, "
          f"{result['counts']['xml']} annotations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
