"""Adapter for HAM10000 (Kaggle: kmader/skin-cancer-mnist-ham10000).

10,015 dermoscopy images of 7 skin-lesion classes:
    akiec, bcc, bkl, df, mel, nv, vasc
Metadata CSV columns used: lesion_id, image_id, dx. Multiple images can share a
lesion_id (same physical lesion re-imaged) - that is why splits group by it.

The adapter discovers layout instead of hardcoding it: on Kaggle the dataset
mounts under /kaggle/input/skin-cancer-mnist-ham10000/ with images split
across HAM10000_images_part_1/ and part_2/, but local mirrors vary.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ..config import PipelineConfig, resolve_data_root
from .base import DataError, DatasetAdapter, ImageRecord

REQUIRED_COLUMNS = {"lesion_id", "image_id", "dx"}

# Safety caps so a scan of an unexpectedly huge root cannot hang a stage.
MAX_CSV_CANDIDATES = 200
MAX_IMAGE_FILES = 100_000


class HAM10000Adapter(DatasetAdapter):
    name = "ham10000"

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    # -- discovery -----------------------------------------------------------
    def _root(self) -> Path:
        root = resolve_data_root(self.cfg)
        if root is None or not root.is_dir():
            raise DataError(
                "dataset root not found. Set [dataset] data_root, or SDF_DATA_ROOT, "
                "or run on Kaggle with the dataset attached "
                f"({self.cfg.dataset.kaggle_slug})."
            )
        # Prefer the slug-named subdirectory when the root is a mount point
        # like /kaggle/input that may hold several datasets.
        slug_tail = self.cfg.dataset.kaggle_slug.split("/")[1]
        candidate = root / slug_tail
        return candidate if candidate.is_dir() else root

    def _find_metadata(self, root: Path) -> Path:
        candidates = []
        for i, p in enumerate(root.rglob("*.csv")):
            if i >= MAX_CSV_CANDIDATES:
                break
            candidates.append(p)
        # Rank: filename mentioning 'metadata' first, then shortest path.
        candidates.sort(key=lambda p: ("metadata" not in p.name.lower(), len(str(p))))
        for p in candidates:
            try:
                with p.open(newline="", encoding="utf-8-sig") as fh:
                    header = next(csv.reader(fh), [])
            except OSError:
                continue
            if REQUIRED_COLUMNS.issubset({c.strip() for c in header}):
                return p
        raise DataError(
            f"no CSV under {root} has columns {sorted(REQUIRED_COLUMNS)} "
            f"(checked {len(candidates)} candidates)"
        )

    def _image_lookup(self, root: Path) -> dict[str, Path]:
        lookup: dict[str, Path] = {}
        count = 0
        for p in root.rglob("*.jpg"):
            count += 1
            if count > MAX_IMAGE_FILES:
                break
            lookup.setdefault(p.stem, p)
        return lookup

    # -- indexing ------------------------------------------------------------
    def index(self) -> tuple[list[ImageRecord], dict]:
        root = self._root()
        metadata = self._find_metadata(root)
        lookup = self._image_lookup(root)

        records: list[ImageRecord] = []
        missing: list[str] = []
        with metadata.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                image_id = (row.get("image_id") or "").strip()
                cls = (row.get("dx") or "").strip()
                group = (row.get("lesion_id") or "").strip() or image_id
                if not image_id or not cls:
                    continue
                path = lookup.get(image_id)
                if path is None:
                    missing.append(image_id)
                records.append(
                    ImageRecord(
                        image_id=image_id,
                        path=path or root / f"{image_id}.jpg",
                        cls=cls,
                        group_id=group,
                        exists=path is not None,
                    )
                )

        indexed_ids = {r.image_id for r in records}
        report = {
            "adapter": self.name,
            "root": str(root),
            "metadata_csv": str(metadata),
            "metadata_rows": len(records),
            "images_on_disk": len(lookup),
            "missing_files": len(missing),
            "missing_sample": missing[:10],
            "on_disk_not_in_metadata": len(set(lookup) - indexed_ids),
        }
        return records, report
