"""Generic adapter for class-per-folder image datasets (private client data).

Handles the two layouts the cond-a/cond-b deliveries use:

    root/<class>/*.jpg                              flat classes
    root/{train,test,valid}/<class>/*.jpg           curated splits (honored)

Curated split dirs map train->train, test->test, valid->val and are HONORED
verbatim via ImageRecord.fixed_split: Roboflow-style exports can contain
augmented near-duplicates of one base image, and re-splitting randomly would
leak variants across the evaluation boundary. Flat-class images are left to
the deterministic splitter.

Sidecar Pascal-VOC XMLs (same stem as the image) are indexed as ROI metadata:
``roi_for(record)`` returns the first bounding box, which is how the
face-safe lip-region crop path gets its regions. XMLs are never ImageRecords.

Files under split dirs WITHOUT a class subfolder (e.g. a flat ``valid/``) are
counted in the report but excluded - there is no label to train against.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import re

from ..config import PipelineConfig, resolve_data_root
from .base import DataError, DatasetAdapter, ImageRecord

# Roboflow-style export suffix: base_jpg.rf.<32 hex>.jpg - every augmented
# variant of one base image shares the prefix before this marker.
_RF_SUFFIX = re.compile(r"_(jpe?g|png|bmp|webp)\.rf\.[0-9a-f]{16,}$", re.IGNORECASE)


def base_stem(filename: str) -> str:
    """Augmentation-invariant base name: variants of one source image map to
    the same value, so the splitter keeps them on one side of every boundary."""
    stem = filename.rsplit(".", 1)[0]
    return _RF_SUFFIX.sub("", stem).lower()

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLIT_DIRS = {"train": "train", "test": "test", "valid": "val", "val": "val"}
MAX_FILES = 200_000


class FolderClassAdapter(DatasetAdapter):
    name = "folder_class"

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def _root(self) -> Path:
        root = resolve_data_root(self.cfg)
        if root is None or not root.is_dir():
            raise DataError(
                "dataset root not found. Set [dataset] data_root or SDF_DATA_ROOT, "
                "or attach the private dataset on Kaggle."
            )
        slug_tail = self.cfg.dataset.kaggle_slug.split("/")[-1]
        candidate = root / slug_tail
        return candidate if candidate.is_dir() else root

    def index(self) -> tuple[list[ImageRecord], dict]:
        root = self._root()
        records: list[ImageRecord] = []
        unlabelled = 0
        xml_count = 0
        seen = 0

        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            seen += 1
            if seen > MAX_FILES:
                raise DataError(f"more than {MAX_FILES} files under {root}")
            if path.suffix.lower() == ".xml":
                xml_count += 1
                continue
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            rel = path.relative_to(root)
            parts = rel.parts
            if len(parts) >= 3 and parts[0].lower() in SPLIT_DIRS:
                # test/valid are honored verbatim; train stays FREE so the
                # deterministic splitter can carve val from it (these exports
                # ship no labelled val, and val must never come from test).
                mapped = SPLIT_DIRS[parts[0].lower()]
                fixed = mapped if mapped in ("test", "val") else ""
                cls = parts[1]
            elif len(parts) == 2 and parts[0].lower() not in SPLIT_DIRS:
                fixed = ""
                cls = parts[0]
            else:
                unlabelled += 1  # split dir without class subfolder, or root file
                continue

            # The relative path itself is the id: injective by construction
            # (a "__"-joined form aliased train/cls/a__b.jpg with
            # train/cls__a/b.jpg). Group by split-scope + class +
            # augmentation-invariant base stem: Roboflow-style augmented
            # siblings (which always share a split dir) can never straddle a
            # carved boundary, while coincidentally equal bare stems across
            # curated splits (different photos both named 1.jpg) stay
            # independent instead of tripping the leakage assertion.
            image_id = rel.as_posix()
            scope = parts[0].lower() if parts[0].lower() in SPLIT_DIRS else "flat"
            records.append(
                ImageRecord(
                    image_id=image_id,
                    path=path,
                    cls=cls,
                    group_id=f"{scope}::{cls}::{base_stem(rel.name)}",
                    exists=True,
                    fixed_split=fixed,
                )
            )

        if not records:
            raise DataError(f"no class-labelled images found under {root}")

        report = {
            "adapter": self.name,
            "root": str(root),
            "metadata_csv": "(folder structure)",
            "metadata_rows": len(records),
            "images_on_disk": len(records),
            "missing_files": 0,
            "missing_sample": [],
            "on_disk_not_in_metadata": unlabelled,
            "voc_xml_sidecars": xml_count,
            "fixed_split_counts": {
                name: sum(1 for r in records if r.fixed_split == name)
                for name in ("train", "val", "test")
            },
        }
        return records, report


def roi_for(record: ImageRecord) -> tuple[int, int, int, int] | None:
    """First VOC bounding box from the image's sidecar XML, as (l, t, r, b)."""
    xml_path = record.path.with_suffix(".xml")
    if not xml_path.exists():
        return None
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return None
    box = tree.find(".//object/bndbox")
    if box is None:
        return None
    try:
        values = {tag: int(float(box.findtext(tag, "nan"))) for tag in
                  ("xmin", "ymin", "xmax", "ymax")}
    except (TypeError, ValueError):
        return None
    left, top, right, bottom = (
        values["xmin"], values["ymin"], values["xmax"], values["ymax"]
    )
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom
