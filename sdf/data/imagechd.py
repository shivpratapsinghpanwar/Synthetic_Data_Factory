"""Adapter for ImageCHD (Kaggle: xiaoweixumedicalai/imagechd).

110 pediatric CT volumes with congenital heart disease labels (16 types,
multi-label per patient) and 7-structure heart segmentations. MICCAI 2020,
hosted on Kaggle by the original author, Apache-2.0. No identifiable data.

This adapter materializes 2D training data from the 3D volumes:

1. Discover ``*_image.nii.gz`` / ``*_label.nii.gz`` pairs (extracting
   ``archive.zip`` first if the Kaggle mount serves the raw zip) and the
   diagnosis sheet (.xlsx, or a .csv fallback for tests).
2. For each volume, select axial slices where the heart segmentation has at
   least ``MIN_HEART_VOXELS`` labelled voxels - slices that actually show the
   heart - capped at ``MAX_SLICES_PER_PATIENT`` evenly spaced.
3. Normalize each volume by robust percentiles (p1-p99) and write grayscale
   PNGs to a cache directory. Slicing runs once; reruns reuse the cache.

Labels: a patient's class is the '+'-joined sorted set of its diagnosis
types (e.g. ``ASD``, ``ASD+VSD``). Single-type classes are the useful
generation targets; the audit stage surfaces the distribution. ``group_id``
is the patient id, so the existing grouped split guarantees patient-level
separation automatically.
"""

from __future__ import annotations

import csv
import os
import re
import zipfile
from pathlib import Path

from ..config import PipelineConfig, resolve_data_root
from .base import DataError, DatasetAdapter, ImageRecord

MIN_HEART_VOXELS = 400
MAX_SLICES_PER_PATIENT = 40
SLICE_SIZE = 256

# Diagnosis columns in the official xlsx (subset we label with).
TYPE_COLUMNS = [
    "ASD", "VSD", "AVSD", "TOF", "PDA", "TGA", "CA", "PAS", "PA", "PuA",
    "DORV", "CAT", "DAA", "APVC", "AAH", "IAA", "DSVC",
]


class ImageCHDAdapter(DatasetAdapter):
    name = "imagechd"

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    # -- roots ---------------------------------------------------------------
    def _root(self) -> Path:
        root = resolve_data_root(self.cfg)
        if root is None or not root.is_dir():
            raise DataError(
                "dataset root not found. Set [dataset] data_root, or SDF_DATA_ROOT, "
                f"or attach {self.cfg.dataset.kaggle_slug} on Kaggle."
            )
        slug_tail = self.cfg.dataset.kaggle_slug.split("/")[1]
        candidate = root / slug_tail
        return candidate if candidate.is_dir() else root

    def _cache_dir(self) -> Path:
        env = os.environ.get("SDF_CACHE_DIR")
        if env:
            return Path(env) / "imagechd_slices"
        for candidate in (Path("/kaggle/temp"), Path.cwd()):
            if candidate.is_dir():
                return candidate / "imagechd_slices"
        return Path.cwd() / "imagechd_slices"

    def _scratch_dir(self) -> Path:
        for candidate in (Path("/kaggle/temp"), Path.cwd()):
            if candidate.is_dir():
                return candidate / "imagechd_extracted"
        return Path.cwd() / "imagechd_extracted"

    # -- discovery -----------------------------------------------------------
    def _ensure_extracted(self, root: Path) -> Path:
        """If the mount serves archive.zip, extract it once to scratch."""
        if any(root.rglob("*_image.nii.gz")):
            return root
        archives = list(root.glob("*.zip")) + list(root.rglob("archive.zip"))
        if not archives:
            return root  # nothing to extract; discovery will fail loudly later
        scratch = self._scratch_dir()
        marker = scratch / ".extracted"
        if not marker.exists():
            scratch.mkdir(parents=True, exist_ok=True)
            print(f"[imagechd] extracting {archives[0].name} -> {scratch}", flush=True)
            with zipfile.ZipFile(archives[0]) as zf:
                zf.extractall(scratch)
            marker.write_text("ok", encoding="utf-8")
        return scratch

    @staticmethod
    def _patient_id(path: Path) -> str:
        match = re.search(r"(\d+)_image", path.name)
        return match.group(1) if match else path.name.split("_")[0]

    def _find_pairs(self, root: Path) -> dict[str, tuple[Path, Path | None]]:
        pairs: dict[str, tuple[Path, Path | None]] = {}
        for img in sorted(root.rglob("*_image.nii.gz")):
            pid = self._patient_id(img)
            label = img.with_name(img.name.replace("_image", "_label"))
            pairs[pid] = (img, label if label.exists() else None)
        return pairs

    # -- diagnosis sheet -------------------------------------------------------
    def _load_diagnosis(self, root: Path) -> tuple[dict[str, str], str]:
        """patient id -> '+'-joined sorted type string. Returns (map, source)."""
        for csv_path in sorted(root.rglob("*.csv")):
            mapping = self._parse_rows_file(csv_path)
            if mapping:
                return mapping, str(csv_path)
        for xlsx in sorted(root.rglob("*.xlsx")):
            try:
                import pandas as pd

                frame = pd.read_excel(xlsx)
            except ImportError as exc:
                raise DataError(
                    f"pandas is needed to read {xlsx.name} (available on Kaggle)"
                ) from exc
            rows = frame.to_dict(orient="records")
            mapping = self._rows_to_mapping(rows)
            if mapping:
                return mapping, str(xlsx)
        raise DataError(f"no diagnosis sheet (.csv/.xlsx) found under {root}")

    def _parse_rows_file(self, path: Path) -> dict[str, str]:
        try:
            with path.open(newline="", encoding="utf-8-sig") as fh:
                rows = list(csv.DictReader(fh))
        except OSError:
            return {}
        return self._rows_to_mapping(rows)

    @staticmethod
    def _rows_to_mapping(rows: list[dict]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for row in rows:
            keys = {str(k).strip().lower(): k for k in row if k is not None}
            id_key = next(
                (keys[k] for k in ("index", "id", "patient", "patient id", "case") if k in keys),
                None,
            )
            if id_key is None:
                return {}
            pid = re.sub(r"\D", "", str(row[id_key]))
            if not pid:
                continue
            types = []
            for col in TYPE_COLUMNS:
                for cand in (col, col.lower()):
                    if cand in keys:
                        value = row[keys[cand]]
                        try:
                            flag = float(value) > 0
                        except (TypeError, ValueError):
                            flag = str(value).strip().lower() in ("1", "yes", "true", "x")
                        if flag:
                            types.append(col)
                        break
            mapping[pid] = "+".join(sorted(set(types))) if types else "NORMAL"
        return mapping

    # -- slicing ---------------------------------------------------------------
    def _materialize(self, pid: str, image_nii: Path, label_nii: Path | None,
                     out_dir: Path) -> list[Path]:
        import numpy as np

        try:
            import nibabel as nib
        except ImportError as exc:
            raise DataError("nibabel is required to read NIfTI volumes") from exc
        from PIL import Image

        done_marker = out_dir / ".done"
        if done_marker.exists():
            return sorted(out_dir.glob("*.png"))
        out_dir.mkdir(parents=True, exist_ok=True)

        volume = np.asanyarray(nib.load(str(image_nii)).dataobj).astype("float32")
        mask = None
        if label_nii is not None:
            mask = np.asanyarray(nib.load(str(label_nii)).dataobj)

        z_count = volume.shape[2]
        if mask is not None:
            heart = [(mask[:, :, z] > 0).sum() for z in range(z_count)]
            candidates = [z for z in range(z_count) if heart[z] >= MIN_HEART_VOXELS]
        else:
            candidates = list(range(z_count // 4, 3 * z_count // 4))
        if not candidates:
            done_marker.write_text("empty", encoding="utf-8")
            return []
        if len(candidates) > MAX_SLICES_PER_PATIENT:
            stride = len(candidates) / MAX_SLICES_PER_PATIENT
            candidates = [candidates[int(i * stride)] for i in range(MAX_SLICES_PER_PATIENT)]

        lo, hi = np.percentile(volume, [1.0, 99.0])
        span = max(float(hi - lo), 1e-6)
        paths = []
        for z in candidates:
            sl = np.clip((volume[:, :, z] - lo) / span, 0.0, 1.0)
            img = Image.fromarray((sl * 255).astype("uint8"), mode="L")
            img = img.resize((SLICE_SIZE, SLICE_SIZE), Image.LANCZOS).convert("RGB")
            path = out_dir / f"{pid}_z{z:03d}.png"
            img.save(path)
            paths.append(path)
        done_marker.write_text("ok", encoding="utf-8")
        return paths

    # -- indexing --------------------------------------------------------------
    def index(self) -> tuple[list[ImageRecord], dict]:
        root = self._ensure_extracted(self._root())
        pairs = self._find_pairs(root)
        if not pairs:
            raise DataError(
                f"no *_image.nii.gz volumes found under {root} "
                "(is the dataset attached / extracted?)"
            )
        diagnosis, diagnosis_source = self._load_diagnosis(root)

        cache = self._cache_dir()
        records: list[ImageRecord] = []
        unlabelled: list[str] = []
        for pid, (image_nii, label_nii) in sorted(pairs.items()):
            cls = diagnosis.get(pid)
            if cls is None:
                unlabelled.append(pid)
                continue
            for path in self._materialize(pid, image_nii, label_nii, cache / pid):
                records.append(
                    ImageRecord(
                        image_id=path.stem,
                        path=path,
                        cls=cls,
                        group_id=pid,
                        exists=True,
                    )
                )

        report = {
            "adapter": self.name,
            "root": str(root),
            "diagnosis_source": diagnosis_source,
            "patients": len(pairs),
            "patients_unlabelled": unlabelled[:10],
            "slice_cache": str(cache),
            "metadata_rows": len(records),
            "images_on_disk": len(records),
            "missing_files": 0,
            "missing_sample": [],
            "on_disk_not_in_metadata": 0,
        }
        return records, report
