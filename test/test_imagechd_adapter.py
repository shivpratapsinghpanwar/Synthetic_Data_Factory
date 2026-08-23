"""Offline tests for the ImageCHD adapter using a synthetic NIfTI fixture."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sdf import config  # noqa: E402
from sdf.data.base import get_adapter  # noqa: E402

VOLUME_SHAPE = (32, 32, 12)
HEART_SLICES = (4, 5, 6, 7)  # slices given a segmentation blob


def build_fixture(root: Path) -> None:
    import nibabel as nib
    import numpy as np

    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    patients = {"1001": {"ASD": 1}, "1002": {"ASD": 1, "VSD": 1}, "1003": {}}

    for pid in patients:
        volume = rng.normal(100, 40, VOLUME_SHAPE).astype("float32")
        mask = np.zeros(VOLUME_SHAPE, dtype="uint8")
        for z in HEART_SLICES:
            mask[3:28, 3:28, z] = 1  # 625 voxels >= MIN_HEART_VOXELS (400)
        nib.save(nib.Nifti1Image(volume, np.eye(4)), str(root / f"ct_{pid}_image.nii.gz"))
        nib.save(nib.Nifti1Image(mask, np.eye(4)), str(root / f"ct_{pid}_label.nii.gz"))

    with (root / "diagnosis.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["Index", "ASD", "VSD", "TOF"])
        writer.writeheader()
        for pid, types in patients.items():
            row = {"Index": pid, "ASD": 0, "VSD": 0, "TOF": 0}
            row.update(types)
            writer.writerow(row)


def _cfg(root: Path, cache: Path) -> config.PipelineConfig:
    cfg = config.load(REPO_ROOT / "pipeline_chd.toml")
    cfg.dataset.data_root = str(root)
    os.environ["SDF_CACHE_DIR"] = str(cache)
    return cfg


def _cleanup():
    os.environ.pop("SDF_CACHE_DIR", None)


def test_imagechd_slices_only_heart_slices(tmp_path):
    build_fixture(tmp_path / "data")
    try:
        records, report = get_adapter(_cfg(tmp_path / "data", tmp_path / "cache")).index()
    finally:
        _cleanup()

    assert report["patients"] == 3
    # 3 patients x 4 heart slices
    assert len(records) == 12
    z_values = {int(r.image_id.split("_z")[1]) for r in records}
    assert z_values == set(HEART_SLICES)
    assert all(r.exists and r.path.suffix == ".png" for r in records)


def test_imagechd_class_labels_and_grouping(tmp_path):
    build_fixture(tmp_path / "data")
    try:
        records, _ = get_adapter(_cfg(tmp_path / "data", tmp_path / "cache")).index()
    finally:
        _cleanup()

    by_pid = {}
    for rec in records:
        by_pid.setdefault(rec.group_id, set()).add(rec.cls)
    assert by_pid["1001"] == {"ASD"}
    assert by_pid["1002"] == {"ASD+VSD"}  # multi-label joins sorted with '+'
    assert by_pid["1003"] == {"NORMAL"}


def test_imagechd_cache_reused(tmp_path):
    build_fixture(tmp_path / "data")
    try:
        adapter = get_adapter(_cfg(tmp_path / "data", tmp_path / "cache"))
        first, _ = adapter.index()
        mtimes = {r.path: r.path.stat().st_mtime_ns for r in first}
        second, _ = adapter.index()
        assert len(second) == len(first)
        for rec in second:  # files were not rewritten
            assert rec.path.stat().st_mtime_ns == mtimes[rec.path]
    finally:
        _cleanup()


def test_imagechd_patient_level_split(tmp_path):
    from sdf.splits import grouped_stratified_split

    build_fixture(tmp_path / "data")
    try:
        records, _ = get_adapter(_cfg(tmp_path / "data", tmp_path / "cache")).index()
    finally:
        _cleanup()

    splits, stats = grouped_stratified_split(records, seed=1, val_frac=0.1, test_frac=0.2)
    owner = {}
    for name, recs in splits.items():
        for rec in recs:
            assert owner.setdefault(rec.group_id, name) == name
    # tiny classes (1 patient each) fall entirely into train, with warnings
    assert stats["warnings"]


def _run_all():
    import inspect
    import tempfile

    mod = sys.modules[__name__]
    tests = [(n, f) for n, f in vars(mod).items() if n.startswith("test_") and callable(f)]
    failures = 0
    for name, fn in tests:
        try:
            if "tmp_path" in inspect.signature(fn).parameters:
                with tempfile.TemporaryDirectory() as td:
                    fn(Path(td))
            else:
                fn()
            print(f"  ok    {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
