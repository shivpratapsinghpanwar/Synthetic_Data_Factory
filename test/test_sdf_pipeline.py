"""Offline tests for the sdf pipeline. No network, no real dataset needed.

Run with:  python -m pytest test/ -q      (or: python test/test_sdf_pipeline.py)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sdf import config, manifest, splits as splits_mod  # noqa: E402
from sdf.data.base import DataError, ImageRecord, get_adapter  # noqa: E402
from sdf.stages import REGISTRY, audit, write_result  # noqa: E402


# ------------------------------------------------------------------- fixture
CLASSES = {
    # cls -> number of lesions (each lesion gets 1-2 images)
    "nv": 30,
    "mel": 12,
    "df": 6,
}


def build_fixture(root: Path) -> int:
    """Create a HAM10000-shaped dataset. Returns total image count."""
    from PIL import Image

    part1 = root / "HAM10000_images_part_1"
    part2 = root / "HAM10000_images_part_2"
    part1.mkdir(parents=True)
    part2.mkdir(parents=True)

    rows = []
    img_no = 0
    for cls, lesions in CLASSES.items():
        for lesion_idx in range(lesions):
            lesion_id = f"HAM_{cls}_{lesion_idx:04d}"
            images_of_lesion = 2 if lesion_idx % 3 == 0 else 1
            for _ in range(images_of_lesion):
                image_id = f"ISIC_{img_no:07d}"
                img_no += 1
                folder = part1 if img_no % 2 else part2
                Image.new("RGB", (60, 45), color=(img_no % 255, 90, 40)).save(
                    folder / f"{image_id}.jpg", "JPEG"
                )
                rows.append({"lesion_id": lesion_id, "image_id": image_id, "dx": cls})

    with (root / "HAM10000_metadata.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["lesion_id", "image_id", "dx"])
        writer.writeheader()
        writer.writerows(rows)

    # A decoy CSV that must NOT be picked as metadata.
    (root / "hmnist_8_8_L.csv").write_text("pixel0,pixel1,label\n1,2,0\n", encoding="utf-8")
    return img_no


def fixture_cfg(root: Path) -> config.PipelineConfig:
    cfg = config.load()
    cfg.dataset.data_root = str(root)
    return cfg


# -------------------------------------------------------------------- config
def test_pipeline_config_loads():
    cfg = config.load()
    assert cfg.dataset.name == "ham10000"
    assert cfg.generator.backend == "sd15_lora"
    assert 0 < cfg.splits.val_frac < 1


def test_pipeline_config_rejects_bad_fractions():
    cfg = config.load()
    cfg.splits.test_frac = 0.6
    try:
        config.validate(cfg)
    except config.ConfigError:
        return
    raise AssertionError("expected ConfigError for test_frac=0.6")


# ------------------------------------------------------------------- adapter
def test_adapter_indexes_fixture(tmp_path):
    total = build_fixture(tmp_path)
    records, report = get_adapter(fixture_cfg(tmp_path)).index()
    assert len(records) == total
    assert report["missing_files"] == 0
    assert report["metadata_csv"].endswith("HAM10000_metadata.csv")
    assert all(r.exists for r in records)


def test_adapter_reports_missing_files(tmp_path):
    build_fixture(tmp_path)
    victim = next((tmp_path / "HAM10000_images_part_1").glob("*.jpg"))
    victim.unlink()
    _, report = get_adapter(fixture_cfg(tmp_path)).index()
    assert report["missing_files"] == 1
    assert len(report["missing_sample"]) == 1


def test_adapter_errors_clearly_when_absent(tmp_path):
    cfg = fixture_cfg(tmp_path / "nowhere")
    try:
        get_adapter(cfg).index()
    except DataError as exc:
        assert "root" in str(exc).lower() or "not found" in str(exc).lower()
        return
    raise AssertionError("expected DataError")


# -------------------------------------------------------------------- splits
def _records(tmp_path):
    build_fixture(tmp_path)
    records, _ = get_adapter(fixture_cfg(tmp_path)).index()
    return records


def test_split_never_leaks_groups(tmp_path):
    records = _records(tmp_path)
    result, _ = splits_mod.grouped_stratified_split(
        records, seed=1, val_frac=0.1, test_frac=0.2
    )
    owner = {}
    for name, recs in result.items():
        for rec in recs:
            assert owner.setdefault(rec.group_id, name) == name, rec.group_id


def test_split_is_deterministic(tmp_path):
    records = _records(tmp_path)
    a, _ = splits_mod.grouped_stratified_split(records, seed=7, val_frac=0.1, test_frac=0.2)
    b, _ = splits_mod.grouped_stratified_split(records, seed=7, val_frac=0.1, test_frac=0.2)
    for name in a:
        assert [r.image_id for r in a[name]] == [r.image_id for r in b[name]]


def test_split_covers_every_record_once(tmp_path):
    records = _records(tmp_path)
    result, _ = splits_mod.grouped_stratified_split(
        records, seed=3, val_frac=0.15, test_frac=0.15
    )
    ids = [r.image_id for recs in result.values() for r in recs]
    assert sorted(ids) == sorted(r.image_id for r in records)


def test_tiny_class_goes_entirely_to_train():
    recs = [
        ImageRecord(f"img{i}", Path(f"img{i}.jpg"), "rare", f"g{i}", True)
        for i in range(3)  # below MIN_GROUPS_TO_SPLIT
    ]
    result, stats = splits_mod.grouped_stratified_split(
        recs, seed=1, val_frac=0.2, test_frac=0.2
    )
    assert len(result["train"]) == 3
    assert stats["warnings"]


# ------------------------------------------------------------------ manifest
def test_manifest_round_trip(tmp_path):
    path = tmp_path / "manifest.jsonl"
    rec = manifest.new_record(
        image_id="syn_0001",
        file="syn_0001.png",
        cls="df",
        backend="sd15_lora",
        base_model="runwayml/stable-diffusion-v1-5",
        checkpoint="lora_df_v1",
        seed=0,  # seed 0 must be accepted
        prompt="dermoscopy image of dermatofibroma",
    )
    manifest.append(path, rec)
    loaded = manifest.read(path)
    assert len(loaded) == 1
    assert loaded[0]["synthetic"] is True
    assert loaded[0]["seed"] == 0
    assert loaded[0]["created_utc"].endswith("Z")


def test_manifest_rejects_incomplete_record():
    try:
        manifest.new_record(image_id="x", file="x.png", cls="df")
    except manifest.ManifestError as exc:
        assert "backend" in str(exc)
        return
    raise AssertionError("expected ManifestError")


def test_manifest_refuses_non_synthetic_write(tmp_path):
    try:
        manifest.append(tmp_path / "m.jsonl", {"synthetic": False})
    except manifest.ManifestError:
        return
    raise AssertionError("expected ManifestError")


# --------------------------------------------------------------------- audit
def test_audit_passes_on_fixture(tmp_path, monkeypatch=None):
    build_fixture(tmp_path)
    cfg = fixture_cfg(tmp_path)
    cfg.generator.rare_class_max_count = 20  # nv is common, mel/df rare
    result = audit.run(cfg)
    assert result.success, result.error
    assert result.metrics["classes"]["nv"] > result.metrics["classes"]["df"]
    assert "df" in result.metrics["rare_classes"]
    assert "nv" not in result.metrics["rare_classes"]
    assert result.metrics["image_sizes_sampled"].get("60x45")


def test_audit_fails_cleanly_without_dataset(tmp_path):
    cfg = fixture_cfg(tmp_path / "missing")
    result = audit.run(cfg)
    assert not result.success
    assert result.error


def test_audit_result_is_json_and_small(tmp_path):
    build_fixture(tmp_path)
    result = audit.run(fixture_cfg(tmp_path))
    path = write_result(result, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["stage"] == "audit"
    assert len(json.dumps(payload)) < 20_000


def test_registry_contains_audit():
    assert "audit" in REGISTRY


# ------------------------------------------------------------------ fallback
def _run_all():
    import inspect
    import tempfile

    mod = sys.modules[__name__]
    tests = [(n, f) for n, f in vars(mod).items() if n.startswith("test_") and callable(f)]
    failures = 0
    for name, fn in tests:
        params = inspect.signature(fn).parameters
        try:
            if "tmp_path" in params:
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
