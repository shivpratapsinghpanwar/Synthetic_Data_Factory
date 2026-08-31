"""Offline tests for the private folder-class adapter and fixed splits."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sdf import config, splits as splits_mod  # noqa: E402
from sdf.data.base import get_adapter  # noqa: E402
from sdf.data.folder_class import roi_for  # noqa: E402

VOC = """<annotation><filename>{name}</filename>
<object><name>{cls}</name><bndbox>
<xmin>10</xmin><ymin>20</ymin><xmax>60</xmax><ymax>90</ymax>
</bndbox></object></annotation>"""


def build_fixture(root: Path) -> None:
    from PIL import Image

    def img(path: Path, shade: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 100), color=(shade, 60, 60)).save(path)

    # curated split layout, with colliding bare stems across splits/classes
    for i in range(20):
        img(root / "train" / "cond-a" / f"{i}.jpg", 30 + i)
        img(root / "train" / "cls_neg" / f"{i}.jpg", 130 + i)
    for i in range(6):
        img(root / "test" / "cond-a" / f"{i}.jpg", 90 + i)
        img(root / "test" / "cls_neg" / f"{i}.jpg", 200 + i)
    # VOC sidecar for one train image
    (root / "train" / "cond-a" / "0.xml").write_text(
        VOC.format(name="0.jpg", cls="cond-a"), encoding="utf-8"
    )
    # flat, label-less valid dir -> excluded but counted
    img(root / "valid" / "stray.jpg", 250)


def _cfg(root: Path) -> config.PipelineConfig:
    cfg = config.load(REPO_ROOT / "pipeline_cond-a.toml")
    cfg.dataset.data_root = str(root)
    return cfg


def test_curated_test_is_honored_and_val_carved_from_train(tmp_path):
    build_fixture(tmp_path)
    records, report = get_adapter(_cfg(tmp_path)).index()

    assert report["metadata_rows"] == 52
    assert report["on_disk_not_in_metadata"] == 1  # the label-less valid stray
    assert report["voc_xml_sidecars"] == 1
    assert report["fixed_split_counts"] == {"train": 0, "val": 0, "test": 12}

    result, stats = splits_mod.grouped_stratified_split(
        records, seed=5, val_frac=0.2, test_frac=0.15
    )
    # every curated test image is in test; nothing else joined it
    test_ids = {r.image_id for r in result["test"]}
    assert len(test_ids) == 12
    assert all(i.startswith("test__") for i in test_ids)
    # val carved from train only
    assert all(r.image_id.startswith("train__") for r in result["val"])
    assert len(result["val"]) == 8  # 0.2 * 40
    assert stats["fixed"]["test"] == 12


def test_colliding_stems_do_not_alias(tmp_path):
    build_fixture(tmp_path)
    records, _ = get_adapter(_cfg(tmp_path)).index()
    ids = [r.image_id for r in records]
    assert len(ids) == len(set(ids))  # "0.jpg" exists in 4 places; all distinct


def test_roi_parsed_from_sidecar(tmp_path):
    build_fixture(tmp_path)
    records, _ = get_adapter(_cfg(tmp_path)).index()
    with_box = [r for r in records if r.image_id == "train__cond-a__0.jpg"]
    assert roi_for(with_box[0]) == (10, 20, 60, 90)
    without = [r for r in records if r.image_id == "train__cond-a__1.jpg"]
    assert roi_for(without[0]) is None


def test_flat_class_layout_left_to_splitter(tmp_path):
    from PIL import Image

    for cls in ("cond-a", "cls_neg"):
        for i in range(10):
            p = tmp_path / cls / f"{cls}_{i}.jpg"
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (50, 50), color=(i * 20, 80, 80)).save(p)

    records, report = get_adapter(_cfg(tmp_path)).index()
    assert report["fixed_split_counts"] == {"train": 0, "val": 0, "test": 0}
    result, _ = splits_mod.grouped_stratified_split(
        records, seed=2, val_frac=0.2, test_frac=0.2
    )
    assert all(len(result[k]) > 0 for k in ("train", "val", "test"))


def test_audit_rare_gate_disabled_at_zero(tmp_path):
    from sdf.stages import audit

    build_fixture(tmp_path)
    cfg = _cfg(tmp_path)
    cfg.generator.rare_class_max_count = 0
    result = audit.run(cfg)
    assert result.success, result.error
    assert result.metrics["rare_classes"] == []


def test_sanitize_strips_exif_and_absolute_paths(tmp_path):
    from PIL import Image

    from sdf import private_upload

    src = tmp_path / "src" / "cls_a"
    src.mkdir(parents=True)
    img = Image.new("RGB", (40, 40), color=(120, 10, 10))
    exif = Image.Exif()
    exif[0x010F] = "SecretCameraMake"       # Make tag
    exif[0x9286] = "patient identifier"     # UserComment
    img.save(src / "a.jpg", exif=exif)
    (src / "a.xml").write_text(
        "<annotation><path>C:/Users/doctor/a.jpg</path>"
        "<object><name>x</name></object></annotation>",
        encoding="utf-8",
    )

    staging = tmp_path / "staged"
    counts = private_upload.sanitize_tree(tmp_path / "src", staging)
    assert counts["images"] == 1 and counts["xml"] == 1

    with Image.open(staging / "cls_a" / "a.jpg") as out:
        assert dict(out.getexif()) == {}  # EXIF gone by construction
    xml_text = (staging / "cls_a" / "a.xml").read_text(encoding="utf-8")
    assert "doctor" not in xml_text


def test_verbatim_mode_copies_bytes_untouched(tmp_path):
    """Owner's terms: the data must not be modified. Default mode complies."""
    from PIL import Image

    from sdf import private_upload

    src = tmp_path / "src" / "cls_a"
    src.mkdir(parents=True)
    img = Image.new("RGB", (40, 40), color=(120, 10, 10))
    exif = Image.Exif()
    exif[0x010F] = "CameraMake"
    img.save(src / "a.jpg", exif=exif)
    original = (src / "a.jpg").read_bytes()

    staging = tmp_path / "staged"
    counts = private_upload.verbatim_tree(tmp_path / "src", staging)
    assert counts["images"] == 1
    assert (staging / "cls_a" / "a.jpg").read_bytes() == original  # byte-identical


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
