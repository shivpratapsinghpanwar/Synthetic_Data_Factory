"""Offline tests for the committee select stage and augment integration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sdf import config, manifest  # noqa: E402
from sdf.stages import augment, select  # noqa: E402


def _syn_pool(root: Path, cls: str, spec: list[tuple[str, str, tuple]]) -> None:
    """Write a synthetic pool: (name, backend, rgb) rows + images."""
    from PIL import Image

    (root / "synthetic" / cls).mkdir(parents=True, exist_ok=True)
    for name, backend, rgb in spec:
        img = Image.new("RGB", (32, 32), color=rgb)
        # deterministic non-flat texture unless rgb is meant to be flat
        if rgb != (10, 10, 10):
            for x in range(0, 32, 4):
                img.putpixel((x, x), (255 - rgb[0], 0, 0))
        img.save(root / "synthetic" / cls / name)
        record = manifest.new_record(
            image_id=name.rsplit(".", 1)[0], file=f"synthetic/{cls}/{name}",
            cls=cls, backend=backend, base_model="x", checkpoint="c",
            seed=1, prompt="p",
        )
        manifest.append(root / "synthetic_manifest.jsonl", record)


def test_select_blends_arms_and_drops_bad(tmp_path, monkeypatch):
    monkeypatch.setenv("SDF_OUTPUT_DIR", str(tmp_path))
    spec = [(f"m{i}.png", "mdx", (40 + i * 9, 80, 120)) for i in range(4)]
    spec += [(f"s{i}.png", "sd15_lora", (200 - i * 9, 60, 30)) for i in range(4)]
    spec += [("flat.png", "mdx", (10, 10, 10))]              # pixel screen: flat
    spec += [("dupe.png", "sd15_lora", (200, 60, 30))]       # duplicate of s0
    spec += [("memo.png", "mdx", (90, 200, 90))]             # gate-flagged
    _syn_pool(tmp_path, "pos", spec)
    (tmp_path / "stage_quality_gate_pos.json").write_text(json.dumps(
        {"metrics": {"memorization": {"flagged": [{"image": "memo.png"}]}}}
    ), encoding="utf-8")

    result = select.run(config.PipelineConfig(), {"cls": "pos", "count": 4})
    assert result.success, result.error
    m = result.metrics
    assert m["per_backend"] == {"mdx": 2, "sd15_lora": 2}  # round-robin blend
    assert m["dropped"]["flagged"] == 1
    assert m["dropped"]["flat"] == 1
    assert m["dropped"]["duplicate"] == 1
    assert m["scored_by_detector"] is False
    saved = json.loads((tmp_path / "selection_pos.json").read_text(encoding="utf-8"))
    assert len(saved["selected"]) == 4
    assert "memo.png" not in saved["selected"] and "flat.png" not in saved["selected"]


def test_select_requires_cls_and_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("SDF_OUTPUT_DIR", str(tmp_path))
    assert not select.run(config.PipelineConfig(), {}).success
    result = select.run(config.PipelineConfig(), {"cls": "nothing"})
    assert not result.success and "no synthetic manifest rows" in result.error


def test_augment_honors_selection(tmp_path, monkeypatch):
    from PIL import Image

    monkeypatch.setenv("SDF_OUTPUT_DIR", str(tmp_path))
    # real dataset: flat folder-class layout
    for cls in ("pos", "neg"):
        for i in range(6):
            p = tmp_path / "data" / cls / f"{i}.jpg"
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (24, 24), color=(i * 30, 60, 60)).save(p)
    cfg = config.PipelineConfig()
    cfg.dataset.name = "folder_class"
    cfg.dataset.data_root = str(tmp_path / "data")

    _syn_pool(tmp_path, "pos", [
        (f"a{i}.png", "mdx", (30 + i * 11, 90, 140)) for i in range(4)
    ])
    (tmp_path / "selection_pos.json").write_text(json.dumps(
        {"cls": "pos", "selected": ["a0.png", "a2.png"]}
    ), encoding="utf-8")

    result = augment.run(cfg, {})
    assert result.success, result.error
    assert result.metrics["accepted_synthetic"] == 2
    assert result.metrics["skipped"]["not_selected"] == 2
    assert result.metrics["per_class"]["pos"]["synthetic"] == 2
