"""Tests for the mdx mixture-of-experts diffusion transformer backend.

Structural tests run anywhere; model-internals tests skip when torch is
absent (the control machine has none) and run on the Kaggle side via
`python -m pytest test/ -q` in a CPU kernel before any GPU spend.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sdf import config  # noqa: E402
from sdf.data.base import ImageRecord  # noqa: E402
from sdf.gen import get_backend  # noqa: E402
from sdf.gen import mdx  # noqa: E402


# ------------------------------------------------------------ torch-free
def test_backend_registered():
    assert get_backend("mdx") is mdx
    assert mdx.ADAPTER_DIR_NAME == "mdx_model"


def test_config_accepts_mdx_and_new_fields(tmp_path):
    (tmp_path / "p.toml").write_text(
        '[dataset]\nname = "folder_class"\nkaggle_slug = "o/d"\n'
        'condition = "cond_x"\n'
        '[generator]\nbackend = "mdx"\nbase_model = "from-scratch"\n'
        "roi_only = true\n",
        encoding="utf-8",
    )
    cfg = config.load(tmp_path / "p.toml")
    assert cfg.generator.backend == "mdx"
    assert cfg.dataset.condition == "cond_x"
    assert cfg.generator.roi_only is True


def test_prompt_template_fills_runtime_class():
    from sdf.gen.prompts import PromptError, prompt_for

    assert prompt_for("df").startswith("a dermatoscopy")  # curated table wins
    assert (
        prompt_for("some_label", "a medical photograph of {cls}")
        == "a medical photograph of some label"
    )
    with pytest.raises(PromptError):
        prompt_for("some_label")


def test_default_patch_keeps_constant_tokens():
    for res in (64, 128, 256):
        assert (res // mdx._default_patch(res)) ** 2 == 256


def _fixture_records(root: Path, cls: str, n: int, with_xml: int = 0) -> list:
    from PIL import Image

    recs = []
    d = root / cls
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        p = d / f"{i}.jpg"
        Image.new("RGB", (100, 80), color=(20 * i % 255, 90, 90)).save(p)
        if i < with_xml:
            p.with_suffix(".xml").write_text(
                "<annotation><object><name>region</name><bndbox>"
                "<xmin>10</xmin><ymin>20</ymin><xmax>60</xmax><ymax>70</ymax>"
                "</bndbox></object></annotation>",
                encoding="utf-8",
            )
        recs.append(ImageRecord(f"{cls}/{i}.jpg", p, cls, f"g{cls}{i}", True))
    return recs


def _mini_cfg(condition: str, roi_only: bool = False):
    cfg = config.PipelineConfig()
    cfg.dataset.name = "folder_class"
    cfg.dataset.condition = condition
    cfg.generator.roi_only = roi_only
    return cfg


def test_roi_only_policy_excludes_boxless_images(tmp_path):
    recs = _fixture_records(tmp_path, "pos", 6, with_xml=2)
    labelled = mdx._labelled_from_records(_mini_cfg("cond_x", roi_only=True), recs)
    assert len(labelled) == 2  # boxless images are OUT, not merely uncropped
    for _path, roi, label, cond in labelled:
        assert roi == (10, 20, 60, 70)
        assert label == "cond_x/pos" and cond == "cond_x"


def test_without_roi_policy_full_images_and_no_crop(tmp_path):
    recs = _fixture_records(tmp_path, "pos", 4, with_xml=2)
    labelled = mdx._labelled_from_records(_mini_cfg("cond_x"), recs)
    assert len(labelled) == 4
    assert all(roi is None for _p, roi, _l, _c in labelled)


def _write_pipeline_toml(path: Path, condition: str, data_root: Path):
    path.write_text(
        "[dataset]\n"
        'name = "folder_class"\n'
        'kaggle_slug = "private/local-overlay"\n'
        f'data_root = "{data_root.as_posix()}"\n'
        f'condition = "{condition}"\n'
        "[generator]\n"
        'backend = "mdx"\n'
        'base_model = "from-scratch"\n',
        encoding="utf-8",
    )


def test_train_joint_stage_rejects_mismatched_data_roots():
    from sdf.stages import train_joint

    result = train_joint.run(
        config.PipelineConfig(),
        {"configs": "a.toml,b.toml", "data_roots": "/one"},
    )
    assert not result.success
    assert "pair up" in result.error


def test_train_joint_stage_rejects_duplicate_condition(tmp_path):
    from sdf.stages import train_joint

    for name in ("d1", "d2"):
        _fixture_records(tmp_path / name, "pos", 6)
        _fixture_records(tmp_path / name, "neg", 6)
        _write_pipeline_toml(tmp_path / f"{name}.toml", "cond_same", tmp_path / name)

    result = train_joint.run(
        config.load(tmp_path / "d1.toml"),
        {"configs": f"{tmp_path / 'd1.toml'},{tmp_path / 'd2.toml'}"},
    )
    assert not result.success
    assert "duplicate condition" in result.error


# ------------------------------------------------------- torch required
def _tiny_arch(vocab, conditions, resolution=64):
    return {
        "resolution": resolution, "patch": 16, "dim": 32, "depth": 2,
        "heads": 2, "mlp_ratio": 2, "expert_rank": 4,
        "vocab": vocab, "conditions": conditions,
    }


def test_forward_shape_and_zero_init_output():
    torch = pytest.importorskip("torch")

    model = mdx.build_model(_tiny_arch(["a/x", "b/y"], ["a", "b"]))
    x = torch.randn(3, 3, 64, 64)
    t = torch.randint(0, 1000, (3,))
    out = model(x, t, torch.tensor([0, 1, 2]), torch.tensor([0, 1, 0]))
    assert out.shape == (3, 3, 64, 64)
    # adaLN-zero + zero-init head: the network starts as an exact zero map.
    assert torch.all(out == 0)


def test_expert_gradients_are_hard_routed():
    torch = pytest.importorskip("torch")

    model = mdx.build_model(_tiny_arch(["a/x", "b/y"], ["a", "b"]))
    # The network is a zero map at init (zero-init head and gates), which
    # would make EVERY gradient zero and the routing assertion vacuous - give
    # experts, gates and the output head signal so gradients actually flow.
    with torch.no_grad():
        for block in model.blocks:
            block.expert_up.normal_(0, 0.1)
            block.ada.weight.normal_(0, 0.1)
        model.final_ada.weight.normal_(0, 0.1)
        model.final_out.weight.normal_(0, 0.1)
    x = torch.randn(4, 3, 64, 64)
    t = torch.randint(0, 1000, (4,))
    cond = torch.zeros(4, dtype=torch.long)  # every sample routed to expert 0
    model(x, t, torch.zeros(4, dtype=torch.long), cond).square().mean().backward()
    for block in model.blocks:
        assert block.expert_up.grad[0].abs().sum() > 0
        assert torch.all(block.expert_up.grad[1] == 0)  # untouched expert
        assert torch.all(block.expert_down.grad[1] == 0)


def test_checkpoint_roundtrip_and_vocab_expansion(tmp_path):
    torch = pytest.importorskip("torch")

    model = mdx.build_model(_tiny_arch(["a/x"], ["a"]))
    with torch.no_grad():
        model.label_emb.weight.normal_(0, 1.0)
    mdx.save_checkpoint(model, tmp_path, extra={"steps_trained": 7})

    loaded, meta = mdx.load_checkpoint(tmp_path)
    assert meta["steps_trained"] == 7
    assert torch.equal(loaded.label_emb.weight, model.label_emb.weight.cpu())

    grown, prev = mdx.load_expanding(tmp_path, _tiny_arch(["a/x", "b/y"], ["a", "b"]))
    assert prev == 7
    row = grown.arch["vocab"].index("a/x")
    assert torch.equal(grown.label_emb.weight[row], model.label_emb.weight[0].cpu())
    # null row carries over to its new position (last).
    assert torch.equal(grown.label_emb.weight[2], model.label_emb.weight[1].cpu())
    for old_block, new_block in zip(model.blocks, grown.blocks):
        assert torch.equal(new_block.expert_down[0], old_block.expert_down[0].cpu())

    with pytest.raises(ValueError, match="drop trained labels"):
        mdx.load_expanding(tmp_path, _tiny_arch(["b/y"], ["b"]))


def test_train_and_sample_tiny_end_to_end(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")

    from sdf import manifest

    recs = _fixture_records(tmp_path / "data", "pos", 4) + _fixture_records(
        tmp_path / "data", "neg", 4
    )
    cfg = _mini_cfg("cond_t")
    labelled = mdx._labelled_from_records(cfg, recs)
    opts = {
        "steps": 2, "resolution": 64, "patch": 16, "dim": 32, "depth": 2,
        "heads": 2, "mlp_ratio": 2, "expert_rank": 4, "batch_size": 4,
        "checkpoint_every": 0,
    }
    report = mdx.train_joint(labelled, tmp_path / "out", opts, seed=1)
    assert report["vocab"] == ["cond_t/neg", "cond_t/pos"]
    assert report["steps_total"] == 2

    adapter = Path(report["adapter_dir"])
    rows = mdx.sample(
        cfg, "pos", adapter, tmp_path / "syn",
        {"count": 2, "sample_steps": 2, "sample_batch": 2, "guidance": 1.5, "seed": 3},
    )
    assert len(rows) == 2
    for row in rows:
        assert (tmp_path / "syn" / row["file"]).exists()
        record = manifest.new_record(**row)  # all required fields present
        assert record["synthetic"] is True

    with pytest.raises(ValueError, match="not in checkpoint vocab"):
        mdx.sample(_mini_cfg("cond_other"), "pos", adapter, tmp_path / "syn2",
                   {"count": 1, "sample_steps": 1})


def test_counterfact_stage_validates_inputs(tmp_path, monkeypatch):
    from sdf.stages import counterfact

    monkeypatch.setenv("SDF_OUTPUT_DIR", str(tmp_path))
    assert not counterfact.run(config.PipelineConfig(), {}).success
    same = counterfact.run(
        config.PipelineConfig(), {"cls_from": "x", "cls_to": "x"}
    )
    assert not same.success and "distinct" in same.error
    missing = counterfact.run(
        config.PipelineConfig(), {"cls_from": "a", "cls_to": "b"}
    )
    assert not missing.success and "checkpoint not found" in missing.error


def test_cf_edit_produces_paired_counterfactuals(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")

    from sdf.stages import counterfact

    monkeypatch.setenv("SDF_OUTPUT_DIR", str(tmp_path))
    data_root = tmp_path / "data"
    recs = _fixture_records(data_root, "pos", 6) + _fixture_records(
        data_root, "neg", 6
    )
    cfg = _mini_cfg("cond_t")
    cfg.dataset.data_root = str(data_root)
    opts = {
        "steps": 2, "resolution": 64, "patch": 16, "dim": 32, "depth": 2,
        "heads": 2, "mlp_ratio": 2, "expert_rank": 4, "batch_size": 4,
        "checkpoint_every": 0,
    }
    report = mdx.train_joint(
        mdx._labelled_from_records(cfg, recs), tmp_path / "lora" / "mdx" / "joint",
        opts, seed=1,
    )

    result = counterfact.run(cfg, {
        "cls_from": "pos", "cls_to": "neg", "count": 3, "strength": 0.5,
        "sample_steps": 2, "sample_batch": 2, "guidance": 1.5, "seed": 5,
        "adapter_dir": report["adapter_dir"],
    })
    assert result.success, result.error
    assert result.metrics["count"] == 3

    from sdf import manifest as manifest_mod

    rows = manifest_mod.read(tmp_path / "synthetic_manifest.jsonl")
    assert len(rows) == 3
    for row in rows:
        assert row["backend"] == "mdx_cfe"
        assert row["cls"] == "neg"
        assert row["source_image"].endswith(".jpg")  # pairing recorded
        assert row["edit_strength"] == 0.5
        assert (tmp_path / row["file"]).exists()

    with pytest.raises(ValueError, match="strength"):
        mdx.edit(cfg, [recs[0].path], "pos", "neg",
                 Path(report["adapter_dir"]), tmp_path / "x",
                 {"strength": 1.5, "sample_steps": 2})


def test_freeze_trunk_trains_only_experts_and_labels(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    import torch

    recs = _fixture_records(tmp_path / "d1", "pos", 4)
    opts = {
        "steps": 1, "resolution": 64, "patch": 16, "dim": 32, "depth": 2,
        "heads": 2, "mlp_ratio": 2, "expert_rank": 4, "batch_size": 4,
        "checkpoint_every": 0,
    }
    first = mdx.train_joint(
        mdx._labelled_from_records(_mini_cfg("cond_1"), recs),
        tmp_path / "out1", opts, seed=1,
    )

    recs2 = _fixture_records(tmp_path / "d2", "pos", 4)
    labelled2 = mdx._labelled_from_records(
        _mini_cfg("cond_1"), recs
    ) + mdx._labelled_from_records(_mini_cfg("cond_2"), recs2)
    frozen_opts = dict(opts)
    frozen_opts.update({"freeze_trunk": 1, "resume_from": first["adapter_dir"]})
    second = mdx.train_joint(labelled2, tmp_path / "out2", frozen_opts, seed=1)
    assert second["freeze_trunk"] is True
    assert second["params_trainable"] < second["params"]
    assert second["steps_total"] == 2  # cumulative across the resume

    # Trunk untouched: patchify weights identical across the onboarding run.
    old = torch.load(Path(first["adapter_dir"]) / mdx.WEIGHTS_TRAIN,
                     map_location="cpu", weights_only=True)
    new = torch.load(Path(second["adapter_dir"]) / mdx.WEIGHTS_TRAIN,
                     map_location="cpu", weights_only=True)
    assert torch.equal(old["patchify.weight"], new["patchify.weight"])
