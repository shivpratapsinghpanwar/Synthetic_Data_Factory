"""Loading and validation of pipeline.toml."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "pipeline.toml"

# Kaggle mounts attached datasets read-only under this root.
KAGGLE_INPUT_ROOT = Path("/kaggle/input")


class ConfigError(RuntimeError):
    """pipeline.toml is missing, malformed, or internally inconsistent."""


@dataclass
class DatasetConfig:
    name: str = "ham10000"
    kaggle_slug: str = "kmader/skin-cancer-mnist-ham10000"
    # Explicit local root for the dataset. Empty -> auto-discover:
    # $SDF_DATA_ROOT, then /kaggle/input (when running on Kaggle).
    data_root: str = ""


@dataclass
class SplitConfig:
    seed: int = 20260823
    val_frac: float = 0.10
    test_frac: float = 0.15
    # HAM10000 has multiple images of the same physical lesion; splitting by
    # image would leak near-duplicates from train into test.
    group_key: str = "lesion_id"


@dataclass
class GeneratorConfig:
    backend: str = "sd15_lora"
    base_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    resolution: int = 512
    # Classes with fewer real training images than this are generation targets.
    rare_class_max_count: int = 600
    # Training defaults (full runs; smoke tests override via --opt).
    train_steps: int = 800
    lr: float = 1e-4
    batch_size: int = 2
    grad_accum: int = 2
    # Sampling defaults.
    sample_count: int = 100
    sample_steps: int = 30
    guidance: float = 7.5


@dataclass
class PipelineConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    path: Path = DEFAULT_CONFIG_PATH


def _section(raw: dict, name: str) -> dict:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"pipeline.toml [{name}] must be a table")
    return value


def _filter(cls, data: dict, section: str):
    known = {f.name for f in cls.__dataclass_fields__.values()}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(
            f"pipeline.toml [{section}] has unknown key(s): {', '.join(sorted(unknown))}"
        )
    return cls(**data)


def load(path: Path | str | None = None) -> PipelineConfig:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise ConfigError(f"config file not found: {cfg_path}")
    with cfg_path.open("rb") as fh:
        raw = tomllib.load(fh)

    cfg = PipelineConfig(
        dataset=_filter(DatasetConfig, _section(raw, "dataset"), "dataset"),
        splits=_filter(SplitConfig, _section(raw, "splits"), "splits"),
        generator=_filter(GeneratorConfig, _section(raw, "generator"), "generator"),
        path=cfg_path,
    )
    validate(cfg)
    return cfg


def validate(cfg: PipelineConfig) -> None:
    s = cfg.splits
    if not (0.0 < s.val_frac < 1.0 and 0.0 < s.test_frac < 1.0):
        raise ConfigError("[splits] val_frac and test_frac must be in (0, 1)")
    if s.val_frac + s.test_frac >= 0.5:
        raise ConfigError("[splits] val_frac + test_frac must be < 0.5")
    if cfg.generator.resolution not in (256, 512):
        raise ConfigError("[generator] resolution must be 256 or 512")
    g = cfg.generator
    for name in ("train_steps", "batch_size", "grad_accum", "sample_count", "sample_steps"):
        if getattr(g, name) <= 0:
            raise ConfigError(f"[generator] {name} must be positive")
    if not (0 < g.lr < 1):
        raise ConfigError("[generator] lr must be in (0, 1)")
    if g.backend not in ("sd15_lora", "ddpm", "sdxl_lora"):
        raise ConfigError(
            f"[generator] unknown backend {g.backend!r} (sd15_lora, ddpm, sdxl_lora)"
        )
    if cfg.dataset.kaggle_slug.count("/") != 1:
        raise ConfigError('[dataset] kaggle_slug must be "owner/dataset-slug"')
    if not cfg.dataset.name:
        raise ConfigError("[dataset] name must not be empty")


def resolve_data_root(cfg: PipelineConfig) -> Path | None:
    """Where to look for the dataset, in priority order. None = nowhere found."""
    if cfg.dataset.data_root:
        return Path(cfg.dataset.data_root)
    env = os.environ.get("SDF_DATA_ROOT")
    if env:
        return Path(env)
    if KAGGLE_INPUT_ROOT.is_dir():
        return KAGGLE_INPUT_ROOT
    return None


def output_dir() -> Path:
    """Where stage results and artifacts go (kaggle_runner sets SDF_OUTPUT_DIR)."""
    return Path(os.environ.get("SDF_OUTPUT_DIR", "."))
