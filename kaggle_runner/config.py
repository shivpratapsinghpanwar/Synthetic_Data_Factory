"""Loading and validation of runner.toml."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "runner.toml"


class ConfigError(RuntimeError):
    """runner.toml is missing, malformed, or internally inconsistent."""


@dataclass
class RepoConfig:
    url: str
    branch: str = "main"
    git_token_secret: str = ""


@dataclass
class KernelConfig:
    owner: str
    slug: str
    title: str
    is_private: bool = True
    enable_internet: bool = True
    enable_gpu: bool = False
    accelerator: str = ""
    timeout_s: int = 1800

    @property
    def ref(self) -> str:
        """The ``owner/slug`` reference the Kaggle CLI expects."""
        return f"{self.owner}/{self.slug}"

    @property
    def url(self) -> str:
        return f"https://www.kaggle.com/code/{self.owner}/{self.slug}"


@dataclass
class JobConfig:
    entrypoint: str
    requirements: str = ""
    fail_kernel_on_error: bool = False


@dataclass
class LocalConfig:
    runs_dir: str = "runs"
    poll_interval_s: int = 15
    poll_backoff_max_s: int = 60
    poll_timeout_s: int = 5400
    log_tail_lines: int = 40
    traceback_max_chars: int = 4000


@dataclass
class Config:
    repo: RepoConfig
    kernel: KernelConfig
    job: JobConfig
    local: LocalConfig = field(default_factory=LocalConfig)
    path: Path = DEFAULT_CONFIG_PATH

    @property
    def runs_path(self) -> Path:
        p = Path(self.local.runs_dir)
        return p if p.is_absolute() else REPO_ROOT / p


def _section(data: dict, name: str) -> dict:
    value = data.get(name)
    if value is None:
        raise ConfigError(f"runner.toml is missing the [{name}] section")
    if not isinstance(value, dict):
        raise ConfigError(f"runner.toml [{name}] must be a table")
    return value


def _filter(cls, data: dict, section: str):
    """Build a dataclass, rejecting unknown keys with a helpful message."""
    known = {f.name for f in cls.__dataclass_fields__.values()}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(
            f"runner.toml [{section}] has unknown key(s): {', '.join(sorted(unknown))}"
        )
    try:
        return cls(**data)
    except TypeError as exc:  # missing required key
        raise ConfigError(f"runner.toml [{section}]: {exc}") from exc


def load(path: Path | str | None = None) -> Config:
    """Read runner.toml and return a validated :class:`Config`."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise ConfigError(f"config file not found: {cfg_path}")

    with cfg_path.open("rb") as fh:
        raw = tomllib.load(fh)

    cfg = Config(
        repo=_filter(RepoConfig, _section(raw, "repo"), "repo"),
        kernel=_filter(KernelConfig, _section(raw, "kernel"), "kernel"),
        job=_filter(JobConfig, _section(raw, "job"), "job"),
        local=_filter(LocalConfig, raw.get("local", {}), "local"),
        path=cfg_path,
    )
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if not cfg.repo.url.startswith(("https://", "http://")):
        raise ConfigError("[repo] url must be an http(s) clone URL")
    if not cfg.kernel.owner or "/" in cfg.kernel.owner:
        raise ConfigError("[kernel] owner must be a bare Kaggle username")
    if not cfg.kernel.slug or "/" in cfg.kernel.slug:
        raise ConfigError("[kernel] slug must be a bare kernel slug")
    if not cfg.job.entrypoint.strip():
        raise ConfigError("[job] entrypoint must not be empty")
    if cfg.kernel.enable_gpu and not cfg.kernel.accelerator:
        raise ConfigError(
            "[kernel] enable_gpu = true requires a non-empty accelerator "
            '(e.g. "NvidiaTeslaT4")'
        )
    if cfg.kernel.accelerator and not cfg.kernel.enable_gpu:
        raise ConfigError(
            "[kernel] accelerator is set but enable_gpu = false; "
            "set enable_gpu = true or clear accelerator"
        )
    if not cfg.kernel.enable_internet:
        raise ConfigError(
            "[kernel] enable_internet must be true - the kernel git-clones the repo"
        )


def kaggle_credentials_present() -> tuple[bool, str]:
    """Best-effort check that the Kaggle CLI will be able to authenticate.

    Returns ``(ok, detail)``. Deliberately does not read any secret value.
    """
    home = Path(os.path.expanduser("~"))
    checks = [
        (bool(os.environ.get("KAGGLE_API_TOKEN")), "KAGGLE_API_TOKEN env var"),
        (
            bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")),
            "KAGGLE_USERNAME + KAGGLE_KEY env vars",
        ),
        ((home / ".kaggle" / "access_token").exists(), "~/.kaggle/access_token"),
        ((home / ".kaggle" / "kaggle.json").exists(), "~/.kaggle/kaggle.json"),
    ]
    found = [label for ok, label in checks if ok]
    if found:
        return True, f"credentials found via {found[0]}"
    return False, "no Kaggle credentials found"
