"""Loading and validation of runner.toml."""

from __future__ import annotations

import os
import re
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
    # Kaggle datasets mounted read-only under /kaggle/input/<slug-tail>/.
    # Attaching is free and avoids re-downloading data on every run.
    dataset_sources: list = field(default_factory=list)
    # Other kernels whose LATEST COMPLETE output mounts read-only under
    # /kaggle/input/<kernel-slug>/. The native way to pass large artifacts
    # between sessions without any local round-trip.
    kernel_sources: list = field(default_factory=list)

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
    # Git-Bash on Windows (MSYS) silently rewrites /kaggle/... path arguments
    # to C:/Program Files/Git/kaggle/... before they reach us; the kernel then
    # executes garbage. Catch it at preflight instead of after queueing.
    # A single drive letter followed by :/ or :\ (not preceded by another
    # letter, so URL schemes like https:// do not match).
    if re.search(r"(?<![A-Za-z0-9])[A-Za-z]:[/\\]", cfg.job.entrypoint):
        raise ConfigError(
            "[job] entrypoint contains a Windows path - MSYS path mangling? "
            "Prefix the command with MSYS_NO_PATHCONV=1 (Git Bash) and retry: "
            f"{cfg.job.entrypoint!r}"
        )
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
    for field_name in ("dataset_sources", "kernel_sources"):
        for src in getattr(cfg.kernel, field_name):
            if not isinstance(src, str) or src.count("/") != 1 or not all(src.split("/")):
                raise ConfigError(
                    f'[kernel] {field_name} entries must be "owner/slug", got: {src!r}'
                )


# Credential files the Kaggle CLI is known to use. `credentials.json` is what
# `kaggle auth login` (OAuth) writes; the others are the token-file and legacy
# forms. This list is a *hint* only - see kaggle_credentials_present().
_CREDENTIAL_FILES = ("credentials.json", "access_token", "kaggle.json")


def kaggle_config_dir() -> Path:
    """Directory the Kaggle CLI reads credentials from."""
    override = os.environ.get("KAGGLE_CONFIG_DIR")
    if override:
        return Path(override)
    return Path(os.path.expanduser("~")) / ".kaggle"


def kaggle_credentials_present() -> tuple[bool, str]:
    """Report where credentials appear to live, without reading any secret.

    This is deliberately advisory. The Kaggle CLI has changed its credential
    filenames before (OAuth login writes ``credentials.json``, which an older
    hardcoded list missed), so a negative result here must never be treated as
    proof that auth will fail - only the live API call in
    ``kaggle_cli.authenticated()`` is authoritative.
    """
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True, "KAGGLE_API_TOKEN env var"
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True, "KAGGLE_USERNAME + KAGGLE_KEY env vars"

    config_dir = kaggle_config_dir()
    for name in _CREDENTIAL_FILES:
        if (config_dir / name).exists():
            return True, f"{config_dir / name}"

    # Any unexpected file in the config dir probably is the credential store
    # under a name this version does not know about.
    if config_dir.is_dir():
        others = [p.name for p in config_dir.iterdir() if p.is_file()]
        if others:
            return True, f"{config_dir} contains {', '.join(sorted(others))}"

    return False, f"no credential file found in {config_dir}"
