"""Builds the kernel payload that gets pushed to Kaggle.

A kernel push needs exactly two things in a folder:
  * the code file (here: a generated copy of ``bootstrap_template.py``)
  * ``kernel-metadata.json``

Keeping generation in one place means the pushed artifact is always a pure
function of (config, commit, run_id) - which is what makes runs reproducible.
"""

from __future__ import annotations

import ast
import json
import pprint
import re
import shutil
from pathlib import Path

from .config import Config, REPO_ROOT

TEMPLATE_PATH = Path(__file__).with_name("bootstrap_template.py")
INJECTION_MARKER = "# SDF_JOB_INJECTION_POINT"
CODE_FILE_NAME = "sdf_bootstrap.py"
BUILD_DIR = REPO_ROOT / ".sdf_build"


class BuildError(RuntimeError):
    pass


def job_spec(cfg: Config, commit: str, run_id: str) -> dict:
    """The dict handed to the in-kernel bootstrap."""
    return {
        "run_id": run_id,
        "commit": commit,
        "repo_url": cfg.repo.url,
        "git_token_secret": cfg.repo.git_token_secret,
        "entrypoint": cfg.job.entrypoint,
        "requirements": cfg.job.requirements,
        "fail_kernel_on_error": cfg.job.fail_kernel_on_error,
        "log_tail_lines": cfg.local.log_tail_lines,
        "traceback_max_chars": cfg.local.traceback_max_chars,
    }


def render_bootstrap(spec: dict) -> str:
    """Inject ``spec`` into the bootstrap template as a Python literal.

    The payload must be a *Python* literal, not JSON: ``json.dumps`` would emit
    ``false``/``true``/``null``, which parse as bare names and blow up at import
    time inside the kernel. ``pprint.pformat`` emits real Python, and the
    round-trip check below proves it before anything is pushed.
    """
    source = TEMPLATE_PATH.read_text(encoding="utf-8")
    pattern = re.compile(r"^SDF_JOB = .*" + re.escape(INJECTION_MARKER) + r"$", re.MULTILINE)
    if not pattern.search(source):
        raise BuildError(f"injection marker not found in {TEMPLATE_PATH}")

    payload = pprint.pformat(spec, indent=4, width=88, sort_dicts=True)
    if ast.literal_eval(payload) != spec:
        raise BuildError("job spec did not survive round-trip to a Python literal")

    replacement = "SDF_JOB = " + payload + "  " + INJECTION_MARKER
    rendered = pattern.sub(lambda _: replacement, source, count=1)

    _verify(rendered, spec)
    return rendered


def _verify(rendered: str, spec: dict) -> None:
    """Prove the rendered file parses *and* that SDF_JOB evaluates to ``spec``.

    ``compile()`` alone is not enough - it only checks syntax, so a JSON-style
    ``false`` would sail through and fail at kernel runtime instead.
    """
    try:
        tree = ast.parse(rendered, filename=CODE_FILE_NAME)
    except SyntaxError as exc:
        raise BuildError(f"rendered bootstrap is not valid Python: {exc}") from exc

    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "SDF_JOB"
        ):
            try:
                value = ast.literal_eval(node.value)
            except ValueError as exc:
                raise BuildError(f"SDF_JOB is not a literal: {exc}") from exc
            if value != spec:
                raise BuildError("SDF_JOB in rendered bootstrap does not match the job spec")
            return

    raise BuildError("rendered bootstrap has no top-level SDF_JOB assignment")


def kernel_metadata(cfg: Config) -> dict:
    """kernel-metadata.json per Kaggle's documented schema.

    Booleans are strings because Kaggle's metadata format specifies them that way.
    """
    return {
        "id": cfg.kernel.ref,
        "title": cfg.kernel.title,
        "code_file": CODE_FILE_NAME,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true" if cfg.kernel.is_private else "false",
        "enable_gpu": "true" if cfg.kernel.enable_gpu else "false",
        "enable_internet": "true" if cfg.kernel.enable_internet else "false",
        "dataset_sources": list(cfg.kernel.dataset_sources),
        "kernel_sources": list(cfg.kernel.kernel_sources),
        "competition_sources": [],
        "model_sources": [],
    }


def build(cfg: Config, commit: str, run_id: str, dest: Path | None = None) -> Path:
    """Write the push folder and return its path."""
    folder = dest or BUILD_DIR
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)

    spec = job_spec(cfg, commit, run_id)
    (folder / CODE_FILE_NAME).write_text(render_bootstrap(spec), encoding="utf-8")
    (folder / "kernel-metadata.json").write_text(
        json.dumps(kernel_metadata(cfg), indent=2), encoding="utf-8"
    )
    return folder
