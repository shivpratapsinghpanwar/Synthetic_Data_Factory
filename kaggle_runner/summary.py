"""Turn a finished Kaggle run into a small, machine-readable result.

The hard constraint here is size: this output is designed to be read by an
automated agent on every loop iteration, so it must stay in the low kilobytes
no matter how much a job logged. Full logs live on disk under ``runs/<run_id>/``
and are referenced by path only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import SCHEMA_VERSION

# Hard ceilings applied regardless of config, so a runaway job can never blow up
# the consumer's context window.
MAX_TRACEBACK_CHARS = 6000
MAX_TAIL_LINES = 60
MAX_TAIL_LINE_CHARS = 400
MAX_MESSAGE_CHARS = 1000


def _clip(text: str, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return "...[truncated]...\n" + text[-limit:]


def _clip_tail(lines: list[str], max_lines: int) -> list[str]:
    tail = [str(line)[:MAX_TAIL_LINE_CHARS] for line in lines[-max_lines:]]
    return tail


@dataclass
class RunPaths:
    run_dir: Path
    output_dir: Path
    result_json: Path
    run_log: Path
    kaggle_log: Path

    @classmethod
    def for_run(cls, runs_root: Path, run_id: str) -> "RunPaths":
        run_dir = runs_root / run_id
        output_dir = run_dir / "output"
        return cls(
            run_dir=run_dir,
            output_dir=output_dir,
            result_json=run_dir / "result.json",
            run_log=output_dir / "sdf_run.log",
            kaggle_log=run_dir / "kaggle_console.log",
        )


def _relative(path: Path) -> str:
    """Repo-relative POSIX path when possible, else absolute."""
    try:
        from .config import REPO_ROOT

        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except (ValueError, ImportError):
        return path.as_posix()


def _list_artifacts(output_dir: Path, limit: int = 25) -> list[dict]:
    if not output_dir.is_dir():
        return []
    files = sorted(p for p in output_dir.rglob("*") if p.is_file())
    listing = [
        {"name": p.relative_to(output_dir).as_posix(), "bytes": p.stat().st_size}
        for p in files[:limit]
    ]
    if len(files) > limit:
        listing.append({"name": f"...and {len(files) - limit} more", "bytes": 0})
    return listing


def load_kernel_result(paths: RunPaths) -> dict | None:
    """Read ``sdf_result.json`` produced inside the kernel, if it arrived."""
    candidate = paths.output_dir / "sdf_result.json"
    if not candidate.exists():
        matches = list(paths.output_dir.rglob("sdf_result.json")) if paths.output_dir.is_dir() else []
        if not matches:
            return None
        candidate = matches[0]
    try:
        return json.loads(candidate.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        return None


def build(
    *,
    run_id: str,
    cfg,
    git_state,
    kernel_status: str,
    kernel_version: int | None,
    kernel_result: dict | None,
    paths: RunPaths,
    wall_s: float,
    queue_s: float,
    infra_error: str = "",
) -> dict:
    """Assemble the final summary dict written to ``runs/<run_id>/result.json``."""
    tail_limit = min(cfg.local.log_tail_lines, MAX_TAIL_LINES)
    tb_limit = min(cfg.local.traceback_max_chars, MAX_TRACEBACK_CHARS)

    if kernel_result:
        success = bool(kernel_result.get("success"))
        exit_code = kernel_result.get("exit_code")
        stage = kernel_result.get("stage", "")
        durations = dict(kernel_result.get("durations_s") or {})
        err = dict(kernel_result.get("error") or {})
        tail = _clip_tail(list(kernel_result.get("log_tail") or []), tail_limit)
        log_bytes = kernel_result.get("log_bytes", 0)
        commit_executed = kernel_result.get("commit_executed", "")
        env = kernel_result.get("env") or {}
    else:
        # No result file: the kernel died before it could write one, or output
        # retrieval failed. Say so explicitly instead of guessing success.
        success = False
        exit_code = None
        stage = "no-result"
        durations = {}
        err = {
            "type": "RunnerError",
            "message": infra_error
            or "kernel produced no sdf_result.json (crashed, cancelled, or timed out)",
            "traceback": "",
        }
        tail = _clip_tail(_read_tail(paths.kaggle_log, tail_limit), tail_limit)
        log_bytes = paths.kaggle_log.stat().st_size if paths.kaggle_log.exists() else 0
        commit_executed = ""
        env = {}

    if infra_error and success:
        success = False
        err = {"type": "RunnerError", "message": infra_error, "traceback": ""}

    if kernel_status in {"error"} and success:
        # Kernel-level failure that the bootstrap did not catch.
        success = False
        err.setdefault("type", "KernelError")
        err["message"] = err.get("message") or f"Kaggle reported kernel status '{kernel_status}'"

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "success": success,
        "status": _overall_status(success, kernel_status, stage),
        "git": {
            "commit": git_state.commit,
            "short": git_state.short,
            "branch": git_state.branch,
            "subject": git_state.subject[:200],
            "dirty": git_state.dirty,
            "commit_executed": commit_executed,
        },
        "kaggle": {
            "kernel": cfg.kernel.ref,
            "version": kernel_version,
            "url": cfg.kernel.url,
            "status": kernel_status,
            "accelerator": cfg.kernel.accelerator or "cpu",
        },
        "exit": {"code": exit_code, "stage": stage},
        "duration_s": {
            "wall": round(wall_s, 1),
            "queue": round(queue_s, 1),
            "clone": durations.get("clone"),
            "deps": durations.get("deps"),
            "entrypoint": durations.get("entrypoint"),
            "kernel_total": durations.get("total"),
        },
        "error": {
            "type": str(err.get("type", ""))[:200],
            "message": str(err.get("message", ""))[:MAX_MESSAGE_CHARS],
            "traceback": _clip(str(err.get("traceback", "")), tb_limit),
        },
        "log_tail": tail,
        "artifacts": {
            "run_dir": _relative(paths.run_dir),
            "run_log": _relative(paths.run_log) if paths.run_log.exists() else "",
            "kaggle_log": _relative(paths.kaggle_log) if paths.kaggle_log.exists() else "",
            "run_log_bytes": log_bytes,
            "files": _list_artifacts(paths.output_dir),
        },
        "env": {"python": env.get("python", ""), "gpu": env.get("gpu", "")},
        "entrypoint": cfg.job.entrypoint,
    }


def _overall_status(success: bool, kernel_status: str, stage: str) -> str:
    if success:
        return "success"
    if kernel_status in {"cancelacknowledged", "cancelrequested", "cancelled"}:
        return "cancelled"
    if stage == "no-result":
        return "no-result"
    if stage.startswith("bootstrap"):
        return "runner-error"
    return f"failed:{stage}" if stage else "failed"


def _read_tail(path: Path, lines: int) -> list[str]:
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return text.splitlines()[-lines:]
