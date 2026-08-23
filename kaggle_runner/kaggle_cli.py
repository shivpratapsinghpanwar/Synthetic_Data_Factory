"""Thin, defensive wrapper around the official Kaggle CLI (`pip install kaggle`).

Design notes
------------
* The CLI is invoked as ``python -m kaggle`` so the runner works regardless of
  whether the ``kaggle`` shim is on PATH (a common Windows papercut).
* ``kernels status`` has no ``--format json`` in kaggle 2.2.x, so its human
  output is parsed with a tolerant regex and unknown text is reported as
  ``unknown`` rather than crashing the loop.
* Nothing here prints kernel logs; callers redirect them to files.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Terminal states reported by the Kaggle kernels API.
TERMINAL_OK = {"complete"}
TERMINAL_BAD = {"error", "cancelacknowledged", "cancelrequested", "cancelled"}
RUNNING = {"queued", "running", "pending"}


class KaggleCliError(RuntimeError):
    def __init__(self, message: str, returncode: int = 1, output: str = ""):
        super().__init__(message)
        self.returncode = returncode
        self.output = output


@dataclass
class CliResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def combined(self) -> str:
        return (self.stdout + "\n" + self.stderr).strip()


def _base_cmd() -> list[str]:
    return [sys.executable, "-m", "kaggle", "--no-warn"]


def run(*args: str, check: bool = True, timeout: int = 900) -> CliResult:
    """Invoke the Kaggle CLI and capture output."""
    cmd = _base_cmd() + list(args)
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env,
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise KaggleCliError(
            "Kaggle CLI not installed. Run: pip install --upgrade kaggle"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise KaggleCliError(f"kaggle {' '.join(args)} timed out after {timeout}s") from exc

    result = CliResult(proc.returncode, proc.stdout or "", proc.stderr or "")
    if check and proc.returncode != 0:
        raise KaggleCliError(
            f"kaggle {' '.join(args)} failed (exit {proc.returncode})",
            returncode=proc.returncode,
            output=result.combined[-2000:],
        )
    return result


def authenticated() -> tuple[bool, str]:
    """Confirm the CLI can actually talk to Kaggle, without leaking secrets."""
    try:
        res = run("kernels", "list", "-m", "--page-size", "1", check=False, timeout=120)
    except KaggleCliError as exc:
        return False, str(exc)
    text = res.combined
    if res.returncode == 0 and "Authentication required" not in text:
        return True, "kaggle CLI authenticated"
    return False, text[-600:] or "kaggle CLI could not authenticate"


def push(folder: Path, timeout_s: int | None = None, accelerator: str = "") -> CliResult:
    """Push a kernel folder and start a run."""
    args = ["kernels", "push", "-p", str(folder)]
    if timeout_s:
        args += ["-t", str(int(timeout_s))]
    if accelerator:
        args += ["--accelerator", accelerator]
    return run(*args, timeout=900)


_VERSION_RE = re.compile(r"version\s+(\d+)", re.IGNORECASE)


def parse_push_version(output: str) -> int | None:
    """Extract the new kernel version number from `kernels push` output."""
    match = _VERSION_RE.search(output)
    return int(match.group(1)) if match else None


_STATUS_RE = re.compile(r'status\s+"([^"]+)"', re.IGNORECASE)
_STATUS_BARE_RE = re.compile(
    r"\b(complete|error|running|queued|pending|cancelAcknowledged|cancelRequested)\b",
    re.IGNORECASE,
)


def parse_status(output: str) -> tuple[str, str]:
    """Return ``(normalised_status, raw_text)`` from `kernels status` output.

    Normalised status is lowercase, or ``"unknown"`` when nothing matched.
    """
    text = output.strip()
    match = _STATUS_RE.search(text) or _STATUS_BARE_RE.search(text)
    return (match.group(1).lower() if match else "unknown"), text


def status(ref: str) -> tuple[str, str]:
    res = run("kernels", "status", ref, check=False, timeout=120)
    return parse_status(res.combined)


def is_terminal(state: str) -> bool:
    return state in TERMINAL_OK or state in TERMINAL_BAD


def output(ref: str, dest: Path, file_pattern: str = "") -> CliResult:
    """Download kernel output files into ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    args = ["kernels", "output", ref, "-p", str(dest), "-o", "-q"]
    if file_pattern:
        args += ["--file-pattern", file_pattern]
    return run(*args, check=False, timeout=1800)


def logs(ref: str) -> str:
    """Fetch the Kaggle-side execution log as text (may be large)."""
    res = run("kernels", "logs", ref, check=False, timeout=600)
    return res.combined
