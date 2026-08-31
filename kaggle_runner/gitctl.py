"""Git inspection used to pin a Kaggle run to an exact revision.

The whole point of the runner is reproducibility: whatever Kaggle executes must
be a commit that exists on the remote, so the run can be reproduced and so a
failure can be attributed to a specific revision.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

from .config import REPO_ROOT


class GitError(RuntimeError):
    pass


def _git(*args: str, cwd: Path | None = None, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if check and proc.returncode != 0:
        raise GitError(
            f"git {' '.join(args)} failed ({proc.returncode}): "
            f"{(proc.stderr or proc.stdout).strip()}"
        )
    return proc.stdout.strip()


@dataclass
class GitState:
    commit: str
    short: str
    branch: str
    remote: str
    dirty: bool
    pushed: bool
    subject: str

    def as_dict(self) -> dict:
        return asdict(self)


def inspect(remote_name: str = "origin") -> GitState:
    """Collect the git facts the runner needs before dispatching a run."""
    commit = _git("rev-parse", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    remote = _git("remote", "get-url", remote_name, check=False)
    subject = _git("log", "-1", "--pretty=%s")
    dirty = bool(_git("status", "--porcelain"))

    # `git branch -r --contains <sha>` is the cheapest local proxy for "the
    # remote can serve this commit". It relies on local remote-tracking refs, so
    # refresh them first; a failed fetch (offline) leaves `pushed` as best-effort.
    _git("fetch", remote_name, "--quiet", check=False)
    contains = _git("branch", "-r", "--contains", commit, check=False)
    pushed = any(line.strip().startswith(f"{remote_name}/") for line in contains.splitlines())

    return GitState(
        commit=commit,
        short=commit[:9],
        branch=branch,
        remote=remote,
        dirty=dirty,
        pushed=pushed,
        subject=subject,
    )


# Directories whose contents must never reach the public repo. The pipeline
# is open source; client/hospital imagery is not.
PRIVATE_PREFIXES = ("Data_to_reproduce_with/", "private_data/")


def tracked_private_files(prefixes: tuple[str, ...] = PRIVATE_PREFIXES) -> list[str]:
    """Tracked paths under private-data prefixes. Non-empty = policy breach."""
    out = _git("ls-files", "--", *prefixes, check=False)
    return [line for line in out.splitlines() if line.strip()]


def push(remote_name: str = "origin", branch: str | None = None) -> str:
    """Push the current branch so Kaggle can clone the target commit."""
    branch = branch or _git("rev-parse", "--abbrev-ref", "HEAD")
    return _git("push", remote_name, f"HEAD:{branch}")
