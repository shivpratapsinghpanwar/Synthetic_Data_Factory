#!/usr/bin/env python3
"""Bootstrap script executed INSIDE a Kaggle kernel. Do not run locally.

This file is a template. ``kaggle_runner.builder`` replaces the marked line
below with a concrete job description and pushes the result as the kernel's
code file. It is kept as valid Python so it can be linted and diffed normally.

Responsibilities, in order:
  1. Clone the repository at an exact commit (no branch tips, no surprises).
  2. Optionally install a requirements file.
  3. Run the job entrypoint, streaming output to a log file on disk.
  4. Write /kaggle/working/sdf_result.json - the machine-readable contract.

It deliberately never raises on job failure (unless ``fail_kernel_on_error``),
so that Kaggle finishes the session normally and the output artifacts - the
result JSON and the full log - remain downloadable.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

# ---------------------------------------------------------------------------
SDF_JOB = {"placeholder": True}  # SDF_JOB_INJECTION_POINT
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1

WORKING = Path("/kaggle/working")
if not WORKING.is_dir():  # local dry-run fallback
    WORKING = Path.cwd()


def _checkout_dir():
    """Somewhere to clone that is NOT the output directory.

    Everything under /kaggle/working is uploaded as kernel output. Cloning there
    would ship the whole repo back on every run, so prefer Kaggle's scratch dir
    and fall back to the system temp dir.
    """
    for candidate in (Path("/kaggle/temp"), Path(tempfile.gettempdir())):
        try:
            if candidate.is_dir():
                target = candidate / "sdf_repo"
                target.mkdir(parents=True, exist_ok=True)
                return target
        except OSError:
            continue
    return WORKING / "_repo"


CHECKOUT = _checkout_dir()
LOG_PATH = WORKING / "sdf_run.log"
RESULT_PATH = WORKING / "sdf_result.json"


def rmtree(path):
    """Remove a tree, tolerating read-only git objects (Windows dry-runs)."""
    def _force(func, target, _exc):
        try:
            os.chmod(target, stat.S_IWRITE)
            func(target)
        except OSError:
            pass

    shutil.rmtree(path, onerror=_force)

TAIL_LINES = int(SDF_JOB.get("log_tail_lines", 40))
TB_MAX_CHARS = int(SDF_JOB.get("traceback_max_chars", 4000))

_tail = deque(maxlen=max(TAIL_LINES, 200))
_log_fh = None


def log(line=""):
    """Write to the run log, the in-memory tail buffer, and Kaggle's console."""
    text = line.rstrip("\n")
    _tail.append(text)
    if _log_fh is not None:
        _log_fh.write(text + "\n")
        _log_fh.flush()
    print(text, flush=True)


def banner(title):
    log("")
    log("=" * 72)
    log("== " + title)
    log("=" * 72)


def stream(cmd, cwd=None, env=None):
    """Run a command, streaming combined output into the log. Returns exit code."""
    shell = isinstance(cmd, str)
    log("$ " + (cmd if shell else " ".join(cmd)))
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        errors="replace",
        env=env,
    )
    for raw in proc.stdout:
        log(raw)
    proc.wait()
    log("[exit %d]" % proc.returncode)
    return proc.returncode


_TB_START = "Traceback (most recent call last):"
_EXC_LINE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Exit|Interrupt))\b:?\s*(.*)$"
)
# Banner/bookkeeping lines the bootstrap itself emits - never a useful diagnosis.
_NOISE = re.compile(r"^(={5,}|==\s|\$\s|\[exit\s|\[env\]|\[info\]|\[clone\]|\[auth\])")
# Broad enough to catch non-Python failures: missing files, bad commands, OOM,
# permission problems, git errors. "Errno" and "can't open" are deliberate -
# a \berror\b word match misses both.
_ERROR_HINT = re.compile(
    r"(error|failed|failure|fatal|exception|errno|traceback"
    r"|no such file|not found|cannot |can't |unable to|denied|refused"
    r"|killed|out of memory|oom|aborted|timed out|segmentation fault"
    r"|assertion|invalid|missing)",
    re.IGNORECASE,
)


def extract_error(lines):
    """Pull the last Python traceback (or last error-looking line) out of a log."""
    start = None
    for idx in range(len(lines) - 1, -1, -1):
        if _TB_START in lines[idx]:
            start = idx
            break

    if start is not None:
        block = lines[start:]
        text = "\n".join(block)
        if len(text) > TB_MAX_CHARS:
            text = "...[truncated]...\n" + text[-TB_MAX_CHARS:]
        exc_type, message = "", ""
        for line in reversed(block):
            match = _EXC_LINE.match(line.strip())
            if match:
                exc_type, message = match.group(1), match.group(2).strip()
                break
        return {"type": exc_type, "message": message[:1000], "traceback": text}

    # No Python traceback. Most non-Python failures (missing file, bad command,
    # OOM kill, git failure) still print one recognisable line - find it.
    candidates = [
        line.strip()
        for line in lines
        if line.strip() and not _NOISE.match(line.strip())
    ]
    for line in reversed(candidates):
        if _ERROR_HINT.search(line):
            return {"type": "", "message": line[:1000], "traceback": ""}

    # Still nothing conclusive: return the tail so the caller is not left blind.
    if candidates:
        context = candidates[-5:]
        return {
            "type": "",
            "message": context[-1][:1000],
            "traceback": "\n".join(context)[-TB_MAX_CHARS:],
        }
    return {"type": "", "message": "", "traceback": ""}


def describe_env():
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "gpu": "none",
    }
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()
            if out:
                info["gpu"] = ", ".join(out.splitlines())
        except Exception:  # present but unusable - not worth failing the run over
            info["gpu"] = "unknown"
    return info


def clone(url, commit, token_env=""):
    """Fetch exactly ``commit`` into CHECKOUT. Returns (exit_code, resolved_sha)."""
    if CHECKOUT.exists():
        rmtree(CHECKOUT)
    CHECKOUT.mkdir(parents=True, exist_ok=True)

    fetch_url = url
    if token_env:
        token = os.environ.get(token_env, "")
        if token and url.startswith("https://"):
            fetch_url = url.replace("https://", "https://x-access-token:" + token + "@", 1)
            log("[auth] using token from secret '%s'" % token_env)
        else:
            log("[auth] secret '%s' not available; cloning anonymously" % token_env)

    steps = [
        ["git", "init", "--quiet"],
        ["git", "remote", "add", "origin", fetch_url],
        # A single-commit fetch: fast, and impossible to accidentally end up on
        # a different revision than the one the runner pinned.
        ["git", "fetch", "--depth", "1", "--quiet", "origin", commit],
        ["git", "checkout", "--quiet", "FETCH_HEAD"],
    ]
    for step in steps:
        safe = [url if part == fetch_url else part for part in step]  # never log the token
        log("$ " + " ".join(safe))
        proc = subprocess.run(step, cwd=str(CHECKOUT), capture_output=True, text=True)
        if proc.stdout.strip():
            log(proc.stdout)
        if proc.returncode != 0:
            log((proc.stderr or "").replace(fetch_url, url))
            return proc.returncode, ""

    resolved = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(CHECKOUT), capture_output=True, text=True
    ).stdout.strip()
    log("[clone] HEAD = " + resolved)
    return 0, resolved


def main():
    global _log_fh

    started = time.time()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))
    WORKING.mkdir(parents=True, exist_ok=True)
    _log_fh = LOG_PATH.open("w", encoding="utf-8", errors="replace")

    job = SDF_JOB
    commit = job["commit"]
    durations = {}
    stage = "startup"
    exit_code = 0
    resolved_sha = ""
    env_info = {}

    banner("SDF runner | commit %s | run %s" % (commit[:9], job.get("run_id", "?")))
    env_info = describe_env()
    for key, value in env_info.items():
        log("[env] %s: %s" % (key, value))

    try:
        # --- 1. clone ------------------------------------------------------
        stage = "clone"
        banner("1/3 clone repository")
        t0 = time.time()
        exit_code, resolved_sha = clone(
            job["repo_url"], commit, job.get("git_token_secret", "")
        )
        durations["clone"] = round(time.time() - t0, 2)

        if exit_code == 0 and resolved_sha and resolved_sha != commit:
            log("[clone] FATAL: expected %s, got %s" % (commit, resolved_sha))
            exit_code = 90

        # --- 2. dependencies ------------------------------------------------
        if exit_code == 0 and job.get("requirements"):
            stage = "deps"
            banner("2/3 install requirements")
            req = CHECKOUT / job["requirements"]
            t0 = time.time()
            if req.exists():
                exit_code = stream(
                    [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
                    cwd=CHECKOUT,
                )
            else:
                log("[deps] FATAL: requirements file not found: %s" % job["requirements"])
                exit_code = 91
            durations["deps"] = round(time.time() - t0, 2)
        else:
            banner("2/3 install requirements (skipped)")
            durations["deps"] = 0.0

        # --- 3. entrypoint ---------------------------------------------------
        if exit_code == 0:
            stage = "entrypoint"
            banner("3/3 run entrypoint: " + job["entrypoint"])
            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONPATH"] = str(CHECKOUT) + os.pathsep + env.get("PYTHONPATH", "")
            env["SDF_COMMIT"] = commit
            env["SDF_RUN_ID"] = str(job.get("run_id", ""))
            env["SDF_OUTPUT_DIR"] = str(WORKING)
            t0 = time.time()
            exit_code = stream(job["entrypoint"], cwd=CHECKOUT, env=env)
            durations["entrypoint"] = round(time.time() - t0, 2)
        else:
            durations.setdefault("entrypoint", 0.0)

    except Exception:  # bootstrap itself broke - still emit a usable result
        import traceback as _tb

        stage = "bootstrap:" + stage
        exit_code = 99
        for line in _tb.format_exc().splitlines():
            log(line)

    finally:
        durations["total"] = round(time.time() - started, 2)
        success = exit_code == 0
        tail = list(_tail)

        result = {
            "schema_version": SCHEMA_VERSION,
            "success": success,
            "stage": "done" if success else stage,
            "commit_requested": commit,
            "commit_executed": resolved_sha,
            "run_id": job.get("run_id", ""),
            "entrypoint": job["entrypoint"],
            "exit_code": exit_code,
            "durations_s": durations,
            "error": (
                {"type": "", "message": "", "traceback": ""}
                if success
                else extract_error(tail)
            ),
            "log_file": LOG_PATH.name,
            "log_bytes": LOG_PATH.stat().st_size if LOG_PATH.exists() else 0,
            "log_tail": tail[-TAIL_LINES:],
            "env": env_info,
            "started_utc": started_utc,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        banner(
            "RESULT: %s (exit %d, stage %s)"
            % ("SUCCESS" if success else "FAILURE", exit_code, result["stage"])
        )
        RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if _log_fh is not None:
            _log_fh.close()

        # The checkout is source, not output - never ship it back.
        rmtree(CHECKOUT)

    return exit_code


if __name__ == "__main__":
    code = main()
    if code != 0 and SDF_JOB.get("fail_kernel_on_error"):
        sys.exit(code)
    # Otherwise exit 0 so Kaggle marks the session complete and the output files
    # (sdf_result.json, sdf_run.log) stay retrievable.
    sys.exit(0)
