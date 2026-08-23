"""Orchestration: preflight -> build -> push -> poll -> collect -> summarise."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from . import kaggle_cli, builder, gitctl, summary
from .config import Config, kaggle_credentials_present


class PreflightError(RuntimeError):
    """A precondition failed; nothing was submitted to Kaggle."""


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def preflight(cfg: Config, *, allow_dirty: bool = False, check_remote: bool = True) -> list[Check]:
    """Verify everything needed for a reproducible run. Never mutates state."""
    checks: list[Check] = []

    try:
        state = gitctl.inspect()
    except gitctl.GitError as exc:
        return [Check("git", False, str(exc))]

    checks.append(Check("git.commit", True, f"{state.short} on {state.branch}: {state.subject}"))
    checks.append(
        Check(
            "git.clean",
            allow_dirty or not state.dirty,
            "working tree clean" if not state.dirty
            else "uncommitted changes present (Kaggle would run the last commit, not your edits)",
        )
    )
    if check_remote:
        checks.append(
            Check(
                "git.pushed",
                state.pushed,
                f"commit {state.short} reachable from origin" if state.pushed
                else f"commit {state.short} is not on origin - Kaggle cannot clone it",
            )
        )

    # The live API call is authoritative. The file probe is only a hint used to
    # explain *why* auth failed, so an unrecognised credential filename can
    # never block a session that actually authenticates fine.
    auth_ok, auth_detail = kaggle_cli.authenticated()
    creds_ok, creds_detail = kaggle_credentials_present()

    checks.append(
        Check(
            "kaggle.credentials",
            auth_ok or creds_ok,
            creds_detail if creds_ok else f"{creds_detail} (CLI may still have its own store)",
        )
    )
    checks.append(Check("kaggle.auth", auth_ok, auth_detail))

    checks.append(
        Check(
            "kernel.accelerator",
            True,
            cfg.kernel.accelerator or "cpu (no GPU quota consumed)",
        )
    )
    return checks


def _make_run_id(commit_short: str) -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + commit_short


# If the CLI's status wording changes again, every poll returns "unknown" and
# the loop would spin until poll_timeout_s. Bail out early and loudly instead.
MAX_UNKNOWN_POLLS = 8


def _poll(cfg: Config, ref: str, on_tick=None) -> tuple[str, float]:
    """Poll kernel status until terminal. Returns (status, seconds_waited)."""
    started = time.time()
    interval = float(cfg.local.poll_interval_s)
    last = ""
    unknown_streak = 0
    last_raw = ""

    while True:
        elapsed = time.time() - started
        if elapsed > cfg.local.poll_timeout_s:
            return "timeout", elapsed

        state, raw = kaggle_cli.status(ref)
        last_raw = raw
        if state != last and on_tick:
            on_tick(state, raw)
        last = state

        if kaggle_cli.is_terminal(state):
            return state, time.time() - started

        if state == "unknown":
            unknown_streak += 1
            if unknown_streak >= MAX_UNKNOWN_POLLS:
                if on_tick:
                    on_tick("unparseable", last_raw[:300])
                return "unknown", time.time() - started
        else:
            unknown_streak = 0

        time.sleep(interval)
        # Gentle backoff: long jobs should not hammer the API.
        interval = min(interval * 1.25, float(cfg.local.poll_backoff_max_s))


def execute(
    cfg: Config,
    *,
    allow_dirty: bool = False,
    auto_push: bool = False,
    keep_build: bool = False,
    on_event=None,
) -> dict:
    """Run the current revision on Kaggle and return the summary dict."""
    emit = on_event or (lambda *_: None)
    wall_start = time.time()

    # --- preflight ---------------------------------------------------------
    if auto_push:
        state = gitctl.inspect()
        if not state.dirty and not state.pushed:
            emit("push", f"pushing {state.short} to origin/{state.branch}")
            gitctl.push(branch=state.branch)

    checks = preflight(cfg, allow_dirty=allow_dirty)
    failed = [c for c in checks if not c.ok]
    if failed:
        raise PreflightError(
            "preflight failed:\n"
            + "\n".join(f"  - {c.name}: {c.detail}" for c in failed)
        )

    git_state = gitctl.inspect()
    run_id = _make_run_id(git_state.short)
    paths = summary.RunPaths.for_run(cfg.runs_path, run_id)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    infra_error = ""
    kernel_status = "unknown"
    kernel_version: int | None = None
    queue_s = 0.0

    # --- build + push ------------------------------------------------------
    emit("build", f"building kernel payload for {git_state.short}")
    folder = builder.build(cfg, git_state.commit, run_id)
    (paths.run_dir / "kernel-metadata.json").write_text(
        (folder / "kernel-metadata.json").read_text(encoding="utf-8"), encoding="utf-8"
    )

    try:
        emit("push", f"pushing to {cfg.kernel.ref} ({cfg.kernel.accelerator or 'cpu'})")
        res = kaggle_cli.push(folder, cfg.kernel.timeout_s, cfg.kernel.accelerator)
        (paths.run_dir / "push.log").write_text(res.combined, encoding="utf-8")
        kernel_version = kaggle_cli.parse_push_version(res.combined)
        emit("push", f"kernel version {kernel_version or '?'} queued")

        # --- poll ----------------------------------------------------------
        kernel_status, queue_s = _poll(
            cfg, cfg.kernel.ref, on_tick=lambda s, _r: emit("status", s)
        )
        if kernel_status == "timeout":
            infra_error = (
                f"kernel did not reach a terminal state within "
                f"{cfg.local.poll_timeout_s}s"
            )
    except kaggle_cli.KaggleCliError as exc:
        infra_error = f"{exc}: {exc.output}"[:1500]
        emit("error", infra_error)

    # --- collect artifacts -------------------------------------------------
    if not infra_error or kernel_status != "unknown":
        emit("collect", "downloading kernel output")
        out = kaggle_cli.output(cfg.kernel.ref, paths.output_dir)
        if out.returncode != 0 and not infra_error:
            infra_error = f"kernels output failed: {out.combined[-500:]}"

        try:
            console = kaggle_cli.logs(cfg.kernel.ref)
            if console:
                paths.kaggle_log.write_text(console, encoding="utf-8", errors="replace")
        except kaggle_cli.KaggleCliError:
            pass  # console log is a nice-to-have; sdf_run.log is the real record

    kernel_result = summary.load_kernel_result(paths)

    # --- summarise ---------------------------------------------------------
    result = summary.build(
        run_id=run_id,
        cfg=cfg,
        git_state=git_state,
        kernel_status=kernel_status,
        kernel_version=kernel_version,
        kernel_result=kernel_result,
        paths=paths,
        wall_s=time.time() - wall_start,
        queue_s=queue_s,
        infra_error=infra_error,
    )

    paths.result_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_latest_pointer(cfg.runs_path, run_id, result)

    if not keep_build:
        shutil.rmtree(folder, ignore_errors=True)

    return result


def collect(cfg: Config, run_id: str, *, slug: str = "", on_event=None) -> dict:
    """Resume collection for a run whose local process died mid-flight.

    Waits for the kernel to reach a terminal state if it is still running,
    then downloads output, fetches logs and writes the summary - identical to
    the tail of execute(). The commit is recovered from the run id suffix.
    """
    emit = on_event or (lambda *_: None)
    wall_start = time.time()
    if slug:
        cfg.kernel.slug = slug

    paths = summary.RunPaths.for_run(cfg.runs_path, run_id)
    if not paths.run_dir.is_dir():
        raise PreflightError(f"unknown run id: {run_id}")
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    # The run id embeds the short sha (<stamp>-<short>); resolve it locally.
    short = run_id.rsplit("-", 1)[-1]
    git_state = gitctl.inspect()
    if not git_state.commit.startswith(short):
        resolved = gitctl._git("rev-parse", short, check=False)
        if resolved:
            from dataclasses import replace

            git_state = replace(
                git_state, commit=resolved, short=resolved[:9],
                subject=gitctl._git("log", "-1", "--pretty=%s", check=False) or "",
            )

    emit("status", f"waiting for {cfg.kernel.ref} to finish")
    kernel_status, queue_s = _poll(cfg, cfg.kernel.ref,
                                   on_tick=lambda st, _r: emit("status", st))
    infra_error = ""
    if kernel_status == "timeout":
        infra_error = f"kernel did not finish within {cfg.local.poll_timeout_s}s"

    emit("collect", "downloading kernel output")
    out = kaggle_cli.output(cfg.kernel.ref, paths.output_dir)
    if out.returncode != 0 and not infra_error:
        infra_error = f"kernels output failed: {out.combined[-500:]}"
    try:
        console = kaggle_cli.logs(cfg.kernel.ref)
        if console:
            paths.kaggle_log.write_text(console, encoding="utf-8", errors="replace")
    except kaggle_cli.KaggleCliError:
        pass

    kernel_result = summary.load_kernel_result(paths)
    result = summary.build(
        run_id=run_id, cfg=cfg, git_state=git_state,
        kernel_status=kernel_status, kernel_version=None,
        kernel_result=kernel_result, paths=paths,
        wall_s=time.time() - wall_start, queue_s=queue_s,
        infra_error=infra_error,
    )
    paths.result_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_latest_pointer(cfg.runs_path, run_id, result)
    return result


def _write_latest_pointer(runs_root: Path, run_id: str, result: dict) -> None:
    """A stable path an automation loop can always read: runs/latest.json."""
    runs_root.mkdir(parents=True, exist_ok=True)
    (runs_root / "latest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (runs_root / "latest.txt").write_text(run_id + "\n", encoding="utf-8")
