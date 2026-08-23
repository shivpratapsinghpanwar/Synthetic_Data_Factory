#!/usr/bin/env python3
"""Minimal CPU-only smoke test executed on Kaggle by the runner.

Purpose: prove the whole loop works end to end - the correct commit was cloned,
the entrypoint ran, output artifacts came back, and the result summary is
accurate. It deliberately does no real work and touches no GPU.

Exit codes:
    0  all assertions passed
    1  an assertion failed

Set ``SDF_SMOKE_FAIL=1`` to make it fail on purpose. That is how you verify the
failure path (error extraction, traceback capture, non-zero exit) without
waiting for a genuine bug.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def check(name: str, ok: bool, detail: str = "") -> dict:
    status = "ok" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" - {detail}" if detail else ""), flush=True)
    return {"name": name, "ok": bool(ok), "detail": detail}


def git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True
    )
    return proc.stdout.strip()


def main() -> int:
    started = time.time()
    print("=" * 60)
    print("Synthetic_Data_Factory :: Kaggle smoke test")
    print("=" * 60)

    expected_commit = os.environ.get("SDF_COMMIT", "")
    actual_commit = git("rev-parse", "HEAD")

    results = [
        check("python", sys.version_info >= (3, 9), platform.python_version()),
        check("repo_root_exists", REPO_ROOT.is_dir(), str(REPO_ROOT)),
        check("readme_present", (REPO_ROOT / "README.md").is_file()),
        check("runner_config_present", (REPO_ROOT / "runner.toml").is_file()),
        check(
            "commit_matches",
            bool(expected_commit) and actual_commit.startswith(expected_commit[:40]),
            f"expected={expected_commit[:9]} actual={actual_commit[:9]}",
        ),
        check("writable_output", _can_write()),
    ]

    # Report what hardware we landed on, so an accidental GPU run is visible in
    # the summary rather than silent.
    accel = "none"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        if out.returncode == 0 and out.stdout.strip():
            accel = out.stdout.strip().replace("\n", ", ")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    print(f"[info] accelerator: {accel}")
    print(f"[info] cwd: {Path.cwd()}")

    if os.environ.get("SDF_SMOKE_FAIL") == "1":
        print("[info] SDF_SMOKE_FAIL=1 - raising on purpose to exercise the failure path")
        raise RuntimeError("intentional smoke-test failure (SDF_SMOKE_FAIL=1)")

    payload = {
        "smoke_test": "synthetic_data_factory",
        "commit": actual_commit,
        "run_id": os.environ.get("SDF_RUN_ID", ""),
        "accelerator": accel,
        "python": platform.python_version(),
        "checks": results,
        "duration_s": round(time.time() - started, 3),
    }

    out_dir = Path(os.environ.get("SDF_OUTPUT_DIR", REPO_ROOT))
    out_path = out_dir / "smoke_test_report.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[info] wrote {out_path}")

    failed = [r["name"] for r in results if not r["ok"]]
    if failed:
        print(f"\nSMOKE TEST FAILED: {', '.join(failed)}")
        return 1

    print(f"\nSMOKE TEST PASSED ({len(results)} checks in {payload['duration_s']}s)")
    return 0


def _can_write() -> bool:
    target = Path(os.environ.get("SDF_OUTPUT_DIR", ".")) / ".sdf_write_probe"
    try:
        target.write_text("ok", encoding="utf-8")
        target.unlink()
        return True
    except OSError:
        return False


if __name__ == "__main__":
    sys.exit(main())
