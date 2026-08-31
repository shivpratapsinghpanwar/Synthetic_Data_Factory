#!/usr/bin/env python3
"""Render rollout videos from trained G1 locomotion checkpoints (Kaggle GPU).

Mount requirements (via the runner):
  --dataset-source shivpratap0007/custom-policy-traning-unitree-rl-mjlab  (code)
  --kernel-source  shivpratap0007/stand-walk-run-take-hits-from-alldirections-g1-uni
                                                                    (checkpoints)

Steps: copy code to scratch -> pip install -e -> find the newest real
checkpoint (>100KB; the kernel file listing shows 1KB stubs, so sizes are
verified on disk) -> run play.py with --video, headless EGL -> collect mp4s
into $SDF_OUTPUT_DIR/media. Exit 0 only if at least one video was produced.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

CODE_MOUNT = Path("/kaggle/input/custom-policy-traning-unitree-rl-mjlab")
CKPT_MOUNT = Path("/kaggle/input/stand-walk-run-take-hits-from-alldirections-g1-uni")
SCRATCH = Path("/kaggle/temp/g1repo")
OUT = Path(os.environ.get("SDF_OUTPUT_DIR", "/kaggle/working")) / "media"
TASK = os.environ.get("G1_TASK", "Unitree-G1-23Dof-Robust-Flat")
VIDEO_LENGTH = os.environ.get("G1_VIDEO_LENGTH", "600")


def sh(cmd: list[str], **kw) -> int:
    print("$", " ".join(map(str, cmd)), flush=True)
    return subprocess.call([str(c) for c in cmd], **kw)


def main() -> int:
    for mount, name in ((CODE_MOUNT, "code dataset"), (CKPT_MOUNT, "checkpoint kernel")):
        if not mount.is_dir():
            print(f"FATAL: {name} not mounted at {mount}")
            print("input mounts:", [p.name for p in Path("/kaggle/input").iterdir()])
            return 2

    # --- code -> scratch (mount is read-only), install -----------------------
    if not SCRATCH.exists():
        print(f"[setup] copying code -> {SCRATCH}", flush=True)
        shutil.copytree(CODE_MOUNT, SCRATCH,
                        ignore=shutil.ignore_patterns("deploy", "simulate", "doc"))
    if sh([sys.executable, "-m", "pip", "install", "-q", "-e", SCRATCH]) != 0:
        print("FATAL: pip install of the task suite failed")
        return 2
    # The suite's pins drag mujoco down to a version mjlab no longer supports
    # (mjENBL_MULTICCD AttributeError). The proven training env upgraded the
    # mujoco stack afterwards - mirror that.
    sh([sys.executable, "-m", "pip", "install", "-q", "-U",
        "mujoco", "mujoco-warp", "warp-lang"])
    sh([sys.executable, "-c",
        "import mujoco, mujoco_warp; "
        "print('[env] mujoco', mujoco.__version__)"])

    # --- find a real checkpoint ---------------------------------------------
    candidates = []
    for pt in CKPT_MOUNT.rglob("model_*.pt"):
        size = pt.stat().st_size
        match = re.search(r"model_(\d+)\.pt$", pt.name)
        step = int(match.group(1)) if match else -1
        candidates.append((size > 100_000, step, size, pt))
    if not candidates:
        print("FATAL: no model_*.pt anywhere under", CKPT_MOUNT)
        return 2
    candidates.sort(reverse=True)
    real, step, size, ckpt = candidates[0]
    print(f"[ckpt] {len(candidates)} checkpoints; best: {ckpt} "
          f"(step {step}, {size / 1e6:.1f} MB, real={real})", flush=True)
    if not real:
        print("FATAL: all checkpoints are tiny stubs - largest is "
              f"{size} bytes. The training kernel output may hold only "
              "placeholders; check the archived logs zip instead.")
        for _, s, sz, p in candidates[:10]:
            print(f"   step {s:>6}  {sz:>10} bytes  {p}")
        return 1

    # --- render --------------------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")
    env["PYTHONUNBUFFERED"] = "1"
    code = sh(
        [sys.executable, str(SCRATCH / "scripts" / "play.py"), TASK,
         f"--checkpoint_file={ckpt}", "--video", f"--video_length={VIDEO_LENGTH}",
         "--num_envs=1"],
        cwd=str(SCRATCH), env=env,
    )
    print(f"[play] exit {code}", flush=True)

    # --- collect whatever was rendered ---------------------------------------
    found = []
    for root in (SCRATCH, Path("/kaggle/working"), Path.cwd()):
        for video in root.rglob("*.mp4"):
            if OUT in video.parents:
                continue
            dest = OUT / f"{TASK}_{step}_{video.name}"
            shutil.copy2(video, dest)
            found.append(dest)
    print(f"[collect] {len(found)} video(s) -> {OUT}")
    for video in found:
        print("   ", video.name, f"{video.stat().st_size / 1e6:.1f} MB")
    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main())
