"""Command line interface: ``python -m kaggle_runner <command>``.

Exit codes are part of the contract so an automation loop can branch on them
without parsing text:

    0  run succeeded
    1  run failed (the job itself errored) - inspect result.json
    2  preflight/infrastructure error - nothing meaningful was executed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, builder, config, gitctl, kaggle_cli, runner

EXIT_OK = 0
EXIT_RUN_FAILED = 1
EXIT_INFRA = 2


def _load(args) -> config.Config:
    cfg = config.load(args.config)
    # CLI overrides, applied after validation of the file itself.
    if getattr(args, "entrypoint", None):
        cfg.job.entrypoint = args.entrypoint
    if getattr(args, "timeout", None):
        cfg.kernel.timeout_s = args.timeout
    if getattr(args, "gpu", None):
        cfg.kernel.enable_gpu = True
        cfg.kernel.accelerator = args.gpu
    if getattr(args, "slug", None):
        # A second kernel lets long sessions run (or queue) in parallel
        # without a new push cancelling the one in flight.
        cfg.kernel.slug = args.slug
        cfg.kernel.title = args.slug
    return cfg


def _print_summary(result: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2))
        return

    mark = "PASS" if result["success"] else "FAIL"
    dur = result["duration_s"]
    print("")
    print(f"[{mark}] {result['status']}  ({result['run_id']})")
    print(f"  commit     {result['git']['short']}  {result['git']['subject']}")
    print(f"  kernel     {result['kaggle']['kernel']} v{result['kaggle']['version']} "
          f"[{result['kaggle']['status']}] {result['kaggle']['accelerator']}")
    print(f"  exit       code={result['exit']['code']} stage={result['exit']['stage']}")
    print(f"  duration   wall={dur['wall']}s queue={dur['queue']}s "
          f"entrypoint={dur['entrypoint']}s")
    if not result["success"]:
        err = result["error"]
        if err["type"] or err["message"]:
            print(f"  error      {err['type']}: {err['message']}")
        if err["traceback"]:
            print("  --- traceback (clipped) ---")
            for line in err["traceback"].splitlines()[-25:]:
                print("  " + line)
    print(f"  artifacts  {result['artifacts']['run_dir']}  "
          f"(full log: {result['artifacts']['run_log'] or 'n/a'}, "
          f"{result['artifacts']['run_log_bytes']} bytes)")
    print(f"  summary    {result['artifacts']['run_dir']}/result.json")
    print("")


# --------------------------------------------------------------------------- run
def cmd_run(args) -> int:
    try:
        cfg = _load(args)
    except config.ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_INFRA

    def on_event(kind: str, detail: str) -> None:
        if not args.quiet:
            print(f"[{kind}] {detail}", flush=True)

    try:
        result = runner.execute(
            cfg,
            allow_dirty=args.allow_dirty,
            auto_push=args.push,
            keep_build=args.keep_build,
            on_event=on_event,
        )
    except runner.PreflightError as exc:
        print(str(exc), file=sys.stderr)
        print("\nhint: `python -m kaggle_runner doctor` explains each check.", file=sys.stderr)
        return EXIT_INFRA
    except kaggle_cli.KaggleCliError as exc:
        print(f"kaggle cli error: {exc}\n{exc.output}", file=sys.stderr)
        return EXIT_INFRA

    _print_summary(result, args.json)
    return EXIT_OK if result["success"] else EXIT_RUN_FAILED


# ------------------------------------------------------------------------ doctor
def cmd_doctor(args) -> int:
    try:
        cfg = _load(args)
    except config.ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_INFRA

    print(f"kaggle_runner {__version__}")
    print(f"config: {cfg.path}")
    print(f"kernel: {cfg.kernel.ref}  ->  {cfg.kernel.url}")
    print(f"entrypoint: {cfg.job.entrypoint}")
    print("")

    checks = runner.preflight(cfg, allow_dirty=args.allow_dirty)
    worst = EXIT_OK
    for check in checks:
        print(f"  [{'ok ' if check.ok else 'FAIL'}] {check.name}: {check.detail}")
        if not check.ok:
            worst = EXIT_INFRA

    # Only offer auth instructions when auth is actually what failed.
    if any(not c.ok for c in checks if c.name.startswith("kaggle.")):
        print("")
        print("To fix Kaggle authentication, do ONE of the following yourself")
        print("(the runner will never handle your token):")
        print("  1. kaggle auth login                       # OAuth, recommended")
        print("  2. Generate a token at https://www.kaggle.com/settings/api")
        print("     then save it to ~/.kaggle/kaggle.json  (or set KAGGLE_API_TOKEN)")
    return worst


# ------------------------------------------------------------------------- build
def cmd_build(args) -> int:
    """Render the kernel payload without touching Kaggle. Useful for review."""
    try:
        cfg = _load(args)
    except config.ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_INFRA

    state = gitctl.inspect()
    folder = builder.build(cfg, state.commit, "dry-run")
    print(f"built kernel payload in {folder}")
    for item in sorted(folder.iterdir()):
        print(f"  {item.name}  ({item.stat().st_size} bytes)")
    return EXIT_OK


# ------------------------------------------------------------------------ result
def cmd_result(args) -> int:
    """Print a stored summary. Default: the most recent run."""
    try:
        cfg = _load(args)
    except config.ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_INFRA

    if args.run_id:
        path = cfg.runs_path / args.run_id / "result.json"
    else:
        path = cfg.runs_path / "latest.json"

    if not path.exists():
        print(f"no result at {path}", file=sys.stderr)
        return EXIT_INFRA

    result = json.loads(path.read_text(encoding="utf-8"))
    _print_summary(result, args.json)
    return EXIT_OK if result["success"] else EXIT_RUN_FAILED


# ----------------------------------------------------------------------- publish
def cmd_publish(args) -> int:
    """Publish a run's output as a version of the private artifacts dataset."""
    from . import artifacts

    try:
        cfg = _load(args)
    except config.ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_INFRA

    run_ids = args.run_id or [(cfg.runs_path / "latest.txt").read_text().strip()]
    try:
        result = artifacts.publish(cfg, run_ids)
    except (artifacts.PublishError, kaggle_cli.KaggleCliError) as exc:
        print(f"publish failed: {exc}", file=sys.stderr)
        return EXIT_INFRA

    print(f"{result['action']}: {result['dataset']}  (run {result['run_id']})")
    print(f"attach in runner.toml dataset_sources or browse {result['url']}")
    return EXIT_OK


# -------------------------------------------------------------------------- logs
def cmd_logs(args) -> int:
    """Print a bounded slice of a stored full log - never the whole thing by default."""
    try:
        cfg = _load(args)
    except config.ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_INFRA

    run_id = args.run_id or (cfg.runs_path / "latest.txt").read_text().strip()
    candidates = [
        cfg.runs_path / run_id / "output" / "sdf_run.log",
        cfg.runs_path / run_id / "kaggle_console.log",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        print(f"no log found for run {run_id}", file=sys.stderr)
        return EXIT_INFRA

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if args.grep:
        import re

        pattern = re.compile(args.grep)
        lines = [line for line in lines if pattern.search(line)]

    selected = lines if args.all else lines[-args.tail:]
    print(f"# {path}  ({len(lines)} lines, showing {len(selected)})")
    print("\n".join(selected))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kaggle_runner",
        description="Execute this repository's current git revision on Kaggle.",
    )
    parser.add_argument("--config", default=None, help="path to runner.toml")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--entrypoint", help="override [job] entrypoint")
        p.add_argument("--allow-dirty", action="store_true",
                       help="permit an uncommitted working tree (run uses HEAD anyway)")

    run_p = sub.add_parser("run", help="push and execute on Kaggle, then report")
    common(run_p)
    run_p.add_argument("--push", action="store_true",
                       help="git push HEAD to origin before running")
    run_p.add_argument("--timeout", type=int, help="kernel time limit in seconds")
    run_p.add_argument("--slug", help="override the kernel slug (parallel sessions)")
    run_p.add_argument("--gpu", metavar="ACCELERATOR",
                       help='opt in to GPU, e.g. "NvidiaTeslaT4" (off by default)')
    run_p.add_argument("--json", action="store_true", help="print the summary as JSON")
    run_p.add_argument("--quiet", action="store_true", help="suppress progress events")
    run_p.add_argument("--keep-build", action="store_true",
                       help="keep .sdf_build/ for inspection")
    run_p.set_defaults(func=cmd_run)

    doc_p = sub.add_parser("doctor", help="check config, git state and Kaggle auth")
    common(doc_p)
    doc_p.set_defaults(func=cmd_doctor)

    build_p = sub.add_parser("build", help="render the kernel payload locally (no network)")
    common(build_p)
    build_p.set_defaults(func=cmd_build)

    res_p = sub.add_parser("result", help="show a stored run summary")
    common(res_p)
    res_p.add_argument("run_id", nargs="?", help="defaults to the latest run")
    res_p.add_argument("--json", action="store_true")
    res_p.set_defaults(func=cmd_result)

    pub_p = sub.add_parser(
        "publish", help="upload a run's output as a private Kaggle dataset version"
    )
    common(pub_p)
    pub_p.add_argument(
        "run_id", nargs="*",
        help="one or more run ids to merge (later wins); defaults to the latest run",
    )
    pub_p.set_defaults(func=cmd_publish)

    log_p = sub.add_parser("logs", help="show part of a stored full log")
    common(log_p)
    log_p.add_argument("run_id", nargs="?", help="defaults to the latest run")
    log_p.add_argument("--tail", type=int, default=80, help="lines from the end (default 80)")
    log_p.add_argument("--grep", help="only lines matching this regex")
    log_p.add_argument("--all", action="store_true", help="print the entire log")
    log_p.set_defaults(func=cmd_logs)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
