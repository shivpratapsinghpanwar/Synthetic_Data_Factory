"""Command line interface: ``python -m sdf <command>``.

Exit codes match the kaggle_runner convention:
    0  stage succeeded
    1  stage ran and failed
    2  configuration/usage error
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__, config
from .stages import REGISTRY, write_result


def cmd_stages(_args) -> int:
    for name in sorted(REGISTRY):
        print(name)
    return 0


def cmd_run_stage(args) -> int:
    try:
        cfg = config.load(args.config)
    except config.ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    if args.data_root:
        cfg.dataset.data_root = args.data_root

    stage_fn = REGISTRY.get(args.stage)
    if stage_fn is None:
        print(
            f"unknown stage {args.stage!r}; available: {', '.join(sorted(REGISTRY))}",
            file=sys.stderr,
        )
        return 2

    opts = {}
    for item in args.opt or []:
        if "=" not in item:
            print(f"--opt must be key=value, got {item!r}", file=sys.stderr)
            return 2
        key, _, value = item.partition("=")
        opts[key.strip()] = _coerce(value.strip())

    result = stage_fn(cfg, opts)
    path = write_result(result)

    print(f"[stage] {result.stage}: {'PASS' if result.success else 'FAIL'} "
          f"({result.duration_s}s) -> {path}")
    if result.error:
        print(f"[stage] error: {result.error}")
    if args.json:
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0 if result.success else 1


def _coerce(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sdf", description="Synthetic Data Factory pipeline."
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    stages_p = sub.add_parser("stages", help="list available stages")
    stages_p.set_defaults(func=cmd_stages)

    run_p = sub.add_parser("run-stage", help="run one pipeline stage")
    run_p.add_argument("stage", help="stage name (see `sdf stages`)")
    run_p.add_argument("--config", default=None, help="path to pipeline.toml")
    run_p.add_argument("--data-root", default=None, help="override dataset root")
    run_p.add_argument("--json", action="store_true", help="print the full result JSON")
    run_p.add_argument(
        "--opt", action="append", metavar="KEY=VALUE",
        help="stage option (repeatable), e.g. --opt cls=df --opt steps=40",
    )
    run_p.set_defaults(func=cmd_run_stage)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
