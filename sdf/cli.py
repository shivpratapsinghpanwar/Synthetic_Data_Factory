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
    suffix = str(opts.get("cls") or opts.get("tag") or "")
    path = write_result(result, suffix=suffix)

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


def cmd_report(args) -> int:
    from pathlib import Path as _P

    from . import report

    src = _P(args.evaluate_json)
    if not src.exists():
        print(f"not found: {src}", file=sys.stderr)
        return 2
    context = {}
    if args.run_id:
        context["run"] = args.run_id
    if args.commit:
        context["commit"] = args.commit
    out = report.build_report(src, _P(args.out), context=context)
    print(f"wrote {out}")
    return 0


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

    rep_p = sub.add_parser("report", help="render an evaluate result to markdown")
    rep_p.add_argument("evaluate_json", help="path to a stage_evaluate*.json")
    rep_p.add_argument("--out", default="docs/results.md")
    rep_p.add_argument("--run-id", default="", help="run id for provenance context")
    rep_p.add_argument("--commit", default="", help="commit for provenance context")
    rep_p.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
