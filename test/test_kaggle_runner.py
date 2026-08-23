"""Offline tests for the Kaggle runner. No network, no Kaggle account needed.

Run with:  python -m pytest test/ -q      (or: python test/test_kaggle_runner.py)
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kaggle_runner import builder, config, kaggle_cli, summary  # noqa: E402


# --------------------------------------------------------------------- config
def test_config_loads():
    cfg = config.load()
    assert cfg.kernel.ref.count("/") == 1
    assert cfg.job.entrypoint
    assert cfg.repo.url.startswith("https://")


def test_config_rejects_gpu_without_accelerator(tmp_path=None):
    cfg = config.load()
    cfg.kernel.enable_gpu = True
    cfg.kernel.accelerator = ""
    try:
        config._validate(cfg)
    except config.ConfigError as exc:
        assert "accelerator" in str(exc)
    else:
        raise AssertionError("expected ConfigError")


def test_gpu_is_off_by_default():
    """Guards requirement: infrastructure work must not consume GPU quota."""
    cfg = config.load()
    assert cfg.kernel.enable_gpu is False
    assert cfg.kernel.accelerator == ""


def test_credentials_probe_recognises_oauth_login_file(tmp_path):
    """Regression: `kaggle auth login` writes credentials.json, not kaggle.json.

    An older hardcoded list missed it and reported "no credentials found" on a
    perfectly authenticated machine.
    """
    import os

    original = os.environ.get("KAGGLE_CONFIG_DIR")
    for var in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        os.environ.pop(var, None)
    try:
        os.environ["KAGGLE_CONFIG_DIR"] = str(tmp_path)

        ok, detail = config.kaggle_credentials_present()
        assert ok is False, detail

        for name in ("credentials.json", "access_token", "kaggle.json"):
            target = tmp_path / name
            target.write_text("{}", encoding="utf-8")
            ok, detail = config.kaggle_credentials_present()
            assert ok is True, f"{name} not recognised: {detail}"
            target.unlink()

        # An unknown filename should still count - the CLI may have renamed it.
        (tmp_path / "some_future_name.json").write_text("{}", encoding="utf-8")
        ok, _ = config.kaggle_credentials_present()
        assert ok is True
    finally:
        if original is None:
            os.environ.pop("KAGGLE_CONFIG_DIR", None)
        else:
            os.environ["KAGGLE_CONFIG_DIR"] = original


# -------------------------------------------------------------------- builder
def test_bootstrap_injection_produces_valid_python():
    cfg = config.load()
    spec = builder.job_spec(cfg, "a" * 40, "testrun")
    rendered = builder.render_bootstrap(spec)

    compile(rendered, "sdf_bootstrap.py", "exec")
    assert '{"placeholder": True}' not in rendered
    assert rendered.count(builder.INJECTION_MARKER) == 1

    # The injected payload must evaluate back to exactly the spec.
    namespace = {}
    exec(compile(rendered.split("SCHEMA_VERSION =")[0], "x", "exec"), namespace)
    assert namespace["SDF_JOB"] == spec


def test_bootstrap_payload_is_python_not_json():
    """Regression: json.dumps emits `false`/`true`/`null`, which are NameErrors.

    compile() does not catch it (they are valid identifiers), so the rendered
    module is actually executed here.
    """
    cfg = config.load()
    spec = builder.job_spec(cfg, "c" * 40, "testrun")
    spec["fail_kernel_on_error"] = False
    spec["requirements"] = ""
    rendered = builder.render_bootstrap(spec)

    assert ": false" not in rendered and ": true" not in rendered
    assert "False" in rendered

    header = rendered.split("SCHEMA_VERSION =")[0]
    namespace = {}
    exec(compile(header, "sdf_bootstrap.py", "exec"), namespace)  # would NameError before
    assert namespace["SDF_JOB"]["fail_kernel_on_error"] is False


def test_builder_rejects_non_literal_payload():
    try:
        builder._verify("SDF_JOB = undefined_name  " + builder.INJECTION_MARKER, {"a": 1})
    except builder.BuildError:
        return
    raise AssertionError("expected BuildError for a non-literal SDF_JOB")


def test_dataset_sources_flow_into_metadata():
    cfg = config.load()
    cfg.kernel.dataset_sources = ["kmader/skin-cancer-mnist-ham10000"]
    meta = builder.kernel_metadata(cfg)
    assert meta["dataset_sources"] == ["kmader/skin-cancer-mnist-ham10000"]


def test_dataset_sources_validation():
    cfg = config.load()
    cfg.kernel.dataset_sources = ["no-slash-here"]
    try:
        config._validate(cfg)
    except config.ConfigError as exc:
        assert "dataset_sources" in str(exc)
    else:
        raise AssertionError("expected ConfigError for bad dataset source")


def test_kernel_metadata_shape():
    cfg = config.load()
    meta = builder.kernel_metadata(cfg)
    assert meta["kernel_type"] == "script"
    assert meta["language"] == "python"
    assert meta["code_file"] == builder.CODE_FILE_NAME
    # Kaggle's metadata schema uses string booleans.
    assert meta["enable_internet"] in {"true", "false"}
    assert meta["enable_gpu"] == "false"


# ---------------------------------------------------------------- cli parsing
def test_parse_status_variants():
    cases = {
        'has status "complete"': "complete",
        'Kernel has status "error"': "error",
        'has status "running"': "running",
        "Kernel is queued": "queued",
        "total gibberish": "unknown",
    }
    for text, expected in cases.items():
        state, _ = kaggle_cli.parse_status(text)
        assert state == expected, f"{text!r} -> {state}"


def test_parse_status_handles_enum_form():
    """Regression: kaggle 2.2.x prints the raw enum, not a bare word.

    'KernelWorkerStatus.COMPLETE' lowercased to
    'kernelworkerstatus.complete', which was not in TERMINAL_OK - so a finished
    kernel never looked terminal and the poller spun until timeout.
    """
    observed = 'shivpratap0007/sdf-runner has status "KernelWorkerStatus.COMPLETE"'
    state, _ = kaggle_cli.parse_status(observed)
    assert state == "complete"
    assert kaggle_cli.is_terminal(state)

    for raw, expected in [
        ("KernelWorkerStatus.ERROR", "error"),
        ("KernelWorkerStatus.RUNNING", "running"),
        ("KernelWorkerStatus.QUEUED", "queued"),
        ("KernelWorkerStatus.CANCEL_ACKNOWLEDGED", "cancel_acknowledged"),
        ("COMPLETE", "complete"),
        ("Complete", "complete"),
    ]:
        assert kaggle_cli.normalise_status(raw) == expected, raw


def test_enum_terminal_states_are_recognised():
    for raw in ("KernelWorkerStatus.COMPLETE", "KernelWorkerStatus.ERROR"):
        state, _ = kaggle_cli.parse_status(f'has status "{raw}"')
        assert kaggle_cli.is_terminal(state), raw
    for raw in ("KernelWorkerStatus.RUNNING", "KernelWorkerStatus.QUEUED"):
        state, _ = kaggle_cli.parse_status(f'has status "{raw}"')
        assert not kaggle_cli.is_terminal(state), raw


def test_terminal_classification():
    assert kaggle_cli.is_terminal("complete")
    assert kaggle_cli.is_terminal("error")
    assert not kaggle_cli.is_terminal("running")
    assert not kaggle_cli.is_terminal("unknown")


def test_parse_push_version():
    assert kaggle_cli.parse_push_version("Kernel version 7 successfully pushed") == 7
    assert kaggle_cli.parse_push_version("no number here") is None


# -------------------------------------------------------------- error extract
def _bootstrap_module():
    """Import the template as a module so its pure helpers can be tested."""
    src = (REPO_ROOT / "kaggle_runner" / "bootstrap_template.py").read_text(encoding="utf-8")
    module = types.ModuleType("sdf_bootstrap_under_test")
    module.__dict__["__name__"] = "sdf_bootstrap_under_test"
    exec(compile(src, "bootstrap_template.py", "exec"), module.__dict__)
    return module


def test_extract_error_finds_traceback():
    bs = _bootstrap_module()
    lines = [
        "some noise",
        "Traceback (most recent call last):",
        '  File "job.py", line 3, in <module>',
        "    import nope",
        "ModuleNotFoundError: No module named 'nope'",
    ]
    err = bs.extract_error(lines)
    assert err["type"] == "ModuleNotFoundError"
    assert "nope" in err["message"]
    assert "Traceback" in err["traceback"]


def test_extract_error_without_traceback():
    bs = _bootstrap_module()
    err = bs.extract_error(["all good", "fatal: repository not found", ""])
    assert "repository not found" in err["message"]


def test_extract_error_keeps_context_for_split_messages():
    """A reverse scan can land on a continuation line; context must survive.

    Real case: cmd.exe splits "'X' is not recognized as an internal or external
    command, / operable program or batch file." across two lines, and only the
    second one is matched last.
    """
    bs = _bootstrap_module()
    err = bs.extract_error([
        "$ SDF_SMOKE_FAIL=1 python job.py",
        "'SDF_SMOKE_FAIL' is not recognized as an internal or external command,",
        "operable program or batch file.",
        "[exit 1]",
    ])
    assert "SDF_SMOKE_FAIL" in err["traceback"], err


def test_extract_error_non_python_failures():
    """Regression: these all lack a \berror\b word but are the real diagnosis."""
    bs = _bootstrap_module()
    cases = [
        ("python: can't open file 'x.py': [Errno 2] No such file or directory", "Errno 2"),
        ("/bin/sh: 1: nosuchcmd: not found", "not found"),
        ("Killed", "Killed"),
        ("git: Permission denied", "denied"),
    ]
    for line, needle in cases:
        err = bs.extract_error(["== banner", "$ some command", line, "[exit 1]"])
        assert needle in err["message"], f"{line!r} -> {err['message']!r}"


def test_extract_error_ignores_bootstrap_noise():
    """Banner and bookkeeping lines must never be reported as the error."""
    bs = _bootstrap_module()
    err = bs.extract_error([
        "=" * 72,
        "== 3/3 run entrypoint: python job.py",
        "=" * 72,
        "$ python job.py",
        "real problem: disk quota exceeded",
        "[exit 1]",
    ])
    assert "disk quota" in err["message"]
    assert not err["message"].startswith("==")


def test_extract_error_clean_log_returns_context_not_blank():
    """With no error hint at all, return the tail rather than nothing."""
    bs = _bootstrap_module()
    err = bs.extract_error(["everything fine", "done"])
    assert err["message"] == "done"


# ------------------------------------------------------------------- summary
class _FakeGit:
    commit = "b" * 40
    short = "b" * 9
    branch = "main"
    subject = "test commit"
    dirty = False


def _summary_for(kernel_result, tmp: Path, status="complete", infra=""):
    cfg = config.load()
    paths = summary.RunPaths.for_run(tmp, "run1")
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    return summary.build(
        run_id="run1",
        cfg=cfg,
        git_state=_FakeGit(),
        kernel_status=status,
        kernel_version=3,
        kernel_result=kernel_result,
        paths=paths,
        wall_s=12.5,
        queue_s=3.0,
        infra_error=infra,
    )


def test_summary_success(tmp_path):
    result = _summary_for(
        {
            "success": True,
            "exit_code": 0,
            "stage": "done",
            "durations_s": {"clone": 1.0, "entrypoint": 2.0, "total": 3.0},
            "error": {"type": "", "message": "", "traceback": ""},
            "log_tail": ["done"],
            "log_bytes": 100,
            "commit_executed": "b" * 40,
            "env": {"python": "3.11.11", "gpu": "none"},
        },
        tmp_path,
    )
    assert result["success"] is True
    assert result["status"] == "success"
    assert result["exit"]["code"] == 0
    assert result["git"]["commit_executed"] == "b" * 40
    assert result["kaggle"]["version"] == 3


def test_summary_failure_carries_traceback(tmp_path):
    result = _summary_for(
        {
            "success": False,
            "exit_code": 1,
            "stage": "entrypoint",
            "durations_s": {"total": 5.0},
            "error": {
                "type": "ValueError",
                "message": "bad input",
                "traceback": "Traceback...\nValueError: bad input",
            },
            "log_tail": ["boom"],
            "log_bytes": 50,
            "commit_executed": "b" * 40,
            "env": {},
        },
        tmp_path,
    )
    assert result["success"] is False
    assert result["status"] == "failed:entrypoint"
    assert result["error"]["type"] == "ValueError"
    assert "ValueError" in result["error"]["traceback"]


def test_summary_handles_missing_result(tmp_path):
    result = _summary_for(None, tmp_path, status="error")
    assert result["success"] is False
    assert result["status"] == "no-result"
    assert result["error"]["type"] == "RunnerError"


def test_summary_is_small_even_with_huge_logs(tmp_path):
    """Requirement: never dump huge Kaggle logs into the agent's context."""
    result = _summary_for(
        {
            "success": False,
            "exit_code": 1,
            "stage": "entrypoint",
            "durations_s": {"total": 5.0},
            "error": {
                "type": "RuntimeError",
                "message": "x" * 50_000,
                "traceback": "y" * 500_000,
            },
            "log_tail": ["z" * 5_000 for _ in range(5_000)],
            "log_bytes": 900_000_000,
            "commit_executed": "b" * 40,
            "env": {},
        },
        tmp_path,
    )
    encoded = json.dumps(result)
    assert len(encoded) < 40_000, f"summary too big: {len(encoded)} bytes"
    assert len(result["log_tail"]) <= summary.MAX_TAIL_LINES
    assert len(result["error"]["traceback"]) <= summary.MAX_TRACEBACK_CHARS + 32


def test_summary_reports_kernel_error_even_if_job_claimed_success(tmp_path):
    result = _summary_for(
        {
            "success": True,
            "exit_code": 0,
            "stage": "done",
            "durations_s": {},
            "error": {},
            "log_tail": [],
            "log_bytes": 0,
            "commit_executed": "b" * 40,
            "env": {},
        },
        tmp_path,
        status="error",
    )
    assert result["success"] is False


# ------------------------------------------------------------------- publish
def test_publish_staging_layout(tmp_path):
    import json as _json

    from kaggle_runner import artifacts

    cfg = config.load()
    src = tmp_path / "output"
    (src / "lora" / "df").mkdir(parents=True)
    (src / "lora" / "df" / "adapter.bin").write_bytes(b"x" * 10)

    import kaggle_runner.artifacts as art
    original = art.STAGING_DIR
    art.STAGING_DIR = tmp_path / "staging"
    try:
        folder = artifacts.stage_folder(cfg, ["run123"], [src])
    finally:
        art.STAGING_DIR = original

    meta = _json.loads((folder / "dataset-metadata.json").read_text())
    assert meta["id"] == artifacts.dataset_slug(cfg)
    assert meta["id"].endswith("-artifacts")
    # content sits at staging root - one run per dataset version
    assert (folder / "lora" / "df" / "adapter.bin").exists()


def test_publish_refuses_missing_source(tmp_path):
    from kaggle_runner import artifacts

    try:
        artifacts.stage_folder(config.load(), ["runX"], [tmp_path / "absent"])
    except artifacts.PublishError:
        return
    raise AssertionError("expected PublishError")


def test_publish_merges_multiple_runs(tmp_path):
    from kaggle_runner import artifacts

    cfg = config.load()
    a = tmp_path / "a"
    (a / "synthetic" / "df").mkdir(parents=True)
    (a / "synthetic" / "df" / "x.png").write_bytes(b"df")
    (a / "shared.json").write_text("from-a", encoding="utf-8")
    b = tmp_path / "b"
    (b / "synthetic" / "vasc").mkdir(parents=True)
    (b / "synthetic" / "vasc" / "y.png").write_bytes(b"vasc")
    (b / "shared.json").write_text("from-b", encoding="utf-8")

    import kaggle_runner.artifacts as art
    original = art.STAGING_DIR
    art.STAGING_DIR = tmp_path / "staging"
    try:
        folder = artifacts.stage_folder(cfg, ["runA", "runB"], [a, b])
    finally:
        art.STAGING_DIR = original

    assert (folder / "synthetic" / "df" / "x.png").exists()
    assert (folder / "synthetic" / "vasc" / "y.png").exists()
    # later run wins on plain collisions
    assert (folder / "shared.json").read_text(encoding="utf-8") == "from-b"


def test_publish_merge_concatenates_manifests_and_keeps_stage_jsons(tmp_path):
    from kaggle_runner import artifacts

    cfg = config.load()
    a = tmp_path / "a"
    a.mkdir()
    (a / "synthetic_manifest.jsonl").write_text('{"cls": "df"}
', encoding="utf-8")
    (a / "stage_quality_gate.json").write_text('{"cls": "df"}', encoding="utf-8")
    b = tmp_path / "b"
    b.mkdir()
    (b / "synthetic_manifest.jsonl").write_text('{"cls": "vasc"}
', encoding="utf-8")
    (b / "stage_quality_gate.json").write_text('{"cls": "bcc"}', encoding="utf-8")

    import kaggle_runner.artifacts as art
    original = art.STAGING_DIR
    art.STAGING_DIR = tmp_path / "staging"
    try:
        folder = artifacts.stage_folder(cfg, ["runA", "runB"], [a, b])
    finally:
        art.STAGING_DIR = original

    # manifests concatenate - dropping rows would silently exclude images
    lines = (folder / "synthetic_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and "df" in lines[0] and "vasc" in lines[1]
    # colliding stage jsons are both kept (augment globs stage_quality_gate*)
    assert (folder / "stage_quality_gate.json").exists()
    assert (folder / "stage_quality_gate.2.json").exists()


# ------------------------------------------------------------------ fallback
def _run_all():
    """Tiny runner so the file works without pytest installed."""
    import inspect
    import tempfile

    mod = sys.modules[__name__]
    tests = [
        (name, fn)
        for name, fn in vars(mod).items()
        if name.startswith("test_") and callable(fn)
    ]
    failures = 0
    for name, fn in tests:
        params = inspect.signature(fn).parameters
        try:
            if "tmp_path" in params:
                with tempfile.TemporaryDirectory() as td:
                    fn(Path(td))
            else:
                fn()
            print(f"  ok    {name}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
