# Kaggle Execution Runner

Infrastructure that executes **this repository's current git revision** on Kaggle
and returns a small, machine-readable result. It exists to make an autonomous
loop possible:

```
edit -> commit -> push -> Kaggle run -> inspect result -> fix -> repeat
```

It contains no synthetic-data functionality. That comes later; this is the
harness that will run it.

---

## 1. Why it is built this way

**Git is the transport.** The runner does not upload your working tree. It pushes
a tiny bootstrap script to Kaggle that `git fetch`es one exact commit from
GitHub. That has three consequences worth stating plainly:

- A run is reproducible: the summary records the commit requested *and* the
  commit actually executed, and the bootstrap aborts if they differ.
- Payload size is constant, no matter how large the repo grows.
- **You must commit and push before running.** The runner refuses to start
  otherwise, rather than silently testing stale code.

**Official API, not browser automation.** Everything goes through the official
Kaggle CLI (`pip install kaggle`, the `Kaggle/kaggle-cli` package). No Chrome
automation is used or needed. See §7 for the one case where a browser is still
required — and it is a one-time human action, not part of the loop.

**Logs stay on disk.** A training run can emit hundreds of megabytes. The
summary returned to the caller is capped at a few kilobytes and references full
logs by path. This is a hard limit in code, not a convention.

---

## 2. One-time setup

```bash
pip install --upgrade kaggle
```

Then authenticate — **you must do this yourself; the runner never handles your
token**. Pick one:

```bash
kaggle auth login                    # OAuth in the browser, recommended
```

or generate a token at <https://www.kaggle.com/settings/api> ("Create New Token")
and save the downloaded `kaggle.json` to:

- Windows: `C:\Users\<you>\.kaggle\kaggle.json`
- Linux/macOS: `~/.kaggle/kaggle.json` (`chmod 600`)

Verify everything at once:

```bash
python -m kaggle_runner doctor
```

It reports git state, whether HEAD is on `origin`, Kaggle credentials, live API
auth, and the configured accelerator. Fix anything marked `FAIL` before running.

> Kaggle requires a **phone-verified account** for notebooks with internet
> access. The bootstrap clones from GitHub, so internet must be on. If pushes
> run but the clone fails with a DNS/connection error, that verification is
> the first thing to check.

---

## 3. Usage

```bash
# Full check without touching Kaggle
python -m kaggle_runner doctor

# Render the exact payload that would be pushed (no network)
python -m kaggle_runner build --keep-build

# Commit, push and run in one step
git commit -am "try a fix"
python -m kaggle_runner run --push

# Machine-readable output for an agent
python -m kaggle_runner run --push --json --quiet

# Re-read the last result without re-running
python -m kaggle_runner result
python -m kaggle_runner result --json

# Look at logs on demand, bounded by default
python -m kaggle_runner logs --tail 100
python -m kaggle_runner logs --grep "Error|Traceback"
python -m kaggle_runner logs --all          # only when you really mean it
```

### Exit codes

| Code | Meaning | Loop should |
|-----:|---------|-------------|
| `0` | Run succeeded | stop, or move to the next task |
| `1` | The job failed on Kaggle | read `error` + `log_tail`, fix, re-run |
| `2` | Preflight/infrastructure error | fix local setup; nothing was executed |

The distinction matters: `2` means your code was never tested.

---

## 4. The result contract

Written to `runs/<run_id>/result.json`, mirrored to `runs/latest.json`, and
printed by `--json`.

```jsonc
{
  "schema_version": 1,
  "run_id": "20260823T142153Z-a1b2c3d4e",
  "success": false,
  "status": "failed:entrypoint",     // success | failed:<stage> | no-result
                                     // | runner-error | cancelled
  "git": {
    "commit": "a1b2c3d4e5...",       // full sha the runner asked Kaggle to run
    "short": "a1b2c3d4e",
    "branch": "main",
    "subject": "try a fix",
    "dirty": false,
    "commit_executed": "a1b2c3d4e5..." // what Kaggle actually checked out
  },
  "kaggle": {
    "kernel": "owner/sdf-runner",
    "version": 12,
    "url": "https://www.kaggle.com/code/owner/sdf-runner",
    "status": "complete",            // Kaggle's own kernel status
    "accelerator": "cpu"
  },
  "exit":   { "code": 1, "stage": "entrypoint" },  // stage: clone|deps|entrypoint
  "duration_s": {
    "wall": 412.3,                   // total, local clock
    "queue": 380.1,                  // waiting for Kaggle
    "clone": 2.1, "deps": 0.0, "entrypoint": 12.4, "kernel_total": 15.2
  },
  "error": {
    "type": "ModuleNotFoundError",
    "message": "No module named 'foo'",
    "traceback": "...clipped to 6000 chars..."
  },
  "log_tail": ["...last 40 lines..."],
  "artifacts": {
    "run_dir": "runs/20260823T142153Z-a1b2c3d4e",
    "run_log": "runs/.../output/sdf_run.log",   // the FULL log
    "kaggle_log": "runs/.../kaggle_console.log",
    "run_log_bytes": 918273,
    "files": [{ "name": "smoke_test_report.json", "bytes": 512 }]
  },
  "env": { "python": "3.11.11", "gpu": "none" },
  "entrypoint": "python kaggle_jobs/smoke_test.py"
}
```

**Size guarantees** (enforced in `summary.py`, not merely configured):
traceback ≤ 6000 chars, ≤ 60 tail lines of ≤ 400 chars each, ≤ 25 artifact
entries. A test asserts a summary stays under 40 KB even when fed a 500 MB log.

### Reading `success` correctly

`success` is `true` only when the entrypoint exited `0`. Note that by default
(`fail_kernel_on_error = false`) the *kernel* still finishes as `complete` even
when the job fails — that is deliberate, because Kaggle reliably preserves
output artifacts for completed sessions. So:

- **trust `success`**, not `kaggle.status`
- `status: "no-result"` means the kernel never wrote its result file (it was
  killed, cancelled, or timed out) — check `artifacts.kaggle_log`

---

## 5. Layout

```
kaggle_runner/
  cli.py                 # argparse front end, exit codes, human/JSON output
  config.py              # runner.toml -> validated dataclasses
  gitctl.py              # commit/branch/dirty/pushed inspection
  builder.py             # renders the kernel payload; verifies it before push
  bootstrap_template.py  # >>> runs INSIDE Kaggle <<<
  kaggle_cli.py          # subprocess wrapper around the official CLI
  runner.py              # orchestration: preflight -> push -> poll -> collect
  summary.py             # size-bounded machine-readable result
kaggle_jobs/
  smoke_test.py          # the default CPU-only job
runs/                    # gitignored: per-run logs, artifacts, result.json
runner.toml              # all configuration
test/test_kaggle_runner.py
```

### What happens on Kaggle

`bootstrap_template.py` is copied with a job spec injected as a Python literal,
then pushed as a **script** kernel. Inside the session it:

1. clones the pinned commit into `/kaggle/temp/sdf_repo` — deliberately *not*
   `/kaggle/working`, so the source tree is never uploaded back as output;
2. optionally `pip install -r <requirements>`;
3. runs the entrypoint with `SDF_COMMIT`, `SDF_RUN_ID` and `SDF_OUTPUT_DIR` set,
   streaming output to `/kaggle/working/sdf_run.log`;
4. writes `/kaggle/working/sdf_result.json` and exits `0` so the artifacts stay
   retrievable.

---

## 6. Configuration (`runner.toml`)

| Key | Purpose |
|---|---|
| `repo.url` / `repo.branch` | what the kernel clones |
| `repo.git_token_secret` | name of a Kaggle Secret holding a PAT (private repos) |
| `kernel.owner` / `slug` | the single kernel reused for every run |
| `kernel.enable_internet` | must stay `true` — the clone needs it |
| `kernel.enable_gpu` / `accelerator` | **off by default**; see §8 |
| `kernel.timeout_s` | hard cap on kernel run time |
| `job.entrypoint` | the command to run — usually the only line you change |
| `job.requirements` | optional requirements file; empty skips pip |
| `job.fail_kernel_on_error` | mark the Kaggle kernel failed too (default `false`) |
| `local.poll_*` | polling cadence, backoff and overall timeout |
| `local.log_tail_lines`, `traceback_max_chars` | summary size (capped in code) |

Config is validated on load: contradictory settings (GPU on with no accelerator,
internet off) are rejected with a specific message rather than failing on Kaggle
several minutes later.

---

## 7. Where a browser is still needed

Only for things the API genuinely cannot do:

| Task | Method |
|---|---|
| Create an API token | **Browser, by you.** Credential creation is never automated. |
| Phone-verify the account | Browser, by you — one time. |
| Everything in the loop | Official CLI. No browser. |

Visual inspection of a run is available at `kaggle.url` in the summary, but
nothing in the loop depends on it.

---

## 8. GPU policy

GPU is **off by default** and a test enforces it. The smoke test is CPU-only and
takes seconds, so iterating on the infrastructure costs no GPU quota.

When a real workload needs one, opt in explicitly and per-run:

```bash
python -m kaggle_runner run --push --gpu NvidiaTeslaT4
```

or set `enable_gpu = true` **and** `accelerator = "NvidiaTeslaT4"` in
`runner.toml`. Both are required together — setting one without the other is a
config error, which prevents both "asked for GPU, silently got CPU" and the
reverse.

> Avoid `NvidiaTeslaP100`: with the default Kaggle image, `torch.cuda.is_available()`
> returns `True` but the first CUDA op fails, because that PyTorch build ships no
> Pascal kernels. Prefer `NvidiaTeslaT4`.

Kaggle's GPU quota is roughly 30 h/week. `kaggle quota` reports the balance.

---

## 9. The iteration loop

The pieces an automated agent needs are all in place:

- a single command that runs the current revision (`run --push`)
- exit codes that separate "your code failed" from "the harness failed"
- a bounded JSON result at a stable path (`runs/latest.json`)
- full logs on disk, reachable via `logs --grep` when the summary is not enough

Sketch:

```bash
for attempt in $(seq 1 5); do
  git commit -am "attempt $attempt" || true
  if python -m kaggle_runner run --push --json --quiet > /tmp/result.json; then
    echo "PASS"; break
  fi
  case $? in
    2) echo "infrastructure problem - not a code failure"; break ;;
    *) python - <<'EOF'
import json; r = json.load(open('/tmp/result.json'))
print(r['error']['type'], r['error']['message'])
EOF
       # ... apply a fix, then loop ...
       ;;
  esac
done
```

Deliberately **not** built yet, because it is the next task and depends on
decisions not yet made: retry policy, when to give up, and whether fixes are
proposed or applied automatically.

---

## 10. Testing

```bash
python test/test_kaggle_runner.py      # no pytest needed
python -m pytest test/ -q              # or with pytest
```

20 offline tests, no network or Kaggle account required. They cover config
validation, payload rendering, status parsing, error extraction, and the summary
size guarantees.

To exercise the failure path end-to-end on Kaggle without introducing a real
bug:

```bash
python -m kaggle_runner run --push --entrypoint "SDF_SMOKE_FAIL=1 python kaggle_jobs/smoke_test.py"
```

It should exit `1` with `error.type == "RuntimeError"`.

---

## 11. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `preflight failed: git.pushed` | commit is local only — use `run --push` |
| `preflight failed: git.clean` | uncommitted edits; commit them, or `--allow-dirty` to knowingly run HEAD |
| `kaggle.auth` fails | run `kaggle auth login`, or place `kaggle.json` (§2) |
| clone fails inside the kernel | internet disabled, account not phone-verified, or the repo is private (set `repo.git_token_secret`) |
| `status: no-result` | kernel killed/timed out — read `artifacts.kaggle_log` |
| `status` stuck `unknown` | Kaggle changed the CLI's status wording; `kaggle_cli.parse_status` needs a new pattern |
| run exceeds the poll window | raise `local.poll_timeout_s` (kernels can queue for a long time) |
