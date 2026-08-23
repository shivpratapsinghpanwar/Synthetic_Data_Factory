# Synthetic_Data_Factory
There is one thing that is still unsolved in deep learning is the creating data real enough to be used for training that can be used in industry products be it medical, security or any anomaly in the scens, this repo will try to fill this gap.

## Kaggle execution runner

This repo can execute its own current git revision on Kaggle and report back a
compact, machine-readable result — the harness for an autonomous
edit → commit → push → run → diagnose → fix loop.

```bash
uv sync                                 # reproducible environment (uv.lock)
uv run kaggle auth login                # one-time, done by you
uv run python -m kaggle_runner doctor   # verify setup
uv run python -m kaggle_runner run --push  # run HEAD on Kaggle (CPU, no GPU quota)
```

Full documentation: [docs/kaggle_runner.md](docs/kaggle_runner.md)

## Synthetic data pipeline

The pipeline that makes the motto true: generate synthetic medical-condition
images (first target: HAM10000 skin lesions, SD 1.5 + LoRA) and measure the
lift they give a condition detector. Stages run on Kaggle via the runner:

```bash
uv run python -m sdf stages                       # list pipeline stages
uv run python -m kaggle_runner run --entrypoint "python -m sdf run-stage audit"
```

Design: [docs/pipeline_design.md](docs/pipeline_design.md)

