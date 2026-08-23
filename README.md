# Synthetic_Data_Factory
There is one thing that is still unsolved in deep learning is the creating data real enough to be used for training that can be used in industry products be it medical, security or any anomaly in the scens, this repo will try to fill this gap.

## Kaggle execution runner

This repo can execute its own current git revision on Kaggle and report back a
compact, machine-readable result — the harness for an autonomous
edit → commit → push → run → diagnose → fix loop.

```bash
pip install -r requirements-runner.txt
kaggle auth login                    # one-time, done by you
python -m kaggle_runner doctor       # verify setup
python -m kaggle_runner run --push   # run HEAD on Kaggle (CPU, no GPU quota)
```

Full documentation: [docs/kaggle_runner.md](docs/kaggle_runner.md)

