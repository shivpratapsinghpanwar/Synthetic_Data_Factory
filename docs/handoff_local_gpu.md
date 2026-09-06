# Handoff: continuing on a local GPU machine

Target: a workstation with an NVIDIA GeForce RTX 2050 (4 GB VRAM, 8 GB RAM).
Decision (owner, 2026-09-06): training moves to local hardware; Kaggle keeps
serving as remote compute for non-face studies and as the private
weights-only checkpoint store. Generated imagery is no longer published to
any Kaggle dataset (the publisher enforces a weights-only allowlist in
`kaggle_runner/artifacts.py`).

## 1. What git carries

Clone and sync — everything public lives in the repo:

```
git clone https://github.com/shivpratapsinghpanwar/Synthetic_Data_Factory.git
cd Synthetic_Data_Factory
uv sync
uv run python -m pytest test/ -q     # 111 passed, 6 torch-gated skips expected
```

Install the ML stack locally (not in pyproject — Kaggle images provide it
there): `uv pip install torch --index-url https://download.pytorch.org/whl/cu121`
plus `diffusers peft transformers accelerate`.

## 2. What git does NOT carry — copy these by hand

These are gitignored on purpose (confidentiality) and must be moved from the
old machine over a private channel (USB / direct copy — never a public host):

| Item | Location | Why |
|---|---|---|
| Kaggle OAuth credentials | `~/.kaggle/credentials.json` | runner + dataset access |
| Runner overlay | `runner.local.toml` | private dataset slugs |
| Study overlays | `pipeline_cond_a.local.toml`, `pipeline_cond_b.local.toml` | real class names, data paths |
| Client data | `Data_to_reproduce_with/` | the studies' images |
| (optional) past runs | `runs/` | local history, evidence JSONs |

Without the overlays, configs load with codenames only and refuse private
runs — that is the intended failure mode.

## 3. Checkpoints

Both current checkpoints live in the private weights-only dataset
(`<owner>/sdf-runner-weights`, flattened at root):

```
kaggle datasets download -d shivpratap0007/sdf-runner-weights -p checkpoints --unzip
# -> checkpoints/mdx/joint/mdx_model/          (M2 joint trunk, 79M)
# -> checkpoints/sd15_lora/<cls>/adapter/      (M2b SD1.5 LoRA)
```

(Kaggle strips the staged `lora/` root folder on extraction — verified
live.) Point stages at them with `--opt adapter_dir=checkpoints/...`; on
Kaggle kernels the mount is `/kaggle/input/sdf-runner-weights/...`.

## 4. What fits in 4 GB VRAM

| Workload | Fits? | Settings |
|---|---|---|
| MDX train 128px (35M, dim 384) | yes | fp16, `--opt batch_size=4 grad_accum=4` |
| MDX train 128px (79M, dim 512) | tight | fp16, `--opt batch_size=1 grad_accum=16` |
| MDX train 256px | no — screen at 128, use Kaggle T4 for 256 | |
| MDX sample / CF-Edit | yes | cheap at any strength |
| SD1.5 sample 512px | tight but yes | attention+VAE slicing are enabled in code |
| SD1.5 LoRA **train** 512px | no (needs ~8 GB+) | keep this arm on Kaggle T4 |
| SDXL anything | no | Kaggle/cloud only |

8 GB system RAM note: don't use diffusers CPU-offload modes (they need more
host RAM than this machine has); fp16 + slicing is the right lever.

## 5. Division of labor going forward

- **Local GPU**: all face-bearing (cond_b) generation and training — face
  pixels never touch Kaggle again; MDX screening runs; CF-Edit sweeps;
  quality gates; sample inspection.
- **Kaggle T4**: cond_a heavy arms (SD1.5/SDXL LoRA training), 256px MDX
  replication runs, detector-arm training at scale, public HAM10000 lane.
- **Weights dataset**: the only artifact bridge — `python -m kaggle_runner
  publish <run_id>` refuses anything that isn't a weight/metadata file.

## 6. First tasks on this machine

1. Verify CUDA torch sees the GPU: `python -c "import torch; print(torch.cuda.get_device_name(0))"`.
2. Re-sample the M2b adapter at its native 512px (the v21 garbage output was
   a config-inheritance bug, fixed in `sd15_lora._sample_defaults`):
   `python -m sdf run-stage sample --config pipeline_cond_a.toml --opt cls=<cls> --opt backend=sd15_lora --opt adapter_dir=checkpoints/sd15_lora/<cls>/adapter --opt count=32`
3. Implement design resolutions R1 (two-regime select) and R4 (CF-Edit
   memorization audit source-exclusion) — see `design/SDF_Design_Document.docx`
   §7 — then run the M3 CF-Edit strength sweep locally at 128px.
