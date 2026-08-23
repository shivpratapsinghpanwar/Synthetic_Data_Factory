# Synthetic Data Pipeline — Design

The repo's goal, made operational: **generate realistic synthetic images of
medical conditions and prove they improve condition/anomaly detectors.**
"Improve" is a measured number, not an impression — see §5.

Execution model: develop and unit-test locally, execute every real stage on
Kaggle through the runner (`docs/kaggle_runner.md`), which pins each run to a
pushed git commit and returns a machine-readable result.

---

## 1. First vertical slice (current defaults)

| Decision | Choice | Why | Change it in |
|---|---|---|---|
| Modality | Skin lesions — HAM10000 | 10,015 dermoscopy images, 7 classes, severe imbalance (dermatofibroma ~115 vs nevi ~6,705) — exactly the rare-condition scarcity synthetic data addresses; fits a T4; best-published augmentation benchmark | `pipeline.toml [dataset]` + one adapter |
| Generator | SD 1.5 + per-class LoRA | Pretrained prior reaches realism in ~1–3 GPU-h/class at 512px; best quality per hour of the ~30 h/week Kaggle GPU quota | `pipeline.toml [generator]` (backend is pluggable) |
| Detector | CNN classifier (e.g. EfficientNet-B0) over the 7 classes | Rare-class recall is the anomaly-detection story stated in supervised terms; simple, strong baseline | evaluation stage config (later) |

Alternatives kept open by design: chest X-ray / MRI / fundus adapters (one
file each, §3); StyleGAN2-ADA or a conditional DDPM as backends (no
natural-image prior → cleaner provenance, more GPU hours).

---

## 2. Stage graph

```
audit ──► train_lora ──► sample ──► quality_gate ──► augment ──► train_detector ──► evaluate
(CPU)     (GPU)          (GPU)      (GPU, cheap)     (CPU)       (GPU)              (CPU)
```

| Stage | Does | Passes when |
|---|---|---|
| `audit` *(implemented)* | Verify dataset presence/structure, class stats, leakage-safe splits, identify rare classes | metadata parses, <1% missing files, rare classes exist |
| `train_lora` | Fine-tune SD 1.5 LoRA on one class's *train-split* images; checkpoint continuously | loss curve sane, checkpoint saved |
| `sample` | Generate N images/class from checkpoints; write provenance manifest rows | N images + manifest complete |
| `quality_gate` | FID/KID vs real held-out per class; near-duplicate screen vs train set (embedding NN) — catches memorization; degenerate-sample screens | FID under threshold, zero near-duplicates above similarity cutoff |
| `augment` | Compose real-train + gated synthetic into a training index (never touches val/test) | index consistent, all synthetic rows manifest-backed |
| `train_detector` | Train the classifier twice: real-only and real+synthetic, same budget/seed set | both runs complete |
| `evaluate` | Per-class recall/F1, macro-F1, AUROC on the *real-only* test split; report the delta | report produced; success of the *project* = positive rare-class delta |

Every stage emits `stage_<name>.json` into `$SDF_OUTPUT_DIR` (the runner
collects it into `runs/<id>/output/`), with the same exit-code contract as the
runner: 0 pass, 1 fail, 2 config error. `audit` is the mandatory cheap gate in
front of any GPU stage.

## 3. Code layout

```
sdf/
  config.py        # pipeline.toml -> validated dataclasses
  splits.py        # grouped, stratified, deterministic splitting
  manifest.py      # provenance for every synthetic image (JSONL)
  data/
    base.py        # ImageRecord + DatasetAdapter interface + registry
    ham10000.py    # HAM10000 adapter (layout auto-discovery)
  stages/
    base.py        # StageResult contract + writer
    audit.py       # implemented; later stages land here one file each
  cli.py           # python -m sdf run-stage <name> / stages
pipeline.toml      # all pipeline configuration
```

A new modality = one adapter file returning `ImageRecord`s; everything
downstream is unchanged. A new generator = one backend module registered under
`[generator] backend`.

## 4. Data discipline

- **Split by physical lesion (`lesion_id`), never by image.** HAM10000
  re-images the same lesion up to 6×; image-level splits leak near-duplicates
  into test and inflate every metric. Enforced by an assertion, unit-tested.
- **Deterministic splits** (stable string-seeded RNG) so val/test never drift
  between runs or machines.
- **Synthetic data only ever enters the train split.** Val and test remain
  100% real, forever — the evaluation would be meaningless otherwise.
- **Test split is untouchable**: no generator, gate, or model-selection step
  may read it before `evaluate`.

## 5. Success metric

Train the detector on (a) real-train only, (b) real-train + gated synthetic;
identical budgets, ≥3 seeds each. Evaluate both on the real test split.

**The pipeline succeeds when rare-class recall/F1 improves with statistical
honesty** (mean ± spread over seeds; no cherry-picking), without degrading
macro metrics. FID is tracked as a diagnostic, not a goal.

## 6. Kaggle execution plan

- Dataset attaches read-only via `runner.toml [kernel] dataset_sources`
  (mounted at `/kaggle/input/skin-cancer-mnist-ham10000/`) — no re-download.
- GPU stages request `--gpu NvidiaTeslaT4` explicitly; audit/augment/evaluate
  stay CPU. GPU budget: LoRA ≈ 1–3 h/class (7 classes, rare ones first);
  sampling ≈ minutes/hundred images; detector ≈ 1–2 h/run.
- Sessions die at ~9–12 h: `train_lora` checkpoints to `/kaggle/working`
  continuously; resume pulls the previous run's checkpoint (uploaded as a
  private Kaggle dataset version — same pattern as the runner's `runs/` flow).
- Heavy deps (torch, diffusers, peft) come from the Kaggle image; the repo's
  own `pyproject.toml` carries only control-plane deps (kaggle, pillow).

## 7. Provenance & ethics

- Every synthetic image has a manifest row (`sdf/manifest.py`) recording
  backend, base model, checkpoint, seed, prompt, timestamp, and
  `synthetic: true`. Writing an unmarked record is a refused operation.
- Synthetic images exist to train detectors. They are never presented as real
  patient imagery, and the manifest makes that verifiable.
- HAM10000 is a public, de-identified research dataset (CC BY-NC 4.0) — usage
  stays research/non-commercial.
- Memorization is treated as a defect: the quality gate's near-duplicate
  screen exists to catch a generator reproducing a real patient's image.

## 8. Milestones

- **M0 — infrastructure** ✅ runner + result contract (`docs/kaggle_runner.md`)
- **M1 — audit green on Kaggle** ✅ dataset attached, stats + splits verified remotely (CPU)
- **M2 — first LoRA** ✅ df (86 train images), 1200 steps @512px on T4 (~20 min);
  100 samples in ~9 min; visually convincing dermatofibroma morphology
- **M3 — quality gate** ✅ ran in-session on Kaggle: 0 flat, 0 duplicates,
  0 memorization flags (max cosine similarity 0.852 vs threshold 0.985);
  FID 328 vs the 12-image df val set, correctly marked unstable
- **M4 — full generation** ✅ all four rare classes at 100 gated images each
  (400 total). Per-class gates: 0 memorization flags everywhere (max cosine
  similarity: df 0.852, vasc 0.865, akiec 0.862, bcc 0.907 vs threshold
  0.985); FID 221-328, all marked unstable (small val references) - tracked
  as diagnostic. Merged into one artifacts-dataset version (manifests
  concatenate on merge; a later-wins policy would have silently dropped a
  run's rows).
- **M5 — the number** ✅ 3 seeds x 2 arms, all four rare classes augmented
  (+400 gated images). Result: macro metrics unchanged; vasc F1
  +0.050 +/- 0.013 (consistent); bcc slightly degraded; strongest consistent
  effect off-target: melanoma recall +0.116 +/- 0.048 via the class-balance
  shift. Full numbers + interpretation: `docs/results.md`.
  Single-seed df-only pilot (run 20260823T105736Z): acc +1.4pt, macro-F1
  +0.4pt, df recall moved by exactly one test image (n=17) - noise-dominated,
  as predicted; treated strictly as pipeline validation. The claimable
  experiment (3 seeds x 2 arms, all rare classes augmented, paired
  mean +/- std deltas) runs via evaluate's `pairs` mode.
- **M6 — scale out** second modality via adapter; second backend if warranted
