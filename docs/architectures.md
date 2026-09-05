# Generative architectures for scarce-data medical imaging

Design space for this factory's research program. Constraint set: private
multi-study clinical datasets of 300-3,000 images, curated evaluation splits,
a single 16GB GPU per session, and one hard metric - paired-seed detector
lift on real test data. Studies are referred to by codename (cond_a,
cond_b, ...).

## A. MDX - mixture-of-experts diffusion transformer (built)

One pixel-space DiT trunk trained jointly across studies; adaLN-zero
conditioning on (timestep, condition/class); per-condition low-rank FFN
experts hard-routed by study id; classifier-free guidance via label dropout;
EMA sampling weights.

Claims this design supports:
1. **Cross-study transfer under scarcity.** The trunk sees every study's
   images; a 500-image study borrows general image statistics from a
   3,000-image one. Test: joint trunk vs equal-compute per-class DDPM
   (the existing baseline arm) on sample quality and detector lift.
2. **Continual onboarding without forgetting - by construction.** New
   studies extend the vocab and expert tables; with the trunk frozen, only
   new rows receive gradients, so previously learned studies are *bitwise*
   unchanged (a hard guarantee most continual-learning work can only
   approximate). Onboarding cost: minutes.
3. Hard routing (not learned soft MoE routing): tiny data cannot estimate a
   router, and a wrong router silently corrupts every study.

Positioning: eDiff-I ensembles denoisers over timestep bands; DiT-MoE scales
token-level soft-routed experts for capacity. Neither targets the
multi-*dataset* scarce-data regime, and neither gives the frozen-trunk
onboarding guarantee. Closest spiritual neighbor is adapter-per-task
continual diffusion; the delta is the shared-trunk joint pretraining across
private clinical studies plus expert routing at the study level.

## B. CF-Edit - counterfactual anomaly synthesis by label-swapped partial denoising (this iteration)

The highest-leverage observation in this problem: clinical datasets come
with paired class structure - the same anatomy imaged the same way, with
and without the condition. Generating an anomalous image from pure noise
wastes model capacity reproducing what real data already provides (pose,
lighting, device signature, background). Instead:

1. take a REAL image of the normal class (train split only - never val/test);
2. diffuse it forward to an intermediate timestep t = strength * T
   (SDEdit-style partial noising: global structure survives, appearance
   detail is destroyed);
3. denoise with the TARGET label ("condition/anomalous-class") under CFG,
   routed through the study's expert.

The output is a counterfactual: the same scene, now exhibiting the
condition. `strength` is an interpretable dial - low keeps more of the real
image (realism up, anomaly subtlety up), high approaches pure generation.

Why this should beat pure generation for detector training:
- **Sample efficiency.** The generator only has to know the *difference*
  between classes, which is exactly what the joint conditional model
  concentrates in its class embeddings and experts.
- **Hard pairs by construction.** Each real normal image yields a
  minimal-difference (normal, anomalous) pair straddling the decision
  boundary - the examples detectors learn the most from. The pairing
  (source image recorded in the manifest) also enables paired training
  objectives later.
- **Label correctness is auditable.** The committee select stage scores
  every counterfactual with the real-only detector; edits too weak to
  express the condition are filtered before training (and the flagged rate
  itself measures the generator's class separation).

Positioning: SDEdit edits with a *prompt*, not a learned per-study class
swap; counterfactual-explanation literature produces per-classifier
gradients, not training data; AnoDDPM-style medical work runs the opposite
direction (remove the anomaly to detect it). Using label-swapped partial
denoising of real normals as a *data factory* for rare-condition detectors,
on sub-1k private datasets, with a memorization audit, is an open slot.

Cost: zero additional training - it reuses the MDX checkpoint. Sampling
cost is lower than pure generation (only (1-strength) of the trajectory).

## C. Generator committee with detector-scored selection (built)

Heterogeneous arms spanning the prior spectrum: prior-free MDX (no
pretraining leakage, data-hungry) vs pretrained LoRA arms at ~860M/~2.6B
(sample-efficient, import a natural-image prior). Arms sample into one
pool; selection drops gate-flagged/degenerate images, scores the rest with
the real-only detector, and blends arms round-robin best-first.

The empirical question this answers - *where along the prior spectrum does
synthetic medical data come from at each dataset size?* - is unresolved and
practically important (privacy and licensing arguments favor prior-free;
quality arguments favor foundation priors). The committee makes it an
ablation table instead of a bet, and CF-Edit slots in as just another arm.

## D. Condition x timestep factorized experts (planned ablation)

eDiff-I showed denoising is stage-like (structure early, texture late).
Factor each expert into a few timestep-band slices (same rank budget,
band-selected at run time). Pure ablation inside MDX; worth one table, not
a paper.

## E. Boundary-seeking guidance (future)

Steer sampling with the downstream detector's uncertainty (classifier
guidance toward low-margin regions), generating exactly the examples the
current detector cannot classify, labelled by construction via CFG.
Active-learning-flavored generation; depends on B+C being solid first.

## Evaluation protocol (all architectures)

- Paired-seed detector deltas (existing evaluate stage): real-only vs
  augmented, same seeds, mean +/- sd of macro-F1 on the curated real test
  split. cond_a baseline to beat: 0.9216 macro-F1.
- Memorization audit is mandatory and reported (max cosine to train
  embeddings; flagged >= 0.985 quarantined). For CF-Edit the audit runs
  against the *source* class too - an edit that fails to move class is a
  near-copy by design and must be caught.
- Balance-matched real-oversampling control arm, so lift cannot be
  explained by class-balance shift alone.

## Publication packaging

Primary claim: **counterfactual pairs from label-swapped diffusion editing
beat pure generation for rare-condition detector training under extreme
data scarcity** (B, evaluated via the protocol, with A as the enabling
model and C as the composition/filtering layer). Secondary: the frozen-trunk
onboarding guarantee (A.2) as a systems contribution. Realistic venues:
MIDL / MICCAI workshop track first, journal extension with more studies as
client data arrives.
