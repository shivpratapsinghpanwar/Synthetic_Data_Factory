"""mdx: mixture-of-experts diffusion transformer (from scratch, prior-free).

One shared pixel-space DiT-style trunk serves every private study at once,
instead of one independent UNet per class:

- **Joint conditioning.** adaLN-zero modulation on (timestep, label), where a
  label is "<condition>/<class>". The trunk learns shared image statistics
  across all studies; small studies borrow capacity from large ones.
- **Hard-routed experts.** Every transformer block carries a low-rank FFN
  expert PER CONDITION (study), selected by the sample's condition id - no
  learned routing, which tiny datasets cannot support. Experts are zero-init
  so they start as an exact no-op on the shared path.
- **Cheap onboarding.** A new study grows the label vocab and expert tables;
  with ``freeze_trunk=1`` only those new rows train, so learned studies are
  untouched and onboarding costs minutes, not a full retrain.
- **Classifier-free guidance** via label dropout to a null token; the expert
  route is kept during dropout so guidance isolates the class signal.

Prior-free by construction: plain torch modules; diffusers supplies only the
noise schedulers (no pretrained weights anywhere). Defaults target a T4:
128px, ~35M-param trunk, batch 16, fp16 autocast, EMA weights for sampling.

Interface matches ddpm: train(cfg, records, cls, out_dir, opts) and
sample(cfg, cls, adapter_dir, out_dir, opts); train_joint() is the
multi-study entry used by the train_joint stage. Heavy imports stay inside
functions - the control machine has no torch.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

ADAPTER_DIR_NAME = "mdx_model"
REPORT_NAME = "training_report.json"
CONFIG_NAME = "mdx_config.json"
WEIGHTS_EMA = "weights.pt"          # EMA weights - what sample() loads
WEIGHTS_TRAIN = "weights_train.pt"  # raw weights - what resume continues from

NULL_LABEL = "<null>"  # classifier-free-guidance drop target (always last row)


# ------------------------------------------------------------------- model
def build_model(arch: dict):
    """Construct the MDX network from an architecture dict.

    arch keys: resolution, patch, dim, depth, heads, mlp_ratio, expert_rank,
    vocab (list of "<condition>/<class>" labels), conditions (list of
    condition tags). The null CFG label is an extra embedding row appended
    after the vocab.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    resolution = int(arch["resolution"])
    patch = int(arch["patch"])
    dim = int(arch["dim"])
    depth = int(arch["depth"])
    heads = int(arch["heads"])
    mlp_ratio = int(arch.get("mlp_ratio", 4))
    rank = int(arch["expert_rank"])
    vocab = list(arch["vocab"])
    conditions = list(arch["conditions"])
    if resolution % patch != 0:
        raise ValueError(f"resolution {resolution} not divisible by patch {patch}")
    tokens = (resolution // patch) ** 2

    def timestep_embedding(t, out_dim):
        half = out_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
            self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim)
            )
            # Per-condition low-rank FFN experts, hard-routed by condition id.
            # Up-projection zero-init: every expert starts as an exact no-op.
            self.expert_down = nn.Parameter(
                torch.randn(len(conditions), dim, rank) * 0.02
            )
            self.expert_up = nn.Parameter(torch.zeros(len(conditions), rank, dim))
            # adaLN-zero: modulation starts at identity, gates at zero.
            self.ada = nn.Linear(dim, 6 * dim)
            nn.init.zeros_(self.ada.weight)
            nn.init.zeros_(self.ada.bias)

        def forward(self, x, cond, cond_idx):
            sh1, sc1, g1, sh2, sc2, g2 = self.ada(F.silu(cond))[:, None, :].chunk(6, dim=-1)
            h = self.norm1(x) * (1 + sc1) + sh1
            x = x + g1 * self.attn(h, h, h, need_weights=False)[0]
            h = self.norm2(x) * (1 + sc2) + sh2
            down = self.expert_down.index_select(0, cond_idx)  # (B, dim, rank)
            up = self.expert_up.index_select(0, cond_idx)      # (B, rank, dim)
            expert = torch.bmm(F.gelu(torch.bmm(h, down)), up)
            return x + g2 * (self.mlp(h) + expert)

    class MDX(nn.Module):
        def __init__(self):
            super().__init__()
            self.arch = dict(arch)
            self.patchify = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
            self.pos = nn.Parameter(torch.randn(1, tokens, dim) * 0.02)
            # +1 row: the null label for classifier-free guidance.
            self.label_emb = nn.Embedding(len(vocab) + 1, dim)
            nn.init.normal_(self.label_emb.weight, std=0.02)
            self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
            self.blocks = nn.ModuleList(Block() for _ in range(depth))
            self.final_norm = nn.LayerNorm(dim, elementwise_affine=False)
            self.final_ada = nn.Linear(dim, 2 * dim)
            self.final_out = nn.Linear(dim, patch * patch * 3)
            nn.init.zeros_(self.final_ada.weight)
            nn.init.zeros_(self.final_ada.bias)
            nn.init.zeros_(self.final_out.weight)
            nn.init.zeros_(self.final_out.bias)

        def forward(self, x, t, label_idx, cond_idx):
            b = x.shape[0]
            x = self.patchify(x).flatten(2).transpose(1, 2) + self.pos
            cond = self.t_mlp(timestep_embedding(t, dim)) + self.label_emb(label_idx)
            for block in self.blocks:
                x = block(x, cond, cond_idx)
            sh, sc = self.final_ada(F.silu(cond))[:, None, :].chunk(2, dim=-1)
            x = self.final_out(self.final_norm(x) * (1 + sc) + sh)
            side = resolution // patch
            x = x.reshape(b, side, side, patch, patch, 3)
            x = x.permute(0, 5, 1, 3, 2, 4).reshape(b, 3, resolution, resolution)
            return x

    return MDX()


def _default_patch(resolution: int) -> int:
    # 64->4, 128->8, 256->16: constant 256-token sequence at every resolution.
    return max(2, resolution // 16)


# ------------------------------------------------------------------- data
def _labelled_dataset(items, resolution: int):
    """(pixels, label_idx, cond_idx) dataset. Items: (path, roi|None, li, ci).

    With an ROI the crop is a square around the box (30% margin, clamped) -
    the region-only policy path. Without, centre-crop like every backend.
    """
    import torch
    from PIL import Image
    from torch.utils.data import Dataset

    class _DS(Dataset):
        def __len__(self):
            return len(items)

        def __getitem__(self, idx):
            path, roi, label_idx, cond_idx = items[idx]
            with Image.open(path) as img:
                img = img.convert("RGB")
                if roi is not None:
                    left, top, right, bottom = roi
                    cx, cy = (left + right) / 2, (top + bottom) / 2
                    side = max(right - left, bottom - top) * 1.3
                    half = side / 2
                    l = max(0, int(cx - half))
                    t = max(0, int(cy - half))
                    r = min(img.width, int(cx + half))
                    b = min(img.height, int(cy + half))
                    img = img.crop((l, t, r, b))
                side = min(img.size)
                left = (img.width - side) // 2
                top = (img.height - side) // 2
                img = img.crop((left, top, left + side, top + side))
                img = img.resize((resolution, resolution), Image.LANCZOS)
                data = torch.frombuffer(
                    bytearray(img.tobytes()), dtype=torch.uint8
                ).reshape(resolution, resolution, 3)
            pixels = (data.permute(2, 0, 1).float() / 127.5) - 1.0
            return pixels, label_idx, cond_idx

    return _DS()


# --------------------------------------------------------------- checkpoint
def save_checkpoint(model, adapter_dir: Path, ema_state=None, extra: dict | None = None):
    import torch

    adapter_dir.mkdir(parents=True, exist_ok=True)
    cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(cpu, adapter_dir / WEIGHTS_TRAIN)
    torch.save(
        {k: v.detach().cpu() for k, v in (ema_state or cpu).items()},
        adapter_dir / WEIGHTS_EMA,
    )
    meta = dict(model.arch)
    meta.update(extra or {})
    (adapter_dir / CONFIG_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_checkpoint(adapter_dir: Path, *, ema: bool = True):
    """Rebuild the model exactly as saved. Returns (model, meta)."""
    import torch

    meta = json.loads((adapter_dir / CONFIG_NAME).read_text(encoding="utf-8"))
    model = build_model(meta)
    name = WEIGHTS_EMA if ema else WEIGHTS_TRAIN
    state = torch.load(adapter_dir / name, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model, meta


def load_expanding(adapter_dir: Path, arch: dict):
    """Resume from a checkpoint whose vocab/conditions may be SMALLER.

    New labels/conditions get fresh rows; overlapping ones keep their trained
    weights (row-mapped by name, so ordering changes are safe). Trunk shapes
    must match. Returns (model, prev_steps).
    """
    import torch

    meta = json.loads((adapter_dir / CONFIG_NAME).read_text(encoding="utf-8"))
    for key in ("resolution", "patch", "dim", "depth", "heads", "expert_rank"):
        if int(meta[key]) != int(arch[key]):
            raise ValueError(
                f"checkpoint {key}={meta[key]} != requested {arch[key]} - "
                "trunk architecture must match to resume"
            )
    missing = [v for v in meta["vocab"] if v not in arch["vocab"]]
    if missing:
        raise ValueError(f"resume would drop trained labels: {missing}")

    model = build_model(arch)
    state = dict(model.state_dict())
    old = torch.load(adapter_dir / WEIGHTS_TRAIN, map_location="cpu", weights_only=True)

    label_row = {lbl: i for i, lbl in enumerate(arch["vocab"])}
    cond_row = {c: i for i, c in enumerate(arch["conditions"])}
    for key, value in old.items():
        if key == "label_emb.weight":
            new = state[key].clone()
            for i, lbl in enumerate(meta["vocab"]):
                new[label_row[lbl]] = value[i]
            new[len(arch["vocab"])] = value[len(meta["vocab"])]  # null row
            state[key] = new
        elif ".expert_down" in key or ".expert_up" in key:
            new = state[key].clone()
            for i, cond in enumerate(meta["conditions"]):
                if cond in cond_row:
                    new[cond_row[cond]] = value[i]
            state[key] = new
        else:
            state[key] = value
    model.load_state_dict(state)
    return model, int(meta.get("steps_trained", 0))


# ------------------------------------------------------------------ training
def train_joint(labelled, out_dir: Path, opts: dict, seed: int) -> dict:
    """Train the shared model across studies.

    ``labelled``: list of (path, roi|None, label_str, condition_str) where
    label_str is "<condition>/<class>". Options: steps(4000),
    resolution(128), patch(auto), dim(384), depth(12), heads(6),
    expert_rank(64), batch_size(16), lr(1e-4), drop_prob(0.1), balance(1),
    checkpoint_every(500), resume_from(path), freeze_trunk(0).
    """
    import torch
    import torch.nn.functional as F
    from diffusers import DDPMScheduler
    from torch.utils.data import DataLoader, WeightedRandomSampler

    steps = int(opts.get("steps", 4000))
    resolution = int(opts.get("resolution", 128))
    if resolution not in (64, 128, 256):
        raise ValueError(f"mdx resolution must be 64/128/256, got {resolution}")
    patch = int(opts.get("patch", _default_patch(resolution)))
    batch_size = int(opts.get("batch_size", 16))
    lr = float(opts.get("lr", 1e-4))
    drop_prob = float(opts.get("drop_prob", 0.1))
    checkpoint_every = int(opts.get("checkpoint_every", 500))
    freeze_trunk = bool(int(opts.get("freeze_trunk", 0)))
    resume_from = str(opts.get("resume_from", ""))
    ema_decay = float(opts.get("ema_decay", 0.999))
    seed = int(opts.get("seed", seed))

    vocab = sorted({label for _p, _r, label, _c in labelled})
    conditions = sorted({cond for _p, _r, _l, cond in labelled})
    if not vocab:
        raise RuntimeError("no labelled training images")
    arch = {
        "resolution": resolution,
        "patch": patch,
        "dim": int(opts.get("dim", 384)),
        "depth": int(opts.get("depth", 12)),
        "heads": int(opts.get("heads", 6)),
        "mlp_ratio": int(opts.get("mlp_ratio", 4)),
        "expert_rank": int(opts.get("expert_rank", 64)),
        "vocab": vocab,
        "conditions": conditions,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    prev_steps = 0
    if resume_from:
        model, prev_steps = load_expanding(Path(resume_from), arch)
    else:
        model = build_model(arch)
    model = model.to(device)
    if freeze_trunk:
        if not resume_from:
            raise ValueError("freeze_trunk=1 requires resume_from=<checkpoint>")
        for name, p in model.named_parameters():
            p.requires_grad = (
                "expert_down" in name or "expert_up" in name or name == "label_emb.weight"
            )
    trainable = [p for p in model.parameters() if p.requires_grad]
    params_total = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in trainable)

    label_row = {lbl: i for i, lbl in enumerate(vocab)}
    cond_row = {c: i for i, c in enumerate(conditions)}
    null_idx = len(vocab)
    items = [
        (path, roi, label_row[label], cond_row[cond])
        for path, roi, label, cond in labelled
    ]
    dataset = _labelled_dataset(items, resolution)

    label_counts: dict[int, int] = {}
    for _p, _r, li, _ci in items:
        label_counts[li] = label_counts.get(li, 0) + 1
    if bool(int(opts.get("balance", 1))) and len(label_counts) > 1:
        weights = [1.0 / label_counts[li] for _p, _r, li, _ci in items]
        sampler = WeightedRandomSampler(
            weights, num_samples=len(items), replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        loader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler,
            num_workers=2 if device == "cuda" else 0, drop_last=False,
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=2 if device == "cuda" else 0, drop_last=False,
        )

    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = out_dir / ADAPTER_DIR_NAME
    drop_gen = torch.Generator(device="cpu").manual_seed(seed + 1)

    per_label = {vocab[i]: n for i, n in sorted(label_counts.items())}
    print(f"[mdx] vocab={vocab} conditions={conditions} images={per_label} "
          f"params={params_total/1e6:.1f}M trainable={params_trainable/1e6:.1f}M",
          flush=True)

    losses: list[float] = []
    step = 0
    t0 = time.time()
    model.train()
    while step < steps:
        for pixels, label_idx, cond_idx in loader:
            if step >= steps:
                break
            pixels = pixels.to(device)
            label_idx = label_idx.to(device)
            cond_idx = cond_idx.to(device)
            # CFG dropout: null the label, keep the expert route.
            drop = torch.rand(label_idx.shape[0], generator=drop_gen) < drop_prob
            label_idx = torch.where(drop.to(device), torch.full_like(label_idx, null_idx), label_idx)

            noise = torch.randn_like(pixels)
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (pixels.shape[0],), device=device
            )
            noisy = scheduler.add_noise(pixels, noise, timesteps)

            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                pred = model(noisy, timesteps, label_idx, cond_idx)
                loss = F.mse_loss(pred.float(), noise.float())

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_sched.step()
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(ema_decay).add_(v.float(), alpha=1 - ema_decay)
                    else:
                        ema[k] = v.detach().clone().float()

            losses.append(float(loss.detach()))
            step += 1
            if step % 100 == 0 or step == steps:
                recent = sum(losses[-100:]) / len(losses[-100:])
                print(f"[mdx] step {step}/{steps} loss={recent:.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            if checkpoint_every and step % checkpoint_every == 0 and step < steps:
                save_checkpoint(model, adapter_dir, ema_state=ema,
                                extra={"steps_trained": prev_steps + step})
                print(f"[mdx] checkpoint at step {step}", flush=True)

    save_checkpoint(model, adapter_dir, ema_state=ema,
                    extra={"steps_trained": prev_steps + steps})

    report = {
        "backend": "mdx",
        "base_model": "from-scratch",
        "vocab": vocab,
        "conditions": conditions,
        "train_images": per_label,
        "steps": steps,
        "steps_total": prev_steps + steps,
        "resolution": resolution,
        "patch": patch,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "params": params_total,
        "params_trainable": params_trainable,
        "freeze_trunk": freeze_trunk,
        "resumed_from": resume_from,
        "device": device,
        "loss_first100": round(sum(losses[:100]) / max(1, len(losses[:100])), 4),
        "loss_last100": round(sum(losses[-100:]) / max(1, len(losses[-100:])), 4),
        "duration_s": round(time.time() - t0, 1),
        "adapter_dir": str(adapter_dir),
    }
    (out_dir / REPORT_NAME).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _condition_tag(cfg) -> str:
    return getattr(cfg.dataset, "condition", "") or cfg.dataset.name


def _labelled_from_records(cfg, records) -> list:
    """Backend-side labelling: (path, roi|None, "<condition>/<cls>", condition).

    When the config sets roi_only, images WITHOUT a region box are excluded
    and every used image is cropped to its box - the region-only policy.
    """
    from ..data.folder_class import roi_for

    condition = _condition_tag(cfg)
    roi_only = bool(getattr(cfg.generator, "roi_only", False))
    labelled = []
    for r in records:
        if not r.exists:
            continue
        roi = roi_for(r) if roi_only else None
        if roi_only and roi is None:
            continue
        labelled.append((r.path, roi, f"{condition}/{r.cls}", condition))
    return labelled


def train(cfg, records, cls: str, out_dir: Path, opts: dict) -> dict:
    """Single-study contract entry (train_lora stage). Joint runs use the
    train_joint stage instead."""
    labelled = _labelled_from_records(cfg, records)
    if not labelled:
        raise RuntimeError(f"no usable training images for class {cls!r}")
    report = train_joint(labelled, out_dir, opts, seed=cfg.splits.seed)
    report["cls"] = cls
    return report


# ------------------------------------------------------------------ sampling
def sample(cfg, cls: str, adapter_dir: Path, out_dir: Path, opts: dict) -> list[dict]:
    import torch
    from diffusers import DDIMScheduler
    from PIL import Image

    count = int(opts.get("count", cfg.generator.sample_count))
    steps = int(opts.get("sample_steps", 60))
    base_seed = int(opts.get("seed", cfg.splits.seed))
    batch = int(opts.get("sample_batch", 16))
    guidance = float(opts.get("guidance", 2.0))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_checkpoint(adapter_dir, ema=True)
    model = model.to(device).eval()
    resolution = int(meta["resolution"])
    vocab = list(meta["vocab"])
    conditions = list(meta["conditions"])

    condition = _condition_tag(cfg)
    label = f"{condition}/{cls}"
    if label not in vocab:
        raise ValueError(f"label {label!r} not in checkpoint vocab {vocab}")
    label_idx = vocab.index(label)
    cond_idx = conditions.index(condition)
    null_idx = len(vocab)

    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    scheduler.set_timesteps(steps, device=device)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t0 = time.time()
    produced = 0
    while produced < count:
        n = min(batch, count - produced)
        seeds = [base_seed + produced + i for i in range(n)]
        generator = torch.Generator(device=device).manual_seed(seeds[0])
        x = torch.randn(n, 3, resolution, resolution, device=device, generator=generator)
        li = torch.full((n,), label_idx, device=device, dtype=torch.long)
        ni = torch.full((n,), null_idx, device=device, dtype=torch.long)
        ci = torch.full((n,), cond_idx, device=device, dtype=torch.long)

        with torch.no_grad():
            for t in scheduler.timesteps:
                tt = t.expand(n) if t.dim() == 0 else t
                with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                    eps_c = model(x, tt, li, ci)
                    if guidance > 1.0:
                        eps_u = model(x, tt, ni, ci)
                        eps = eps_u + guidance * (eps_c - eps_u)
                    else:
                        eps = eps_c
                x = scheduler.step(eps.float(), t, x).prev_sample

        images = ((x.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for i in range(n):
            seed = seeds[i]
            image_id = f"syn_{cls}_mdx_{seed}"
            file_name = f"{image_id}.png"
            Image.fromarray(images[i]).save(out_dir / file_name)
            rows.append(
                {
                    "image_id": image_id,
                    "file": file_name,
                    "cls": cls,
                    "backend": "mdx",
                    "base_model": "from-scratch",
                    "checkpoint": str(adapter_dir),
                    "seed": seed,
                    "prompt": f"mdx conditional ({label}, cfg={guidance})",
                    "sample_steps": steps,
                    "resolution": resolution,
                }
            )
        produced += n
        print(f"[mdx-sample] {label} {produced}/{count} ({time.time() - t0:.0f}s)", flush=True)
    return rows
