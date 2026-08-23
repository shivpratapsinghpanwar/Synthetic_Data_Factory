"""From-scratch per-class DDPM backend: no pretrained prior whatsoever.

Why it exists alongside sd15_lora:
- **Sensitive domains.** A model that has only ever seen the licensed medical
  training images cannot leak a natural-image prior (faces, scenes) into its
  samples. For pediatric/identifiable anatomy this is the defensible option.
- **Science.** Comparing prior-free DDPM samples against SD-prior LoRA samples
  on the same class separates 'what the pretrained prior contributes' from
  'what the medical data itself supports'.

Cost: needs more steps than LoRA to reach usable quality (no prior to lean
on). Defaults target a T4: 128px, ~55M-param UNet, batch 16, fp16 autocast.

Interface matches sd15_lora: train(cfg, records, cls, out_dir, opts) and
sample(cfg, cls, adapter_dir, out_dir, opts). Heavy imports stay inside
functions - the control machine has no torch.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

ADAPTER_DIR_NAME = "ddpm_unet"
REPORT_NAME = "training_report.json"


def _build_unet(resolution: int):
    from diffusers import UNet2DModel

    return UNet2DModel(
        sample_size=resolution,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 384, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )


# ------------------------------------------------------------------ training
def train(cfg, records, cls: str, out_dir: Path, opts: dict) -> dict:
    import torch
    import torch.nn.functional as F
    from diffusers import DDPMScheduler
    from torch.utils.data import DataLoader

    from .sd15_lora import _ImageDataset  # same pixel pipeline, reused

    steps = int(opts.get("steps", 4000))
    resolution = int(opts.get("resolution", 128))
    batch_size = int(opts.get("batch_size", 16))
    lr = float(opts.get("lr", 1e-4))
    seed = int(opts.get("seed", cfg.splits.seed))
    checkpoint_every = int(opts.get("checkpoint_every", 500))

    if resolution not in (64, 128, 256):
        raise ValueError(f"ddpm resolution must be 64/128/256, got {resolution}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    unet = _build_unet(resolution).to(device)
    params = sum(p.numel() for p in unet.parameters())
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    dataset = _ImageDataset([r.path for r in records if r.exists], resolution)
    if len(dataset) == 0:
        raise RuntimeError(f"no training images on disk for class {cls!r}")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False
    )

    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = out_dir / ADAPTER_DIR_NAME

    losses: list[float] = []
    step = 0
    t0 = time.time()
    unet.train()
    while step < steps:
        for pixels in loader:
            if step >= steps:
                break
            pixels = pixels.to(device)
            noise = torch.randn_like(pixels)
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (pixels.shape[0],), device=device
            )
            noisy = scheduler.add_noise(pixels, noise, timesteps)

            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                pred = unet(noisy, timesteps).sample
                loss = F.mse_loss(pred.float(), noise.float())

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_sched.step()

            losses.append(float(loss.detach()))
            step += 1
            if step % 100 == 0 or step == steps:
                recent = sum(losses[-100:]) / len(losses[-100:])
                print(f"[ddpm] {cls} step {step}/{steps} loss={recent:.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            if checkpoint_every and step % checkpoint_every == 0 and step < steps:
                unet.save_pretrained(str(adapter_dir))
                print(f"[ddpm] {cls} checkpoint at step {step}", flush=True)

    unet.save_pretrained(str(adapter_dir))
    scheduler.save_pretrained(str(adapter_dir / "scheduler"))

    report = {
        "backend": "ddpm",
        "cls": cls,
        "base_model": "from-scratch",
        "prompt": f"unconditional ddpm ({cls})",
        "steps": steps,
        "resolution": resolution,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "unet_params": params,
        "train_images": len(dataset),
        "device": device,
        "loss_first100": round(sum(losses[:100]) / max(1, len(losses[:100])), 4),
        "loss_last100": round(sum(losses[-100:]) / max(1, len(losses[-100:])), 4),
        "duration_s": round(time.time() - t0, 1),
        "adapter_dir": str(adapter_dir),
    }
    (out_dir / REPORT_NAME).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# ------------------------------------------------------------------ sampling
def sample(cfg, cls: str, adapter_dir: Path, out_dir: Path, opts: dict) -> list[dict]:
    import torch
    from diffusers import DDIMScheduler, UNet2DModel
    from PIL import Image

    count = int(opts.get("count", cfg.generator.sample_count))
    steps = int(opts.get("sample_steps", 60))
    base_seed = int(opts.get("seed", cfg.splits.seed))
    batch = int(opts.get("sample_batch", 16))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    unet = UNet2DModel.from_pretrained(str(adapter_dir)).to(device).eval()
    resolution = unet.config.sample_size

    scheduler_dir = adapter_dir / "scheduler"
    if scheduler_dir.is_dir():
        scheduler = DDIMScheduler.from_pretrained(str(scheduler_dir))
    else:
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

        with torch.no_grad():
            for t in scheduler.timesteps:
                with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                    eps = unet(x, t).sample
                x = scheduler.step(eps.float(), t, x).prev_sample

        images = ((x.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for i in range(n):
            seed = seeds[i]
            image_id = f"syn_{cls}_ddpm_{seed}"
            file_name = f"{image_id}.png"
            Image.fromarray(images[i]).save(out_dir / file_name)
            rows.append(
                {
                    "image_id": image_id,
                    "file": file_name,
                    "cls": cls,
                    "backend": "ddpm",
                    "base_model": "from-scratch",
                    "checkpoint": str(adapter_dir),
                    "seed": seed,
                    "prompt": f"unconditional ddpm ({cls})",
                    "sample_steps": steps,
                    "resolution": resolution,
                }
            )
        produced += n
        print(f"[ddpm-sample] {cls} {produced}/{count} ({time.time() - t0:.0f}s)", flush=True)
    return rows
