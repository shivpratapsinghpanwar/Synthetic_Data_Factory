"""SDXL + LoRA backend: the capacity-upgrade axis (backbone-scale ablation).

T4 16GB fittings (all mandatory - diffusers PR #4470 established this recipe):
- UNet-only LoRA, text encoders frozen and NEVER trained (does not fit)
- fp16 autocast + gradient checkpointing
- fp16-safe VAE (madebyollin/sdxl-vae-fp16-fix); the stock SDXL VAE NaNs in fp16
- batch 1 + gradient accumulation; 768px default (1024 fits but is ~2x slower)

Expect ~4-10 s/step on a T4: a 1500-step LoRA is a multi-hour session. Use
deliberately - the DDPM and SD 1.5 backends are the cheap iteration paths.

Interface matches the other backends. Heavy imports stay inside functions.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .prompts import prompt_for
from .sd15_lora import _ImageDataset, neutralize_broken_torchao

ADAPTER_DIR_NAME = "adapter"
REPORT_NAME = "training_report.json"

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
FP16_VAE = "madebyollin/sdxl-vae-fp16-fix"


def _encode_prompt(prompt: str, tokenizers, text_encoders, device):
    """SDXL dual-encoder prompt embedding (computed once, reused every step)."""
    import torch

    embeds_list = []
    pooled = None
    with torch.no_grad():
        for tokenizer, encoder in zip(tokenizers, text_encoders):
            tokens = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            out = encoder(tokens, output_hidden_states=True)
            pooled = out[0]  # final pooled output of the *last* encoder wins
            embeds_list.append(out.hidden_states[-2])
    prompt_embeds = torch.cat(embeds_list, dim=-1)
    return prompt_embeds, pooled


# ------------------------------------------------------------------ training
def train(cfg, records, cls: str, out_dir: Path, opts: dict) -> dict:
    import torch
    import torch.nn.functional as F
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import (
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPTokenizer,
    )

    neutralize_broken_torchao()
    from peft import LoraConfig, get_peft_model

    steps = int(opts.get("steps", 1200))
    resolution = int(opts.get("resolution", 768))
    grad_accum = int(opts.get("grad_accum", 4))
    lr = float(opts.get("lr", 1e-4))
    seed = int(opts.get("seed", cfg.splits.seed))
    checkpoint_every = int(opts.get("checkpoint_every", 200))
    base = str(opts.get("base_model", BASE_MODEL))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    tokenizers = [
        CLIPTokenizer.from_pretrained(base, subfolder="tokenizer"),
        CLIPTokenizer.from_pretrained(base, subfolder="tokenizer_2"),
    ]
    text_encoders = [
        CLIPTextModel.from_pretrained(
            base, subfolder="text_encoder", torch_dtype=torch.float16
        ).to(device),
        CLIPTextModelWithProjection.from_pretrained(
            base, subfolder="text_encoder_2", torch_dtype=torch.float16
        ).to(device),
    ]
    vae = AutoencoderKL.from_pretrained(FP16_VAE, torch_dtype=torch.float16).to(device)
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet").to(device)
    scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler")

    vae.requires_grad_(False)
    for encoder in text_encoders:
        encoder.requires_grad_(False)
    unet.requires_grad_(False)
    unet.enable_gradient_checkpointing()

    lora = LoraConfig(
        r=int(opts.get("lora_rank", 8)),
        lora_alpha=int(opts.get("lora_alpha", 8)),
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora)
    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)

    prompt = prompt_for(cls)
    prompt_embeds, pooled = _encode_prompt(prompt, tokenizers, text_encoders, device)
    prompt_embeds = prompt_embeds.float()
    pooled = pooled.float()

    # Free the text encoders - embeddings are cached, and 16GB is tight.
    del text_encoders
    torch.cuda.empty_cache() if device == "cuda" else None

    # SDXL micro-conditioning: original size / crop / target size.
    add_time_ids = torch.tensor(
        [[resolution, resolution, 0, 0, resolution, resolution]],
        device=device, dtype=torch.float32,
    )

    dataset = _ImageDataset([r.path for r in records if r.exists], resolution)
    if len(dataset) == 0:
        raise RuntimeError(f"no training images on disk for class {cls!r}")
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad], lr=lr
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    scaling = vae.config.scaling_factor

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
            with torch.no_grad():
                latents = (
                    vae.encode(pixels.half()).latent_dist.sample().float() * scaling
                )
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device
            )
            noisy = scheduler.add_noise(latents, noise, timesteps)

            cond = {
                "text_embeds": pooled.expand(latents.shape[0], -1),
                "time_ids": add_time_ids.expand(latents.shape[0], -1),
            }
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                pred = unet(
                    noisy,
                    timesteps,
                    encoder_hidden_states=prompt_embeds.expand(latents.shape[0], -1, -1),
                    added_cond_kwargs=cond,
                ).sample
                loss = F.mse_loss(pred.float(), noise.float()) / grad_accum

            scaler.scale(loss).backward()
            if (step + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            losses.append(float(loss.detach()) * grad_accum)
            step += 1
            if step % 25 == 0 or step == steps:
                recent = sum(losses[-25:]) / len(losses[-25:])
                print(f"[sdxl_lora] {cls} step {step}/{steps} loss={recent:.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            if checkpoint_every and step % checkpoint_every == 0 and step < steps:
                unet.save_pretrained(str(adapter_dir))
                print(f"[sdxl_lora] {cls} checkpoint at step {step}", flush=True)

    unet.save_pretrained(str(adapter_dir))

    report = {
        "backend": "sdxl_lora",
        "cls": cls,
        "base_model": base,
        "vae": FP16_VAE,
        "prompt": prompt,
        "steps": steps,
        "resolution": resolution,
        "batch_size": 1,
        "grad_accum": grad_accum,
        "lr": lr,
        "seed": seed,
        "lora_rank": lora.r,
        "trainable_params": trainable,
        "train_images": len(dataset),
        "device": device,
        "loss_first25": round(sum(losses[:25]) / max(1, len(losses[:25])), 4),
        "loss_last25": round(sum(losses[-25:]) / max(1, len(losses[-25:])), 4),
        "duration_s": round(time.time() - t0, 1),
        "adapter_dir": str(adapter_dir),
    }
    (out_dir / REPORT_NAME).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# ------------------------------------------------------------------ sampling
def sample(cfg, cls: str, adapter_dir: Path, out_dir: Path, opts: dict) -> list[dict]:
    import torch
    from diffusers import AutoencoderKL, StableDiffusionXLPipeline

    neutralize_broken_torchao()
    from peft import PeftModel

    count = int(opts.get("count", cfg.generator.sample_count))
    steps = int(opts.get("sample_steps", 30))
    guidance = float(opts.get("guidance", cfg.generator.guidance))
    resolution = int(opts.get("resolution", 768))
    base_seed = int(opts.get("seed", cfg.splits.seed))
    base = str(opts.get("base_model", BASE_MODEL))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    vae = AutoencoderKL.from_pretrained(FP16_VAE, torch_dtype=dtype)
    pipe = StableDiffusionXLPipeline.from_pretrained(base, vae=vae, torch_dtype=dtype)
    merged = PeftModel.from_pretrained(pipe.unet, str(adapter_dir)).merge_and_unload()
    pipe.unet = merged
    pipe = pipe.to(device)
    pipe.enable_vae_slicing()
    pipe.set_progress_bar_config(disable=True)

    prompt = prompt_for(cls)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t0 = time.time()
    for i in range(count):
        seed = base_seed + i
        generator = torch.Generator(device=device).manual_seed(seed)
        image = pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=resolution,
            width=resolution,
            generator=generator,
        ).images[0]
        image_id = f"syn_{cls}_sdxl_{seed}"
        file_name = f"{image_id}.png"
        image.save(out_dir / file_name)
        rows.append(
            {
                "image_id": image_id,
                "file": file_name,
                "cls": cls,
                "backend": "sdxl_lora",
                "base_model": base,
                "checkpoint": str(adapter_dir),
                "seed": seed,
                "prompt": prompt,
                "guidance": guidance,
                "sample_steps": steps,
                "resolution": resolution,
            }
        )
        if (i + 1) % 5 == 0 or i + 1 == count:
            print(f"[sdxl-sample] {cls} {i + 1}/{count} ({time.time() - t0:.0f}s)", flush=True)
    return rows
