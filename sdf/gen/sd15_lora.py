"""SD 1.5 + LoRA backend: class-specific fine-tuning and sampling.

Runs on a Kaggle GPU kernel; torch/diffusers/peft come from the Kaggle image.
All heavy imports happen inside functions so the control machine (no torch)
can import and unit-test the surrounding plumbing.

Training: standard epsilon-prediction latent-diffusion objective. Everything
frozen except LoRA adapters on the UNet attention projections. fp16 autocast
(T4 has no bf16), gradient accumulation for an effective batch on 16GB.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .prompts import prompt_for

ADAPTER_DIR_NAME = "adapter"
REPORT_NAME = "training_report.json"


def neutralize_broken_torchao() -> bool:
    """Work around the Kaggle GPU image shipping torchao 0.10.0.

    peft >= 0.19 probes torchao inside its LoRA layer dispatcher and RAISES
    ImportError when the installed version is older than it supports - even
    though we never asked for torchao quantization. If the installed torchao
    is too old to be usable, patch peft's availability probes to report it as
    absent, which restores the correct 'no torchao' code path.

    Returns True if a patch was applied. Safe no-op everywhere else.
    """
    try:
        from importlib.metadata import version

        installed = version("torchao")
    except Exception:
        return False  # torchao absent - nothing to neutralize

    try:
        import peft.import_utils as import_utils

        probe = import_utils.is_torchao_available
        probe()  # healthy installs return bool; broken ones raise ImportError
        return False
    except ImportError:
        pass  # the exact failure we are here to fix
    except Exception:
        return False

    patched = False
    import peft.import_utils as import_utils

    import_utils.is_torchao_available = lambda: False
    patched = True
    try:  # the dispatcher imported the symbol into its own namespace
        import peft.tuners.lora.torchao as lora_torchao

        lora_torchao.is_torchao_available = lambda: False
    except Exception:
        pass
    print(f"[env] neutralized incompatible torchao {installed} for peft", flush=True)
    return patched


# ------------------------------------------------------------------ training
def train(cfg, records, cls: str, out_dir: Path, opts: dict) -> dict:
    """Fine-tune a LoRA for one class on its train-split records.

    Returns a JSON-safe report; writes the adapter + report under ``out_dir``.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

    neutralize_broken_torchao()
    from peft import LoraConfig, get_peft_model
    from transformers import CLIPTextModel, CLIPTokenizer

    gen = cfg.generator
    steps = int(opts.get("steps", gen.train_steps))
    resolution = int(opts.get("resolution", gen.resolution))
    batch_size = int(opts.get("batch_size", gen.batch_size))
    grad_accum = int(opts.get("grad_accum", gen.grad_accum))
    lr = float(opts.get("lr", gen.lr))
    seed = int(opts.get("seed", cfg.splits.seed))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    base = gen.base_model
    tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet").to(device)
    scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    lora = LoraConfig(
        r=int(opts.get("lora_rank", 16)),
        lora_alpha=int(opts.get("lora_alpha", 16)),
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora)
    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)

    # One fixed prompt per class -> encode once, reuse for every batch.
    prompt = prompt_for(cls)
    with torch.no_grad():
        tokens = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        prompt_embeds = text_encoder(tokens)[0]

    dataset = _ImageDataset([r.path for r in records if r.exists], resolution)
    if len(dataset) == 0:
        raise RuntimeError(f"no training images on disk for class {cls!r}")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False
    )

    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad], lr=lr
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    scaling = vae.config.scaling_factor

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
                latents = vae.encode(pixels).latent_dist.sample() * scaling
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device
            )
            noisy = scheduler.add_noise(latents, noise, timesteps)
            embeds = prompt_embeds.expand(latents.shape[0], -1, -1)

            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                pred = unet(noisy, timesteps, encoder_hidden_states=embeds).sample
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
                print(f"[train_lora] {cls} step {step}/{steps} loss={recent:.4f}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = out_dir / ADAPTER_DIR_NAME
    unet.save_pretrained(str(adapter_dir))

    report = {
        "cls": cls,
        "base_model": base,
        "prompt": prompt,
        "steps": steps,
        "resolution": resolution,
        "batch_size": batch_size,
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


def _ImageDataset(paths, resolution: int):
    """Tiny torch Dataset: centre-crop to square, resize, scale to [-1, 1]."""
    import torch
    from PIL import Image
    from torch.utils.data import Dataset

    class _DS(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, idx):
            with Image.open(paths[idx]) as img:
                img = img.convert("RGB")
                side = min(img.size)
                left = (img.width - side) // 2
                top = (img.height - side) // 2
                img = img.crop((left, top, left + side, top + side))
                img = img.resize((resolution, resolution), Image.LANCZOS)
                data = torch.frombuffer(
                    bytearray(img.tobytes()), dtype=torch.uint8
                ).reshape(resolution, resolution, 3)
            return (data.permute(2, 0, 1).float() / 127.5) - 1.0

    return _DS()


# ------------------------------------------------------------------ sampling
def sample(cfg, cls: str, adapter_dir: Path, out_dir: Path, opts: dict) -> list[dict]:
    """Generate images from a trained adapter. Returns manifest-ready dicts
    (without file writes to the manifest itself - the stage owns that)."""
    import torch
    from diffusers import StableDiffusionPipeline

    neutralize_broken_torchao()
    from peft import PeftModel

    gen = cfg.generator
    count = int(opts.get("count", gen.sample_count))
    steps = int(opts.get("sample_steps", gen.sample_steps))
    guidance = float(opts.get("guidance", gen.guidance))
    resolution = int(opts.get("resolution", gen.resolution))
    base_seed = int(opts.get("seed", cfg.splits.seed))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # safety_checker=None: the checker false-positives heavily on medical
    # imagery (skin close-ups) and returns black frames. These are private
    # research artifacts, provenance-tracked as synthetic; see design doc §7.
    pipe = StableDiffusionPipeline.from_pretrained(
        gen.base_model, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
    )
    merged = PeftModel.from_pretrained(pipe.unet, str(adapter_dir)).merge_and_unload()
    pipe.unet = merged
    pipe = pipe.to(device)
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
        image_id = f"syn_{cls}_{seed}"
        file_name = f"{image_id}.png"
        image.save(out_dir / file_name)
        rows.append(
            {
                "image_id": image_id,
                "file": file_name,
                "cls": cls,
                "backend": "sd15_lora",
                "base_model": gen.base_model,
                "checkpoint": str(adapter_dir),
                "seed": seed,
                "prompt": prompt,
                "guidance": guidance,
                "sample_steps": steps,
                "resolution": resolution,
            }
        )
        if (i + 1) % 10 == 0 or i + 1 == count:
            print(f"[sample] {cls} {i + 1}/{count} ({time.time() - t0:.0f}s)", flush=True)
    return rows
