"""probe_ml stage: CPU-cheap verification of the ML execution environment.

Runs on Kaggle *before* any GPU stage is allowed to spend quota. Checks that
the heavy imports exist at usable versions, whether CUDA is present, and that
the configured base model is actually reachable on the Hugging Face Hub
(metadata only - no weight download).
"""

from __future__ import annotations

import time

from ..config import PipelineConfig
from ..gen.prompts import CLASS_PROMPTS
from .base import StageResult


def run(cfg: PipelineConfig, opts: dict | None = None) -> StageResult:
    started = time.time()
    metrics: dict = {"base_model": cfg.generator.base_model}
    problems: list[str] = []

    versions = {}
    for module in ("torch", "diffusers", "peft", "transformers", "huggingface_hub"):
        try:
            mod = __import__(module)
            versions[module] = getattr(mod, "__version__", "?")
        except ImportError as exc:
            versions[module] = None
            problems.append(f"import {module} failed: {exc}")
    metrics["versions"] = versions

    if versions.get("torch"):
        import torch

        metrics["cuda_available"] = torch.cuda.is_available()
        metrics["gpus"] = (
            [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else []
        )

    if versions.get("huggingface_hub"):
        from huggingface_hub import model_info

        try:
            info = model_info(cfg.generator.base_model)
            metrics["hub_model"] = {
                "id": info.id,
                "gated": bool(getattr(info, "gated", False)),
            }
            if getattr(info, "gated", False):
                problems.append(f"base model {cfg.generator.base_model} is gated")
        except Exception as exc:  # noqa: BLE001 - any hub failure blocks GPU spend
            problems.append(f"hub lookup failed for {cfg.generator.base_model}: {exc}")

    metrics["prompt_classes"] = sorted(CLASS_PROMPTS)

    for line in (
        f"[probe_ml] versions: {versions}",
        f"[probe_ml] cuda: {metrics.get('cuda_available')} {metrics.get('gpus', [])}",
        f"[probe_ml] hub: {metrics.get('hub_model', 'n/a')}",
    ):
        print(line, flush=True)

    return StageResult(
        stage="probe_ml",
        success=not problems,
        metrics=metrics,
        error="; ".join(problems),
        duration_s=round(time.time() - started, 2),
    )
