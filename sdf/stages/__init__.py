"""Pipeline stages. Each stage: (PipelineConfig, opts: dict|None) -> StageResult."""

from . import audit, probe_ml, sample, train_lora
from .base import StageResult, write_result  # noqa: F401

REGISTRY = {
    "audit": audit.run,
    "probe_ml": probe_ml.run,
    "train_lora": train_lora.run,
    "sample": sample.run,
}
